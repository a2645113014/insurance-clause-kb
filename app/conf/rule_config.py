"""规则层可配置参数 —— Phase 2.3 规则工具化的唯一切换面板

为什么单独建这个文件
--------------------
理赔规则工具（按产品 + 事故类型查材料清单 / 查时限）会接入查询图，一旦接入就
必须能关掉它做对照 —— 「有工具 vs 无工具」的检索与问答指标差异，是这个模块
唯一能证明自己价值的证据。把开关做进 `.env`，做对照时源码零改动。

另一层原因是**降级策略要可调**：17 款产品里有 3 款条款没单列理赔材料，
工具遇到它们时是「按险种取同类材料」还是「直接拒答」，这个口径会随
评测结论变化，不该写死在代码里。

取值优先级
----------
`.env` > 本文件默认值，口径与 `app/conf/retrieval_config.py` 完全一致
（解析器直接从那边复用，保证两个面板对 `.env` 的容错行为相同）。
"""

import os
from dataclasses import dataclass, asdict

from dotenv import load_dotenv

# 复用检索层的解析器：只读导入，不改变其行为。
# 这样 .env 里写成 "True"/"true"/"1" 在两层的行为完全一致，
# 不会出现「检索层的布尔能识别、规则层的识别不了」这种坑。
from app.conf.retrieval_config import _parse_bool, _parse_int
from app.core.logger import logger

load_dotenv()

# ---------------------------------------------------------------- 默认值
# 规则集合名。刻意与 Milvus 侧 insurance_clauses 区分开 ——
# 前者是结构化规则表（MongoDB），后者是条款原文切片（向量库）。
DEFAULT_RULE_COLLECTION = "insurance_rules"
# 规则工具总开关。关掉后规则分支直接短路，图退化为改造前的纯检索链路。
DEFAULT_RULE_TOOL_ENABLED = True
# 产品级查不到时的降级开关：
#   True  → 按 insurance_type 取同类产品的通用材料，返回值标注 matched_by
#   False → 直接返回空结果，由上层走拒答
DEFAULT_RULE_FALLBACK_TO_TYPE = True
# 意图识别至少命中几个关键词才触发规则分支。
# 取 1 是因为关键词表本身已经足够特异（「理赔材料」「多久能赔」这类短语
# 在非理赔问题里几乎不会出现）；调高会漏召，调低到 0 则等于全量走规则。
DEFAULT_RULE_MIN_KEYWORD_HITS = 1
# 理赔意图关键词表。命中的问题会被路由到规则工具而不是条款检索。
# 这份表是 Phase 2.3「最小接入」的妥协产物 —— Phase 3.3 上 Planner 后，
# 路由改由模型决策，届时这张表退化为兜底规则。
DEFAULT_RULE_INTENT_KEYWORDS = (
    "理赔材料", "理赔资料", "需要什么材料", "需要哪些材料", "要什么材料",
    "什么证明和资料", "索赔材料", "索赔资料", "理赔流程", "怎么理赔",
    "多久能赔", "多久赔付", "多久到账", "几天到账", "多长时间赔付",
    "赔付时限", "核定期限", "什么时候能赔", "多久能拿到钱",
)


def _parse_keywords(raw, default, name):
    """解析逗号分隔的关键词表，容忍中文逗号与多余空白。

    `.env` 常被中文输入法改坏，`，` 与 `,` 混用是常态。
    空串与解析失败都回落默认表 —— 关键词表为空会让规则分支永不触发，
    这是静默失效，比报错更难查，所以必须兜底。
    """
    if raw is None or str(raw).strip() == "":
        return default
    parts = [x.strip() for x in str(raw).replace("，", ",").split(",")]
    parts = [x for x in parts if x]
    if not parts:
        logger.warning(f"环境变量 {name} 解析后为空，回落默认关键词表（共 {len(default)} 条）")
        return default
    return tuple(parts)


# -------------------------------------------------------------- 配置类
@dataclass(frozen=True)
class RuleConfig:
    """规则层参数快照。frozen 的理由同检索层：开关必须在运行中不可变，
    否则「关掉工具再跑一遍」的结果无法归因。"""

    collection: str
    tool_enabled: bool
    fallback_to_insurance_type: bool
    min_keyword_hits: int
    intent_keywords: tuple

    def as_dict(self) -> dict:
        """转成可 JSON 序列化的字典，供评测脚本记录本轮配置。"""
        data = asdict(self)
        data["intent_keywords"] = list(self.intent_keywords)
        data["intent_keyword_count"] = len(self.intent_keywords)
        return data

    def signature(self) -> str:
        """配置指纹，用于给「有工具 vs 无工具」的对照跑分结果打标签。"""
        return (
            f"rule{int(self.tool_enabled)}"
            f"_fb{int(self.fallback_to_insurance_type)}"
            f"_kw{len(self.intent_keywords)}x{self.min_keyword_hits}"
            f"_{self.collection}"
        )


rule_config = RuleConfig(
    collection=os.getenv("RULE_COLLECTION") or DEFAULT_RULE_COLLECTION,
    tool_enabled=_parse_bool(os.getenv("RULE_TOOL_ENABLED"), DEFAULT_RULE_TOOL_ENABLED, "RULE_TOOL_ENABLED"),
    fallback_to_insurance_type=_parse_bool(
        os.getenv("RULE_FALLBACK_TO_INSURANCE_TYPE"),
        DEFAULT_RULE_FALLBACK_TO_TYPE,
        "RULE_FALLBACK_TO_INSURANCE_TYPE",
    ),
    min_keyword_hits=_parse_int(
        os.getenv("RULE_MIN_KEYWORD_HITS"), DEFAULT_RULE_MIN_KEYWORD_HITS, "RULE_MIN_KEYWORD_HITS"
    ),
    intent_keywords=_parse_keywords(
        os.getenv("RULE_INTENT_KEYWORDS"), DEFAULT_RULE_INTENT_KEYWORDS, "RULE_INTENT_KEYWORDS"
    ),
)

# 导入即打印生效值：跑对照实验时，日志里必须能直接看到这轮用的什么配置
logger.info(f"规则层配置生效 | 指纹={rule_config.signature()} | 关键词数={len(rule_config.intent_keywords)}")


if __name__ == "__main__":
    import json

    print(json.dumps(rule_config.as_dict(), ensure_ascii=False, indent=2))
    print("signature:", rule_config.signature())
