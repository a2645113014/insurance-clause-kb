"""合规审查层可配置参数 —— Phase 3.2 合规审查链路的唯一切换面板

为什么单独建这个文件
--------------------
合规判定要找的不是「最相似的条目」，而是「**判定依据是否给全了**」。
这个区别决定了本层有两个必须可调的旋钮：

1. **喂几条进去（top_k）**：给太少 → 漏掉适用条目 → 误判合规；
   给太多 → 无关条目稀释判断 → 误判违规。取几最合适只能靠评测定，不能写死。
2. **兜底要不要开（通用 / 险种匹配）**：105 条里相当一部分 scope=通用，
   对任何产品都适用，但它们未必与待审表述语义相似 —— 只靠向量 top-k 会漏。
   兜底能补上，代价是条目数变多。两个方向都要能单独开关做对照。

取值优先级
----------
`.env` > 本文件默认值。解析器直接从 `retrieval_config` 复用，保证三个面板
（检索层 / 规则层 / 合规层）对 `.env` 的容错行为完全一致 ——
不会出现「检索层认 True、合规层认不了」这种坑。
"""

import os
from dataclasses import dataclass, asdict

from dotenv import load_dotenv

from app.conf.retrieval_config import _parse_bool, _parse_int
from app.core.logger import logger

load_dotenv()

# ---------------------------------------------------------------- 默认值
# 合规条目集合名。与 insurance_rules（理赔规则）、insurance_clauses（条款切片）
# 三者互不相干，各有各的读写封装。
DEFAULT_COMPLIANCE_COLLECTION = "compliance_rules"
# 合规工具总开关。关掉后合规分支短路，可以做「有工具 / 无工具」对照。
DEFAULT_COMPLIANCE_TOOL_ENABLED = True
# 向量检索取前几条。105 条里一次给 12 条：实测相关的通常 2~4 条，
# 留出余量给「语义不那么像但确实适用」的条目。
#
# ⚠️ 实测（2026-09-24 消融，17 条带金标用例）：这个旋钮**当前是死的**。
# k6 / k12 / k24 三组的召回条数都是 40.0 —— 恰好等于 MAX_ITEMS。因为 scope=通用
# 的条目有 59 条，通用兜底一路就能把清单填到上限，top_k 只改了顺序没改集合。
# 想让它生效，必须先降 MAX_ITEMS 或关掉 INCLUDE_UNIVERSAL，否则调它没有意义。
# 三组金标位次 2.71 / 2.82 / 3.18，差异在噪声量级。
DEFAULT_COMPLIANCE_TOP_K = 12
# 是否恒带 scope=通用 的条目。关掉它会让「条款文字冗长」这类通用条款
# 在语义不相似时被漏掉 —— 默认开。
#
# ⚠️ 实测悖论：关掉它（no_universal）召回率仍是 1.0（一条不漏），清单从 40 条
# 降到 14.5 条（省 64% 输入），金标位次还从 2.82 改善到 2.47。也就是说
# **在当前金标集上它是纯成本、零收益**。
# 但**默认保持开着**，因为金标集只有 17 条带金标用例，测不出「适用但语义不像」
# 的长尾 —— 而那正是通用兜底存在的理由。合规场景漏检的代价远大于多喂几条
# 噪音条目，所以宁可保留冗余。要关它请先扩充金标集。
DEFAULT_COMPLIANCE_INCLUDE_UNIVERSAL = True
# 是否按产品险种补充 scope 匹配的条目（医疗保险产品带上 scope=医疗保险 的条目）。
#
# ✅ 实测这是**唯一真正有取舍的旋钮**：关掉（no_scope）全金标召回率从 1.0
# 掉到 0.9412（漏掉 B01 的条 33），但金标位次从 2.82 改善到 1.69。
# 即「召回全 vs 金标靠前」的直接权衡 —— 默认开，因为合规优先保召回。
DEFAULT_COMPLIANCE_INCLUDE_SCOPE_MATCH = True
# 兜底之后的总条数上限。防止险种匹配过宽（如「健康保险」命中一大片）时
# 把 prompt 撑爆 —— 上限一到就按相似度分数截断。
#
# ⚠️ 实测 base 组 27 条用例**全部恰好 40 条**，说明这个上限每轮都被顶满，
# 它同时也是把 top_k 架空的元凶（见上）。
DEFAULT_COMPLIANCE_MAX_ITEMS = 40


# -------------------------------------------------------------- 配置类
@dataclass(frozen=True)
class ComplianceConfig:
    """合规层参数快照。frozen 的理由同前两层：开关必须在运行中不可变，
    否则「开兜底 / 关兜底」两轮的结果无法归因。"""

    collection: str
    tool_enabled: bool
    top_k: int
    include_universal: bool
    include_scope_match: bool
    max_items: int

    def as_dict(self) -> dict:
        """转成可 JSON 序列化的字典，供评测脚本记录本轮配置。"""
        return asdict(self)

    def signature(self) -> str:
        """配置指纹，用于给每轮判定评测的结果打标签。"""
        return (
            f"comp{int(self.tool_enabled)}"
            f"_k{self.top_k}"
            f"_uni{int(self.include_universal)}"
            f"_scope{int(self.include_scope_match)}"
            f"_max{self.max_items}"
            f"_{self.collection}"
        )


compliance_config = ComplianceConfig(
    collection=os.getenv("COMPLIANCE_COLLECTION") or DEFAULT_COMPLIANCE_COLLECTION,
    tool_enabled=_parse_bool(
        os.getenv("COMPLIANCE_TOOL_ENABLED"),
        DEFAULT_COMPLIANCE_TOOL_ENABLED,
        "COMPLIANCE_TOOL_ENABLED",
    ),
    top_k=_parse_int(
        os.getenv("COMPLIANCE_TOP_K"), DEFAULT_COMPLIANCE_TOP_K, "COMPLIANCE_TOP_K"
    ),
    include_universal=_parse_bool(
        os.getenv("COMPLIANCE_INCLUDE_UNIVERSAL"),
        DEFAULT_COMPLIANCE_INCLUDE_UNIVERSAL,
        "COMPLIANCE_INCLUDE_UNIVERSAL",
    ),
    include_scope_match=_parse_bool(
        os.getenv("COMPLIANCE_INCLUDE_SCOPE_MATCH"),
        DEFAULT_COMPLIANCE_INCLUDE_SCOPE_MATCH,
        "COMPLIANCE_INCLUDE_SCOPE_MATCH",
    ),
    max_items=_parse_int(
        os.getenv("COMPLIANCE_MAX_ITEMS"), DEFAULT_COMPLIANCE_MAX_ITEMS, "COMPLIANCE_MAX_ITEMS"
    ),
)

# 导入即打印生效值：跑对照时日志里必须能直接看到这轮用的什么配置
logger.info(f"合规层配置生效 | 指纹={compliance_config.signature()}")


if __name__ == "__main__":
    import json

    print(json.dumps(compliance_config.as_dict(), ensure_ascii=False, indent=2))
    print("signature:", compliance_config.signature())
