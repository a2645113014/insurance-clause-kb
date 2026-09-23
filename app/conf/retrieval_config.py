"""检索层可配置参数 —— 消融实验的唯一切换面板

为什么单独建这个文件
--------------------
改造前这些阈值散落在各节点里硬编码：混合检索权重写死在
`node_search_embedding.py` 第 45 行，断崖 TopK 的四个阈值写死在
`node_rerank.py` 第 8-14 行。做消融（对比不同权重/不同 TopK 策略）
必须改代码 —— 而改代码这个动作本身就会引入变量，测出来的差异到底来自
配置还是来自手抖，事后分不清。

集中到一处并从 `.env` 读取后，消融只需要改 `.env` 一行、重启服务、跑评测，
源码零改动，跑分可复现。

取值优先级
----------
`.env` > 本文件默认值。默认值取「推荐值」，`.env` 里写「当前部署值」，
两者允许不同。模块导入时会打印实际生效值与配置指纹 `signature()`，
用于给每轮消融跑分结果标注配置来源（溯源用）。
"""

import os
from dataclasses import dataclass, asdict

from dotenv import load_dotenv

from app.core.logger import logger

load_dotenv()

# ---------------------------------------------------------------- 默认值
# 混合检索加权：依次对应 (稠密向量, 稀疏向量)，各自必须落在 [0, 1]
DEFAULT_RANKER_WEIGHTS = (0.6, 0.4)
# HyDE 分支总开关 —— 关掉它可以直接量出「HyDE 到底有没有用」
DEFAULT_HYDE_ENABLED = True
# 混合检索最终返回条数（送进 RRF 的候选数）
DEFAULT_RETRIEVAL_LIMIT = 5
# 单路（稠密 / 稀疏）召回条数，须 >= RETRIEVAL_LIMIT，否则融合无料可用
DEFAULT_REQ_LIMIT = 10
# 断崖式动态 TopK 的四个阈值
DEFAULT_RERANK_MAX_TOPK = 10
DEFAULT_RERANK_MIN_TOPK = 1
DEFAULT_RERANK_GAP_RATIO = 0.5
# 改造前的硬编码值就是 2。这里保持 2 作为默认以复现原行为，
# 路线图建议试 1（条款切片比商品手册自包含，分数分布更紧凑），
# 该试验值写在 .env 里，由消融实验去验证它到底该取几。
# 注意：reranker 返回的是 logits（实测 -0.85 ~ -7.32），这个绝对阈值实际几乎不触发，
# 真正起作用的是 RERANK_GAP_RATIO —— 做消融时别只盯 gap_abs。
DEFAULT_RERANK_GAP_ABS = 2.0

# ------------------------------------------------- 联网检索参与方式（三态）
# 为什么需要这个开关
# ----------------
# 改造前 web_search_docs 与本地条款切片被混进同一个 rerank 池统一打分，
# 但两者量纲与可信度都不同：bge-reranker 对「网页标题 + 营销文风」系统性给正分
# （实测 +2.59 / +3.49 / +3.81），对规范条款句式给负 logits（实测 -0.85 ~ -7.32），
# 于是联网软文稳定占据 top1，再叠加断崖 TopK 把本地切片清零 ——
# 模型最终拿一段互联网营销文案当条款依据，答出与条款原文相悖、且引用编号伪造的结论。
# 保险条款合规问答要求「以条款原文为唯一依据」，因此默认行为必须是：
# 本地条款命中时，联网结果一律不进上下文。
#
# 三态取值
# --------
#   off       完全不发起联网检索（最省，也最隔绝外网不确定性）
#   fallback  照常检索，但只有当本地条款一条都没命中时才拿它兜底（默认）
#   always    与本地结果一起进上下文（改造前行为，仅留给消融做对照）
DEFAULT_WEB_SEARCH_MODE = "fallback"
VALID_WEB_SEARCH_MODES = ("off", "fallback", "always")
# fallback / always 模式下，联网结果最多带进上下文几条。
# 取 2 是因为它只承担「条款库没有的周边信息」这一窄职责，不需要更多。
DEFAULT_WEB_MAX_TOPK = 2


# ------------------------------------------------------------ 解析工具
def _parse_float(raw, default, name):
    """解析浮点配置。解析失败告警并回落默认值，绝不让 .env 写坏整条链路。"""
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except Exception:
        logger.warning(f"环境变量 {name} 解析失败（原值 {raw!r}），回落默认值 {default}")
        return default


def _parse_int(raw, default, name):
    """解析整型配置。容忍 '3.0' 这种写法（部分编辑器会自动补小数位）。"""
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(float(str(raw).strip()))
    except Exception:
        logger.warning(f"环境变量 {name} 解析失败（原值 {raw!r}），回落默认值 {default}")
        return default


def _parse_bool(raw, default, name):
    """解析布尔配置，接受 1/true/yes/on 与 0/false/no/off（大小写不敏感）。"""
    if raw is None or str(raw).strip() == "":
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    logger.warning(f"环境变量 {name} 取值无法识别（原值 {raw!r}），回落默认值 {default}")
    return default


def _parse_enum(raw, default, name, allowed):
    """解析枚举配置（如 WEB_SEARCH_MODE）。

    非法取值一律回落默认值并告警，不做「猜你想写的是哪个」的模糊匹配 ——
    拼错的开关必须被看见，否则会静默退回默认行为，让消融结论对不上号。
    """
    if raw is None or str(raw).strip() == "":
        return default
    text = str(raw).strip().lower()
    if text not in allowed:
        logger.warning(f"环境变量 {name} 取值必须是 {allowed} 之一（实得 {raw!r}），回落默认值 {default}")
        return default
    return text


def _parse_weights(raw, default, name="RANKER_WEIGHTS"):
    """解析 '0.8,0.2' → (0.8, 0.2)。

    校验规则刻意宽松：只要求是两个数、各自落在 [0, 1]。
    不强制和为 1 —— `WeightedRanker` 本身不要求，且消融时可能想试
    (0.6, 0.5) 这类非归一组合。但和明显不为 1 时会告警，
    避免对照表上出现「不知道为什么权重和是 1.1」的组合。
    """
    if raw is None or str(raw).strip() == "":
        return default
    # 容忍中文逗号：.env 常在中文输入法下被改
    parts_raw = str(raw).replace("，", ",").split(",")
    try:
        parts = [float(x.strip()) for x in parts_raw]
    except Exception:
        logger.warning(f"环境变量 {name} 解析失败（原值 {raw!r}），回落默认值 {default}")
        return default
    if len(parts) != 2:
        logger.warning(f"环境变量 {name} 需要恰好两个权重（稠密,稀疏），实得 {len(parts)} 个，回落默认值 {default}")
        return default
    if any(p < 0 or p > 1 for p in parts):
        logger.warning(f"环境变量 {name} 的权重必须落在 [0,1]，实得 {parts}，回落默认值 {default}")
        return default
    if abs(sum(parts) - 1.0) > 1e-6:
        logger.warning(f"环境变量 {name} 两项之和为 {sum(parts):.4f}，不为 1；WeightedRanker 允许，但请确认这是有意的")
    return (parts[0], parts[1])


# -------------------------------------------------------------- 配置类
@dataclass(frozen=True)
class RetrievalConfig:
    """检索层参数快照。frozen 是为了防止运行中被某处悄悄改掉 —— 消融跑分必须可溯源。"""

    ranker_weights: tuple
    hyde_enabled: bool
    retrieval_limit: int
    req_limit: int
    rerank_max_topk: int
    rerank_min_topk: int
    rerank_gap_ratio: float
    rerank_gap_abs: float
    # 联网检索参与方式：off / fallback / always，见文件顶部说明
    web_search_mode: str
    web_max_topk: int

    def as_dict(self) -> dict:
        """转成可 JSON 序列化的字典，供评测脚本落盘记录本轮配置。"""
        data = asdict(self)
        data["ranker_weights"] = list(self.ranker_weights)
        return data

    def signature(self) -> str:
        """配置指纹，用于给消融跑分结果命名/打标签，一眼看出这行是哪个配置跑出来的。"""
        w = self.ranker_weights
        return (
            f"w{w[0]:g}-{w[1]:g}"
            f"_hyde{int(self.hyde_enabled)}"
            f"_gapabs{self.rerank_gap_abs:g}"
            f"_gapratio{self.rerank_gap_ratio:g}"
            f"_lim{self.retrieval_limit}"
            f"_web{self.web_search_mode}"
            f"_webmax{self.web_max_topk}"
        )


retrieval_config = RetrievalConfig(
    ranker_weights=_parse_weights(os.getenv("RANKER_WEIGHTS"), DEFAULT_RANKER_WEIGHTS),
    hyde_enabled=_parse_bool(os.getenv("HYDE_ENABLED"), DEFAULT_HYDE_ENABLED, "HYDE_ENABLED"),
    retrieval_limit=_parse_int(os.getenv("RETRIEVAL_LIMIT"), DEFAULT_RETRIEVAL_LIMIT, "RETRIEVAL_LIMIT"),
    req_limit=_parse_int(os.getenv("RETRIEVAL_REQ_LIMIT"), DEFAULT_REQ_LIMIT, "RETRIEVAL_REQ_LIMIT"),
    rerank_max_topk=_parse_int(os.getenv("RERANK_MAX_TOPK"), DEFAULT_RERANK_MAX_TOPK, "RERANK_MAX_TOPK"),
    rerank_min_topk=_parse_int(os.getenv("RERANK_MIN_TOPK"), DEFAULT_RERANK_MIN_TOPK, "RERANK_MIN_TOPK"),
    rerank_gap_ratio=_parse_float(os.getenv("RERANK_GAP_RATIO"), DEFAULT_RERANK_GAP_RATIO, "RERANK_GAP_RATIO"),
    rerank_gap_abs=_parse_float(os.getenv("RERANK_GAP_ABS"), DEFAULT_RERANK_GAP_ABS, "RERANK_GAP_ABS"),
    web_search_mode=_parse_enum(os.getenv("WEB_SEARCH_MODE"), DEFAULT_WEB_SEARCH_MODE, "WEB_SEARCH_MODE", VALID_WEB_SEARCH_MODES),
    web_max_topk=_parse_int(os.getenv("WEB_MAX_TOPK"), DEFAULT_WEB_MAX_TOPK, "WEB_MAX_TOPK"),
)

# 导入即打印生效值：消融跑分日志里必须能直接看到这轮用的什么配置
logger.info(f"检索层配置生效 | 指纹={retrieval_config.signature()} | {retrieval_config.as_dict()}")


if __name__ == "__main__":
    import json
    print(json.dumps(retrieval_config.as_dict(), ensure_ascii=False, indent=2))
    print("signature:", retrieval_config.signature())
