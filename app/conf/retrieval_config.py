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
DEFAULT_RERANK_GAP_ABS = 2.0


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
)

# 导入即打印生效值：消融跑分日志里必须能直接看到这轮用的什么配置
logger.info(f"检索层配置生效 | 指纹={retrieval_config.signature()} | {retrieval_config.as_dict()}")


if __name__ == "__main__":
    import json
    print(json.dumps(retrieval_config.as_dict(), ensure_ascii=False, indent=2))
    print("signature:", retrieval_config.signature())
