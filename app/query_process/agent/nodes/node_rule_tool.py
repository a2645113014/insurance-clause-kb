"""规则工具节点 —— Phase 2.3 的最小接入

在查询图里的位置
----------------
    node_item_name_confirm ─┬─(理赔意图)─> node_rule_tool ──────────────> node_answer_output
                            ├─(已有 answer)────────────────────────────> node_answer_output
                            └─(否则)────> 三路检索 ─> rrf ─> rerank ─> node_answer_output

为什么命中规则就**跳过检索**
----------------------------
「身故理赔要交什么材料」的答案在条款 3.3 里唯一确定。再走一遍向量检索不但没有
增益，反而有风险：检索可能召回**另一款产品**的材料条款，而模型未必察觉两份清单
来自不同产品 —— 那会给出一个看起来完整、实际错误的合规答案。
确定性查询的价值就在于绕过这个概率环节。

边界
----
本节点是「最小接入」：路由靠关键词表（`rule_config.intent_keywords` 判是否理赔类，
本文件的两组提示词细分是问材料还是问时限）。Phase 3.3 上 Planner 后由模型决策
路由，这里的逻辑退化为兜底。
"""

import os
import sys
from functools import lru_cache

if __name__ == "__main__":
    # 单文件自测时把项目根加进 path（被当作模块 import 时不需要）
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.conf.rule_config import rule_config
from app.core.logger import logger, node_log, step_log
from app.query_process.agent.state import QueryGraphState
from app.tool.insurance_rule_tools import (
    ACCIDENT_ALIAS,
    TOOL_CLAIM_TIMELINE,
    TOOL_REQUIRED_DOCS,
    get_claim_timeline,
    get_required_docs,
)
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_done_task, add_running_task

# 意图判定用「词根组合」而不是长短语精确匹配。
# 长短语很脆：「多久能拿到钱」匹配不上「理赔要多久**才**能拿到钱」——少一个「才」就漏判。
# 拆成词根后，「多久」「材料」这类稳定片段能覆盖各种问法。
CLAIM_ROOTS = ("理赔", "赔付", "索赔", "保险金申请", "给付")
DOCS_ROOTS = ("材料", "资料", "证明", "证件", "文件", "手续")
TIMELINE_ROOTS = ("多久", "多长时间", "几天", "多少天", "时限", "期限", "什么时候", "多长")
# 事故类型词。单独出现不构成理赔意图，但与材料/时限词根同时出现时就成立 ——
# 「住院了要交什么材料」一句没提「理赔」，意图却毫无歧义。
ACCIDENT_ROOTS = ("住院", "门诊", "身故", "全残", "伤残", "重疾", "重大疾病",
                  "满期", "护理", "死亡", "医疗费用", "特药", "靶向")

# 塞进 prompt 的原文依据长度上限。太长会挤占检索上下文的预算，
# 太短又不足以让模型引用具体条款表述。
SPAN_PREVIEW_CHARS = 320

# 事故类型别名按长度倒序 —— 「意外身故」必须先于「身故」被匹配到，
# 否则只会返回「身故」这个更宽泛的别名。
_ALIAS_SORTED = sorted(ACCIDENT_ALIAS, key=len, reverse=True)


def detect_intent(query):
    """判断问题是不是理赔类，以及问的是材料还是时限。

    判定分两层：
    1. **必须先有理赔语义**（`CLAIM_ROOTS`）。否则「这款产品的犹豫期是多久」
       会因为命中「多久」而被误路由到规则表 —— 而犹豫期不是理赔问题，
       它的答案在条款正文里，该走检索。
    2. 再看问的是材料还是时限（`DOCS_ROOTS` / `TIMELINE_ROOTS`），两者都命中就都查。
       都没命中时（如「理赔流程是怎样的」）按问材料处理。

    `rule_config.intent_keywords` 是补充路径，用于词根覆盖不到的说法。

    :return: (命中的配置关键词, 类型列表)。类型列表为空表示不走规则分支。
    """
    text = query or ""
    if not any(r in text for r in CLAIM_ROOTS):
        return [], []

    kinds = []
    if any(r in text for r in DOCS_ROOTS):
        kinds.append("docs")
    if any(r in text for r in TIMELINE_ROOTS):
        kinds.append("timeline")

    hits = [kw for kw in rule_config.intent_keywords if kw in text]

    if kinds:
        return hits, kinds
    # 是理赔问题但没问材料也没问时限 —— 由配置的关键词表兜底
    if len(hits) >= max(1, rule_config.min_keyword_hits):
        return hits, ["docs"]
    return [], []


def extract_accident_type(query):
    """从问题里找事故类型线索。找到返回别名原文，否则 None。

    只做「用户话 → 查询词」的映射，不改变条款原文用词。
    """
    text = query or ""
    for alias in _ALIAS_SORTED:
        if alias in text:
            return alias
    return None


def is_rule_query(state: QueryGraphState) -> bool:
    """图的路由判断：这个问题该不该走规则工具。

    两个必要条件：理赔意图 + 已识别出产品名。
    没有产品名就没法做产品级规则查询 —— 而「理赔要什么材料」这类问题的答案
    恰恰是产品特有的，此时交给检索链路更合适。
    """
    if not rule_config.tool_enabled:
        return False
    if not (state.get("item_names") or []):
        return False
    hits, kinds = detect_intent(state.get("rewritten_query") or state.get("original_query"))
    return bool(kinds)


@lru_cache(maxsize=256)
def _lookup_insurance_type(item_name):
    """从条款库查产品所属险种。降级查询需要它。

    查不到返回 None —— 降级路径自然禁用，不影响产品级主路径。
    """
    try:
        client = get_milvus_client()
        if client is None:
            return None
        rows = client.query(
            collection_name=milvus_config.chunks_collection,
            filter=f'item_name == "{escape_milvus_string(item_name)}"',
            output_fields=["insurance_type"],
            limit=1,
            consistency_level="Strong",
        )
        return rows[0].get("insurance_type") if rows else None
    except Exception as e:
        logger.warning(f"查询产品险种失败（{item_name}）：{e}")
        return None


def _fmt_docs(res):
    """把 get_required_docs 的结果转成供模型阅读的文本。

    未命中时**必须把 available_accident_types 一起写进去** —— note 里让模型
    「从可选类型里反问用户」，但如果这个列表没进 context，模型就只能干看着
    一句做不到的指示，最后要么拒答要么瞎猜。指令必须与数据同行。
    """
    if not res.get("found"):
        lines = [f"【理赔材料】查询未命中。{res.get('note')}"]
        avail = res.get("available_accident_types") or []
        if avail:
            lines.append(f"该产品条款覆盖的事故类型：{'、'.join(avail)}")
        lines.append(f"免责声明：{res['disclaimer']}")
        return "\n".join(lines)

    lines = []
    for e in res["entries"]:
        lines.append(
            f"事故类型：{e['accident_type']}"
            f"（依据条款 {e['source_clause_no']}，来源产品：{e['source_item_name']}）"
        )
        for i, d in enumerate(e["docs"], 1):
            lines.append(f"  {i}. {d}")
        # 原文依据逐条给。「重疾」会扩展成「重大疾病/特定疾病」而命中多条，
        # 只打第一条的原文会让另外几条的条款号失去出处。
        span = e.get("source_span") or ""
        if span:
            lines.append(f"原文依据：{span[:SPAN_PREVIEW_CHARS]}")
    if res.get("note"):
        lines.append(f"⚠️ {res['note']}")
    lines.append(f"免责声明：{res['disclaimer']}")
    return "\n".join(lines)


def _fmt_timeline(res):
    """把 get_claim_timeline 的结果转成供模型阅读的文本。"""
    if not res.get("found"):
        # 未命中也要带免责声明：模型可能据此回答，红线不能因为分支不同而漏掉
        return f"【理赔时限】查询未命中。{res.get('note')}\n免责声明：{res['disclaimer']}"
    lines = [f"产品：{res['item_name']}（依据条款 {res['source_clause_no']}）"]
    for _key, item in (res.get("timeline") or {}).items():
        if item:
            lines.append(f"  - {item['text']}")
    if res.get("note"):
        lines.append(f"⚠️ {res['note']}")
    lines.append(f"原文依据：{(res.get('source_span') or '')[:SPAN_PREVIEW_CHARS]}")
    lines.append(f"免责声明：{res['disclaimer']}")
    return "\n".join(lines)


@node_log("node_rule_tool")
def node_rule_tool(state: QueryGraphState):
    """理赔材料 / 理赔时限走 MongoDB 确定性查询，结果写入 rule_context。

    无论走哪条路径都必须调 add_done_task —— 早期返回漏掉它会把任务永久
    卡在「运行中」（HyDE 节点踩过这个坑，这里用 try/finally 兜住）。
    """
    add_running_task(state["session_id"], "node_rule_tool", state["is_stream"])
    try:
        if not rule_config.tool_enabled:
            logger.info("规则工具已关闭（RULE_TOOL_ENABLED=false），本节点直接返回空")
            return {"rule_tool_result": {}, "rule_context": ""}

        query = state.get("rewritten_query") or state.get("original_query") or ""
        item_names = state.get("item_names") or []
        item_name = item_names[0] if item_names else None
        accident_type = extract_accident_type(query)
        hits, kinds = detect_intent(query)

        if not item_name:
            logger.info("未识别出产品名，规则查询跳过")
            return {"rule_tool_result": {}, "rule_context": ""}

        insurance_type = _lookup_insurance_type(item_name)
        logger.info(
            f"规则查询 | 产品={item_name} | 险种={insurance_type} | "
            f"事故类型线索={accident_type} | 命中关键词={hits} | 类型={kinds}"
        )

        blocks, tool_results = [], {}
        if "docs" in kinds:
            res = get_required_docs(item_name, accident_type, insurance_type)
            tool_results[TOOL_REQUIRED_DOCS] = res
            blocks.append(_fmt_docs(res))
        if "timeline" in kinds:
            res = get_claim_timeline(item_name, insurance_type)
            tool_results[TOOL_CLAIM_TIMELINE] = res
            blocks.append(_fmt_timeline(res))

        context = "\n\n".join(blocks)
        logger.info(f"规则查询完成 | 命中工具={list(tool_results)} | context 长度={len(context)}")
        return {"rule_tool_result": tool_results, "rule_context": context}
    finally:
        add_done_task(state["session_id"], "node_rule_tool", state["is_stream"])


if __name__ == "__main__":
    # 单文件自测：验证意图判断与事故类型抽取，不依赖图
    CASES = [
        "身故理赔需要提交哪些材料？",
        "理赔要多久才能拿到钱？",
        "得了重疾理赔需要什么资料，多久能赔下来",
        "这款产品的犹豫期是多久",
        "帮我推荐一款适合我妈的保险",
        "身故理赔需要什么材料",  # 配合 item_names 才走规则
    ]
    for q in CASES:
        h, k = detect_intent(q)
        print(f"{q}\n    intent={k} accident={extract_accident_type(q)} hits={h}\n")
