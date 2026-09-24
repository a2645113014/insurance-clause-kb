"""合规审查节点 —— Phase 3.3 Planner 分流的合规分支

在查询图里的位置
----------------
    node_item_name_confirm ─┬─(合规意图 + 待审表述在问题里)──> node_compliance_check ─┐
                            ├─(已有 answer)──────────────────────────────────────> node_answer_output
                            ├─(理赔意图)──> node_rule_tool ──────────────────────> node_answer_output
                            └─(否则)──> 三路检索 ─> rrf ─> rerank ─┬─(合规意图)─> node_compliance_check
                                                                    └─(其余)───> node_answer_output

待审表述从哪来（两条路径，一条节点吃下）
----------------------------------------
1. **用户直接给**：问题里就带着条款原文（「帮我审查这段条款：……」）。
   这条路径**不需要产品名** —— 审查对象是这段表述本身，与它属于哪款产品无关。
2. **检索取回**：用户只说「审查《XX》第 3.3 条」，先走三路检索把条款捞出来，
   再拿 top 切片当待审表述。这条路径复用检索链路，不另建一套召回。

判定为什么放在本节点而不是工具层
--------------------------------
`check_compliance` 只保证「依据给全了」，判定必须由模型对照条文原文做 ——
判定标准是监管条文，不是相似度分数。工具层越权下判定会把合规审查退化成
检索排序，那是这类系统最隐蔽的失效方式。

三条容易被削掉的合规红线（本节点负责守住）
------------------------------------------
1. **子集声明必须真的进 prompt**。`check_compliance` 已把「这是子集，未列出
   不代表不适用」写进 `subset_note`，但如果装配时忘了拼进去，模型就会把
   「不在清单里」读成「合规」—— 漏召回被洗成合规，方向恰好是最危险的一侧。
   本节点只经由 `build_negative_list_block()` 取文本，不自己拼。
2. **召回为空必须显式说空**。空清单 ≠ 查过了没问题。工具层已区分，
   本节点据此短路为「无法判定」，不去调模型（没有依据的判定必然是编的）。
3. **免责声明由代码追加，不指望模型**。`compliance_check.prompt` 硬性规则第 8 条
   要求只输出 JSON，所以声明不能写在 prompt 里让模型带出来 —— 那会和
   「只输出 JSON」直接冲突，模型只能二选一。改为节点在模型输出之后追加，
   归属明确：模型负责判定，代码负责合规声明。
"""

import json
import os
import re
import sys

if __name__ == "__main__":
    # 单文件自测时把项目根加进 path（被当作模块 import 时不需要）
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from app.clients.milvus_utils import get_milvus_client
from app.conf.compliance_config import compliance_config
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.nodes.node_rule_tool import _lookup_insurance_type
from app.query_process.agent.state import QueryGraphState
from app.tool.compliance_tools import (
    DISCLAIMER_COMPLIANCE,
    check_compliance,
    build_negative_list_block,
)
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_done_task, add_running_task

# 判定用 prompt 名（对应 prompts/compliance_check.prompt）
COMPLIANCE_PROMPT_NAME = "compliance_check"

# ---------------------------------------------------------------- 意图判定
# 判定分两层：**动作词**说明用户想干合规这件事，**对象词**说明审的是条款表述。
# 只命中动作词不成立 —— 「检查一下这款产品的保险期间」里「检查」是日常用语，
# 审的不是合规性；必须两层同时命中才路由到合规分支。
COMPLIANCE_ACTION_ROOTS = (
    "合规", "违规", "负面清单", "监管要求", "监管规定", "监管红线",
    "审查", "审核", "判定", "检查", "是否符合规定", "是否违反",
)
COMPLIANCE_TARGET_ROOTS = (
    "条款", "表述", "措辞", "文字", "内容", "这句话", "这段话", "这段",
    "这一条", "这条", "以下", "下列",
)
# 「负面清单」本身就是合规审查的专有名词，单独出现即成立，不必再配对象词
COMPLIANCE_STRONG_ROOTS = ("负面清单",)

# 内联待审表述的最小长度。低于这个长度不会是条款原文，
# 而是「这段文字合规吗」这类指代 —— 那种情况应该走检索取回原文。
INLINE_MIN_CHARS = 24
# 从检索结果拼待审表述时的字符上限，防止把 prompt 挤爆
MAX_CLAUSE_CHARS = 3000

# 内联表述的提取：带引号的整段、或显式引导语之后的部分
_QUOTE_PATTERN = re.compile(r"[「『\"“]([^」』\"”]{8,})[」』\"”]")
# 引导语：动词 + 可选指代 + 对象词 + 冒号，冒号之后就是待审表述
_LEAD_PATTERN = re.compile(
    r"(?:审查|审核|检查|判定|看看|帮我审|请审)"
    r"(?:一下|下|这)?(?:段|条|个)?(?:以下|下列)?"
    r"(?:条款|表述|措辞|文字|内容|话|句子)?\s*[：:]\s*"
)
# 条款号引用（「第 3.3 条」「第六条」）。它是**定位信息**不是待审表述本身 ——
# 命中它说明用户指的是某条已有条款，得先检索取回原文，不能当成原文。
_CLAUSE_REF_PATTERN = re.compile(r"第\s*[0-9一二三四五六七八九十百零〇.]+\s*条")
# 待审表述几乎必然是完整句子，句号是最稳的判别特征；没有句号的多半是
# 「审查《X》第 3.3 条」这类指令 + 定位，应当走检索。
_SENTENCE_END = "。"


def is_compliance_query(state: QueryGraphState) -> bool:
    """图的路由判断：这个问题该不该走合规审查分支。

    工具关闭时一律返回 False —— 合规问题退化为普通条款问答，
    而不是让节点去说「工具关了」。路由与开关同步，语义才自洽。
    """
    if not compliance_config.tool_enabled:
        return False
    query = state.get("rewritten_query") or state.get("original_query") or ""
    if not query:
        return False
    if any(r in query for r in COMPLIANCE_STRONG_ROOTS):
        return True
    has_action = any(r in query for r in COMPLIANCE_ACTION_ROOTS)
    if not has_action:
        return False
    # 对象词命中，或直接引用了某个条款号（「审查《XX》第 3.3 条」）——
    # 后者没有「条款」二字，但审查意图毫无歧义
    has_target = (
        any(r in query for r in COMPLIANCE_TARGET_ROOTS)
        or bool(_CLAUSE_REF_PATTERN.search(query))
    )
    # 用户直接贴了整段条款原文时，也不必再要求出现对象词 ——
    # 「审查：申请身故保险金时须提供火化证明……」里没有「条款」二字，
    # 但意图毫无歧义。
    return has_target or extract_clause_text(query) is not None


def extract_clause_text(query):
    """从问题里抽出用户直接给出的待审表述。抽不到返回 None。

    按可靠性从高到低试四种写法：
    1. 整段带引号 —— 边界最明确，用户主动划定范围
    2. 显式引导语 + 冒号 —— 「审查以下条款：……」
    3. 冒号后一段完整句子，且冒号前是短指令 —— 「这条表述是否违规：……」
    4. 整体就够长且含句号 —— 用户直接贴了原文，没加任何引导
    都不成立返回 None，调用方据此改走「检索取回原文」那条路径。
    """
    text = (query or "").strip()
    if not text:
        return None

    # 1. 带引号的整段：取最长的一段（用户可能同时引了条号与正文）
    quoted = [m.strip() for m in _QUOTE_PATTERN.findall(text)]
    quoted = [q for q in quoted if len(q) >= INLINE_MIN_CHARS]
    if quoted:
        return max(quoted, key=len)

    # 2. 引导语之后的部分
    m = _LEAD_PATTERN.search(text)
    if m:
        rest = text[m.end():].strip()
        if len(rest) >= INLINE_MIN_CHARS:
            return rest

    # 3. 冒号后取尾段。限定「冒号前是短指令」是因为待审表述自身也可能带冒号，
    #    那种情况下冒号后的内容不是一段独立表述，切了就丢上下文。
    tail_match = re.search(r"[：:]\s*([^：:]{%d,})\s*$" % INLINE_MIN_CHARS, text)
    if tail_match and len(text[:tail_match.start()].strip()) <= 40:
        return tail_match.group(1).strip()

    # 4. 整体即原文。要求含句号 —— 否则「审查《X》第 3.3 条」这种
    #    「指令 + 条款号」也会被当成原文，把定位信息当审查对象。
    #    末尾仍是提问口气（吗 / ？）的说明用户在指代某段文字，不算给出原文。
    if (
        len(text) >= INLINE_MIN_CHARS
        and _SENTENCE_END in text
        and not text.endswith(("吗", "？", "?"))
    ):
        return text
    return None


def _clause_refs(query):
    """抽出问题里引用的条款号（「第 2.3 条」→ `2.3`）。

    语料侧 `clause_no` 实测 966/966 都是规范 `N.M`（原文里的「第六条」已被
    归一化），所以这里只取数字点号形态；中文数字的引用匹配不上，会在下面
    自然退化为「不做过滤」，不会误删切片。
    """
    return [m.strip() for m in re.findall(r"第\s*([0-9][0-9.]*)\s*条", query or "")]


def _lookup_clause_by_refs(item_name, refs):
    """按「产品 + 条款号」直接定位切片（确定性查询，不走向量召回）。

    为什么用户点明条款号时不该走向量检索：待审对象是唯一确定的，而向量检索
    给的是「语义最像的几条」。实测拿「审查《XX》第 1.4 条」去检索，回来的
    是 1.1 与 2.6 —— 审的对象直接错了。这与规则层同样的取舍：能用确定性
    键值查到的，就不要交给概率模型。

    查不到返回空字符串，调用方退化为用 rerank 结果（宁可审得宽，不可审错）。
    """
    if not item_name or not refs:
        return ""
    try:
        client = get_milvus_client()
        if client is None:
            return ""
        refs_expr = ", ".join(f'"{escape_milvus_string(r)}"' for r in refs)
        rows = client.query(
            collection_name=milvus_config.chunks_collection,
            filter=(
                f'item_name == "{escape_milvus_string(item_name)}" '
                f"and clause_no in [{refs_expr}]"
            ),
            output_fields=["clause_no", "clause_title", "chunk_seq", "text"],
            limit=16,
            consistency_level="Strong",
        )
    except Exception as e:
        logger.warning(f"按条款号定位切片失败（{item_name} / {refs}）：{e}")
        return ""

    # 同一 clause_no 可能被三级切分拆成多片，按 (条款号, 片序) 排序后原样拼回
    rows.sort(key=lambda r: (str(r.get("clause_no") or ""), r.get("chunk_seq") or 0))
    parts, used = [], 0
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        head = f"【第 {r.get('clause_no')} 条】"
        title = r.get("clause_title")
        if title:
            head += f"（{title}）"
        piece = f"{head}{text}"
        if used + len(piece) > MAX_CLAUSE_CHARS:
            break
        parts.append(piece)
        used += len(piece)
    return "\n\n".join(parts)


def _clause_text_from_docs(docs, refs=None):
    """从检索结果拼出待审表述。只取本地条款切片，联网条目一律排除。

    为什么排除 web：合规审查的判定依据必须是条款原文。联网检索回来的可能是
    营销软文或二手解读，拿它当「待审表述」会把审查对象本身搞错 ——
    那比判错结论更离谱。条款号随正文一起带上，便于用户核对审的是哪一条。

    `refs` 非空时只取条款号命中的切片 —— 用户说「审查第 2.3 条」，
    就该只审 2.3，把整款产品 top-5 切片拼成一坨送去审等于审了整个产品。
    命中为空则不过滤（宁可多审，不可漏审）。
    """
    pool = [d for d in (docs or []) if d.get("source") != "web"]
    if refs:
        hit = [d for d in pool if (d.get("clause_no") or "").strip() in refs]
        if hit:
            pool = hit
    parts, used = [], 0
    for doc in pool:
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        clause_no = doc.get("clause_no") or ""
        head = f"【第 {clause_no} 条】" if clause_no else ""
        piece = f"{head}{text}"
        if used + len(piece) > MAX_CLAUSE_CHARS:
            break
        parts.append(piece)
        used += len(piece)
    return "\n\n".join(parts)


def _parse_verdict(answer):
    """把模型输出解析成 dict。解析失败返回 None（不抛异常）。

    模型偶尔仍会裹 ```json 代码块，虽然 prompt 第 8 条明令禁止 ——
    挡住即可，没必要为此让整条链路失败。
    """
    if not answer:
        return None
    text = answer.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    try:
        data = json.loads(text)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _fmt_answer(verdict, tool_result, source):
    """把判定结果整理成对用户可读的答复，并追加合规免责声明。

    模型只输出 JSON（prompt 硬性规则 8），直接丢给用户是一坨裸 JSON。
    这里补一层人话铺陈，JSON 原文原样保留在末尾 —— 结构化结果要能被
    下游解析，可读性交给这层。
    """
    lines = []
    if not verdict:
        lines.append("合规判定未能生成结构化结论，以下为本次可用的判定依据与说明。")
    else:
        label = verdict.get("verdict") or "无法判定"
        item_no = verdict.get("violated_item_no")
        lines.append(f"【判定结论】{label}")
        if item_no is not None and label == "违规":
            lines.append(f"【违反条目】负面清单第 {item_no} 条")
        reason = verdict.get("reason")
        if reason:
            lines.append(f"【判定依据】{reason}")
        suggestion = verdict.get("suggestion")
        if suggestion:
            lines.append(f"【修改建议】{suggestion}")

    # 召回为空时把「为什么判不了」说清楚，而不是让用户以为查过了没问题
    if not tool_result.get("found"):
        lines.append(
            f"【关于依据】本次未从全量 {tool_result.get('total_in_list', 0)} 条负面清单中"
            "检索到可用条目，因此不应据此认为该表述合规。"
        )
    if tool_result.get("note"):
        lines.append(f"【提示】{tool_result['note']}")

    source_label = {
        "inline": "用户在问题中直接给出",
        "clause_no": "按产品与条款号从条款库精确取回",
        "retrieval": "从条款库检索取回",
    }.get(source, source)
    lines.append(f"【审查对象来源】{source_label}")
    lines.append("")
    lines.append("【结构化结果】")
    lines.append(json.dumps(verdict, ensure_ascii=False) if verdict else "null")
    lines.append("")
    lines.append(DISCLAIMER_COMPLIANCE)
    return "\n".join(lines)


@step_log("step_1_resolve_clause_text")
def step_1_resolve_clause_text(state: QueryGraphState):
    """确定待审表述与来源标记。返回 (clause_text, source, refs)。

    来源优先级：用户给的原文 > 按条款号确定性定位 > 检索取回的 top 切片。
    越靠前的越确定，确定性最高的那条排在第一。
    """
    query = state.get("rewritten_query") or state.get("original_query") or ""
    inline = extract_clause_text(query)
    if inline:
        return inline, "inline", []

    refs = _clause_refs(query)
    item_names = state.get("item_names") or []
    if refs and item_names:
        exact = _lookup_clause_by_refs(item_names[0], refs)
        if exact:
            return exact, "clause_no", refs

    retrieved = _clause_text_from_docs(state.get("reranked_docs") or [], refs)
    return retrieved, "retrieval", refs


@node_log("node_compliance_check")
def node_compliance_check(state: QueryGraphState):
    """对照负面清单判定待审表述，把结果写入 answer。

    无论走哪条返回路径都必须调 add_done_task —— 早期返回漏掉它会把任务
    永久卡在「运行中」（HyDE 节点踩过这个坑，这里用 try/finally 兜住）。
    """
    add_running_task(state["session_id"], "node_compliance_check", state["is_stream"])
    try:
        clause_text, source, refs = step_1_resolve_clause_text(state)
        logger.info(
            f"合规审查 | 待审表述来源={source} | 长度={len(clause_text or '')} "
            f"| 条款号引用={refs} | 配置指纹={compliance_config.signature()}"
        )
        if not clause_text:
            answer = (
                "未能取得待审的条款表述，无法进行合规审查。\n\n"
                "请用以下任一方式提供：\n"
                "1. 直接把条款原文贴在问题里（可用引号括起）；\n"
                "2. 指明产品名称与条款号（如「审查《XX 保险》第 3.3 条」），"
                "我从条款库取回原文后判定。\n\n"
                f"{DISCLAIMER_COMPLIANCE}"
            )
            logger.warning("合规审查：待审表述为空，直接返回提示")
            return {"compliance_result": {}, "compliance_context": "", "answer": answer}

        # 产品险种用于「按险种兜底召回」。取不到不影响主路径（向量路 + 通用兜底）。
        item_names = state.get("item_names") or []
        product_scope = _lookup_insurance_type(item_names[0]) if item_names else None

        tool_result = check_compliance(clause_text, product_scope)
        negative_list_block = build_negative_list_block(tool_result)
        logger.info(
            f"合规审查召回 | 条目={tool_result['retrieved']}/{tool_result['total_in_list']} "
            f"| 来源分布={tool_result['picked_by_counts']} | 险种={product_scope}"
        )

        # 空召回短路：没有依据的判定必然是编的，不去调模型
        verdict = None
        if tool_result.get("found"):
            prompt = load_prompt(
                COMPLIANCE_PROMPT_NAME,
                clause_text=clause_text,
                negative_list_items=negative_list_block,
            )
            state["prompt"] = prompt
            try:
                response = get_llm_client().invoke(prompt)
                verdict = _parse_verdict(response.content)
            except Exception as e:
                logger.error(f"合规判定调用模型失败：{e}", exc_info=True)
            if tool_result.get("found") and verdict is None:
                logger.warning("合规判定未返回可解析的 JSON，答案将只含依据与说明")
        else:
            logger.warning("合规审查：未召回任何依据，跳过模型判定")

        # 模型判定与工具结果一起入 state：前者供前端读结论，后者供溯源
        result = dict(tool_result)
        result["judgment"] = verdict
        result["source"] = source
        result["clause_refs"] = refs
        # 审查对象原文一并入 state：合规审查必须能回答「你到底审的是哪段字」，
        # 只留判定结论的审计链是断的
        result["clause_text"] = clause_text
        result["clause_text_chars"] = len(clause_text)
        result["product_scope"] = product_scope

        answer = _fmt_answer(verdict, tool_result, source)
        return {
            "compliance_result": result,
            "compliance_context": negative_list_block,
            "answer": answer,
        }
    finally:
        add_done_task(state["session_id"], "node_compliance_check", state["is_stream"])


if __name__ == "__main__":
    # 单文件自测：只验路由判定与待审表述抽取，不连数据库、不调模型
    CASES = [
        "帮我审查这段条款：「申请身故保险金时，申请人除提供死亡证明外，还须提供火化证明与丧葬证明。」",
        "审查以下条款：本产品保险期间届满时，本公司如未收到投保人的不续保申请，则视同续保。",
        "这条表述是否违反监管要求：因细菌或病毒感染引发的保险事故，本公司不承担给付责任。",
        "负面清单里关于受益人的规定是什么",
        "审查《太保福有余（2025）终身寿险》第 3.3 条",
        "这款产品的犹豫期是多久",
        "保险期间是几年",
        "检查一下这款产品的等待期规定",
        "这段文字合规吗",
    ]
    for q in CASES:
        state = {"rewritten_query": q}
        inline = extract_clause_text(q)
        print(
            f"{'★' if is_compliance_query(state) else ' '} {q}\n"
            f"    合规意图={is_compliance_query(state)} "
            f"内联表述={'有(' + str(len(inline)) + '字)' if inline else '无'}\n"
        )
