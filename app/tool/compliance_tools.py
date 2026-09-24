"""合规审查工具 —— Phase 3.2 合规审查层对外暴露的可调用函数

它做什么、不做什么
------------------
做：给定一段**待审条款表述**，从 105 条负面清单里召回可能适用的条目，
    并装配成 prompt 可直接吃下的一段文本。
不做：**不下判定**。「违规 / 合规 / 无法判定」由 LLM 按
    `prompts/compliance_check.prompt` 逐条比对后给出 —— 判定标准是监管条文，
    不是相似度分数，工具层越权下判定会把合规审查退化成检索排序。

合规红线（固化在返回值与装配文本里，不靠 prompt 单方面叮嘱）
------------------------------------------------------------
1. **子集声明必须与数据同行**。判定标准只给了筛出来的一小撮，如果只说
   「逐条比对给定条目，都不违反就是合规」，那么**检索漏召回会直接表现为
   把违规判成合规** —— 这是掩盖漏召回的方向性错误，比答不出来危险得多。
   所以装配文本末尾恒定附一句「这是从 105 条中检索出的子集，未列出不代表
   不适用；依据不足一律判无法判定」。
2. **恒带 `disclaimer`**：本判定不构成法律意见或监管认定结论。
3. **召回为空时显式说空**，不允许把空清单当作「查过了，没问题」。
4. **条号原样透传**。`item_no` 是答案引用条号的唯一依据，不得重排或改号 ——
   prompt 明确禁止编造条号，工具层一旦改号，模型引用得再准也是错的。

为什么格式化函数放在工具层（与规则层的做法相反）
------------------------------------------------
规则层把 `_fmt_*` 放在 `node_rule_tool.py`，因为那边有唯一的消费节点。
合规层目前没有节点（图接入留待 Phase 3.3），消费方有两个 ——
`test/13_compliance_eval.py` 与将来的合规节点 —— 且它产出的文本就是
prompt 的输入契约（`{negative_list_items}` 占位符）。放在工具层才能保证
两条路径看到的是同一套文本，不会各自拼一遍。
"""

import json
import os
import sys

if __name__ == "__main__":
    # 单文件自测时把项目根加进 path（被 import 时不需要）
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from app.clients.mongo_compliance_utils import get_compliance_tool
from app.conf.compliance_config import compliance_config
from app.core.logger import logger

TOOL_COMPLIANCE_CHECK = "check_compliance"

DISCLAIMER_COMPLIANCE = (
    "以上为基于《人身保险产品负面清单（2026版）》条文的文本比对结果，"
    "不构成法律意见，也不能替代监管机构的认定结论。合规定稿请以监管原文为准。"
)

# 装配进 prompt 的子集声明。措辞照抄 `insurance_kb/12_compliance_schema.json`
# 的 `retrieval.why_subset_declaration` 的要义：把漏召回从「静默放过」转成
# 「显式说不知道」。改这里等于改合规审查的漏检率，动之前先想清楚。
SUBSET_NOTE_TEMPLATE = (
    "⚠️ 以上 {retrieved} 条是从全量 {total} 条负面清单中**检索出的子集**，"
    "未列出不代表不适用。逐条比对后若无一条构成违规，且你确信本次召回已覆盖"
    "待审表述所涉全部方面，方可判「合规」；只要依据不足以支撑结论，"
    "一律判「无法判定」并说明缺少什么依据。"
)


def embed_text(text):
    """把待审表述编码成 bge-m3 稠密向量。失败返回 None。

    延迟 import：`app.lm.embedding_utils` 会拉起 2GB 级模型，
    只在真正要检索时才加载，避免 import 本模块就触发模型初始化。

    为什么编码失败不抛异常：一次编码失败不该让整条合规审查链路崩掉。
    调用方拿到 None 会跳过向量路、只走通用/险种兜底 —— 给出的是粗清单，
    但配合子集声明仍然安全（判不出就判「无法判定」）。
    """
    try:
        from app.lm.embedding_utils import generate_embeddings

        result = generate_embeddings([text])
        dense = result.get("dense") or []
        return dense[0] if dense else None
    except Exception as e:
        logger.error(f"待审表述向量化失败：{e}", exc_info=True)
        return None


def check_compliance(clause_text, product_scope=None, **overrides):
    """取「判定这段表述是否违反负面清单」所需的依据条目。

    :param clause_text: 待审的条款表述原文（逐字，不要概括 —— 概括会丢掉
            构成要件里最关键的限定语）
    :param product_scope: 产品险种，如「医疗保险」「年金保险」。用于险种兜底，
            留空则只走向量 + 通用两路
    :param overrides: 覆盖配置面板取值（top_k / include_universal /
            include_scope_match / max_items / categories），仅供评测脚本做消融；
            生产调用一律不传。其中 categories 是**刻意没进配置面板**的
            大类过滤，理由见 `mongo_compliance_utils.search`。
    :return: dict。items 为空表示**没有召回到任何依据**，此时不得判「合规」。
    """
    result = {
        "tool": TOOL_COMPLIANCE_CHECK,
        "found": False,
        "clause_text": clause_text,
        "product_scope": product_scope,
        "items": [],
        "retrieved": 0,
        "total_in_list": 0,
        "picked_by_counts": {},
        "subset_note": "",
        "note": "",
        "disclaimer": DISCLAIMER_COMPLIANCE,
    }

    if not clause_text or not clause_text.strip():
        result["note"] = "未提供待审条款表述，无法检索判定依据。"
        return result

    if not compliance_config.tool_enabled:
        # 关掉工具时**不是**返回「无违规」，而是明确告知依据不可得。
        # 这两者在合规语义上天差地别。
        result["note"] = "合规工具已关闭（COMPLIANCE_TOOL_ENABLED=false），本次未检索任何判定依据。"
        return result

    try:
        tool = get_compliance_tool()
    except Exception as e:
        result["note"] = f"合规条目库不可用：{e}。本次未检索任何判定依据，应判「无法判定」。"
        logger.error(result["note"], exc_info=True)
        return result

    total = tool.rules.count_documents({})
    result["total_in_list"] = total
    if total == 0:
        result["note"] = (
            "合规条目库为空（未执行 test/12_build_compliance_rules.py 导入）。"
            "本次未检索任何判定依据，应判「无法判定」。"
        )
        logger.warning(result["note"])
        return result

    query_vector = embed_text(clause_text)
    if query_vector is None:
        # 记下来：本次结果里没有任何一条走向量路，召回质量天然打折
        result["note"] = "待审表述向量化失败，本次仅用通用/险种兜底召回，召回可能不完整。"

    items = tool.search(
        query_vector,
        product_scope=product_scope,
        top_k=overrides.get("top_k"),
        include_universal=overrides.get("include_universal"),
        include_scope_match=overrides.get("include_scope_match"),
        max_items=overrides.get("max_items"),
        categories=overrides.get("categories"),
    )

    result["items"] = items
    result["retrieved"] = len(items)
    result["found"] = bool(items)
    counts = {}
    for it in items:
        counts[it["picked_by"]] = counts.get(it["picked_by"], 0) + 1
    result["picked_by_counts"] = counts

    if items:
        result["subset_note"] = SUBSET_NOTE_TEMPLATE.format(retrieved=len(items), total=total)
    else:
        # 空召回必须与「召回了但都不适用」区分开 —— prompt 硬性规则第 4 条
        # 正是靠这个区分来决定判「无法判定」而不是「合规」。
        result["subset_note"] = (
            f"本次未从全量 {total} 条负面清单中召回任何条目。"
            "**不得据此判为「合规」**，应判「无法判定」并说明缺少判定依据。"
        )
    return result


def build_negative_list_block(result):
    """把 `check_compliance` 的结果装配成 prompt 的 `{negative_list_items}` 文本。

    格式刻意保持紧凑：`【条 12】产品条款表述｜适用范围：通用` + 换行 + 原文。
    条号必须写得显眼 —— prompt 要求 reason 里引用条号，条号不显眼模型就会
    在长文本里数错。
    """
    items = result.get("items") or []
    if not items:
        return result.get("subset_note") or "（本次未召回任何负面清单条目）"

    lines = []
    for it in items:
        # 标注位次便于人工复核；模型侧无意义，但不影响它读取条号
        lines.append(
            f"【条 {it['item_no']}】{it.get('category_name') or ''}"
            f"｜适用范围：{it.get('scope') or '未标注'}"
        )
        lines.append(f"  {it.get('text') or ''}")
    if result.get("note"):
        lines.append(f"注：{result['note']}")
    # 子集声明放在最后一行 —— prompt 的硬性规则读到这里，与数据贴得最近
    lines.append(result.get("subset_note") or "")
    return "\n".join([ln for ln in lines if ln is not None])


# ---------------------------------------------------------------- 自测
def _brief(res):
    """自测输出用：把结果压成几行，不打印条目全文。"""
    lines = [
        f"tool={res['tool']} found={res['found']} retrieved={res['retrieved']}"
        f"/{res['total_in_list']} picked_by={res['picked_by_counts']}"
    ]
    for it in res["items"][:8]:
        lines.append(
            f"  【条 {it['item_no']}】sim={it['similarity']} "
            f"by={it['picked_by']} scope={it['scope']} | {(it['text'] or '')[:38]}"
        )
    if res["retrieved"] > 8:
        lines.append(f"  …另 {res['retrieved'] - 8} 条")
    if res.get("note"):
        lines.append(f"  note: {res['note']}")
    return "\n".join(lines)


if __name__ == "__main__":
    CONFIG = compliance_config.as_dict()
    print("配置指纹:", compliance_config.signature())
    print(json.dumps(CONFIG, ensure_ascii=False))

    CASES = [
        # 与条 15「变相增加保险金给付条件」构成要件高度重合，应当召回该条
        ("疑似违规·分期给付身故保险金",
         "被保险人身故后，本公司将按本合同约定的标准分期给付身故保险金，"
         "而非一次性全额给付。"),
        # 与条 16「视同续保侵害选择权」重合
        ("疑似违规·未申请视同续保",
         "本产品保险期间届满时，若本公司未收到投保人的不续保申请，则视同续保一年。",
         "医疗保险"),
        # 正常条款表述，不应有强相关条目
        ("正常表述·犹豫期",
         "自投保人签收保险单之日起十五日内为犹豫期，投保人在犹豫期内可以解除本合同。"),
        # 空输入
        ("空输入", ""),
    ]
    for case in CASES:
        title, text = case[0], case[1]
        scope = case[2] if len(case) > 2 else None
        print("=" * 72)
        print(f"### {title} | scope={scope}")
        try:
            print(_brief(check_compliance(text, scope)))
        except Exception as e:
            print(f"  !! 异常：{e}")
