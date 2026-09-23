"""理赔规则工具 —— Phase 2.3 规则层对外暴露的两个可调用函数

为什么这两个问题值得做成「工具」而不是再走一次向量检索
------------------------------------------------------
「身故理赔要交什么材料」「多久能赔下来」这两个问题有**唯一确定的答案**，
写在条款的固定位置（3.3 保险金申请 / 3.4 保险金给付），一次键值查询就能取到。
用向量检索回答它，等于把一个精确问题退化成概率问题 —— 检索可能召回
另一款产品的材料清单，而模型未必能察觉。这就是总纲 §2.1 把
`insurance_rules` 放在 MongoDB 而不是 Milvus 的原因。

合规红线（固化在返回值里，不靠 prompt 叮嘱）
------------------------------------------
1. **不判断能不能赔**。「符不符合赔付条件」的证据在条款正文与理赔实务里，
   本规则表只答「要准备什么」与「多久处理」。
2. **不承诺金额**。规则表里根本没有金额数据 —— 这是刻意的，见 01_改造清单 §0。
3. **每个结果都带 `source_clause_no`**，答案可以且必须引用条款号。
4. **恒带 `disclaimer`**，且写明「不构成赔付承诺」。

查不到时的行为
--------------
不编造、不猜。返回 `found=False` + `note` 说明原因；如果产品有规则但没匹配上
指定的事故类型，会把**该产品覆盖的全部事故类型**一并返回，让上层能反问用户
而不是瞎答。
"""

import json
import os
import sys

if __name__ == "__main__":
    # 单文件自测时把项目根加进 path（被 import 时不需要）
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from app.clients.mongo_rule_utils import get_rule_mongo_tool
from app.conf.rule_config import rule_config

TOOL_REQUIRED_DOCS = "get_required_docs"
TOOL_CLAIM_TIMELINE = "get_claim_timeline"

DISCLAIMER_DOCS = (
    "以上为条款约定的申请材料清单，实际理赔以保险公司的核定要求为准。"
    "本结果不构成任何赔付承诺。"
)
DISCLAIMER_TIMELINE = (
    "以上为条款约定的理赔处理时限，实际时效以保险公司的核定结果为准。"
    "本结果不构成任何赔付承诺。"
)

# 口语 → 条款用词。用户不会说「身故保险金」，他会说「人没了」。
# 只做「查询词扩展」，不反过来改写条款原文 —— 保真优先。
ACCIDENT_ALIAS = {
    "死亡": ["身故"],
    "去世": ["身故"],
    "身故": ["身故"],
    "意外身故": ["身故"],
    "残疾": ["全残", "伤残"],
    "伤残": ["全残", "伤残"],
    "全残": ["全残"],
    "残废": ["全残", "伤残"],
    "满期": ["满期"],
    "到期": ["满期"],
    "住院": ["住院"],
    "门诊": ["门诊"],
    "医疗费": ["医疗费用"],
    "医药费": ["医疗费用"],
    "看病": ["医疗费用"],
    "重疾": ["重大疾病", "特定疾病"],
    "大病": ["重大疾病", "特定疾病"],
    "轻症": ["轻症"],
    "中症": ["中症"],
    "护理": ["护理"],
    "失能": ["护理"],
    "津贴": ["津贴"],
    "特药": ["特定药品"],
    "靶向药": ["特定药品"],
    "质子重离子": ["质子重离子"],
    "转诊": ["转诊"],
    "异地": ["异地"],
}

# 时限单位的中文表述。**必须区分工作日与自然日** ——
# 30 个工作日 ≈ 6 周，30 个自然日 = 1 个月，抹掉单位的答案会误导用户。
UNIT_TEXT = {"workday": "个工作日", "day": "个自然日"}

TIMELINE_LABELS = [
    ("settle", "核定"),
    ("settle_complex", "情形复杂的核定"),
    ("pay", "达成给付协议后给付"),
    ("reject_notice", "拒赔通知"),
]


def expand_accident_keywords(raw):
    """把用户口语里的事故类型扩展成一组候选关键词。

    「死亡」→ ["死亡", "身故"]；「重疾」→ ["重疾", "重大疾病", "特定疾病"]。
    扩展出来的键交给 Mongo 做「或」关系的子串匹配。
    """
    text = (raw or "").strip()
    if not text:
        return []
    keys = {text}
    for alias, targets in ACCIDENT_ALIAS.items():
        if alias in text:
            keys.update(targets)
    return [k for k in keys if k]


def _entry(row):
    """把一条 required_docs 规则整理成对外结构。source_span 保留完整 ——
    它是答案能引用条款号的依据，截断溯源就断了。"""
    return {
        "accident_type": row.get("accident_type"),
        "docs": row.get("docs") or [],
        "docs_count": row.get("docs_count"),
        "source_item_name": row.get("item_name"),
        "source_clause_no": row.get("source_clause_no"),
        "source_pk": row.get("source_pk"),
        "source_span": row.get("source_span"),
    }


def describe_timeline(row):
    """把时限四元组转成带单位和标签的人话。"""
    out = {}
    for key, label in TIMELINE_LABELS:
        value = row.get(f"{key}_value")
        unit = row.get(f"{key}_unit")
        if value is None:
            out[key] = None
            continue
        out[key] = {
            "label": label,
            "value": value,
            "unit": unit,
            "text": f"{label}：{value}{UNIT_TEXT.get(unit, unit or '')}",
        }
    return out


def get_required_docs(item_name, accident_type=None, insurance_type=None):
    """查「申请某项保险金需要哪些证明和资料」。

    :param item_name: 产品名称，必须与 insurance_clauses.item_name 同源
    :param accident_type: 事故类型，可口语化（「死亡」/「重疾」/「住院」）；
           留空则返回该产品全部事故类型的清单
    :param insurance_type: 险种，仅在产品级查不到时用于降级查询
    :return: dict。found=False 时看 note 与 available_accident_types，
             不要凭常识补答案。
    """
    result = {
        "tool": TOOL_REQUIRED_DOCS,
        "found": False,
        "matched_by": None,
        "item_name": item_name,
        "insurance_type": insurance_type,
        "query_accident_type": accident_type,
        "entries": [],
        "available_accident_types": [],
        "note": "",
        "disclaimer": DISCLAIMER_DOCS,
    }
    if not item_name:
        result["note"] = "未提供产品名称，无法查询。请先确认是哪款产品。"
        return result

    tool = get_rule_mongo_tool()
    keys = expand_accident_keywords(accident_type)

    rows = tool.find_by_item(item_name, "required_docs", keys)
    if rows:
        result["found"] = True
        result["matched_by"] = "item_name"
        result["entries"] = [_entry(r) for r in rows]
        return result

    # 产品有规则，但没匹配到用户问的事故类型 —— 把可选范围交出去，让上层反问
    all_rows = tool.find_by_item(item_name, "required_docs")
    if all_rows:
        result["available_accident_types"] = [r.get("accident_type") for r in all_rows]
        result["note"] = (
            f"该产品条款按事故类型分列材料清单，但没有与「{accident_type}」匹配的条目。"
            f"请从 available_accident_types 里确认用户问的是哪一种，不要自行推定。"
        )
        return result

    # 产品级完全没有 —— 视开关决定是否降级到险种级
    if rule_config.fallback_to_insurance_type and insurance_type:
        rows = tool.find_by_insurance_type(insurance_type, "required_docs", keys)
        if rows:
            # 只取一条代表。同险种下往往有多款产品，全给出去会让答案自相矛盾
            # ——用户读不出哪条适用于自己，模型也容易把它们揉成一份不存在的清单。
            row = rows[0]
            result["found"] = True
            result["matched_by"] = "insurance_type"
            result["entries"] = [_entry(row)]
            result["note"] = (
                f"「{item_name}」的条款未单列理赔材料。以下取自同险种（{insurance_type}）"
                f"产品「{row.get('item_name')}」的条款，仅供参考，"
                f"**不等同于该产品的约定**，回答时必须说明这一点。"
            )
            return result

    result["note"] = f"规则表中没有「{item_name}」的理赔材料规则。应拒答并建议用户查阅条款原件或咨询客服。"
    return result


def get_claim_timeline(item_name, insurance_type=None):
    """查「理赔要多久」—— 核定期限 / 复杂案件核定期限 / 给付期限 / 拒赔通知期限。

    :param item_name: 产品名称
    :param insurance_type: 险种，仅在产品级查不到时用于降级查询
    :return: dict。found=True 时 timeline 里每个非空项都带 text（含单位的成句表述）。
    """
    result = {
        "tool": TOOL_CLAIM_TIMELINE,
        "found": False,
        "matched_by": None,
        "item_name": item_name,
        "insurance_type": insurance_type,
        "timeline": {},
        "source_clause_no": None,
        "source_pk": None,
        "source_span": "",
        "note": "",
        "disclaimer": DISCLAIMER_TIMELINE,
    }
    if not item_name:
        result["note"] = "未提供产品名称，无法查询。请先确认是哪款产品。"
        return result

    tool = get_rule_mongo_tool()
    rows = tool.find_by_item(item_name, "claim_timeline")
    # 显式记降级来源，不靠 note 是否为空去猜
    matched_by = "item_name"

    if not rows and rule_config.fallback_to_insurance_type and insurance_type:
        rows = tool.find_by_insurance_type(insurance_type, "claim_timeline")
        if rows:
            matched_by = "insurance_type"

    if not rows:
        result["note"] = f"规则表中没有「{item_name}」的理赔时限规则。应拒答并建议用户咨询客服。"
        return result

    row = rows[0]
    result["found"] = True
    result["matched_by"] = matched_by
    if matched_by == "insurance_type":
        result["note"] = (
            f"「{item_name}」的条款未单列理赔时限。以下取自同险种（{insurance_type}）"
            f"产品「{row.get('item_name')}」的条款，**不等同于该产品的约定**，"
            f"回答时必须说明这一点。"
        )
    result["timeline"] = describe_timeline(row)
    result["source_clause_no"] = row.get("source_clause_no")
    result["source_pk"] = row.get("source_pk")
    result["source_span"] = row.get("source_span")
    return result


# ---------------------------------------------------------------- 自测
def _brief(res):
    """自测输出用：把结果压成几行，避免把 source_span 全打出来。"""
    lines = [f"tool={res['tool']} found={res['found']} matched_by={res['matched_by']} "
             f"item={res['item_name']}"]
    if res["tool"] == TOOL_REQUIRED_DOCS:
        for e in res["entries"]:
            lines.append(f"  [{e['accident_type']}] {e['docs_count']} 项 条款{e['source_clause_no']}")
            for d in e["docs"][:3]:
                lines.append(f"      - {d}")
            if e["docs_count"] > 3:
                lines.append(f"      …另 {e['docs_count'] - 3} 项")
        if res["available_accident_types"]:
            lines.append(f"  可选事故类型：{res['available_accident_types']}")
    else:
        for key, item in (res["timeline"] or {}).items():
            if item:
                lines.append(f"  {item['text']}")
        if res["source_clause_no"]:
            lines.append(f"  出条款 {res['source_clause_no']}")
    if res["note"]:
        lines.append(f"  note: {res['note']}")
    return "\n".join(lines)


if __name__ == "__main__":
    CASES = [
        ("身故（口语「死亡」）", lambda: get_required_docs("太保盈有余（2026A）年金保险（互联网）", "死亡")),
        ("不给事故类型", lambda: get_required_docs("太保盈有余（2026A）年金保险（互联网）")),
        ("事故类型对不上", lambda: get_required_docs("太保盈有余（2026A）年金保险（互联网）", "质子重离子")),
        ("最细的产品", lambda: get_required_docs("太保城市定制医保补充团体医疗保险（互联网）")),
        ("降级到险种级（养老年金·产品条款无材料规则）",
         lambda: get_required_docs("中英人寿福临门养老年金保险", "身故",
                                   insurance_type="年金保险")),
        ("降级到险种级·时限（产品不在规则表）",
         lambda: get_claim_timeline("某虚构年金保险", insurance_type="年金保险")),
        ("时限·太保系", lambda: get_claim_timeline("太保盈有余（2026A）年金保险（互联网）")),
        ("时限·中英系", lambda: get_claim_timeline("中英人寿福临门两全保险A款")),
        ("不存在的产品", lambda: get_required_docs("某不存在的产品")),
    ]
    for title, fn in CASES:
        print("=" * 70)
        print(f"### {title}")
        try:
            print(_brief(fn()))
        except Exception as e:
            print(f"  !! 异常：{e}")
