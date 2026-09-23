"""构建理赔规则表（MongoDB insurance_rules）—— Phase 2.3 的数据侧入口

做两件事
--------
1. `claim_timeline`：从「保险金给付」条款抽理赔时限（正则，覆盖 17/18）
2. `required_docs` ：从材料条款抽材料清单（正则，覆盖 17/17 款产品）

为什么最终全用正则，而不是 LLM 抽材料清单
------------------------------------------
最初的设计是「时限走正则、材料走 LLM」，理由是材料清单各家表述不同。实际探底后
发现**表述差异只有三种，都能写成正则**（见 TITLE_PATTERNS），加上按 clause_no
把被切散的片拼回来，即可全覆盖：

    太保系   生存保险金申请所需的证明和资料
    乐享系   3.3.1一般医疗费用保险金的申请
    中英系   1、申请身故保险金时

既然规则能做到全覆盖且逐字可溯，就没有理由引入 LLM 的额外不确定性
（幻觉风险、调用成本、外部依赖）。`insurance_kb/07_prompts_v2/claim_docs_extract.prompt`
保留下来，供后续 0.6B 微调做条款要素抽取时复用 —— 但构建规则表这一步不需要它。

⚠️ 2026-09-23 修正：不要按 clause_type 过滤取池
---------------------------------------------
第一版的取池条件是 `clause_type == "保险金申请"`（18 片，覆盖 14/17 款产品）。
这个条件是错的 —— 材料条款的 clause_type 标签本身就不准：
「保险金及保险费豁免申请」被打成 `保险费`、「如何申请保险金」被打成 `其他`。
于是 3 款产品的材料清单虽然写着，却因为取不到而只能靠险种级降级或直接拒答：

    太保福有余（2025）终身寿险   3.3  type=保险费  漏抽
    太保阿基米德（2025）重大疾病   3.3  type=保险费  漏抽
    中英人寿福临门养老年金保险    6.3  type=其他    漏抽

改为**全量取池 + 材料形态门**后：规则 43 → 50 条，覆盖 14/17 → 17/17，
且原有 43 条逐字不变（零回归）。过滤职责由正则承担，见 MATERIAL_GUARD。

两个必须处理的语料事实
----------------------
1. **长条款被 L2 切散**：L1 切片超过上限字符数时会下探 L2，一条「保险金申请」
   被切成 3 片，而小标题只在第一片里。不把同 clause_no 的片拼回来，
   后两片的材料会静默漏掉（实测太保乐享无忧、太保城市定制都是三片结构）。
2. **时限有「工作日 / 日」两种单位**：混用会算错（30 个工作日 ≠ 30 天）。

反幻觉门
--------
`source_span` 直接取自原文切片，天然满足「逐字命中」；脚本仍做断言校验，
一旦不通过即丢弃并留痕 —— 这条门是给未来替换抽取方式时留的保险。

用法（在项目根目录执行）
----------------------
    .venv\\Scripts\\python.exe test/11_build_claim_rules.py --dry-run     # 只抽取不写库
    .venv\\Scripts\\python.exe test/11_build_claim_rules.py               # 抽取并写库
    .venv\\Scripts\\python.exe test/11_build_claim_rules.py --only docs   # 只跑材料清单
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 沙箱代理会劫持 gRPC 到本地 Milvus 的连接，必须在导入客户端前清掉
for _k in list(os.environ):
    if _k.lower() in {"http_proxy", "https_proxy", "all_proxy", "grpc_proxy", "no_proxy"}:
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# 必须在清代理之后再 load_dotenv，否则 .env 里的代理变量又会回来
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from app.clients.milvus_utils import get_milvus_client
from app.clients.mongo_rule_utils import get_rule_mongo_tool
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

SCHEMA_VERSION = "1.0"
OUT_ROOT = os.path.join(ROOT, "output")

FIELDS = [
    "pk", "doc_id", "item_name", "insurance_type", "company", "product_code",
    "clause_type", "clause_no", "clause_no_norm", "clause_title", "text",
]


def norm_ws(text):
    """归一化空白。PDF 转出来的条款里满是换行与断词空格，
    比对 source_span 时必须先抹平，否则「逐字命中」会被无关空格判死。
    """
    return re.sub(r"\s+", "", text or "")


# ============================================================ 材料清单：标题识别
# 三种标题写法。配合全量取池 + MATERIAL_GUARD，实测覆盖 17/17 款产品。
# 每条正则的捕获组即事故类型（如「身故保险金」）。
TITLE_PATTERNS = [
    # 太保系：生存保险金申请所需的证明和资料
    # 排除「：:」是为了避开它前面的引导句
    # 「…并提供下列证明和资料：生存保险金申请所需的证明和资料」——
    # 不排除的话会把「并提供下列证明和资料」整体吞成事故类型。
    re.compile(r"([^；。，、\s：:]{2,30}?)申请所需的证明和资料"),
    # 乐享系：3.3.1一般医疗费用保险金的申请  /  3.3.2恶性肿瘤质子重离子医疗费用保险金的申请
    re.compile(r"\d+(?:\.\d+){2}([^（(；。，]{2,40}?)的申请"),
    # 中英系：1、申请身故保险金时，由身故保险金受益人作为申请人…
    re.compile(r"\d+、申请([^，,；。]{2,20}?)时"),
]

# 材料条目：(1)保险合同；  —— 全角/半角括号都要认
# 捕获到「；」或「。」为止；条目内部允许出现释义引用（见13.13）这类小括号
DOC_ITEM_RE = re.compile(r"[（(]\d+[)）]((?:(?![（(]\d+[)）])[^；;。])*)")

# 事故类型前缀清洗：命中组里可能带小标题编号，如「3.3.2疾病身故保险金」
ACC_PREFIX_RE = re.compile(r"^\d+(?:\.\d+){0,3}")

# PDF 抽文阅读顺序错乱的检测。
# 正常形态：「身故保险金申请所需的证明和资料」——「申」「请」紧挨着。
# 错乱形态：「身故保险金申(1)保险合同；请所需的证明和资料」——表格列被插进了标题中间。
# 后果很严重：这个标题识别不到，它的材料会被并进**前一个**事故类型，
# 于是「满期保险金」的清单里出现了「死亡证明」——会给出错误的合规答案。
# 这类段落必须整条丢弃，不能靠猜顺序去修：修完 span 就不再逐字来自原文，溯源断掉。
BROKEN_TITLE_RE = re.compile(r"保险金申(?=.{1,60}?请所需的证明和资料)")

# 材料形态门（2026-09-23 新增）。全量取池后必须有这道门，否则会抽到假材料。
#
# 背景：条款里除了「申请材料清单」，还有别的段落也长得像清单 ——
# 例如乐享无忧 3.5/3.6「恶性肿瘤特定药品费用保险金的申请」，
# 正文写的是「购药资格理赔审核通过后…」，被模式的 (1)(2) 编号捞出来就是
# 「项和第」「项外的全部材料」这类垃圾。而它的 accident_type 又和 3.3 的
# 真清单同名，会在 (item_name, accident_type) 这个业务主键上**覆盖掉正确答案**。
#
# 判据：全库实测，条款里的申请材料清单**恒以「保险合同」开头**（其次可能是
# 保险单/投保单）。这不是拍脑袋的启发式，而是理赔条款的行文惯例：
# 索赔材料第一项永远是保单本身。
MATERIAL_GUARD = re.compile(r"^(保险合同|保险单|投保单)")


def clean_accident_type(raw):
    """清洗事故类型：去掉小标题编号前缀与两端空白。保留原文用词，不做归并。"""
    text = ACC_PREFIX_RE.sub("", (raw or "").strip())
    return text.strip("、，,。：: ")


def find_titles(flat):
    """在归一化后的条款文本里找出所有「事故类型 → 材料清单」的标题位置。

    :return: [(title_start, title_end, accident_type)]，按出现顺序排列
    """
    hits = []
    for pat in TITLE_PATTERNS:
        for m in pat.finditer(flat):
            acc = clean_accident_type(m.group(1))
            if acc:
                hits.append((m.start(), m.end(), acc))
    # 同一位置可能被多条正则命中（例如乐享系的标题也含「保险金」），按起点去重
    hits.sort(key=lambda x: (x[0], x[1]))
    deduped, last_end = [], -1
    for start, end, acc in hits:
        if start >= last_end:
            deduped.append((start, end, acc))
            last_end = end
    return deduped


def extract_doc_items(segment):
    """从一段文本里抽出材料条目。

    丢弃空条目与明显是释义注脚的超长条目（>120 字）——
    注释里常含括号编号，容易被误当成材料。
    """
    items = []
    for m in DOC_ITEM_RE.finditer(segment):
        text = m.group(1).strip("；;。 ")
        if not text or len(text) > 120:
            continue
        items.append(text)
    return items


def build_docs_rules(merged, method="regex"):
    """把一条（聚合后的）「保险金申请」条款拆成若干 required_docs 规则。

    :param merged: group_by_clause 产出的聚合条目，含 text / source_pks / item_name 等
    :return: (rules, rejected)
    """
    flat = merged["text"]
    titles = find_titles(flat)
    rules, rejected = [], []

    for idx, (start, end, acc) in enumerate(titles):
        # 本条材料列表的终点 = 下一个标题的起点（没有下一个则到文本末尾）
        seg_end = titles[idx + 1][0] if idx + 1 < len(titles) else len(flat)
        segment = flat[end:seg_end]
        docs = extract_doc_items(segment)
        span = flat[start:seg_end]

        # 抽文顺序错乱先判 —— 这类问题比「抽不到材料」更危险：
        # 它会把别的事故类型的材料混进来，答案看起来完整但归属是错的。
        if BROKEN_TITLE_RE.search(span):
            rejected.append({
                "item_name": merged.get("item_name"),
                "accident_type": acc,
                "reason": "PDF 抽文顺序错乱：标题被材料条目切断，材料归属不可判定",
                "clause_no": merged.get("clause_no"),
                "span_head": span[:90],
            })
            continue

        if not docs:
            rejected.append({
                "item_name": merged.get("item_name"),
                "accident_type": acc,
                "reason": "标题命中但材料列表为空",
                "clause_no": merged.get("clause_no"),
            })
            continue

        # 材料形态门：首条必须是保单类凭证。挡掉「…的申请」模式在非材料段落上的误命中，
        # 这类误命中会以相同的 accident_type 覆盖掉真清单，比漏抽更危险。
        if not MATERIAL_GUARD.match(docs[0]):
            rejected.append({
                "item_name": merged.get("item_name"),
                "accident_type": acc,
                "reason": f"材料形态门未通过：首条不是保单类凭证（{docs[0][:40]}）",
                "clause_no": merged.get("clause_no"),
            })
            continue

        if not verify_span(span, flat):
            rejected.append({
                "item_name": merged.get("item_name"),
                "accident_type": acc,
                "reason": "span 未逐字命中原文",
                "clause_no": merged.get("clause_no"),
            })
            continue

        rules.append({
            "rule_id": f"{merged['item_name']}|required_docs|{acc}",
            "rule_type": "required_docs",
            "item_name": merged["item_name"],
            "insurance_type": merged.get("insurance_type"),
            "product_code": merged.get("product_code"),
            "company": merged.get("company"),
            "accident_type": acc,
            "docs": docs,
            "docs_count": len(docs),
            "source_clause_no": merged.get("clause_no"),
            # 主溯源片 = 标题所在的那片；材料跨片时，其余片记在 source_pks
            "source_pk": (merged.get("source_pks") or [None])[0],
            "source_pks": merged.get("source_pks") or [],
            "source_span": span,
            "extract_method": method,
            "verified": True,
            "schema_version": SCHEMA_VERSION,
        })
    return rules, rejected


# ============================================================ 时限抽取（正则）
# 四条正则对应时限四元组。注意「工作日」与「日」是两种单位，
# 混用会算错（30 个工作日 ≈ 6 周，30 日 = 1 个月），必须分开匹配。
PAT_SETTLE_WORKDAY = re.compile(r"在(\d+)个工作日内作出核定")
PAT_SETTLE_DAY = re.compile(r"在(\d+)日内作出核定")
PAT_SETTLE_COMPLEX = re.compile(r"情形复杂的?[，,]?在(\d+)日内作出核定")
PAT_PAY = re.compile(r"协议后(\d+)日内[，,]?履行给付")
PAT_REJECT = re.compile(r"作出核定后(\d+)个工作日内.{0,8}?发出拒绝")


def extract_timeline(text):
    """从「保险金给付」条款抽四组时限。

    关键处理：**先把「情形复杂的…」那句摘掉**，再匹配常规核定期限。
    否则同一条里「5个工作日」与「30日」都会命中同一条正则，
    单位还不同，后一次匹配会静默覆盖前一次。

    :return: (时限字典, source_span)。时限字典四项全 None 表示这条不是时限条款。
    """
    flat = norm_ws(text)
    result = {
        "settle_value": None, "settle_unit": None,
        "settle_complex_value": None, "settle_complex_unit": None,
        "pay_value": None, "pay_unit": None,
        "reject_notice_value": None, "reject_notice_unit": None,
    }
    hits = []

    m = PAT_SETTLE_COMPLEX.search(flat)
    if m:
        result["settle_complex_value"] = int(m.group(1))
        result["settle_complex_unit"] = "day"
        hits.append(m.span())
        rest = flat[:m.start()] + flat[m.end():]
    else:
        rest = flat

    m = PAT_SETTLE_WORKDAY.search(rest)
    if m:
        result["settle_value"] = int(m.group(1))
        result["settle_unit"] = "workday"
        hits.append(m.span())
    else:
        # 退一步找不带「工作日」的写法（部分产品只写「30日内」）
        m = PAT_SETTLE_DAY.search(rest)
        if m:
            result["settle_value"] = int(m.group(1))
            result["settle_unit"] = "day"
            hits.append(m.span())

    m = PAT_PAY.search(flat)
    if m:
        result["pay_value"] = int(m.group(1))
        result["pay_unit"] = "day"
        hits.append(m.span())

    m = PAT_REJECT.search(flat)
    if m:
        result["reject_notice_value"] = int(m.group(1))
        result["reject_notice_unit"] = "workday"
        hits.append(m.span())

    # source_span 取「覆盖全部命中位置的最小片段 + 少量上下文」，
    # 既能证明数字有出处，又不至于把整条条款塞进规则表
    span = ""
    if hits:
        lo = max(0, min(h[0] for h in hits) - 24)
        hi = min(len(flat), max(h[1] for h in hits) + 12)
        span = flat[lo:hi]

    return result, span


def build_timeline_rule(row, tl, span):
    """组装时限规则。四项全 None 说明这条根本不是时限条款（实测有 1 条误分类），
    此时返回 None 而不是存一条空壳规则 —— 空壳规则会让工具误报「查到了」。
    """
    values = [tl["settle_value"], tl["settle_complex_value"],
              tl["pay_value"], tl["reject_notice_value"]]
    if all(v is None for v in values):
        return None
    rule = {
        "rule_id": f"{row['item_name']}|claim_timeline",
        "rule_type": "claim_timeline",
        "item_name": row["item_name"],
        "insurance_type": row.get("insurance_type"),
        "product_code": row.get("product_code"),
        "company": row.get("company"),
        "source_clause_no": row.get("clause_no"),
        "source_pk": row.get("pk"),
        "source_pks": [row.get("pk")],
        "source_span": span,
        "extract_method": "regex",
        # 正则从原文里抠出来的数字，span 必然命中；标 True 是为了字段齐整
        "verified": True,
        "schema_version": SCHEMA_VERSION,
    }
    rule.update(tl)
    return rule


# ============================================================ 通用
def verify_span(span, text):
    """反幻觉门：span 归一化空白后必须逐字出现在原文里。"""
    s, t = norm_ws(span), norm_ws(text)
    return bool(s) and s in t


def chunk_seq(pk):
    """从 pk 末尾取切片序号。pk 形如 `10_CPIC_医疗_xxx_3.3_2` → 2。

    用于把同一条款被 L2 切散的多个片按原文顺序拼回来。
    """
    m = re.search(r"_(\d+)$", pk or "")
    return int(m.group(1)) if m else 0


def group_by_clause(rows):
    """按 (item_name, clause_no) 聚合，把被 L2 切散的同一条款拼回完整文本。

    这是本脚本能覆盖 18/18 的关键一步 —— 不聚合的话，
    「保险金申请」只有第一片带小标题，后两片的材料会被静默漏掉。
    """
    groups = {}
    for r in rows:
        groups.setdefault((r.get("item_name"), r.get("clause_no")), []).append(r)

    merged = []
    for (item_name, clause_no), items in groups.items():
        items.sort(key=lambda x: chunk_seq(x.get("pk")))
        merged.append({
            **items[0],
            "item_name": item_name,
            "clause_no": clause_no,
            "text": "".join(norm_ws(x.get("text")) for x in items),
            "chunk_count": len(items),
            "source_pks": [x.get("pk") for x in items],
        })
    return merged


def pull_clauses(client, clause_type=None):
    """从 Milvus 拉条款。clause_type 传 None 表示**不限类型，全量拉取**。

    ⚠️ 为什么材料抽取必须全量拉、不能按 clause_type 过滤（2026-09-23 修正）：
    材料清单所在条款的 clause_type 标签并不可靠 —— 实测「保险金及保险费豁免申请」
    被打成 `保险费`、「如何申请保险金」被打成 `其他`。按 `clause_type == "保险金申请"`
    过滤会**静默漏掉 3 款产品**（太保福有余终身寿险 / 太保阿基米德重疾 /
    中英福临门养老年金），它们明明在条款里写全了材料清单，却只能靠险种级降级
    或直接拒答。改为全量取池、让标题正则自己裁决后覆盖率 14/17 → 17/17。
    代价是池子从 18 片涨到 966 片 —— 但过滤动作由正则承担，见 MATERIAL_GUARD。

    ⚠️ Milvus query 的 limit 上限是 16384，写 20000 会直接报
    `invalid max query result window`。
    """
    kwargs = dict(
        collection_name=milvus_config.chunks_collection,
        output_fields=FIELDS,
        limit=16384,
        consistency_level="Strong",
    )
    if clause_type:
        kwargs["filter"] = f'clause_type == "{clause_type}"'
    else:
        # 空 filter 必须带 limit，否则 Milvus 拒绝执行
        kwargs["filter"] = ""
    return client.query(**kwargs)


# ============================================================ 主流程
def main():
    ap = argparse.ArgumentParser(description="构建理赔规则表（MongoDB insurance_rules）")
    ap.add_argument("--dry-run", action="store_true", help="只抽取不写库，打印详情")
    ap.add_argument("--only", choices=["timeline", "docs"], default=None, help="只跑其中一类")
    args = ap.parse_args()

    started = time.time()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"rule_build_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    client = get_milvus_client()
    if client is None:
        logger.error("Milvus 连接失败，终止。先确认 milvus-standalone 是否在运行")
        return 2

    all_rules, rejected, notes = [], [], []

    # ---------------- claim_timeline ----------------
    if args.only in (None, "timeline"):
        rows = pull_clauses(client, "保险金给付")
        logger.info(f"拉取「保险金给付」{len(rows)} 条")
        built, skipped = [], []
        for r in rows:
            tl, span = extract_timeline(r.get("text") or "")
            rule = build_timeline_rule(r, tl, span)
            (built if rule else skipped).append(rule or {
                "pk": r.get("pk"), "item_name": r.get("item_name"), "clause_no": r.get("clause_no")})
        all_rules.extend(built)
        notes.append(f"claim_timeline：源条款 {len(rows)} 条 → 规则 {len(built)} 条，"
                     f"跳过 {len(skipped)} 条（无时限结构）")
        if skipped:
            notes.append(f"  跳过：{[str(s.get('item_name')) + ' ' + str(s.get('clause_no')) for s in skipped]}")
        combo = Counter(
            (b["settle_value"], b["settle_unit"], b["settle_complex_value"],
             b["pay_value"], b["reject_notice_value"]) for b in built
        )
        notes.append("  时限组合分布（核定值/单位/复杂/给付/拒赔通知）：")
        for k, v in combo.most_common():
            notes.append(f"    {v:3d}  {k}")

    # ---------------- required_docs ----------------
    if args.only in (None, "docs"):
        # 全量取池，不按 clause_type 过滤 —— 材料条款的 clause_type 标签不可靠，
        # 过滤会静默漏产品。裁决交给 TITLE_PATTERNS + MATERIAL_GUARD。
        raw_rows = pull_clauses(client)
        merged = group_by_clause(raw_rows)
        logger.info(f"全量拉取 {len(raw_rows)} 片 → 聚合为 {len(merged)} 条完整条款")
        built, no_title, no_docs = [], [], []
        for m in merged:
            rules, rej = build_docs_rules(m)
            # rej 必须逐条收进循环内 —— 写在循环外只会拿到最后一次迭代的空列表，
            # 丢弃记录（串行错乱 / 形态门 / span 校验失败）会全部静默丢失
            rejected.extend(rej)
            if rules:
                built.extend(rules)
                title_hint = "+".join(r["accident_type"] for r in rules)
                logger.info(f"  {m['item_name']} [{m['clause_no']}] {m['chunk_count']} 片 "
                            f"→ {len(rules)} 条规则：{title_hint}")
            else:
                (no_title if not find_titles(m["text"]) else no_docs).append(
                    {"item_name": m["item_name"], "clause_no": m["clause_no"],
                     "chunk_count": m["chunk_count"]})
        all_rules.extend(built)
        notes.append(f"required_docs：源 {len(raw_rows)} 片（全量）→ 聚合 {len(merged)} 条条款 "
                     f"→ 规则 {len(built)} 条；无标题 {len(no_title)} 条，"
                     f"有标题无材料 {len(no_docs)} 条，质检丢弃 {len(rejected)} 条")
        # 全量取池后「无标题」必然占绝大多数（绝大多数条款与理赔材料无关），
        # 只报数不铺清单，避免日志被 800 多行淹没
        if no_docs:
            notes.append(f"  有标题无材料（需人工看）：{[x['item_name'] for x in no_docs]}")
        gate_rej = [x for x in rejected if "形态门" in (x.get("reason") or "")]
        if gate_rej:
            notes.append(f"  形态门拦下 {len(gate_rej)} 条："
                         f"{[(x['item_name'], x['accident_type']) for x in gate_rej]}")
        # 每款产品覆盖的事故类型数，一眼看出谁的条款更细
        per_product = Counter(r["item_name"] for r in built)
        notes.append(f"  材料清单覆盖 {len(per_product)} 款产品，"
                     f"每款事故类型数：{dict(per_product.most_common())}")

    # ---------------- 汇总 ----------------
    by_type = Counter(r["rule_type"] for r in all_rules)
    products = sorted({r["item_name"] for r in all_rules})
    summary = {
        "stamp": stamp,
        "elapsed_sec": round(time.time() - started, 1),
        "dry_run": bool(args.dry_run),
        "total_rules": len(all_rules),
        "by_rule_type": dict(by_type),
        "products_covered": len(products),
        "products": products,
        "rejected": rejected,
        "notes": notes,
    }

    with open(os.path.join(out_dir, "rules.json"), "w", encoding="utf-8") as f:
        json.dump(all_rules, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for line in notes:
        logger.info(line)
    logger.info(f"产物目录：{out_dir}")

    if args.dry_run:
        logger.info("--dry-run 模式，未写库。")
        print("\n".join(notes))
        return 0

    if not all_rules:
        logger.error("没有生成任何规则，未写库。")
        return 1

    tool = get_rule_mongo_tool()
    result = tool.upsert_rules(all_rules)
    logger.info(f"写库完成：{result}")
    stats = tool.stats()
    logger.info(f"规则表现状：{stats}")
    print("\n".join(notes))
    print(f"写库结果：{result}")
    print(f"规则表现状：{stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
