"""构造检索层评测标注集 —— 消融实验的标尺

产出
----
  data/eval/retrieval_cases.jsonl            标注集本体
  data/eval/retrieval_cases_rejected.jsonl   被质检丢弃的生成结果（留痕，用于迭代出题 prompt）

为什么需要它
------------
`insurance_kb/06_eval_cases.csv` 是照中保协示范条款写的，其 `expected_clause_no` 是
「第六条 / 第十四条」式中文条号；而本项目语料（太平洋人寿 10 款 + 中英人寿 7 款）
966 片切片里**含「条」的为 0** —— 全是 `N.M` 数字层级。且该用例集未绑产品。
所以它无法直接支撑 Recall@k / MRR 的计算，必须先重建标注。

两条来源
--------
A 锚点集（人工绑定，见 ANCHORS）
  用旧用例自带的 `expected_keywords` 到全库回验，把确实能在语料里定位到的用例
  重绑到真实 `doc_id` + `clause_no`。作用是给标注集钉一组「真人问法」，
  防止全靠 LLM 出题导致问法同质化。
  每条都带 `must_contain` 断言：绑错会直接抛错，不会静默产出脏标注。

B 扩量集（LLM 反向出题 + 程序质检）
  对 L1 条款切片抽样，让 LLM 反着出题（给条款 → 问什么能被它回答），
  并要求同时摘录「答案原话」。两道质检：

  ① 幻觉门：答案原话必须逐字出现在切片原文中。
     LLM 编的内容不可能逐字命中原文 —— 这是零成本的幻觉检测器。

  ② 难度门（去词面重叠）：问题与切片原文的最长公共子串 <= MAX_LCS，
     且问题的 2-gram 落在原文里的比例 <= MAX_COVER。
     为什么必须有这道门：第一版没设这道门时，LLM 出的题几乎在照抄条款用词
     （最长公共子串中位 6、最大 24，问题中位 35 字），跑出来 Recall@1 = 0.967、
     RRF 阶段满分 1.0 —— 标注太简单，任何配置都是满分，消融实验失去区分度。
     阈值取自「人写锚点」的实测分布（最长公共子串中位 4、上界 7）。
     两道门互相制衡出好题：答案必须逐字来自原文（保证可判定），
     问法却又不能复述原文（保证有难度）。

为什么评测必须按产品过滤
------------------------
语料里多份文档都有几乎一致的「1.4 犹豫期」，若不按产品过滤，
「犹豫期多久」这道题的标准答案有十几个，指标无从计算。
线上检索本身就是 `item_name in [...]` 精确过滤，评测按同一口径才是真实场景。

用法
----
  .venv\\Scripts\\python.exe test/09_build_retrieval_cases.py                # 全量重建
  .venv\\Scripts\\python.exe test/09_build_retrieval_cases.py --anchors-only # 只做锚点集（不调 LLM）
  .venv\\Scripts\\python.exe test/09_build_retrieval_cases.py --per-doc 30   # 调整每份语料出题数
"""

import os
import re
import sys
import csv
import json
import random
import argparse
import io
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NO_PROXY", "*")

from pymilvus import MilvusClient  # noqa: E402

from app.conf.milvus_config import milvus_config  # noqa: E402
from app.core.logger import logger  # noqa: E402

# ------------------------------------------------------------------ 常量
OUT_DIR = ROOT / "data" / "eval"
ANCHOR_NOTE = "重绑自 insurance_kb/06_eval_cases.csv，见脚本内 ANCHORS 表"

# 出题 prompt 放在设计资产目录（与 clause_extract.prompt 同类，都是数据构造 prompt），
# 不放进 prompts/ —— 那里是运行时注入检索链路的 prompt
PROMPT_PATH = ROOT / "insurance_kb" / "07_prompts_v2" / "retrieval_question_gen.prompt"

# 抽样与并发
DEFAULT_PER_DOC = 20
MAX_WORKERS = 6
SEED = 20260918
MIN_CHUNK_CHARS = 40       # 太短的切片（标题残片）出不了好题
MIN_QUESTION_CHARS = 8
MAX_QUESTION_CHARS = 40
MIN_SPAN_CHARS = 4

# 难度门阈值：对齐「人写锚点」的实测上界（最长公共子串上界 7、2-gram 覆盖率上界 0.53）
MAX_LCS = 8
MAX_COVER = 0.55

# ------------------------------------------------------------------ 锚点表
# (case_id, 来源用例, 问题, doc_id 前缀, clause_no_norm, must_contain)
# doc_id 前缀必须唯一命中一份语料；must_contain 里的关键词必须全部落在目标切片原文里，
# 任一条件不满足就抛错 —— 宁可脚本失败，也不要静默产出绑错的标注。
ANCHORS = [
    ("H01", "F001", "这款产品的等待期是多少天？", "04_CPIC_护理", "2.4", ["90日"]),
    ("H02", "F002", "犹豫期有多久？", "01_CPIC_年金", "1.4", ["15日"]),
    ("H03", "F003", "保单贷款最多能贷多少？", "09_CPIC_终身寿", "5.2", ["80%"]),
    ("H04", "F004", "等待期内确诊特定疾病会怎么处理？", "04_CPIC_护理", "2.4",
     ["90日", "不承担护理保险金"]),
    ("H05", "F006", "申请保险金需要提交哪些证明和资料？", "05_CPIC_医疗", "3.3",
     ["申请书", "保险合同", "诊断证明", "身份证件"]),
    ("H06", "F007", "责任免除包含哪些情形？", "01_CPIC_年金", "2.5",
     ["故意杀害", "故意犯罪", "毒品", "酒后驾驶"]),
    ("H07", "F008", "经社保结算和未经社保结算的赔付比例分别是多少？", "10_CPIC_医疗", "2.7",
     ["100%", "60%"]),
    ("H08", "F009", "这款产品的年免赔额是多少？", "10_CPIC_医疗", "2.5",
     ["年免赔额", "10000元"]),
    ("H09", "F010", "既往症是怎么定义的？", "05_CPIC_医疗", "10.12",
     ["本合同生效日之前", "已患且已知晓"]),
    ("H10", "F013", "发生保险事故后最晚多久要通知保险公司？", "01_CPIC_年金", "3.2", ["10日"]),
    ("H11", "F014", "合同争议解决方式是仲裁还是诉讼？", "11_AC_两全", "12",
     ["协商", "仲裁", "人民法院"]),
    ("H12", "F016", "这款产品的投保年龄范围是多少？", "03_CPIC_重疾", "1.3",
     ["28日", "55周岁"]),
]

# 明确记录哪些旧用例无法绑定，避免下次又去试一遍
UNBINDABLE = {
    "F005": "语料无「意外伤害不受等待期限制」表述",
    "F011": "语料无「继续有效 / 以一次为限」对应表述",
    "F012": "语料无「自动续保 / 调整保险费率」条款",
    "F015": "语料无「二年内自杀不赔 / 无民事行为能力人除外」表述",
    "C001-C005": "coref 用例是省略式追问（「那它的免赔额呢」），非自包含查询，不适合做检索评测",
    "R001-R008": "应拒答用例，其正确答案是「拒答」而非某个切片，不参与检索指标",
    "P001-P004": "合规判定用例，属 Phase 3 合规链路，不参与检索指标",
}


# ------------------------------------------------------------------ 工具
def nospace(s: str) -> str:
    """去掉全部空白。语料里条款号与数字常被换行切开，比对前必须归一。"""
    return re.sub(r"\s+", "", s or "")


def lcs_len(a: str, b: str) -> int:
    """最长公共子串长度（滚动数组）。衡量问题是否在照抄原文用词。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def bigram_cover(question: str, doc: str) -> float:
    """问题里的 2-gram 有多大比例出现在原文里（0~1）。"""
    if len(question) < 2:
        return 0.0
    grams = [question[i:i + 2] for i in range(len(question) - 1)]
    return sum(1 for g in grams if g in doc) / len(grams)


def load_chunks() -> list:
    """从 Milvus 拉全量切片。注意 pk 在 query 结果里是直接返回的（与 search 不同）。"""
    client = MilvusClient(uri=milvus_config.milvus_url)
    collection_name = milvus_config.chunks_collection
    client.load_collection(collection_name=collection_name)
    chunks = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=[
            "pk", "text", "clause_no", "clause_no_norm", "clause_path",
            "clause_title", "clause_type", "doc_id", "item_name", "chunk_level",
            "chunk_seq", "char_len",
        ],
        limit=5000,
        consistency_level="Strong",
    )
    for c in chunks:
        c["_flat"] = nospace(c.get("text"))
    return chunks


def build_anchors(chunks_by_doc: dict) -> tuple:
    """把 ANCHORS 表解析成带 pk 的标注条目。任一断言失败直接抛错。"""
    items, rejected = [], []
    for case_id, origin, query, doc_prefix, clause_norm, must in ANCHORS:
        docs = [d for d in chunks_by_doc if d.startswith(doc_prefix)]
        if len(docs) != 1:
            raise ValueError(f"[{case_id}] doc_id 前缀 {doc_prefix!r} 命中 {len(docs)} 份语料，必须唯一")
        doc = docs[0]
        cands = [c for c in chunks_by_doc[doc] if c.get("clause_no_norm") == clause_norm]
        if not cands:
            raise ValueError(f"[{case_id}] 语料 {doc} 中找不到 clause_no_norm={clause_norm}")
        # 同一 clause_no 可能被三级切分拆成多片（L1 母片 + L2/L3 子片），
        # 取「同时命中最多关键词」的那一片作为标准答案
        scored = sorted(
            ((sum(1 for k in must if nospace(k) in c["_flat"]), len(c["_flat"]), c) for c in cands),
            key=lambda x: (-x[0], x[1]),
        )
        hits, _, best = scored[0]
        if hits < len(must):
            raise ValueError(
                f"[{case_id}] 关键词未全中（{hits}/{len(must)}）：{doc} clause_no_norm={clause_norm}；"
                f"缺 {[k for k in must if nospace(k) not in best['_flat']]}"
            )
        items.append({
            "case_id": case_id,
            "source": "anchor",
            "origin": f"06_eval_cases.csv#{origin}",
            "query": query,
            "doc_id": best["doc_id"],
            "item_name": best["item_name"],
            "gold_clause_no": best["clause_no"],
            "gold_clause_no_norm": best["clause_no_norm"],
            "gold_clause_type": best["clause_type"],
            "gold_clause_path": best["clause_path"],
            "gold_pk": best["pk"],
            "gold_span": "；".join(must),
            "note": ANCHOR_NOTE,
        })
        logger.info(f"[{case_id}] 锚点绑定成功：{best['doc_id'][:26]} / clause_no={best['clause_no']} / pk={best['pk']}")
    return items, rejected


# ------------------------------------------------------------------ LLM 出题
def load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"出题 prompt 不存在：{PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")


def render_prompt(template: str, clause_text: str) -> str:
    """用 replace 而不是 str.format —— prompt 里含 JSON 示例的裸花括号，
    format() 会把它们当占位符直接抛 KeyError。"""
    return template.replace("{clause_text}", clause_text)


def parse_llm_json(raw: str):
    """解析模型输出。容忍偶发的 ```json 包裹与前后缀说明。"""
    text = (raw or "").strip()
    try:
        return json.loads(text), None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, "输出中找不到 JSON 对象"
    try:
        return json.loads(m.group(0)), None
    except Exception as e:
        return None, f"JSON 解析失败：{e}"


def verify_generated(data: dict, chunk: dict):
    """两道质检：① 答案原话逐字命中原文（反幻觉）；② 问题不照抄原文用词（保难度）。"""
    question = (data.get("question") or "").strip() if isinstance(data, dict) else ""
    span = (data.get("answer_span") or "").strip() if isinstance(data, dict) else ""
    if not question or not span:
        return None, "模型判定无有效问题（question/answer_span 为空）"
    q_flat = nospace(question)
    if not (MIN_QUESTION_CHARS <= len(q_flat) <= MAX_QUESTION_CHARS):
        return None, f"问题长度异常（{len(q_flat)} 字）"
    s_flat = nospace(span)
    if len(s_flat) < MIN_SPAN_CHARS:
        return None, f"答案片段过短（{len(s_flat)} 字）"
    # ① 反幻觉门
    if s_flat not in chunk["_flat"]:
        return None, "答案原话未逐字命中原文（疑似幻觉）"
    # ② 难度门
    lcs = lcs_len(q_flat, chunk["_flat"])
    if lcs > MAX_LCS:
        return None, f"问题照抄原文用词（最长公共子串 {lcs} > {MAX_LCS}）"
    cover = bigram_cover(q_flat, chunk["_flat"])
    if cover > MAX_COVER:
        return None, f"问题词面重叠过高（2-gram 覆盖 {cover:.2f} > {MAX_COVER}）"
    return {"question": question, "answer_span": span, "lcs": lcs, "cover": round(cover, 3)}, None


def generate_one(llm, template: str, chunk: dict, retries: int = 2):
    """对单个切片出题。返回 (记录, 丢弃原因)。"""
    last_err = ""
    for attempt in range(retries + 1):
        try:
            resp = llm.invoke(render_prompt(template, chunk["text"]))
            data, err = parse_llm_json(getattr(resp, "content", "") or "")
            if err:
                last_err = err
                continue
            item, err = verify_generated(data, chunk)
            if err:
                return None, err
            return {
                "source": "llm_gen",
                "query": item["question"],
                "doc_id": chunk["doc_id"],
                "item_name": chunk["item_name"],
                "gold_clause_no": chunk["clause_no"],
                "gold_clause_no_norm": chunk["clause_no_norm"],
                "gold_clause_type": chunk["clause_type"],
                "gold_clause_path": chunk["clause_path"],
                "gold_pk": chunk["pk"],
                "gold_span": item["answer_span"],
                "chunk_level": chunk["chunk_level"],
                "max_lcs": item["lcs"],
                "bigram_cover": item["cover"],
                "note": f"LLM 反向出题（第 {attempt + 1} 次尝试通过）",
            }, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return None, last_err


def sample_chunks(chunks_by_doc: dict, per_doc: int) -> list:
    """按文档分层抽样 L1 切片。用固定种子，保证可复现。"""
    rng = random.Random(SEED)
    picked = []
    for doc in sorted(chunks_by_doc):
        cands = [
            c for c in chunks_by_doc[doc]
            if c.get("chunk_level") == "L1" and len(c["_flat"]) >= MIN_CHUNK_CHARS
        ]
        cands.sort(key=lambda c: c["pk"])  # 固定顺序，抵消 dict 遍历随机性
        k = min(per_doc, len(cands))
        picked.extend(rng.sample(cands, k))
    rng.shuffle(picked)
    return picked


def main():
    parser = argparse.ArgumentParser(description="构造检索层评测标注集")
    parser.add_argument("--per-doc", type=int, default=DEFAULT_PER_DOC, help="每份语料抽多少条 L1 切片出题")
    parser.add_argument("--anchors-only", action="store_true", help="只做锚点集，不调用 LLM")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    chunks_by_doc = defaultdict(list)
    for c in chunks:
        chunks_by_doc[c["doc_id"]].append(c)
    logger.info(f"语料载入：{len(chunks)} 片 / {len(chunks_by_doc)} 份文档")

    # ------------------------------ A 锚点集
    anchor_items, _ = build_anchors(chunks_by_doc)
    logger.info(f"锚点集：{len(anchor_items)} 条绑定成功")

    if args.anchors_only:
        gen_items, rejected = [], []
    else:
        # ------------------------------ B 扩量集
        template = load_prompt_template()
        picked = sample_chunks(chunks_by_doc, args.per_doc)
        logger.info(f"扩量集：抽样 {len(picked)} 片 L1 切片，开始出题（并发 {MAX_WORKERS}）")

        from app.lm.lm_utils import get_llm_client
        llm = get_llm_client(json_mode=True)

        gen_items, rejected = [], []
        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(generate_one, llm, template, c): c for c in picked}
            for fut in as_completed(futures):
                chunk = futures[fut]
                item, reason = fut.result()
                done += 1
                if item:
                    gen_items.append(item)
                else:
                    rejected.append({
                        "doc_id": chunk["doc_id"],
                        "gold_pk": chunk["pk"],
                        "clause_no": chunk["clause_no"],
                        "reason": reason,
                        "clause_text": chunk["text"][:300],
                    })
                if done % 25 == 0:
                    logger.info(f"  出题进度 {done}/{len(picked)}，通过 {len(gen_items)}，丢弃 {len(rejected)}")

    # ------------------------------ 去重 + 编号
    seen, final = set(), []
    for item in anchor_items + gen_items:
        key = nospace(item["query"])
        if key in seen:
            continue
        seen.add(key)
        final.append(item)

    for idx, item in enumerate((x for x in final if x["source"] == "llm_gen"), start=1):
        item["case_id"] = f"G{idx:03d}"

    final.sort(key=lambda x: x["case_id"])

    cases_path = OUT_DIR / "retrieval_cases.jsonl"
    with open(cases_path, "w", encoding="utf-8") as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    rejected_path = OUT_DIR / "retrieval_cases_rejected.jsonl"
    with open(rejected_path, "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------------------ 汇总
    n_anchor = sum(1 for x in final if x["source"] == "anchor")
    n_gen = sum(1 for x in final if x["source"] == "llm_gen")
    logger.info("=" * 70)
    logger.info(f"标注集已生成：{cases_path}")
    logger.info(f"  锚点集 {n_anchor} 条 + 扩量集 {n_gen} 条 = 合计 {len(final)} 条")
    logger.info(f"  覆盖文档 {len(set(x['doc_id'] for x in final))} 份 / "
                f"覆盖条款 {len(set(x['gold_pk'] for x in final))} 个")
    gen_final = [x for x in final if x["source"] == "llm_gen"]
    if gen_final:
        lcs = sorted(x["max_lcs"] for x in gen_final)
        cov = sorted(x["bigram_cover"] for x in gen_final)
        logger.info(f"  扩量集难度：最长公共子串 中位={lcs[len(lcs) // 2]} 上界={lcs[-1]}；"
                    f"2-gram 覆盖 中位={cov[len(cov) // 2]:.2f} 上界={cov[-1]:.2f}")
    logger.info(f"  丢弃 {len(rejected)} 条（通过率 "
                f"{(len(gen_final) / (len(gen_final) + len(rejected)) * 100 if (len(gen_final) + len(rejected)) else 0):.1f}%），"
                f"明细见 {rejected_path}")
    if rejected:
        for reason, cnt in Counter(r["reason"] for r in rejected).most_common(6):
            logger.info(f"    丢弃原因 × {cnt}：{reason}")
    logger.info("  条款类型分布：" + json.dumps(
        dict(Counter(x["gold_clause_type"] for x in final).most_common()), ensure_ascii=False))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
