"""构建合规条目表（MongoDB compliance_rules）—— Phase 3.2 的数据侧入口

数据来源与去向
--------------
来源：`insurance_kb/04_negative_list_2026.json`（《人身保险产品负面清单（2026版）》
      105 条，四大类）
去向：MongoDB `compliance_rules` 集合 —— 条目原文 + 导入时预算的 bge-m3 稠密向量

为什么向量在这里算、而不在查询时算
----------------------------------
105 条是**固定不变的判定标准**，每次查询都重新编码等于把固定成本摊到每次请求上。
更重要的是：**向量必须在导入时算好并与条目同批落库**，否则「换模型只重导条款库、
忘了重导合规库」会让两边向量落在不同空间里，余弦还算得出数、但已经无意义 ——
这类错误不报错、只出错。`embed_model` 字段就是为这件事留的痕迹。

为什么只导一次就够、不需要像条款库那样分批
------------------------------------------
105 条一次 encode 完，CPU 上约 10 秒。条款库要分批是因为 966 片 + L2 拼接，
规模差一个量级。

用法（在项目根目录执行）
----------------------
    .venv\\Scripts\\python.exe test/12_build_compliance_rules.py --dry-run     # 只组装不写库
    .venv\\Scripts\\python.exe test/12_build_compliance_rules.py               # 组装并写库
    .venv\\Scripts\\python.exe test/12_build_compliance_rules.py --no-vector   # 跳过向量（只修文本字段）

幂等：`_id` 取业务键 `NL-{item_no:03d}`，重复跑只更新不新增。
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Windows 控制台默认是 GBK，print 里出现 ✓ / ⚠ 这类符号会直接抛
# UnicodeEncodeError 把整个脚本打断（实测过一次）。
# ⚠️ 这里**不能把 encoding 改成 utf-8**：PowerShell 管道按控制台代码页解码，
# 改 utf-8 会让中文全变乱码。只兜住编码错误即可 —— 编不出的字符退化成 ?，
# 中文照常按本地编码输出。
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

# 沙箱代理会劫持到本地服务与本机模型的连接，必须在导入客户端前清掉
for _k in list(os.environ):
    if _k.lower() in {"http_proxy", "https_proxy", "all_proxy", "grpc_proxy", "no_proxy"}:
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# 必须在清代理之后再 load_dotenv，否则 .env 里的代理变量又会回来
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from app.clients.mongo_compliance_utils import SCHEMA_VERSION, get_compliance_tool
from app.conf.compliance_config import compliance_config
from app.conf.embedding_config import embedding_config
from app.core.logger import logger

SOURCE_FILE = os.path.join(ROOT, "insurance_kb", "04_negative_list_2026.json")
OUT_REPORT = os.path.join(ROOT, "output", "compliance_build_report.json")

# scope 按「/」切分。全角斜杠也要认 —— 源文件里两种都出现过
RE_SCOPE_SPLIT = re.compile(r"[/／]")


def load_source():
    """读负面清单源文件，返回 (categories, meta)。"""
    if not os.path.exists(SOURCE_FILE):
        raise FileNotFoundError(f"找不到负面清单源文件：{SOURCE_FILE}")
    with open(SOURCE_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("categories") or [], data.get("meta") or {}


def split_scope(scope):
    """把 scope 原文按「/」切成标签列表。

    **保留原文，不做归并** —— 「长期险」与「人寿保险」在监管口径下不必然等价，
    归并等于替监管做了一次语义解释。标签只用于召回，判定仍看原文。
    """
    parts = [seg.strip() for seg in RE_SCOPE_SPLIT.split(scope or "")]
    seen, tags = set(), []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            tags.append(part)
    return tags


def build_docs(categories, meta):
    """把源 json 组装成待写库的条目文档列表（不含向量）。

    2026 版变动标记取自 `meta.changes_in_2026` —— 不在条目里逐条写死，
    避免源文件改版时两处不一致。
    """
    new_nos = {int(x["no"]) for x in (meta.get("changes_in_2026", {}) or {}).get("new_items", [])}
    mod_nos = {int(x["no"]) for x in (meta.get("changes_in_2026", {}) or {}).get("modified_items", [])}

    docs = []
    for cat in categories:
        cat_id = str(cat.get("id") or "")
        cat_name = cat.get("name") or ""
        for item in cat.get("items") or []:
            item_no = int(item["no"])
            scope = (item.get("scope") or "").strip()
            docs.append(
                {
                    "item_no": item_no,
                    "category_id": cat_id,
                    "category_name": cat_name,
                    # 原文逐字保留。它是判定依据，概括或改写即失真
                    "text": item["text"],
                    "scope": scope,
                    "scope_tags": split_scope(scope),
                    "is_2026_new": item_no in new_nos,
                    "is_2026_modified": item_no in mod_nos,
                }
            )
    docs.sort(key=lambda d: d["item_no"])
    return docs


def attach_vectors(docs):
    """为条目预算稠密向量。整批一次 encode。

    用 `encode_documents`（文档侧）而不是 query 侧编码函数：与条款库导入侧
    保持同一路径，避免新旧向量落在不同空间。
    """
    from app.lm.embedding_utils import generate_embeddings

    texts = [d["text"] for d in docs]
    started = time.time()
    result = generate_embeddings(texts)
    dense = result.get("dense") or []
    if len(dense) != len(texts):
        raise RuntimeError(f"向量条数不匹配：输入 {len(texts)} 条，返回 {len(dense)} 条")

    model_path = embedding_config.bge_m3_path or "BAAI/bge-m3"
    for doc, vec in zip(docs, dense):
        doc["dense_vector"] = [float(x) for x in vec]
        doc["embed_model"] = model_path
    logger.success(
        f"向量生成完成 | 条数={len(dense)} | 维度={len(dense[0]) if dense else 0} "
        f"| 耗时={time.time() - started:.1f}s"
    )
    return docs


def verify_docs(docs, expect_total=105):
    """写库前的体检。任何一条不过就不写 —— 半份判定标准比没有判定标准更危险。

    :return: (problems, summary)。problems 非空即中止。
    """
    problems = []
    if len(docs) != expect_total:
        problems.append(f"条数不符：期望 {expect_total}，实得 {len(docs)}")

    nos = [d["item_no"] for d in docs]
    missing = [n for n in range(1, expect_total + 1) if n not in set(nos)]
    if missing:
        problems.append(f"条号缺失：{missing}")
    dup = [n for n, c in Counter(nos).items() if c > 1]
    if dup:
        problems.append(f"条号重复：{dup}")

    for d in docs:
        if not (d.get("text") or "").strip():
            problems.append(f"条 {d['item_no']} 原文为空")
        if not d.get("scope_tags"):
            problems.append(f"条 {d['item_no']} scope_tags 为空（scope={d.get('scope')!r}）")
        if not d.get("category_name"):
            problems.append(f"条 {d['item_no']} 缺 category_name")
        vec = d.get("dense_vector")
        if vec is not None and len(vec) == 0:
            problems.append(f"条 {d['item_no']} 向量为空")

    dims = {len(d["dense_vector"]) for d in docs if d.get("dense_vector")}
    if len(dims) > 1:
        problems.append(f"向量维度不一致：{sorted(dims)}")

    summary = {
        "total": len(docs),
        "by_category": dict(Counter(d["category_name"] for d in docs)),
        "universal": sum(1 for d in docs if "通用" in (d.get("scope_tags") or [])),
        "new_2026": sum(1 for d in docs if d.get("is_2026_new")),
        "modified_2026": sum(1 for d in docs if d.get("is_2026_modified")),
        "vector_dim": sorted(dims)[0] if len(dims) == 1 else None,
        "scope_tag_kinds": len({t for d in docs for t in d.get("scope_tags") or []}),
    }
    return problems, summary


def main():
    ap = argparse.ArgumentParser(description="构建合规条目表（MongoDB compliance_rules）")
    ap.add_argument("--dry-run", action="store_true", help="只组装不写库，打印详情")
    ap.add_argument("--no-vector", action="store_true",
                    help="跳过向量生成（只修文本字段时用，注意：会让存量向量与原文不一致）")
    args = ap.parse_args()

    started = time.time()
    print("=" * 78)
    print("合规条目表构建 | 指纹:", compliance_config.signature())
    print("  源文件:", SOURCE_FILE)

    categories, meta = load_source()
    docs = build_docs(categories, meta)
    print(f"  组装条数: {len(docs)} | 源文件标题: {meta.get('title')}")

    if not args.no_vector:
        docs = attach_vectors(docs)
    else:
        print("  [!] --no-vector：不生成向量，只更新文本字段。")
        print("     若原文有改动，存量向量将与新原文不对应，务必随后重跑一次带向量的导入。")

    problems, summary = verify_docs(docs)
    print("-" * 78)
    print("  体检:", json.dumps(summary, ensure_ascii=False))
    if problems:
        print("  [FAIL] 体检不通过，中止写库：")
        for p in problems:
            print("    -", p)
        return 1
    print("  [OK] 体检通过")

    if args.dry_run:
        print("-" * 78)
        print("  --dry-run：不写库。抽样 3 条：")
        for d in docs[:3]:
            print(f"    【条 {d['item_no']}】{d['category_name']}｜{d['scope']}"
                  f"｜tags={d['scope_tags']}｜vec={len(d.get('dense_vector') or [])}")
            print(f"      {d['text'][:60]}")
        return 0

    tool = get_compliance_tool()
    result = tool.upsert_items(docs)
    print("-" * 78)
    print(f"  写库结果: {json.dumps(result, ensure_ascii=False)}")
    if result["skipped"]:
        print(f"  [!] 跳过 {result['skipped']} 条（缺 item_no）")

    stats = tool.stats()
    print("  库内现状:", json.dumps(stats, ensure_ascii=False))
    if stats["total"] != len(docs):
        print(f"  [!] 库内条数({stats['total']})与本次组装条数({len(docs)})不一致 —— "
              f"可能存在历史残留条目，需人工核对。")
    if result["upserted"] == 0:
        print("  [OK] 幂等验证：本次 upserted=0，说明重复构建不会新增条目。")

    report = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_signature": compliance_config.signature(),
        "schema_version": SCHEMA_VERSION,
        "source_file": os.path.relpath(SOURCE_FILE, ROOT).replace("\\", "/"),
        "summary": summary,
        "upsert": result,
        "db_stats": stats,
        "elapsed_sec": round(time.time() - started, 1),
    }
    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告已落盘: {OUT_REPORT}")
    print(f"  总耗时: {report['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
