import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# 支持以 `python test/06_import_test.py` 直接运行：
# 直接执行脚本时 sys.path[0] 指向 test/ 目录，找不到 app 包，需手动把项目根加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger
from app.utils.path_util import PROJECT_ROOT

"""
条款语料批量导入入口：把 data/clauses/pdf 下的全部条款 PDF 跑完整导入链路
（entry → pdf2md → md_img → document_split → item_name_recognition → bge_embedding → import_milvus）
并写入 Milvus 的 insurance_clauses 集合。

用法（在项目根目录执行）：
    .venv\\Scripts\\python.exe test\\06_import_test.py                  # 全量：遍历语料目录
    .venv\\Scripts\\python.exe test\\06_import_test.py --only 03        # 只跑文件名含 "03" 的，用于单份调试

四个设计要点：
1. 全量必须跑在同一个进程里。BGE-M3 是模块级单例（app/lm/embedding_utils.py），进程内只加载一次；
   若改成循环起子进程，18 次模型加载会把总耗时从分钟级推到小时级。
2. 每份 PDF 独立 try/except。单份失败不阻断整批 —— 16 号是已知的「字体缺 ToUnicode 映射」样本，
   会被解析节点主动拒收，这属于预期行为而不是批量任务失败。
3. 每份的 task_id 必须唯一。进度追踪表（app/utils/task_utils.py）按 task_id 分组，复用同一个 id
   会让各份的节点进度互相覆盖，排查失败时看不到真实进度。
4. 跑完写 import_manifest.json（逐份状态 + 切片数 + pk 集合指纹）。再跑第二遍时脚本会自动与上一遍
   比对指纹并把结论打出来 —— 幂等验证从「人工盯着条数」变成「脚本直接给结论」。
"""

# 语料目录：条款 PDF 的落盘位置（由 data/collect_clauses.py 幂等维护）
PDF_DIR = Path(PROJECT_ROOT) / "data" / "clauses" / "pdf"

# 幂等比对只看 pk 集合指纹，不比对切片正文 —— 正文变化会体现在 pk 数量或 clause_no 上
PK_DIGEST_LEN = 16


def parse_args():
    parser = argparse.ArgumentParser(description="条款语料批量导入（写入 Milvus insurance_clauses）")
    parser.add_argument(
        "--only",
        default="",
        help="只跑文件名包含该字符串的 PDF，用于单份调试（此时清单写入 import_manifest_partial.json）",
    )
    return parser.parse_args()


def run_one_pdf(pdf_path_obj: Path, seq: int, batch_dir_obj: Path):
    """
    跑一份 PDF 的完整导入链路，返回本次导入的结果字典。

    用 stream_mode="updates" 而不是 "values"：
    updates 只把「本节点新产生的增量」交给调用方，values 则在每个节点结束后把整个 state 深拷贝一份。
    state 里装着几百条 1024 维向量时，后者为了打印一条进度就要反复拷贝整份状态，白白付出开销。
    """
    result = {
        "seq": seq,
        "file": pdf_path_obj.name,
        "doc_id": pdf_path_obj.stem,
        "status": "OK",
        "chunks": 0,
        "item_name": "",
        "pk_digest": "",
        "nodes": [],
        "last_done_node": "",
        "seconds": 0.0,
        "error": "",
    }
    # 每份任务独立 task_id，避免进度表互相覆盖
    state = ImportGraphState({
        "task_id": f"batch_{seq:02d}_{pdf_path_obj.stem}",
        "local_file_path": str(pdf_path_obj),
        "local_dir": str(batch_dir_obj),
        "is_pdf_read_enabled": True,   # node_entry 会按后缀重写这两个开关，这里显式给出意图
        "is_md_read_enabled": False,
    })

    started_at = time.time()
    final_chunks = []
    try:
        for update in kb_import_app.stream(state, stream_mode="updates"):
            for node_name, node_output in update.items():
                result["nodes"].append(node_name)
                if not isinstance(node_output, dict):
                    continue
                # 记录最后一个产出 chunks / item_name 的节点结果
                if node_output.get("chunks"):
                    final_chunks = node_output["chunks"]
                if node_output.get("item_name"):
                    result["item_name"] = node_output["item_name"]
    except Exception as e:
        result["status"] = "FAILED"
        # 注意：updates 流里不会出现「抛异常的那个节点」—— 异常时该节点的增量根本不会产出。
        # 所以这里只能记录「最后一个成功完成的节点」，真正的失败点在其紧邻的下游。
        # 字段名据此叫 last_done_node，不叫 failed_node，避免排查时被误导到错误的节点上。
        result["last_done_node"] = result["nodes"][-1] if result["nodes"] else "（无节点完成）"
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        result["seconds"] = round(time.time() - started_at, 1)

    if result["status"] == "OK":
        pk_list = sorted(str(chunk.get("pk")) for chunk in final_chunks if chunk.get("pk"))
        result["chunks"] = len(final_chunks)
        result["pk_digest"] = hashlib.sha256("\n".join(pk_list).encode("utf-8")).hexdigest()[:PK_DIGEST_LEN]
        # 切片为空说明链路走完了但没产出，按失败计 —— 空文档进库等于什么都没导
        if not final_chunks:
            result["status"] = "FAILED"
            result["last_done_node"] = "（全链路已完成）"
            result["error"] = "链路执行完成但切片数为 0，未产生任何可入库数据"
    return result


def compare_with_previous(previous_manifest, results):
    """把本遍结果与上一遍清单比对，给出幂等结论。返回 (新增, 一致, 变化) 三个 doc_id 列表"""
    previous_by_doc = {
        doc.get("doc_id"): doc for doc in (previous_manifest or {}).get("documents", [])
    }
    added, unchanged, changed = [], [], []
    for result in results:
        if result["status"] != "OK":
            continue
        previous = previous_by_doc.get(result["doc_id"])
        if not previous or previous.get("status") != "OK":
            added.append(result["doc_id"])
        elif previous.get("pk_digest") == result["pk_digest"]:
            unchanged.append(result["doc_id"])
        else:
            changed.append(result["doc_id"])
    return added, unchanged, changed


def report_milvus_stats():
    """从 Milvus 侧独立复核落库结果，给出不看应用日志也能对上的条数基线"""
    from app.clients.milvus_utils import get_milvus_client
    from app.conf.milvus_config import milvus_config

    collection_name = milvus_config.chunks_collection
    try:
        client = get_milvus_client()
    except Exception as e:
        logger.error(f"连接 Milvus 失败，跳过落库统计：{e}")
        return

    if not client.has_collection(collection_name):
        logger.warning(f"集合不存在，跳过落库统计：{collection_name}")
        return

    client.load_collection(collection_name=collection_name)

    # 【实测坑】这里必须显式指定 Strong 一致性。
    # 集合默认 Bounded 一致性：紧跟本遍 delete + insert 之后的普通 query，可能命中「删除已提交、
    # 插入尚未可见」的中间快照而返回 0 条；而 count(*) 走聚合路径读的是最新状态，于是出现
    # 「count(*) = 44、明细却 0 条」的自相矛盾结果（同一份数据两种查法给出两个答案）。
    # 实测证据：脚本内紧跟着查返回空，27 秒后同样的查询返回 44 条。
    strong = "Strong"

    count_result = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["count(*)"],
        consistency_level=strong,
    )
    total = count_result[0].get("count(*)") if count_result else 0

    # 分页拉明细：Milvus 单次 offset+limit 有上限（16384），条款量级下按 1000 一页拉几轮即可
    fields = ["doc_id", "clause_no_norm", "clause_type", "needs_review"]
    rows, offset, page_size = [], 0, 1000
    while True:
        batch = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=fields,
            limit=page_size,
            offset=offset,
            consistency_level=strong,
        )
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    doc_count = len({row.get("doc_id") for row in rows})
    # 「覆盖多少条款」的口径：同一条款号在不同文档里是两个条款，故按 (doc_id, clause_no_norm) 组合去重
    clause_count = len({(row.get("doc_id"), row.get("clause_no_norm")) for row in rows})
    review_count = sum(1 for row in rows if row.get("needs_review"))
    type_counter = Counter(row.get("clause_type") or "（空）" for row in rows)

    logger.info("-" * 80)
    logger.info("===== Milvus 落库复核（insurance_clauses）=====")
    logger.info(f"集合：{collection_name}")
    logger.info(f"切片总条数 count(*)：{total}")
    logger.info(f"覆盖文档数（distinct doc_id）：{doc_count}")
    logger.info(f"覆盖条款数（distinct doc_id + clause_no_norm）：{clause_count}")
    logger.info(f"标记待人工复核（needs_review）：{review_count}")
    logger.info(f"条款类型分布（共 {len(type_counter)} 类）：")
    for clause_type, count in type_counter.most_common():
        logger.info(f"    {count:>5}  {clause_type}")
    logger.info("-" * 80)


def main():
    args = parse_args()

    pdf_list = sorted(PDF_DIR.glob("*.pdf"))
    if args.only:
        pdf_list = [pdf_obj for pdf_obj in pdf_list if args.only in pdf_obj.name]
    if not pdf_list:
        logger.error(f"没有匹配的 PDF。语料目录：{PDF_DIR}，--only={args.only!r}")
        return 1

    # 中间产物（MD / images / chunks.json）集中放在按日期分的批次目录下，
    # 不与 output/ 下历史通用商品手册的产物混在一起，便于整批清理与回溯
    batch_dir_obj = Path(PROJECT_ROOT) / "output" / f"batch_{datetime.now().strftime('%Y%m%d')}"
    batch_dir_obj.mkdir(parents=True, exist_ok=True)

    manifest_name = "import_manifest_partial.json" if args.only else "import_manifest.json"
    manifest_path_obj = batch_dir_obj / manifest_name

    # 读上一遍清单：存在就做幂等比对，不存在（首次全量）就跳过
    previous_manifest = None
    if manifest_path_obj.exists():
        try:
            previous_manifest = json.loads(manifest_path_obj.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"上一遍清单解析失败，本遍不做幂等比对：{e}")

    logger.info("=" * 80)
    logger.info(f"条款语料批量导入：共 {len(pdf_list)} 份 | 语料目录 {PDF_DIR}")
    logger.info(f"中间产物目录：{batch_dir_obj}")
    logger.info(f"清单文件：{manifest_path_obj}")
    logger.info("=" * 80)

    results = []
    batch_started_at = time.time()
    for seq, pdf_path_obj in enumerate(pdf_list, start=1):
        logger.info(f"[{seq}/{len(pdf_list)}] 开始导入：{pdf_path_obj.name}")
        result = run_one_pdf(pdf_path_obj, seq, batch_dir_obj)
        results.append(result)
        if result["status"] == "OK":
            logger.success(
                f"[{seq}/{len(pdf_list)}] 完成 | {result['chunks']} 片 | "
                f"{result['seconds']}s | 产品名「{result['item_name']}」| {pdf_path_obj.name}"
            )
        else:
            logger.error(
                f"[{seq}/{len(pdf_list)}] 失败 | 完成到「{result['last_done_node']}」，其后节点失败 | "
                f"{result['seconds']}s | {pdf_path_obj.name} | {result['error']}"
            )

    ok_results = [r for r in results if r["status"] == "OK"]
    failed_results = [r for r in results if r["status"] != "OK"]
    total_chunks = sum(r["chunks"] for r in ok_results)
    batch_seconds = round(time.time() - batch_started_at, 1)

    logger.info("=" * 80)
    logger.info(
        f"批量导入结束：成功 {len(ok_results)} / 失败 {len(failed_results)} / 合计 {len(results)} 份，"
        f"切片 {total_chunks} 条，总耗时 {batch_seconds}s"
    )
    if failed_results:
        logger.warning("失败明细：")
        for result in failed_results:
            logger.warning(
                f"    {result['file']} | 完成到「{result['last_done_node']}」 | {result['error']}"
            )

    # 幂等比对：本遍与上一遍的逐份 pk 指纹是否一致
    if previous_manifest:
        added, unchanged, changed = compare_with_previous(previous_manifest, results)
        logger.info(
            f"与上一遍清单比对：新增 {len(added)} 份 / 指纹一致 {len(unchanged)} 份 / "
            f"指纹变化 {len(changed)} 份"
        )
        if added:
            logger.info(f"    新增：{added}")
        if changed:
            # 指纹变化意味着同一份文档的切片边界或数量变了，属于需要确认的改动（切分逻辑调整等）
            logger.warning(f"    指纹变化（切片边界或数量已变，需确认是否预期）：{changed}")
        if not added and not changed:
            logger.success("    幂等校验通过：全部文档的 pk 集合与上一遍完全一致")
    else:
        logger.info("未找到上一遍清单（本次为首次全量），跳过幂等比对")

    # 写出本遍清单（清单本身就是这批导入的台账，供后续回滚与复现）
    manifest = {
        "batch_dir": str(batch_dir_obj),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pdf_dir": str(PDF_DIR),
        "partial": bool(args.only),
        "totals": {
            "documents": len(results),
            "ok": len(ok_results),
            "failed": len(failed_results),
            "chunks": total_chunks,
            "seconds": batch_seconds,
        },
        "documents": results,
    }
    manifest_path_obj.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"本遍清单已写出：{manifest_path_obj}")

    # 从数据库侧独立复核，避免只信应用日志（复核本身失败不应改变导入的成败结论）
    try:
        report_milvus_stats()
    except Exception as e:
        logger.error(f"落库复核失败（不影响本次导入结果判定）：{e}", exc_info=True)

    logger.info("=" * 80)
    return 0 if not failed_results else 2


if __name__ == "__main__":
    sys.exit(main())
