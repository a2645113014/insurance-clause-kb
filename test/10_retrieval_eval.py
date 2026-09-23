"""检索层评测台 —— 用 data/eval/retrieval_cases.jsonl 给每一个配置改动打分

指标
----
Recall@k : Top-k 里是否出现标准答案切片 —— 看「有没有找到」
MRR@10   : 标准答案排名的倒数平均 —— 看「排得够不够前」
两个必须一起看：Recall 高而 MRR 低 = 答案都召回了但全堆在后面，上游会把噪声喂给生成层。

三个打分阶段
------------
retrieval : 单路混合检索的输出（node_search_embedding）—— 权重只作用在这一层，主看这里
rrf       : 普通 + HyDE 两路经 RRF 融合后的输出（node_rrf）—— HyDE 消融看这里
rerank    : RRF 结果经重排 + 断崖 TopK 后的输出（node_rerank）—— 端到端上下文质量，含 TopK 消融

只测一个阶段会掩盖另一个阶段的问题：权重的影响会在 RRF/rerank 后被稀释，
所以要看清权重的作用必须看 retrieval 阶段；而断崖 TopK 的截断效果只在 rerank 阶段可见。

实现要点
--------
1) **直接驱动真实节点**，不复制一份检索逻辑 —— 评的必须是线上代码。
   权重、HyDE 开关、条数上限通过 `dataclasses.replace` 造配置后注入各节点模块的命名空间，
   因此跑完整消融矩阵既不用改 .env 也不用重启进程。
2) **三层缓存**：query 向量、HyDE 假设文档、reranker 打分。
   三者都不随权重变化，缓存后每换一个权重配置只重跑 Milvus 检索，秒级出分。
   缓存落 output/eval/_cache/eval_cache.json，可安全删除（删了重算而已）。

用法
----
  .venv\\Scripts\\python.exe test/10_retrieval_eval.py                     # 用 .env 配置跑一轮
  .venv\\Scripts\\python.exe test/10_retrieval_eval.py --weights 0.6,0.4 --no-hyde
  .venv\\Scripts\\python.exe test/10_retrieval_eval.py --ablation          # 权重 3 组 × HyDE 开/关 = 6 配置
  .venv\\Scripts\\python.exe test/10_retrieval_eval.py --ablation --with-rerank
  .venv\\Scripts\\python.exe test/10_retrieval_eval.py --no-cache          # 不使用缓存，全量重算
"""

import os
import re
import sys
import json
import time
import argparse
import dataclasses
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NO_PROXY", "*")

from app.conf.retrieval_config import retrieval_config  # noqa: E402
from app.lm.embedding_utils import generate_embeddings  # noqa: E402
from app.core.logger import logger  # noqa: E402

import app.query_process.agent.nodes.node_search_embedding as se_module  # noqa: E402
import app.query_process.agent.nodes.node_search_embedding_hyde as hyde_module  # noqa: E402
import app.query_process.agent.nodes.node_rrf as rrf_module  # noqa: E402
import app.query_process.agent.nodes.node_rerank as rerank_module  # noqa: E402

CASES_PATH = ROOT / "data" / "eval" / "retrieval_cases.jsonl"
CACHE_PATH = ROOT / "output" / "eval" / "_cache" / "eval_cache.json"
OUT_ROOT = ROOT / "output" / "eval"

RECALL_KS = (1, 3, 5, 10)
MRR_K = 10
HYDE_WORKERS = 6
ABLATION_WEIGHTS = [(0.8, 0.2), (0.6, 0.4), (0.5, 0.5)]


# ==================================================================== 缓存
class Cache:
    """三层缓存。sparse 向量的 key 在 JSON 里必须转字符串，取出时转回 int。"""

    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.vectors = {}
        self.hyde = {}
        self.rerank = {}
        if enabled and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.vectors = data.get("vectors", {})
                self.hyde = data.get("hyde", {})
                self.rerank = data.get("rerank", {})
                logger.info(f"缓存载入：向量 {len(self.vectors)} / HyDE {len(self.hyde)} / "
                            f"reranker 打分 {len(self.rerank)}")
            except Exception as e:
                logger.warning(f"缓存读取失败（将重建）：{e}")

    def save(self):
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"vectors": self.vectors, "hyde": self.hyde, "rerank": self.rerank}
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.info(f"缓存已落盘：{self.path}")

    # -------- 向量
    def embed(self, texts):
        """批量取向量：命中缓存直接用，未命中的一次性补算。"""
        missing = [t for t in texts if t not in self.vectors]
        if missing:
            emb = generate_embeddings(missing)
            for i, t in enumerate(missing):
                self.vectors[t] = {
                    "dense": emb["dense"][i],
                    "sparse": {str(k): v for k, v in emb["sparse"][i].items()},
                }
            logger.info(f"补算向量 {len(missing)} 条（缓存命中 {len(texts) - len(missing)} 条）")
        dense, sparse = [], []
        for t in texts:
            rec = self.vectors[t]
            dense.append(rec["dense"])
            sparse.append({int(k): v for k, v in rec["sparse"].items()})
        return {"dense": dense, "sparse": sparse}

    # -------- HyDE 假设文档
    def hyde_doc(self, query):
        return self.hyde.get(query, {}).get("doc")


def make_cached_embed(cache: Cache):
    """替换节点模块里的 generate_embeddings，使其走缓存。"""
    def _gen(texts):
        return cache.embed(texts)
    return _gen


def rerank_key(query, item):
    pk = item.get("pk") or item.get("doc_id") or item.get("text", "")[:48]
    return f"{query}||{pk}"


def make_cached_rerank(cache: Cache):
    """替换 step_2_rerank_docs：逻辑与线上一致，只是打分走缓存。"""
    def _step2(state, doc_items):
        query = state.get("rewritten_query") or state.get("original_query")
        if not query or not doc_items:
            return []
        pending = [it for it in doc_items if rerank_key(query, it) not in cache.rerank]
        if pending:
            from app.lm.reranker_utils import get_reranker_model
            model = get_reranker_model()
            scores = model.compute_score([[query, it["text"]] for it in pending])
            if not isinstance(scores, (list, tuple)):
                scores = [scores]
            for it, s in zip(pending, scores):
                cache.rerank[rerank_key(query, it)] = float(s)
        # ⚠️ 必须与线上 node_rerank.step_2_rerank_docs 保持同一套字段转发方式：
        # 用 dict(it) 整体转发而不是逐字段白名单 —— 白名单曾经把 clause_no 丢掉，
        # 导致评测台与线上行为分叉（评测台看不出引用编号缺失）。改线上时必须同步这里。
        scored = []
        for it in doc_items:
            record = dict(it)
            record["score"] = cache.rerank[rerank_key(query, it)]
            scored.append(record)
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored
    return _step2


# ==================================================================== 用例
def load_cases():
    if not CASES_PATH.exists():
        raise FileNotFoundError(
            f"标注集不存在：{CASES_PATH}\n"
            f"请先跑：.venv\\Scripts\\python.exe test/09_build_retrieval_cases.py"
        )
    cases = []
    with open(CASES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def prepare_hyde_docs(cases, cache: Cache):
    """并行预生成 HyDE 假设文档（这是唯一需要联网调 LLM 的步骤，最慢，必须缓存）。"""
    todo = [c["query"] for c in cases if c["query"] not in cache.hyde]
    if not todo:
        logger.info(f"HyDE 假设文档全部命中缓存：{len(cache.hyde)} 条")
        return
    logger.info(f"需生成 HyDE 假设文档 {len(todo)} 条，并发 {HYDE_WORKERS}")

    from app.lm.lm_utils import get_llm_client
    llm = get_llm_client(json_mode=False)

    def one(q):
        try:
            return q, hyde_module.step_1_create_hyde_doc(q)
        except Exception as e:
            logger.warning(f"HyDE 生成失败，按跳过处理：{q} / {e}")
            return q, None

    done = 0
    with ThreadPoolExecutor(max_workers=HYDE_WORKERS) as pool:
        for q, doc in pool.map(one, todo):
            cache.hyde[q] = {"doc": doc}
            done += 1
            if done % 25 == 0:
                logger.info(f"  HyDE 进度 {done}/{len(todo)}")
    sentinel = sum(1 for q in todo if not cache.hyde[q]["doc"])
    logger.info(f"HyDE 生成完成：{len(todo)} 条，其中 {sentinel} 条被判定为非条款问题（哨兵/空）")


# ==================================================================== 指标
def rank_of(gold_pk, items, key):
    """返回标准答案在结果里的排名（1 起）；未命中返回 None。"""
    for i, it in enumerate(items, start=1):
        if key(it) == gold_pk:
            return i
    return None


def metrics(ranks):
    n = len(ranks) or 1
    out = {f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 4) for k in RECALL_KS}
    out[f"mrr@{MRR_K}"] = round(
        sum((1.0 / r) if (r and r <= MRR_K) else 0.0 for r in ranks) / n, 4)
    out["hit@10"] = sum(1 for r in ranks if r and r <= MRR_K)
    out["total"] = len(ranks)
    return out


def subset_metrics(per_case, stage, source=None):
    ranks = [rec[stage]["rank"] for rec in per_case
             if source is None or rec["source"] == source]
    return metrics(ranks)


# ==================================================================== 跑一个配置
def run_config(cases, base_cfg, cache: Cache, weights, use_hyde, stages, max_k):
    cfg = dataclasses.replace(
        base_cfg,
        ranker_weights=weights,
        hyde_enabled=use_hyde,
        retrieval_limit=max_k,
        req_limit=max(max_k, base_cfg.req_limit),
    )
    # 注入配置与缓存：注入后节点内部所有阈值/开关都以本轮配置为准
    se_module.retrieval_config = cfg
    hyde_module.retrieval_config = cfg
    rerank_module.retrieval_config = cfg
    se_module.generate_embeddings = make_cached_embed(cache)
    hyde_module.generate_embeddings = make_cached_embed(cache)
    hyde_module.step_1_create_hyde_doc = lambda q: cache.hyde_doc(q)
    rerank_module.step_2_rerank_docs = make_cached_rerank(cache)

    per_case, started = [], time.time()
    for idx, case in enumerate(cases, 1):
        query, item_name, gold = case["query"], case["item_name"], case["gold_pk"]
        sid = f"eval_{case['case_id']}"
        state = {
            "session_id": sid,
            "original_query": query,
            "rewritten_query": query,
            "item_names": [item_name],
            "web_search_docs": [],
            "embedding_chunks": [],
            "hyde_embedding_chunks": [],
            "is_stream": False,
        }
        rec = {
            "case_id": case["case_id"], "source": case["source"], "query": query,
            "gold_pk": gold, "gold_clause_no_norm": case["gold_clause_no_norm"],
            "doc_id": case["doc_id"],
        }

        # ---- 阶段 1：单路混合检索
        se_res = se_module.node_search_embedding(state) or {}
        plain = se_res.get("embedding_chunks") or []
        state["embedding_chunks"] = plain
        rec["retrieval"] = {
            "rank": rank_of(gold, plain, lambda h: getattr(h, "id", None)),
            "top1": getattr(plain[0], "id", None) if plain else None,
            "n": len(plain),
        }

        # ---- 阶段 2：RRF 融合（HyDE 关时这一路为空，融合自然退化成单路）
        hy_res = hyde_module.node_search_embedding_hyde(state) or {}
        state["hyde_embedding_chunks"] = hy_res.get("hyde_embedding_chunks") or []
        rec["hyde_doc_len"] = len(hy_res.get("hyde_doc") or "")
        rrf_res = rrf_module.node_rrf(state) or {}
        fused = rrf_res.get("rrf_chunks") or []
        state["rrf_chunks"] = fused
        rec["rrf"] = {
            "rank": rank_of(gold, fused, lambda d: d.get("pk")),
            "top1": fused[0].get("pk") if fused else None,
            "n": len(fused),
        }

        # ---- 阶段 3：重排 + 断崖 TopK
        if "rerank" in stages:
            rr_res = rerank_module.node_rerank(state) or {}
            ranked = rr_res.get("reranked_docs") or []
            rec["rerank"] = {
                "rank": rank_of(gold, ranked, lambda d: d.get("pk")),
                "top1": ranked[0].get("pk") if ranked else None,
                "n": len(ranked),
            }

        per_case.append(rec)
        if idx % 50 == 0:
            logger.info(f"  评测进度 {idx}/{len(cases)}")

    summary = {
        "config": {
            "ranker_weights": list(weights),
            "hyde_enabled": use_hyde,
            "rerank_gap_abs": cfg.rerank_gap_abs,
            "rerank_gap_ratio": cfg.rerank_gap_ratio,
            "rerank_min_topk": cfg.rerank_min_topk,
            "rerank_max_topk": cfg.rerank_max_topk,
            "retrieval_limit": cfg.retrieval_limit,
        },
        "signature": cfg.signature(),
        "seconds": round(time.time() - started, 1),
        "stages": {},
        "by_source": {},
        "per_case": per_case,
    }
    for stage in stages:
        if stage not in per_case[0]:
            continue
        summary["stages"][stage] = subset_metrics(per_case, stage)
        summary["by_source"][stage] = {
            src: subset_metrics(per_case, stage, src)
            for src in sorted(set(r["source"] for r in per_case))
        }
    return summary


# ==================================================================== 报表
def print_table(results):
    header = f"{'配置':<34}{'阶段':<10}" + "".join(f"{'R@' + str(k):>9}" for k in RECALL_KS) + f"{'MRR@10':>9}{'命中/总':>11}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        cfg = r["config"]
        tag = f"w={cfg['ranker_weights'][0]:g}/{cfg['ranker_weights'][1]:g} hyde={'on' if cfg['hyde_enabled'] else 'off'}"
        for stage, m in r["stages"].items():
            line = f"{tag:<34}{stage:<10}" + "".join(f"{m['recall@' + str(k)]:>9.4f}" for k in RECALL_KS)
            line += f"{m['mrr@10']:>9.4f}{str(m['hit@10']) + '/' + str(m['total']):>11}"
            print(line)
        print()


def save_report(results, tag):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"{stamp}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = ["| 权重 | HyDE | 阶段 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | 命中/总数 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["config"]
        for stage, m in r["stages"].items():
            rows.append(
                f"| {c['ranker_weights'][0]:g}/{c['ranker_weights'][1]:g} "
                f"| {'开' if c['hyde_enabled'] else '关'} | {stage} "
                f"| {m['recall@1']:.4f} | {m['recall@3']:.4f} | {m['recall@5']:.4f} "
                f"| {m['recall@10']:.4f} | {m['mrr@10']:.4f} | {m['hit@10']}/{m['total']} |")
    (out_dir / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out_dir


def print_group_breakdown(result):
    """锚点集（真人问法）与扩量集（LLM 出题）分组看，避免一类拖累另一类看不出来。"""
    for stage, groups in result["by_source"].items():
        parts = []
        for src, m in groups.items():
            parts.append(f"{src}: R@5={m['recall@5']:.3f} MRR={m['mrr@10']:.3f} (n={m['total']})")
        print(f"  [{stage}] " + " | ".join(parts))


# ==================================================================== main
def main():
    parser = argparse.ArgumentParser(description="检索层评测台")
    parser.add_argument("--weights", type=str, default=None, help="覆盖权重，如 0.6,0.4")
    parser.add_argument("--no-hyde", action="store_true", help="关闭 HyDE 分支")
    parser.add_argument("--with-rerank", action="store_true", help="加上重排 + 断崖 TopK 阶段")
    parser.add_argument("--ablation", action="store_true", help="跑完整消融矩阵（权重 3 组 × HyDE 开/关）")
    parser.add_argument("--no-cache", action="store_true", help="不使用缓存")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条用例（冒烟测试用）")
    args = parser.parse_args()

    cases = load_cases()
    if args.limit:
        cases = cases[:args.limit]
    logger.info(f"标注集载入：{len(cases)} 条用例"
                f"（锚点 {sum(1 for c in cases if c['source'] == 'anchor')} / "
                f"LLM 出题 {sum(1 for c in cases if c['source'] == 'llm_gen')}）")

    stages = ["retrieval", "rrf"] + (["rerank"] if args.with_rerank else [])
    max_k = max(RECALL_KS)

    cache = Cache(CACHE_PATH, enabled=not args.no_cache)

    # 跑哪些配置
    if args.ablation:
        grid = [(w, hyde) for w in ABLATION_WEIGHTS for hyde in (True, False)]
    else:
        w = tuple(float(x) for x in args.weights.split(",")) if args.weights else tuple(retrieval_config.ranker_weights)
        grid = [(w, not args.no_hyde)]

    # HyDE 假设文档只生成一次（与权重无关），先把要用 HyDE 的查询全部备好
    if any(hyde for _, hyde in grid):
        prepare_hyde_docs(cases, cache)
        cache.save()

    results = []
    for weights, use_hyde in grid:
        logger.info(f"===== 评测配置：权重 {weights} / HyDE {'开' if use_hyde else '关'} / 阶段 {stages} =====")
        res = run_config(cases, retrieval_config, cache, weights, use_hyde, stages, max_k)
        results.append(res)
        cache.save()
        for stage, m in res["stages"].items():
            logger.info(f"  [{stage}] " + " ".join(f"{k}={v}" for k, v in m.items()))
        print_group_breakdown(res)

    print_table(results)
    out_dir = save_report(results, "ablation" if args.ablation else "single")
    logger.info(f"报告已落盘：{out_dir}")

    if len(results) > 1:
        base = results[0]
        best = max(results, key=lambda r: r["stages"].get("retrieval", {}).get("mrr@10", 0))
        print("\n最佳配置（按 retrieval MRR@10）：", best["signature"])
        print("对照基准：", base["signature"])


if __name__ == "__main__":
    main()
