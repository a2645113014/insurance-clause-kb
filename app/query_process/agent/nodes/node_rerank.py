from app.core import logger
from app.core.logger import node_log, step_log
from app.lm.reranker_utils import get_reranker_model
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.5
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 2

@step_log("step_1_merge_docs")
def step_1_merge_docs(state):
    # 分别获取状态中web_search_docs和rrf_chunks
    web_search_docs = state["web_search_docs"]
    rrf_chunks = state["rrf_chunks"]
    # 创建存储最终合并数据的列表
    doc_items = []
    # 遍历rrf_chunks，将其中的数据转换为固定的格式
    for chunk in rrf_chunks:
        # 获取chunk中存储的entity
        entity = chunk.get("entity") if isinstance(chunk, dict) and "entity" in chunk else chunk
        # 判断entity是否是字典
        if not isinstance(entity, dict):
            continue
        # 获取content
        content = entity.get("content")
        # 判断content是否为空
        if not content:
            continue
        # 分别获取chunk_id和title(item_name)
        chunk_id = entity.get("chunk_id") or entity.get("id")
        title = entity.get("title") or entity.get("item_name")
        # 将数据转为固定的结构并存储到doc_items
        doc_items.append(
            {
                "text": content,
                "title": title,
                "doc_id": chunk_id,
                "chunk_id": chunk_id,
                "url": "",
                "source": "local"
            }
        )
    # 遍历web_search_docs，将其中的数据转换为固定的格式
    for doc in web_search_docs:
        # 分别获取网络搜索结果中的snippet摘要，title标题，url网址
        snippet = (doc.get("snippet") or doc.get("content")).strip()
        title = (doc.get("title") or "").strip()
        url = (doc.get("url") or "").strip()
        # 使用固定的格式存储数据
        doc_items.append(
            {
                "text": snippet,
                "title": title,
                "doc_id": "",
                "chunk_id": "",
                "url": url,
                "source": "web"
            }
        )
    return doc_items

@step_log("step_2_rerank_docs")
def step_2_rerank_docs(state, doc_items):
    # 获取状态中的rewritten_query或者original_query
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    # 判断rewritten_query或doc_items是否为空
    if not rewritten_query or not doc_items:
        return []
    # 获取需要进行融合排序的文本
    texts = [item["text"] for item in doc_items]
    try:
        # 获取reranker模型对象
        reranker_model = get_reranker_model()
        # 将要进行重排序的数据转换为[[query, text],...]
        sentence_pairs = [[rewritten_query, text] for text in texts]
        # 对数据进行重排序，返回的是每条数据的分数组成的列表
        scores = reranker_model.compute_score(sentence_pairs)
        # 创建存储最终结果的列表
        scored_docs = []
        # 将scores、texts、doc_items进行压缩且遍历
        for score, text, item in zip(scores, texts, doc_items):
            scored_docs.append(
                {
                    "text": text,
                    "score": float(score),
                    "doc_id": item["doc_id"],
                    "chunk_id": item["chunk_id"],
                    "url": item["url"],
                    "title": item["title"],
                    "source": item["source"]
                }
            )
        # 将最终的结果进行排序
        scored_docs.sort(key=lambda doc: doc["score"], reverse=True)
        return scored_docs
    except Exception as e:
        logger.error(f"使用reranker模型重排序失败，{e}")
        for item in doc_items:
            item["score"] = 0.0
        return doc_items

@step_log("step_3_topk")
def step_3_topk(scored_docs):
    # 硬上限：最多取前10条，取全局常量与实际文档数的较小值（避免索引越界）
    # 注：max_topk从全局常量读取，不依赖外部状态，保证逻辑一致性
    max_topk = min(RERANK_MAX_TOPK, len(scored_docs))
    min_topk = RERANK_MIN_TOPK  # 硬下限：至少保留的文档数量（全局常量配置）
    gap_ratio = RERANK_GAP_RATIO  # 相对断崖阈值：分数下降的相对比例阈值（全局常量配置）
    gap_abs = RERANK_GAP_ABS  # 绝对断崖阈值：分数下降的绝对差值阈值（全局常量配置）
    # 创建表示真正动态topk的变量
    topk = max_topk
    # 判断现有的数据是否能够满足硬下限
    # 若能够满足，则获取动态的上限
    # 若无法满足，则提供现有的所有数据
    if topk > min_topk:
        # 循环遍历相邻的数据的分数差以及分数差比例
        for i in range(min_topk - 1, topk - 1):
            # 分别获取相邻的数据的分数
            score1 = scored_docs[i]["score"]
            score2 = scored_docs[i + 1]["score"]
            # 分别获取分数差值和分数差值的比例
            gap = score1 - score2
            rel = gap / (abs(score1) + 1e-6)
            # 判断分数差值和分数差值的比例是否大于等于绝对断崖阈值和相对断崖阈值
            if gap >= gap_abs or rel >= gap_ratio:
                # 获取动态的topk
                topk = i + 1
                break
    scored_docs = scored_docs[:topk]
    return scored_docs



@node_log("node_rerank")
def node_rerank(state: QueryGraphState):
    # 记录当前任务的状态为进行中
    add_running_task(state["session_id"], "node_rerank", state["is_stream"])
    # 阶段一：合并文档，[{text,title,doc_id,chunk_id,url,source}]
    doc_items = step_1_merge_docs(state)
    # 阶段二：对文档进行重排序，[{text,score,title,doc_id,chunk_id,url,source}]
    scored_docs = step_2_rerank_docs(state, doc_items)
    # 阶段三：动态 TopK
    topk_docs = step_3_topk(scored_docs)
    # 记录当前任务的状态为已完成
    add_done_task(state["session_id"], "node_rerank", state["is_stream"])
    return {"reranked_docs": topk_docs}

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)

    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [
        {"entity":{"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9}},
        {"entity":{"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8}},
        {"entity":{"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1}}  # 预期低分
    ]

    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"}  # 预期低分
    ]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",  # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:30]}...")

        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")