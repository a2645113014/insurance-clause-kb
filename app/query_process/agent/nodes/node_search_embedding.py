from app.clients.milvus_utils import hybrid_search, get_milvus_client, create_hybrid_search_requests
from app.conf.milvus_config import milvus_config
from app.core.logger import logger, node_log
from app.lm.embedding_utils import generate_embeddings
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task


@node_log("node_search_embedding")
def node_search_embedding(state: QueryGraphState):
    # 记录当前任务的状态为进行中
    add_running_task(state["session_id"], "node_search_embedding", state["is_stream"])
    # 分别获取rewritten_query和item_names
    rewritten_query = state["rewritten_query"]
    item_names = state["item_names"]
    # 判断item_names是否为空
    if not item_names:
        logger.warning("item_names为空")
        return {"embedding_chunks": []}
    # 将rewritten_query转换为向量
    embeddings = generate_embeddings([rewritten_query])
    # 分别获取稠密向量和稀疏向量
    dense_vector = embeddings["dense"][0]
    sparse_vector = embeddings["sparse"][0]
    # 将item_names中的所有的产品主体拼接为字符串
    expr_data = ", ".join(f"'{item_name}'" for item_name in item_names)
    # 将item_name作为检索的条件，拼接条件
    expr = f"item_name in [{expr_data}]"
    # 设置稠密向量和稀疏向量的检索方式
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=expr,
        limit=5
    )
    # 获取Milvus客户端对象
    milvus_client = get_milvus_client()
    # 进行混合检索
    results = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.8, 0.2),
        norm_score=True,
        limit=5,
        output_fields=["chunk_id", "content", "item_name"]
    )
    # 记录当前任务的状态为已完成
    add_done_task(state["session_id"], "node_search_embedding", state["is_stream"])
    return {"embedding_chunks": results[0] if results else []}


if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机使用说明",  # 模拟改写后的查询
        "item_names": ["HAK 180 烫金机"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(chunks)
        # print(chunks[0])
        # print(chunks[0].entity.to_dict()["content"])
        # print(chunks[0].entity.to_dict()["chunk_id"])
        # print(chunks[0].entity.to_dict()["item_name"])
    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)