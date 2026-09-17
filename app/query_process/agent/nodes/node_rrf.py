from typing import List, Dict, Any

from app.core.logger import node_log, step_log
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 将向量检索结果进行处理，转换为只包含实体信息的列表
def _as_entity_list(chunks):
    # 创建存储转换之后实体字典的列表
    out: List[Dict[str, Any]] = []
    # 对chunks进行遍历
    for chunk in chunks:
        # 判断chunk是否为空
        if not chunk:
            continue
        # 创建存储转换之后实体字典的变量
        final_entity = {}
        # ==============================================
        # 情况A：处理 Milvus 返回的 Hit 对象（含 entity、chunk_id、distance）
        # ==============================================
        if hasattr(chunk, "entity") and hasattr(chunk, "chunk_id"):
            # 获取Hit对象的entity属性值
            entity = chunk.entity
            # 判断entity是否有to_dict属性
            if hasattr(entity, "to_dict"):
                final_entity = entity.to_dict()
            elif isinstance(entity, dict):
                final_entity = entity.copy()
            else:
                try:
                    final_entity = dict(entity)
                except:
                    pass
            # 在实体字典中补充chunk_id
            if "chunk_id" not in final_entity:
                final_entity["chunk_id"] = chunk.chunk_id
            # 在实体字典中补充score
            if hasattr(chunk, "distance"):
                final_entity["score"] = chunk.distance
        # ==============================================
        # 情况B：doc 已经是字典（模拟数据 / 已格式化数据）
        # ==============================================
        elif isinstance(chunk, dict):
            # 获取entity
            entity = chunk["entity"]
            # 判断entity是否是字典
            if isinstance(entity, dict):
                final_entity = entity.copy()
                # 在实体字典中补充chunk_id
                if "chunk_id" not in final_entity:
                    final_entity["chunk_id"] = chunk["chunk_id"]
                # 在实体字典中补充score
                if "distance" in chunk:
                    final_entity["score"] = chunk["distance"]
            else:
                final_entity = chunk
        # ==============================================
        # 情况C：支持 .get() 方法的其他对象
        # ==============================================
        elif hasattr(chunk, "get"):
            entity = chunk.get("entity") or chunk
            if isinstance(entity, dict):
                final_entity = entity
        # 判断final_entity是否为空，是否为字典
        if final_entity and isinstance(final_entity, dict):
            out.append(final_entity)
    return out

@step_log("step_2_rrf")
def step_2_rrf(source_weights, k, max_results):
    # 创建存储分数和数据的字典
    score_map = {}
    chunk_map = {}
    # 对source_weights进行遍历
    for chunks, weight in source_weights:
        # 对chunks进行遍历，rank表示排名，
        for rank, chunk in enumerate(chunks, start=1):
            # 获取chunk_id
            chunk_id = chunk["chunk_id"]
            # 计算chunk_id所对应数据的分数
            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + weight * (1.0 / (rank + k))
            # 存储chunk_id所对应的数据
            chunk_map.setdefault(chunk_id, chunk)
    # 创建存储数据以及所对应分数的列表
    merged_chunks = []
    # 对score_map进行遍历
    for chunk_id, score in score_map.items():
        # 从chunk_map中通过chunk_id获取所对应的数据
        chunk = chunk_map.get(chunk_id)
        # 存储数据以及所对应的分数
        merged_chunks.append((chunk, score))
    # 对merged_chunks中的数据进行排序
    merged_chunks.sort(key=lambda item: item[1], reverse=True)
    # 对merged_chunks进行截断，保留max_results个结果
    merged_chunks = merged_chunks[:max_results]
    return merged_chunks

@node_log("node_rrf")
def node_rrf(state: QueryGraphState):
    # 记录当前任务的状态为进行中
    add_running_task(state["session_id"], "node_rrf", state["is_stream"])
    # 步骤1：将向量检索结果进行处理，转换为只包含实体信息的列表
    embedding_chunks = _as_entity_list(state.get("embedding_chunks"))
    hyde_embedding_chunks = _as_entity_list(state.get("hyde_embedding_chunks"))
    # 设置多路检索的结果的权重
    source_weights = [
        (embedding_chunks, 1.0),
        (hyde_embedding_chunks, 1.0),
    ]
    # 步骤2：进行rrf融合以及排序
    rrf_result = step_2_rrf(source_weights, k=60, max_results=10)
    # 步骤3：获取rrf融合排序之后的数据
    rrf_chunks = [chunk for chunk, score in rrf_result]
    # 记录当前任务的状态为已完成
    add_done_task(state["session_id"], "node_rrf", state["is_stream"])
    return {"rrf_chunks": rrf_chunks}

if __name__ == "__main__":
    init_state = {
        "session_id": "abc",
        "rewritten_query": "HAK 180 烫金机如何安全使用",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False,
    }
    embedding_chunks = node_search_embedding(init_state)["embedding_chunks"]
    hyde_embedding_chunks = node_search_embedding_hyde(init_state)["hyde_embedding_chunks"]
    init_state["embedding_chunks"] = embedding_chunks
    init_state["hyde_embedding_chunks"] = hyde_embedding_chunks
    result = node_rrf(init_state)
    print(result)