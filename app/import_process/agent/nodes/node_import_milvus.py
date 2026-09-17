import sys

from pymilvus import DataType, MilvusClient

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

# Milvus中存储切片相关数据的集合的名称
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection

@step_log("step_1_validate_input")
def step_1_validate_input(state):
    # 获取chunks
    chunks = state.get("chunks")
    # 判断chunks是否为空
    if not chunks:
        raise ValueError("切片数据异常，请重试")
    return chunks

@step_log("step_2_prepare_collection")
def step_2_prepare_collection(milvus_client: MilvusClient):
    # 判断Milvus中切片集合是否存在，若不存在则创建
    if not milvus_client.has_collection(CHUNKS_COLLECTION_NAME):
        # 创建结构对象
        schema = milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        # 添加字段
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 准备索引
        index_params = milvus_client.prepare_index_params()
        # 添加索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="COSINE",
        )
        # 添加索引
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        # 创建集合
        milvus_client.create_collection(
            collection_name=CHUNKS_COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )

@step_log("step_3_delete_old_data")
def step_3_delete_old_data(milvus_client: MilvusClient, item_name: str):
    # 删除数据
    milvus_client.delete(
        collection_name=CHUNKS_COLLECTION_NAME,
        filter=f"item_name == '{item_name}'",
    )
    # 重新加载集合
    milvus_client.load_collection(
        collection_name=CHUNKS_COLLECTION_NAME,
    )

@step_log("step_4_insert_collections")
def step_4_insert_collections(milvus_client: MilvusClient, chunks):
    # 添加数据
    insert_result = milvus_client.insert(
        collection_name=CHUNKS_COLLECTION_NAME,
        data=chunks,
    )
    logger.info(f"成功添加了{insert_result['insert_count']}条数据")
    # 获取添加的数据的id并回填
    ids = insert_result["ids"]
    if len(ids) == len(chunks):
        for index, chunk in enumerate(chunks):
            chunk["chunk_id"] = ids[index]
    return chunks


@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    # 记录任务状态为运行中
    add_running_task(state["task_id"], "node_import_milvus")
    # 步骤1：检查数据 chunks 是否存在
    chunks = step_1_validate_input(state)
    # 步骤2：前置准备工作 创建 Milvus 集合和字段
    milvus_client = get_milvus_client()
    step_2_prepare_collection(milvus_client)
    # 步骤3：删除旧数据
    step_3_delete_old_data(milvus_client, state['item_name'])
    # 步骤4：保存新数据
    with_id_chunks = step_4_insert_collections(milvus_client, chunks)
    # 更新状态
    state["chunks"] = with_id_chunks
    # 记录任务状态为已完成
    add_done_task(state["task_id"], "node_import_milvus")
    return state

if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "item_name":"测试项目_Milvus",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
,
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus2",  # 必须有 item_name，用于幂等清理
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf2",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")