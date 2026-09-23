import os
import sys
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.ledger_utils import load_product_name, sanitize_item_name
from app.utils.task_utils import add_running_task, add_done_task

# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500

@step_log("step_1_get_chunks_and_file_title")
def step_1_get_chunks_and_file_title(state):
    # 分别获取chunks和file_tile
    file_title = state.get("file_title")
    chunks = state.get("chunks")
    # 判断chunks是否为空
    if not chunks:
        raise RuntimeError("chunks为空，即没有任何的切片")
    # 判断file_title是否为空
    if not file_title:
        file_title = Path(state.get("md_path")).stem
        state["file_title"] = file_title
    return chunks, file_title

@step_log("step_2_build_context")
def step_2_build_context(chunks):
    # 创建存储切片处理之后的数据的变量
    parts = []
    # 记录当前切片的字符数
    total_chars = 0
    # 对前DEFAULT_ITEM_NAME_CHUNK_K个chunks进行遍历
    for idx, chunk in enumerate(chunks, start=1):
        # 获取切片的标题和内容。
        # 字段名与 02_clause_schema.json 对齐：切分节点的产出是 clause_title / clause_path 与 text。
        # 用 get 而非下标 —— 章级切片可能没有小标题，此时标题退化为层级路径，不因缺字段中断识别
        title = chunk.get("clause_title") or chunk.get("clause_path") or ""
        content = chunk.get("text") or ""
        # 将切片组装为：切片:idx，标题:title，内容:content
        data = f"切片：{idx}，标题：{title}，内容：{content}"
        # 存储data
        parts.append(data)
        # 记录已存储的切片的总字符数
        total_chars += len(data)
        # 判断total_chars是否超过了指定的阈值CONTEXT_TOTAL_MAX_CHARS
        if total_chars >= CONTEXT_TOTAL_MAX_CHARS:
            break
    # 将处理之后的切片拼接为字符串
    context = "\n\n".join(parts)
    # 进行兜底处理，防止上下文超过指定阈值
    context = context[:CONTEXT_TOTAL_MAX_CHARS]
    return context

@step_log("step_3b_llm_fallback")
def _call_llm_for_item_name(context, file_title):
    """台账缺行时的兜底路径：让大模型从条款正文里识别产品名。"""
    # 分别获取用户提示词和系统提示词
    human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
    system_prompt = load_prompt("product_recognition_system")
    # 将human_prompt和system_prompt组成提示词
    messages = [
        SystemMessage(system_prompt),
        HumanMessage(human_prompt),
    ]
    # 获取大模型对象
    llm = get_llm_client()
    # 创建链对象
    chain = llm | StrOutputParser()
    # 调用链对象
    return chain.invoke(messages)


@step_log("step_3_resolve_item_name")
def step_3_resolve_item_name(context, file_title):
    """确定 item_name：**台账备案名优先**，大模型只做兜底。

    为什么不让大模型主导
--------------------
    item_name 不是展示字段，是**键**：条款库靠它做检索过滤（``expr = item_name in [...]``），
    产品名库靠它做向量对齐，MongoDB 规则表靠它精确查规则。三处必须逐字一致。

    而大模型输出格式不受控 —— 换模型后同一份文档开始返回带成对引号的空串 ``""``，
    5 份文档的 item_name 因此塌缩成同一个键：265 个切片挤在一起（跨产品互相召回），
    产品名库多出孤儿行，6 款产品的理赔规则全部查不到。**服务不报错，只是答不出东西。**

    台账「产品名称」列是监管备案名，实测与历史大模型产出 17/17 逐字一致，
    所以改用它不改变任何已有数据，却让 item_name 摆脱对模型行为的依赖。

    :return: (item_name, 来源标记) —— 来源会打进日志，便于事后判断走没走兜底
    """
    from_ledger = load_product_name(file_title)
    if from_ledger:
        return from_ledger, "ledger"

    # 台账缺行（新采语料尚未登记）时才调大模型，且必须清洗后再用
    raw = _call_llm_for_item_name(context, file_title)
    logger.warning(
        f"台账中未找到 file_title={file_title}，item_name 回退大模型识别：{raw!r}"
    )
    return sanitize_item_name(raw, file_title), "llm"

@step_log("step_4_update_chunks_and_state")
def step_4_update_chunks_and_state(state, item_name, chunks):
    # 更新状态中的item_name
    state["item_name"] = item_name
    # 更新每个切片，添加item_name信息
    for chunk in chunks:
        chunk["item_name"] = item_name
    # 更新状态中的chunks
    state["chunks"] = chunks

@step_log("step_5_generate_embeddings")
def step_5_generate_embeddings(item_name):
    # 将item_name生成稠密向量和稀疏向量
    embeddings = generate_embeddings([item_name])
    # 返回稠密向量和稀疏向量
    return embeddings["dense"][0], embeddings["sparse"][0]

@step_log("step_6_save_to_vector_db")
def step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector):
    # 获取milvus的客户端对象
    milvus_client = get_milvus_client()
    # 若milvus中没有kb_item_names集合，则创建
    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
        # 设置集合的结构
        schema = milvus_client.create_schema(
            auto_id=True,  # 集合中的主键自增
            enable_dynamic_field=True,  # 开启动态字段，允许向向量数据库不存在的字段进行赋值
        )
        # 设置集合的字段
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 设置集合的索引
        index_params = milvus_client.prepare_index_params()
        # 设置稠密向量的索引
        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",
            index_name="dense_vector_index",
            metric_type="COSINE",
        )
        # 设置稀疏向量的索引
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_vector_index",
            metric_type="IP",
        )
        # 创建集合
        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,
            index_params=index_params
        )
    # 将item_name相关的数据删除
    milvus_client.delete(
        collection_name=milvus_config.item_name_collection,
        filter=f"item_name == '{item_name}'"
    )
    # 准备数据
    data = {
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector,
    }
    # 保存数据
    milvus_client.insert(
        collection_name=milvus_config.item_name_collection,
        data=[data]
    )


@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)

    产出 state["item_name"]，并把它回填到每个切片上。这个字段同时承担三个角色，
    所以它的取值必须是**确定性**的：

    1. 检索过滤键 —— 查询侧用它做 `expr = item_name in [...]`，隔离到用户选中的产品；
    2. 产品名库的向量对齐目标 —— 用户口语化的产品名靠它对齐到库里的产品；
    3. MongoDB 规则表的查询键 —— 理赔材料/时限规则按它精确查。

    取值来源：语料台账的「产品名称」列（监管备案名）。大模型只在台账缺行时兜底，
    且兜底结果必须过 sanitize_item_name —— 大模型输出格式不受控，实测换模型后
    会返回带引号的空串，一旦这种值当上键，整条链路会静默失效（详见 step_3）。
    """
    # 记录任务的状态为运行中
    add_running_task(state["task_id"], "node_item_name_recognition")
    # 步骤1：校验和取值 （file_title,chunks）
    chunks, file_title = step_1_get_chunks_and_file_title(state)
    # 步骤2：构建上下文环境  chunks -> top 5 -> 拼接成context文本
    context = step_2_build_context(chunks)
    # 步骤3：确定 item_name —— 台账备案名优先，大模型仅兜底
    item_name, item_name_source = step_3_resolve_item_name(context, file_title)
    logger.info(f"item_name 来源={item_name_source} → {item_name}")
    # 步骤4：产品主体回填，修改state chunks -> item_name
    step_4_update_chunks_and_state(state, item_name, chunks)
    # 步骤5：item_name生成向量（稠密/稀疏）
    dense_vector, sparse_vector = step_5_generate_embeddings(item_name)
    # 步骤6：存储向量到向量数据库 kb_item_name (id / file_title / item_name / 稠密 和 稀疏)
    step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector)
    # 记录任务的状态为已完成
    add_done_task(state["task_id"], "node_item_name_recognition")
    return state

if __name__ == "__main__":
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切分节点产出，字段为 clause_title / clause_path / text）
            "chunks": [
                {
                    "clause_title": "产品简介",
                    "clause_path": "1 产品说明 > 1.1 产品简介",
                    "text": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "clause_title": "拍照功能",
                    "clause_path": "1 产品说明 > 1.2 拍照功能",
                    "text": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "clause_title": "电池参数",
                    "clause_path": "1 产品说明 > 1.3 电池参数",
                    "text": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)