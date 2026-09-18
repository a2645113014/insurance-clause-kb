from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from app.conf.milvus_config import milvus_config
# 检索阈值（权重 / 召回条数 / HyDE 开关）统一由配置面板提供
from app.conf.retrieval_config import retrieval_config
from app.core.load_prompt import load_prompt
# 原写法误从 torch.distributed.checkpoint 借用 logger（标准 logging 对象），
# 虽能打印但语义错位、且与本项目日志格式不一致，改回项目自己的 logger
from app.core.logger import logger, node_log, step_log
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

# HyDE 改写版 prompt 要求「不属于条款文本能回答的问题」直接输出这个哨兵值。
# 它不是一个假设性文档，而是整个 HyDE 分支的短路信号 —— 必须在这里拦掉：
# 否则会把这串字面量当作文档送去向量化，检索回一堆语义无关的切片，
# 再经 RRF 混进最终结果，等于给答案里掺噪声。
NOT_A_CLAUSE_QUERY_SENTINEL = "NOT_A_CLAUSE_QUERY"
# 改写版 prompt 要求的假设性文档长度区间（用于校验，越界只告警不拦截）
HYDE_DOC_MIN_CHARS = 60
HYDE_DOC_MAX_CHARS = 400


@step_log("step_1_create_hyde_doc")
def step_1_create_hyde_doc(rewritten_query):
    """生成「假设性条款表述」。返回 None 表示该问题应跳过 HyDE 分支。

    为什么要生成「条款文体」而不是「假设答案」：
    常规 HyDE 让模型写一个假设答案，但保险场景下用户问的是口语
    （「犹豫期有多久」），目标语料是法律文体（「自您签收本合同之日起十五日内」），
    两者文体分布差得远。改成让模型直接写「这段话如果写在条款里会怎么写」，
    假设文档就落进了目标语料的文体分布里，向量检索才拉得近。
    """
    # 获取提示词
    prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    # 获取大模型对象
    llm = get_llm_client()
    # 调用大模型
    response = llm.invoke(prompt)
    # 获取模型返回的结果，即假设性文档
    hyde_doc = (response.content or "").strip()
    # 模型判定该问题不属于条款能回答的范围（产品推荐 / 保费测算 / 产品对比 / 投诉渠道等）
    if NOT_A_CLAUSE_QUERY_SENTINEL in hyde_doc:
        logger.info(f"HyDE 判定为非条款问题，跳过 HyDE 分支：{rewritten_query}")
        return None
    # 空文档同样短路：拿空串去向量化没有意义，只会给 HyDE 分支掺噪声
    if not hyde_doc:
        logger.warning(f"HyDE 返回空文档，跳过 HyDE 分支：{rewritten_query}")
        return None
    # 长度越界不是错误（模型偶尔写长写短），只留痕便于后续调 prompt
    if not (HYDE_DOC_MIN_CHARS <= len(hyde_doc) <= HYDE_DOC_MAX_CHARS):
        logger.warning(f"HyDE 假设文档长度 {len(hyde_doc)} 未落在建议区间 "
                       f"[{HYDE_DOC_MIN_CHARS}, {HYDE_DOC_MAX_CHARS}]：{rewritten_query}")
    return hyde_doc

@step_log("step_2_search_embedding_hyde")
def step_2_search_embedding_hyde(
    rewritten_query: str,
    hyde_doc: str,
    item_names=None,
    req_limit: int = None,      # 稠密向量和稀疏向量各自召回的数据量，None 时取配置值
    limit: int = None,          # 混合检索最终返回的数据量，None 时取配置值
    ranker_weights=None,        # 加权融合权重，None 时取配置值（改造前硬编码 (0.8, 0.2)）
    norm_score: bool = True,    # 必须开启：不归一化时权重会被两个向量的分数量级差异淹没
    output_fields=None,         # 默认字段在函数体内给（避免可变对象当默认参数被跨调用共享）
):
    """拼接 query 与假设性文档后做混合检索。"""
    # 默认值统一从配置面板取，使消融只需改 .env
    if req_limit is None:
        req_limit = retrieval_config.req_limit
    if limit is None:
        limit = retrieval_config.retrieval_limit
    if ranker_weights is None:
        ranker_weights = retrieval_config.ranker_weights
    if output_fields is None:
        output_fields = [
            "text", "item_name", "product_name", "clause_no", "clause_no_norm",
            "clause_path", "clause_title", "clause_type", "doc_id",
        ]
    # 拼接rewritten_query和hyde_doc
    text = rewritten_query + " " + hyde_doc
    # 获取text所对应的稠密向量和稀疏向量
    embeddings = generate_embeddings([text])
    dense_vector = embeddings["dense"][0]
    sparse_vector = embeddings["sparse"][0]
    # 拼接item_name作为检索条件
    expr_data = ", ".join(f"'{item_name}'" for item_name in item_names)
    expr = f"item_name in [{expr_data}]"
    # 设置稠密向量和稀疏向量的检索方式
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=expr,
        limit=req_limit
    )
    # 获取milvus客户端
    milvus_client = get_milvus_client()
    # 进行混合检索
    result = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=ranker_weights,
        norm_score=norm_score,
        limit=limit,
        output_fields=output_fields,
    )
    return result

@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state: QueryGraphState):
    # 记录当前任务的状态为进行中
    add_running_task(state["session_id"], "node_search_embedding_hyde", state["is_stream"])
    try:
        # 总开关：关掉后本分支直接返回空。RRF 收不到这一路，
        # 结果自然退化成「只有普通混合检索」，这就是 HyDE 消融的对照组
        if not retrieval_config.hyde_enabled:
            logger.info("HyDE 分支已关闭（HYDE_ENABLED=false），跳过")
            return {"hyde_embedding_chunks": [], "hyde_doc": ""}
        # 分别获取rewritten_query和item_names
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        # 如果rewritten_query为空，使用original_query兜底
        if not rewritten_query:
            rewritten_query = state.get("original_query")
        if not rewritten_query:
            logger.warning("假设性文档检索缺失用户的问题")
            return {"hyde_embedding_chunks": [], "hyde_doc": ""}
        # 判断item_names是否为空
        if not item_names:
            logger.warning("假设性文档检索缺失item_names")
            return {"hyde_embedding_chunks": [], "hyde_doc": ""}
        # 步骤1：通过rewritten_query获取假设性文档；返回 None 表示这一步已判定该跳过本分支
        try:
            hyde_doc = step_1_create_hyde_doc(rewritten_query)
        except Exception as e:
            logger.error(f"获取假设性文档失败，{e}")
            return {"hyde_embedding_chunks": [], "hyde_doc": ""}
        if not hyde_doc:
            # 非条款问题（或模型返回空）：不产出 HyDE 召回，由普通混合检索兜底。
            # 注意这里不能塞一个空字符串进 step_2 —— 空文档的向量是无效召回源
            return {"hyde_embedding_chunks": [], "hyde_doc": ""}
        # 步骤2：通过hyde_doc和rewritten_query转换为向量检索数据
        try:
            result = step_2_search_embedding_hyde(
                rewritten_query=rewritten_query,
                hyde_doc=hyde_doc,
                item_names=item_names,
            )
        except Exception as e:
            logger.error(f"获取假设性文档检索结果失败，{e}")
            return {"hyde_embedding_chunks": [], "hyde_doc": hyde_doc}
        return {
            "hyde_embedding_chunks": result[0] if result else [],
            "hyde_doc": hyde_doc,
        }
    finally:
        # 无论从哪个出口返回都要落「已完成」——
        # 原写法只在最后一条路径上有 finally，早期返回会把任务永久卡在「运行中」
        add_done_task(state["session_id"], "node_search_embedding_hyde", state["is_stream"])


if __name__ == "__main__":
    # 本地测试代码
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_embedding_hyde 本地测试")
    print("=" * 50)

    # 模拟输入状态
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_search_embedding_hyde(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print("hyde_doc:")
        print(result["hyde_doc"])
        print("hyde_embedding_chunks:")
        print(result["hyde_embedding_chunks"])
    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")