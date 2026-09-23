from datetime import datetime
from pathlib import Path

from pymilvus import DataType, MilvusClient

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.ledger_utils import load_ledger_row
from app.utils.task_utils import add_running_task, add_done_task

"""
本节点职责：把切分 + 向量化后的条款切片写入 Milvus 的 insurance_clauses 集合。

与原通用商品库版本的三处本质差异：
1. 主键从自增 INT64 `chunk_id` 改为业务构造的 VARCHAR `pk`（`{doc_id}_{clause_no_norm}_{chunk_seq}`）。
   自增主键无法表达「同一条款重复导入应当覆盖而非新增」，而条款库必须支持按文档幂等重导。
   代价是 auto_id 必须关闭，且 Milvus 不再回传主键，原「插入后回填 chunk_id」的逻辑随之作废。
2. 字段从 9 个通用字段扩展为 27 个领域字段，定义来自 insurance_kb/02_clause_schema.json ——
   该文件同时是微调模型要素抽取的输出契约，一份定义两处使用，避免 RAG 与 SFT 之间字段漂移。
3. 幂等删除的依据从 LLM 推理出的 item_name 改为确定性的 doc_id（源文件名）。
   item_name 由大模型生成，同一份文档两次识别可能不一致，拿它做删除条件会导致旧数据清不干净；
   且产品级删除会误伤同一产品的其他文档（主条款 + 附加条款）。
"""

# Milvus中存储条款切片数据的集合名称
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection

# 建库时的 schema 版本，与 02_clause_schema.json 的 version 字段同步
SCHEMA_VERSION = "1.2"

# 台账中文列名 → Milvus 字段名
META_COLUMN_MAP = {
    "保司": "company",
    "产品名称": "product_name",
    "产品注册号": "product_code",
    "报备文件编号": "filing_no",
}

# 台账「险种」是业务简称，02_clause_schema.json 的 insurance_type 枚举是监管口径，需要一层收敛。
# 未收录的值落到「其他」而非原样存自由文本 —— 枚举一旦被自由文本污染，按险种过滤就失去意义
INSURANCE_TYPE_MAP = {
    "重疾": "健康保险-疾病保险",
    "疾病": "健康保险-疾病保险",
    "医疗": "健康保险-医疗保险",
    "护理": "健康保险-护理保险",
    "失能": "健康保险-失能收入损失保险",
    "意外": "意外伤害保险",
    "定期寿险": "人寿保险-定期寿险",
    "终身寿险": "人寿保险-终身寿险",
    "两全": "人寿保险-两全保险",
    "养老": "年金保险-养老年金保险",
    "年金": "年金保险",
}

# 产品级字段：(字段名, 类型, 最大字节长度)。导入时整批写入该产品的所有切片
PRODUCT_FIELDS = [
    ("company", DataType.VARCHAR, 128),
    ("product_name", DataType.VARCHAR, 256),
    ("product_code", DataType.VARCHAR, 128),
    ("filing_no", DataType.VARCHAR, 128),
    ("insurance_type", DataType.VARCHAR, 64),
    ("design_type", DataType.VARCHAR, 32),
    ("effective_date", DataType.VARCHAR, 32),
    ("stop_date", DataType.VARCHAR, 32),
]

# 条款级字段：(字段名, 类型, 最大字节长度)。除主键外逐一对应 clause_level.fields
CLAUSE_FIELDS = [
    ("vector", DataType.FLOAT_VECTOR, 1024),  # 第三位为向量维度
    ("sparse", DataType.SPARSE_FLOAT_VECTOR, None),
    ("text", DataType.VARCHAR, 8192),
    ("clause_no", DataType.VARCHAR, 32),
    ("clause_no_norm", DataType.VARCHAR, 32),
    ("clause_type", DataType.VARCHAR, 64),
    ("clause_path", DataType.VARCHAR, 256),
    ("clause_title", DataType.VARCHAR, 128),
    ("doc_id", DataType.VARCHAR, 128),
    ("page_no", DataType.INT64, None),
    ("chunk_seq", DataType.INT64, None),
    ("chunk_level", DataType.VARCHAR, 16),
    ("char_len", DataType.INT64, None),
    ("content_hash", DataType.VARCHAR, 64),
    ("import_batch", DataType.VARCHAR, 64),
    ("needs_review", DataType.BOOL, None),
    ("schema_version", DataType.VARCHAR, 16),
    ("item_name", DataType.VARCHAR, 256),
]

# 需要建标量倒排索引的字段（来自 02_clause_schema.json 的 index_config.scalar_inverted）。
# 建它们是为了让 hybrid_search 的 expr 过滤走索引而不是全表扫描
SCALAR_INVERTED_FIELDS = [
    "company",
    "insurance_type",
    "clause_type",
    "product_code",
    "design_type",
]

# pk 字段长度上限
PK_MAX_LEN = 128


def _clip(value, max_bytes):
    """
    按 UTF-8 字节数裁剪字符串。
    Milvus 的 VARCHAR max_length 以字节计，一个汉字占 3 字节 —— 中文标题很容易在「看着不长」的情况下超限，
    超限会让整批 insert 直接失败。统一入口裁剪，把失败挡在写入之前。
    """
    if value is None:
        return ""
    text = str(value)
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore")


@step_log("step_0_load_product_meta")
def step_0_load_product_meta(doc_id: str):
    """
    步骤0：按源文件名从语料台账读取产品级元数据。

    台账缺失或查不到该文件时不报错，返回全空字段 —— 产品级元数据是「锦上添花」，
    不该因为一次元数据缺失就让整份条款导不进去。

    读取统一走 app/utils/ledger_utils：node_item_name_recognition 也从台账取 item_name，
    两处必须用同一套 doc_id 匹配规则，否则会出现「切片的 item_name 与 product_name
    指向不同产品」——不报错，但按产品过滤和按产品查规则会同时失效。
    """
    empty_meta = {field_name: "" for field_name, _, _ in PRODUCT_FIELDS}

    row = load_ledger_row(doc_id)
    if row is None:
        logger.warning(f"语料台账中未找到 doc_id={doc_id} 的行，产品级字段将留空")
        return empty_meta

    meta = dict(empty_meta)
    for column_name, field_name in META_COLUMN_MAP.items():
        meta[field_name] = _clip((row.get(column_name) or "").strip(), 256)
    meta["insurance_type"] = INSURANCE_TYPE_MAP.get(
        (row.get("险种") or "").strip(), "其他"
    )
    logger.info(
        f"产品级元数据命中台账：{meta['company']} / {meta['product_name']} "
        f"/ {meta['product_code']} / {meta['insurance_type']}"
    )
    return meta


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
    """步骤2：集合不存在则按 02_clause_schema.json 创建（27 字段 / VARCHAR 主键 / auto_id 关闭）"""
    # 判断Milvus中切片集合是否存在，若不存在则创建
    if milvus_client.has_collection(CHUNKS_COLLECTION_NAME):
        return

    # 创建结构对象。auto_id 必须为 False —— Milvus 不允许 VARCHAR 主键自增
    schema = milvus_client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    # 主键：业务构造的复合键，保证同一份文档重复导入得到完全相同的 pk，从而可幂等覆盖
    schema.add_field(
        field_name="pk", datatype=DataType.VARCHAR, max_length=PK_MAX_LEN, is_primary=True
    )
    # 条款级字段
    for field_name, datatype, size in CLAUSE_FIELDS:
        if datatype == DataType.VARCHAR:
            schema.add_field(field_name=field_name, datatype=datatype, max_length=size)
        elif datatype == DataType.FLOAT_VECTOR:
            schema.add_field(field_name=field_name, datatype=datatype, dim=size)
        else:
            schema.add_field(field_name=field_name, datatype=datatype)
    # 产品级字段
    for field_name, datatype, size in PRODUCT_FIELDS:
        schema.add_field(field_name=field_name, datatype=datatype, max_length=size)

    # 准备并添加向量索引，参数取自 02_clause_schema.json 的 index_config
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_name="vector_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    index_params.add_index(
        field_name="sparse",
        index_name="sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"drop_ratio_search": 0.2},
    )
    # 创建集合
    milvus_client.create_collection(
        collection_name=CHUNKS_COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    logger.info(
        f"集合 {CHUNKS_COLLECTION_NAME} 已创建："
        f"{1 + len(CLAUSE_FIELDS) + len(PRODUCT_FIELDS)} 字段，schema_version={SCHEMA_VERSION}"
    )

    # 标量倒排索引单独补建，并允许失败降级：
    # 与集合创建解耦，是为了避免某个标量字段建索引失败导致整个集合建不出来
    try:
        scalar_index_params = milvus_client.prepare_index_params()
        for field_name in SCALAR_INVERTED_FIELDS:
            scalar_index_params.add_index(
                field_name=field_name,
                index_name=f"{field_name}_index",
                index_type="INVERTED",
            )
        milvus_client.create_index(
            collection_name=CHUNKS_COLLECTION_NAME,
            index_params=scalar_index_params,
        )
        logger.info(f"标量倒排索引已建立：{SCALAR_INVERTED_FIELDS}")
    except Exception as e:
        logger.warning(f"标量倒排索引建立失败（不影响向量检索，只是过滤会走全扫）：{e}")


@step_log("step_3_delete_old_data")
def step_3_delete_old_data(milvus_client: MilvusClient, doc_id: str):
    """
    步骤3：按源文档标识删除旧数据，实现幂等重导。

    用 doc_id 而非产品名：同一产品可能对应多份文档（主条款 + 附加条款 + 费率表），
    按产品名删会误伤同产品的其他文档；按文件删则只覆盖「自己这份」。
    """
    delete_result = milvus_client.delete(
        collection_name=CHUNKS_COLLECTION_NAME,
        filter=f"doc_id == '{doc_id}'",
    )
    # 不同 pymilvus 小版本返回结构不同（dict 带 delete_count 或直接返回 pk 列表），两种都兼容
    delete_count = (
        delete_result.get("delete_count") if isinstance(delete_result, dict) else "未知"
    )
    logger.info(f"清理旧数据：doc_id={doc_id}，删除 {delete_count} 条")
    # 重新加载集合
    milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)


def build_milvus_row(chunk, item_name, product_meta, import_batch):
    """把一个切片字典转换为 Milvus 插入行。字段名严格对齐 02_clause_schema.json"""
    doc_id = chunk.get("doc_id") or chunk.get("file_title") or ""
    clause_no_norm = chunk.get("clause_no_norm") or "0"
    chunk_seq = int(chunk.get("chunk_seq") or 1)
    text = chunk.get("text") or ""

    row = {
        # pk 构造规则与 02_clause_schema.json 的 pk.note 一致；改了它等于破坏幂等语义
        "pk": _clip(f"{doc_id}_{clause_no_norm}_{chunk_seq}", PK_MAX_LEN),
        "vector": chunk["vector"],
        "sparse": chunk["sparse"],
        "text": _clip(text, 8192),
        "clause_no": _clip(chunk.get("clause_no"), 32),
        "clause_no_norm": _clip(clause_no_norm, 32),
        "clause_type": _clip(chunk.get("clause_type") or "其他", 64),
        "clause_path": _clip(chunk.get("clause_path"), 256),
        "clause_title": _clip(chunk.get("clause_title"), 128),
        "doc_id": _clip(doc_id, 128),
        # 页码暂缺：切分节点当前未保留页边界，等解析节点回传页码后回填。
        # 置 -1 而不是 0，是为了不把「没有页码」和真实的第 1 页混为一谈
        "page_no": int(chunk.get("page_no") or -1),
        "chunk_seq": chunk_seq,
        "chunk_level": _clip(chunk.get("chunk_level"), 16),
        "char_len": int(chunk.get("char_len") or len(text)),
        "content_hash": _clip(chunk.get("content_hash"), 64),
        "import_batch": _clip(import_batch, 64),
        "needs_review": bool(chunk.get("needs_review", False)),
        "schema_version": SCHEMA_VERSION,
        # item_name 由上游 LLM 识别，是检索侧的过滤键；product_name 来自台账，是权威备案名
        "item_name": _clip(item_name, 256),
    }
    row.update(product_meta)
    return row


@step_log("step_4_insert_collections")
def step_4_insert_collections(milvus_client: MilvusClient, chunks, item_name, product_meta, import_batch):
    """
    步骤4：批量写入。

    注意这里不再有「插入后回填自增主键」这一步（原实现靠 insert_result["ids"] 回填 chunk_id）。
    新主键 pk 在插入前就已确定，插入是纯粹的幂等覆盖语义 —— 相同 pk 重复写入不会产生重复行。
    """
    rows = [build_milvus_row(chunk, item_name, product_meta, import_batch) for chunk in chunks]
    insert_result = milvus_client.insert(
        collection_name=CHUNKS_COLLECTION_NAME,
        data=rows,
    )
    logger.info(f"成功添加了{insert_result['insert_count']}条数据")

    # 把主键回填到切片，保持 state 的 chunks 携带主键（下游测试与调试依赖这一点）
    for chunk, row in zip(chunks, rows):
        chunk["pk"] = row["pk"]
    return chunks


@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    实现要点:
    1. 从语料台账补产品级元数据（公司 / 备案名 / 注册号 / 报备文号 / 险种）。
    2. 按 doc_id 删除旧数据，保证同一份文档重复导入时覆盖而非堆积。
    3. 按 02_clause_schema.json 建集合，批量插入带 pk 的条款切片。
    """
    # 记录任务状态为运行中
    add_running_task(state["task_id"], "node_import_milvus")
    # 步骤1：检查数据 chunks 是否存在
    chunks = step_1_validate_input(state)
    # 源文档标识：切分节点写入 file_title，缺失时退回 md 文件名
    doc_id = state.get("file_title") or Path(state.get("md_path", "")).stem
    if not doc_id:
        raise ValueError("无法确定源文档标识 doc_id，无法保证导入幂等")
    # 产品名：优先用 LLM 识别结果，缺失时退回文件名
    item_name = state.get("item_name") or doc_id
    # 步骤0：从语料台账读取产品级元数据
    product_meta = step_0_load_product_meta(doc_id)
    # 导入批次号：供整批回滚与问题溯源
    import_batch = datetime.now().strftime("%Y%m%d%H%M%S")
    # 客户端
    milvus_client = get_milvus_client()
    # 步骤2：前置准备工作，创建集合与字段
    step_2_prepare_collection(milvus_client)
    # 步骤3：删除该文档的旧数据
    step_3_delete_old_data(milvus_client, doc_id)
    # 步骤4：写入新数据
    with_pk_chunks = step_4_insert_collections(
        milvus_client, chunks, item_name, product_meta, import_batch
    )
    # 更新状态
    state["chunks"] = with_pk_chunks
    # 记录任务状态为已完成
    add_done_task(state["task_id"], "node_import_milvus")
    return state


if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证新 schema 建集合 + 幂等删除 + 带 pk 写入三个阶段。
    # 会真实写入 Milvus 的 insurance_clauses 集合，跑两次应当得到相同的总条数（幂等）
    import hashlib
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    dim = 1024
    test_doc_id = "test_clause_doc.pdf"

    def _make_chunk(seq, text):
        return {
            "text": text,
            "clause_no": f"第{seq}条",
            "clause_no_norm": str(seq),
            "clause_title": f"测试条款{seq}",
            "clause_path": f"{seq} 测试条款{seq}",
            "clause_type": "其他",
            "chunk_seq": 1,
            "chunk_level": "L1",
            "char_len": len(text),
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "doc_id": test_doc_id,
            "needs_review": False,
            "vector": [0.1] * dim,
            "sparse": {1: 0.5, 10: 0.8},
        }

    test_state = {
        "task_id": "test_milvus_task",
        "file_title": test_doc_id,
        "item_name": "测试产品_保险条款",
        "md_path": "",
        "chunks": [
            _make_chunk(1, "第一条 在本合同保险期间内，我们按约定承担给付保险金的责任。"),
            _make_chunk(2, "第二条 因下列情形之一导致被保险人身故的，我们不承担给付保险金的责任。"),
        ],
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数（连跑两次，验证幂等）
            result_state = node_import_milvus(test_state)
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("pk"):
                print(f"✅ 第一次导入通过，pk 示例：{chunks[0]['pk']}")
            else:
                print("❌ 测试失败：未能生成 pk")

            result_state_2 = node_import_milvus(dict(test_state, chunks=test_state["chunks"]))
            chunks_2 = result_state_2.get("chunks", [])
            if chunks_2 and chunks_2[0].get("pk") == chunks[0].get("pk"):
                print(f"✅ 第二次导入 pk 一致（幂等）：{chunks_2[0]['pk']}")
            else:
                print("❌ 幂等校验失败：两次导入的 pk 不一致")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
