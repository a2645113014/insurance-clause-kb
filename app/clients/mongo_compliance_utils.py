"""compliance_rules 集合（MongoDB）的读写封装 —— Phase 3.2 合规审查层的数据出口

为什么又是 MongoDB，不是 Milvus
--------------------------------
见 `insurance_kb/12_compliance_schema.json` 的 `storage.why_not_milvus`。三句话：
规模只有 105 条（430KB 向量），内存里做一次暴力余弦是微秒级；这 105 条是
**判定标准**（拿它去比对待审表述），不是**待检语料**（拿相似度去召回），
做成索引库反而引入召回误差，而这里的召回误差直接等于漏判。

检索策略三路合并
----------------
1. **向量 top-k**：待审表述编码后与 105 条算余弦，取前 `COMPLIANCE_TOP_K`
2. **通用兜底**：`scope_tags` 含「通用」的条目对任何产品都适用，但它们未必
   与待审表述语义相似 —— 只靠向量会漏。开关 `COMPLIANCE_INCLUDE_UNIVERSAL`
3. **险种兜底**：按产品险种双向包含匹配补条目。开关 `COMPLIANCE_INCLUDE_SCOPE_MATCH`

三路按 `item_no` 去重，最终按相似度降序截断到 `COMPLIANCE_MAX_ITEMS`。
**为什么必须先合并再截断**：如果先截断再合并，向量路会把 40 个位置占满，
兜底路一条都进不来 —— 兜底等于没做。

职责边界
--------
本模块只做「存取 + 召回」，不做「判定」：
- 「这批条目意味着违规还是合规」→ LLM，prompt 见 `prompts/compliance_check.prompt`
- 「召回漏了怎么办」→ 由 `app/tool/compliance_tools.py` 在返回值里写子集声明，
  把漏召回从「静默放过」转成「显式说不知道」

字段定义以 `insurance_kb/12_compliance_schema.json` 为准（项目决策 C）。
本文件的索引清单与之保持同步，改 schema 时两边一起改。
"""

import os
from datetime import datetime, timezone

import numpy as np
from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient, UpdateOne

from app.conf.compliance_config import compliance_config
from app.core.logger import logger

load_dotenv()

# 与 insurance_kb/12_compliance_schema.json 的 version 字段保持一致
SCHEMA_VERSION = "1.0"

# 「通用」标签：表示该条适用全部产品，检索时恒带（受开关控制）
UNIVERSAL_TAG = "通用"

# 各种召回路径的标注。答案里要能看出某条是「因为相似」还是「因为通用」进来的 ——
# 后者对模型的判定同样有效，但人工复核时归因不同。
PICKED_BY_VECTOR = "vector"
PICKED_BY_UNIVERSAL = "universal"
PICKED_BY_SCOPE = "scope"


class ComplianceMongoTool:
    """负面清单条目的 MongoDB 读写工具类（原生 PyMongo 实现）。

    初始化做三件事：连库、取集合、建索引。另维护一份**全量条目内存缓存** ——
    105 条只在首次检索时载入，之后每次检索直接算余弦。
    """

    def __init__(self):
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")
            if not self.mongo_url or not self.db_name:
                raise ValueError("缺少 MONGO_URL 或 MONGO_DB_NAME 环境变量配置")

            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            # 集合名从 compliance_config 取，不硬编码 —— 换库名只改 .env
            self.rules = self.db[compliance_config.collection]

            # 内存缓存：None 表示未载入，[] 表示载入过但库里是空的
            self._cache = None
            self._cache_matrix = None
            # 向量矩阵每一行对应的条号。矩阵只装「有向量」的条目，
            # 行序与 items 行序不同，所以必须单独记条号来对齐结果
            self._dense_nos = []

            self._ensure_indexes()
            logger.info(
                f"合规条目库连接成功 | db={self.db_name} "
                f"| collection={compliance_config.collection}"
            )
        except Exception as e:
            logger.error(f"合规条目库连接失败：{e}", exc_info=True)
            raise

    def _ensure_indexes(self):
        """建查询索引。清单与 12_compliance_schema.json 的 indexes 段一一对应。

        为什么是这三条：`item_no` 供答案引用的条号反查原文；`category_id` 供
        按大类抽检统计；`scope_tags` 是 multikey 索引，供险种兜底取条目。
        向量检索不走索引 —— 全量载入内存算余弦，索引帮不上也不会成为瓶颈。
        """
        self.rules.create_index([("item_no", ASCENDING)], name="idx_item_no")
        self.rules.create_index([("category_id", ASCENDING)], name="idx_category")
        self.rules.create_index([("scope_tags", ASCENDING)], name="idx_scope_tags")

    # ------------------------------------------------------------ 写入
    def upsert_items(self, docs):
        """幂等写入合规条目。

        `_id` 取业务键 `NL-{item_no:03d}`，因此重复构建只更新不新增 ——
        这是「同一份负面清单跑两遍，条目表仍是 105 条」的保证。

        :param docs: 条目文档列表，每条必须含 item_no
        :return: dict，含 matched / upserted / modified / skipped 计数
        """
        if not docs:
            return {"matched": 0, "upserted": 0, "modified": 0, "skipped": 0}

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        ops, skipped = [], 0
        for raw in docs:
            doc = dict(raw)
            item_no = doc.get("item_no")
            if item_no is None:
                # 缺业务键就没法幂等，宁可跳过也不能让 Mongo 自己生成 _id
                logger.warning("跳过一条缺 item_no 的合规条目")
                skipped += 1
                continue
            doc["_id"] = f"NL-{int(item_no):03d}"
            doc.setdefault("schema_version", SCHEMA_VERSION)
            # `created_at` 走 `$setOnInsert` 而不是 `$set`：它是「创建时间」，
            # 复跑时被刷新就变成了「最后构建时间」，审计时答不出「这批条目
            # 是什么时候落库的」。复跑 modified 因此也只统计真正变化的字段。
            doc.pop("created_at", None)
            ops.append(
                UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": doc, "$setOnInsert": {"created_at": now}},
                    upsert=True,
                )
            )

        if not ops:
            return {"matched": 0, "upserted": 0, "modified": 0, "skipped": skipped}

        result = self.rules.bulk_write(ops, ordered=False)
        self.refresh()  # 写库后必须让内存缓存失效，否则后续检索还是旧条目
        return {
            "matched": result.matched_count,
            "upserted": len(result.upserted_ids) if result.upserted_ids else 0,
            "modified": result.modified_count,
            "skipped": skipped,
        }

    # ------------------------------------------------------------ 缓存
    def refresh(self):
        """清空内存缓存。写库后、或人工改库后调用。"""
        self._cache = None
        self._cache_matrix = None

    def load_all(self):
        """载入全量条目并按条号排序，构建归一化向量矩阵。结果缓存复用。

        :return: (items, matrix)。items 是去掉 _id 的文档列表，matrix 是
                 (N, 1024) float32 且已按行 L2 归一化（点积即余弦）。
                 库为空时返回 ([], None)。
        """
        if self._cache is not None:
            return self._cache, self._cache_matrix

        rows = list(self.rules.find({}, {"_id": 0}).sort("item_no", ASCENDING))
        if not rows:
            self._cache, self._cache_matrix = [], None
            return self._cache, self._cache_matrix

        # 只有带向量的条目才能参与向量路。缺向量的条目仍保留在 items 里 ——
        # 它们是「通用兜底」的合法候选，不该因为没向量就从清单里消失。
        matrix, self._dense_nos = _build_matrix(rows)

        self._cache, self._cache_matrix = rows, matrix
        logger.info(
            f"合规条目载入内存 | 总数={len(rows)} | 含向量={len(self._dense_nos)} "
            f"| 维度={matrix.shape[1] if matrix is not None else 0}"
        )
        return self._cache, self._cache_matrix

    # ------------------------------------------------------------ 召回
    def search(self, query_vector, product_scope=None, top_k=None,
               include_universal=None, include_scope_match=None, max_items=None,
               categories=None):
        """三路合并召回适用条目。

        :param query_vector: 待审表述的稠密向量（1024 维，模型已归一化）；
                传 None 表示编码失败 —— 此时跳过向量路，只走两个兜底路，
                宁可给一份粗清单，也不要因为一次编码失败就判「无依据」
        :param product_scope: 产品险种，如「医疗保险」。用于险种兜底匹配
        :param top_k / include_universal / include_scope_match / max_items:
                覆盖配置面板的取值，仅供评测脚本做消融；生产调用一律传 None
        :param categories: 只保留这些 `category_id` 的条目（如 ["1","2"]）。
                **刻意不放进配置面板**：它不是稳定的生产口径，而是评测要量的
                一个假设 —— 负面清单第 3/4 大类（费率厘定 36 条 + 报送管理 19 条）
                判的是精算报告与备案材料，条款文本无从对应，把它们混进候选
                只会制造误判风险。是否固化成生产开关，等 test/13 的对照数据说话。
                **对照结果（2026-09-24）**：cat12 组清单从 40 条降到 31.2 条，
                全金标召回率仍是 1.0、金标位次 2.53（base 2.82）—— 砍掉 55 条
                不可判条目**不损失召回**，还少喂 9 条噪音。但仍未固化为生产开关：
                金标集只有 17 条带金标用例，不足以支撑「永久排除两整类」这种
                不可逆的口径收窄，留待扩量后决定。
        :return: 条目列表，每项含 similarity / picked_by；按相似度降序
        """
        items, matrix = self.load_all()
        if not items:
            return []

        if categories:
            allowed = {str(c) for c in categories}
            items = [it for it in items if str(it.get("category_id")) in allowed]
            if not items:
                return []
            # 向量矩阵是按全量条目行序建的，过滤后行序对不上 —— 必须重建。
            # 重建结果只放在局部变量里：写回 self._dense_nos 会污染后续不带
            # 过滤的调用（load_all 缓存命中时不重建它，条号会整体错位）。
            matrix, dense_nos = _build_matrix(items)
        else:
            dense_nos = self._dense_nos

        top_k = compliance_config.top_k if top_k is None else top_k
        include_universal = (
            compliance_config.include_universal if include_universal is None else include_universal
        )
        include_scope_match = (
            compliance_config.include_scope_match if include_scope_match is None else include_scope_match
        )
        max_items = compliance_config.max_items if max_items is None else max_items

        # 每条候选统一记成 {item_no: (doc, similarity, picked_by)}
        # 用 dict 天然完成「按条号去重」，先到的不被后到的覆盖 ——
        # 所以向量路必须最先跑，它的相似度最高、归因也最准。
        picked = {}

        # ---- 第 1 路：向量 top-k
        sims = {}
        if query_vector is not None and matrix is not None:
            qv = np.asarray(query_vector, dtype="float32")
            norm = float(np.linalg.norm(qv))
            if norm > 0:
                qv = qv / norm
                all_sims = matrix @ qv
                # 按条号存，不依赖矩阵行序与 items 行序一致
                for pos, item_no in enumerate(dense_nos):
                    sims[item_no] = float(all_sims[pos])
                ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
                by_no = {it.get("item_no"): it for it in items}
                for item_no, score in ranked[: max(0, top_k)]:
                    doc = by_no.get(item_no)
                    if doc is not None:
                        picked[item_no] = (doc, score, PICKED_BY_VECTOR)

        # ---- 第 2 路：通用兜底
        if include_universal:
            for doc in items:
                tags = doc.get("scope_tags") or []
                if UNIVERSAL_TAG not in tags:
                    continue
                no = doc.get("item_no")
                if no in picked:
                    continue
                # 通用条目里也有「与本次待审表述很相关」的，它们的相似度已经在
                # 向量路体现；这里补进来的是**向量路没排上号**的那些。
                picked[no] = (doc, sims.get(no, 0.0), PICKED_BY_UNIVERSAL)

        # ---- 第 3 路：险种兜底
        if include_scope_match and product_scope:
            for doc in items:
                tags = doc.get("scope_tags") or []
                if not _scope_match(tags, product_scope):
                    continue
                no = doc.get("item_no")
                if no in picked:
                    continue
                picked[no] = (doc, sims.get(no, 0.0), PICKED_BY_SCOPE)

        merged = sorted(picked.values(), key=lambda t: t[1], reverse=True)[: max(0, max_items)]
        return [
            {
                "item_no": doc.get("item_no"),
                "category_id": doc.get("category_id"),
                "category_name": doc.get("category_name"),
                "text": doc.get("text"),
                "scope": doc.get("scope"),
                "scope_tags": doc.get("scope_tags") or [],
                "similarity": round(score, 4),
                "picked_by": picked_by,
            }
            for doc, score, picked_by in merged
        ]

    # ------------------------------------------------------------ 统计
    def stats(self):
        """条目表体检：总条数与按大类 / 召回来源的分组计数。

        拿去回答「105 条到底导进来没有、四大类分布对不对」。
        """
        total = self.rules.count_documents({})
        by_category = {
            row["_id"]: row["n"]
            for row in self.rules.aggregate(
                [{"$group": {"_id": "$category_name", "n": {"$sum": 1}}}]
            )
        }
        with_vector = self.rules.count_documents({"dense_vector": {"$exists": True}})
        universal = self.rules.count_documents({"scope_tags": UNIVERSAL_TAG})
        items, matrix = self.load_all()
        return {
            "collection": compliance_config.collection,
            "db": self.db_name,
            "total": total,
            "by_category": by_category,
            "with_vector": with_vector,
            "universal": universal,
            "vector_dim": int(matrix.shape[1]) if matrix is not None else 0,
            "loaded_in_memory": len(items),
        }


def _build_matrix(items):
    """按给定条目顺序构建**行 L2 归一化**的向量矩阵，并返回行序对应的条号列表。

    归一化的必要性：入库时模型已做 L2 归一化，这里再除一次范数纯属防御 ——
    万一有旧数据未归一化，点积就不再等于余弦，相似度会整体失真且不报错。

    :return: (matrix, nos)。没有任何条目带向量时返回 (None, [])。
    """
    nos, vecs = [], []
    for it in items:
        vec = it.get("dense_vector")
        if isinstance(vec, (list, tuple)) and len(vec) > 0:
            nos.append(it.get("item_no"))
            vecs.append(np.asarray(vec, dtype="float32"))
    if not vecs:
        return None, []
    matrix = np.vstack(vecs)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12), nos


def _scope_match(tags, product_scope):
    """判断条目 scope 标签与产品险种是否匹配（双向包含）。

    为什么要双向：「医疗保险」产品应当匹配标签「医疗保险」，而标签
    「附加两全保险」在 scope 写成「长期险」时也该被包含关系捞到。
    只做单向 `tag in product_scope` 会漏掉后者。

    「通用」标签由专门一路处理，这里排除 —— 否则通用条目会被算两次，
    且归因（picked_by）会失真。
    """
    scope = (product_scope or "").strip()
    if not scope:
        return False
    for tag in tags or []:
        tag = (tag or "").strip()
        if not tag or tag == UNIVERSAL_TAG:
            continue
        if tag in scope or scope in tag:
            return True
    return False


# ---------------------------------------------------------------- 单例
_compliance_tool = None

try:
    _compliance_tool = ComplianceMongoTool()
except Exception as e:
    # 模块加载阶段连不上不该炸掉整个服务 —— 保留懒加载兜底，
    # 与 mongo_rule_utils.py 的处理方式一致。
    logger.warning(f"合规条目库在模块加载阶段未就绪，将在首次调用时重试：{e}")


def get_compliance_tool() -> ComplianceMongoTool:
    """获取合规条目库工具单例。模块加载时未连上则在此重试一次。"""
    global _compliance_tool
    if _compliance_tool is None:
        _compliance_tool = ComplianceMongoTool()
    return _compliance_tool


if __name__ == "__main__":
    import json

    tool = get_compliance_tool()
    print(json.dumps(tool.stats(), ensure_ascii=False, indent=2))
