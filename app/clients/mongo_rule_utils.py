"""insurance_rules 集合（MongoDB）的读写封装 —— Phase 2.3 规则层工具化的数据出口

为什么用 MongoDB 而不是 Milvus
------------------------------
见 `insurance_kb/01_改造清单.md` §2.1：理赔规则要的是**确定性键值查询**
（按产品 + 事故类型取材料清单），不是相似度检索。放进向量库会把一个
可以精确回答的问题变成概率问题 —— 「身故要交死亡证明」这件事没有
「相似度 0.83」可言，它要么在规则表里，要么不在。

职责边界
--------
本模块只做「存取」，不做「判断」：
- 「这句话是不是在问理赔材料」→ `app/query_process/agent/nodes/node_rule_tool.py`
- 「查不到时该降级还是拒答」→ `app/conf/rule_config.py` 的开关 + 工具层

字段定义以 `insurance_kb/11_claim_rule_schema.json` 为准（项目决策 C）。
本文件的索引清单与之保持同步，改 schema 时两边一起改。
"""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient, UpdateOne

from app.conf.rule_config import rule_config
from app.core.logger import logger

load_dotenv()

# 与 insurance_kb/11_claim_rule_schema.json 的 version 字段保持一致
SCHEMA_VERSION = "1.0"


class RuleMongoTool:
    """保险理赔规则的 MongoDB 读写工具类（原生 PyMongo 实现）。

    初始化时完成三件事：连库、取集合、建索引。索引创建自带幂等性，
    重复初始化不会报错也不会重复建。
    """

    def __init__(self):
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")
            if not self.mongo_url or not self.db_name:
                raise ValueError("缺少 MONGO_URL 或 MONGO_DB_NAME 环境变量配置")

            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            # 集合名从 rule_config 取，不硬编码 —— 换库名只改 .env
            self.rules = self.db[rule_config.collection]

            self._ensure_indexes()
            logger.info(
                f"保险规则库连接成功 | db={self.db_name} | collection={rule_config.collection}"
            )
        except Exception as e:
            logger.error(f"保险规则库连接失败：{e}", exc_info=True)
            raise

    def _ensure_indexes(self):
        """建查询索引。清单与 11_claim_rule_schema.json 的 indexes 段一一对应。

        为什么是这四条：
        - 工具的两条主查询路径是「产品+规则类型」与「产品+规则类型+事故类型」
        - 降级路径走「险种+规则类型+事故类型」
        - `source_pk` 单列索引供溯源反查（从规则回到原文切片）
        复合索引的字段顺序按「等值过滤 → 等值过滤」排，Mongo 的前缀匹配
        规则下 ①② 两个索引可互相覆盖，不必再单独建 item_name 索引。
        """
        self.rules.create_index([("item_name", ASCENDING), ("rule_type", ASCENDING)],
                                name="idx_item_ruletype")
        self.rules.create_index([("item_name", ASCENDING), ("rule_type", ASCENDING),
                                 ("accident_type", ASCENDING)],
                                name="idx_item_ruletype_accident")
        self.rules.create_index([("insurance_type", ASCENDING), ("rule_type", ASCENDING),
                                 ("accident_type", ASCENDING)],
                                name="idx_intype_ruletype_accident")
        self.rules.create_index([("source_pk", ASCENDING)], name="idx_source_pk")

    # ------------------------------------------------------------ 写入
    def upsert_rules(self, docs):
        """幂等写入规则文档。

        `_id` 取业务键 `rule_id`，因此重复构建只更新不新增 ——
        这是「同一份语料跑两遍，规则表条数不变」的保证。

        :param docs: 规则文档列表，每条必须含 rule_id
        :return: dict，含 matched / upserted / modified / skipped 计数
        """
        if not docs:
            return {"matched": 0, "upserted": 0, "modified": 0, "skipped": 0}

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        ops, skipped = [], 0
        for raw in docs:
            doc = dict(raw)
            rule_id = doc.get("rule_id")
            if not rule_id:
                # 缺业务键就没法幂等，宁可跳过也不能让 Mongo 自己生成 _id
                logger.warning("跳过一条缺 rule_id 的规则文档")
                skipped += 1
                continue
            doc["_id"] = rule_id
            doc.setdefault("schema_version", SCHEMA_VERSION)
            doc.setdefault("created_at", now)
            ops.append(UpdateOne({"_id": rule_id}, {"$set": doc}, upsert=True))

        if not ops:
            return {"matched": 0, "upserted": 0, "modified": 0, "skipped": skipped}

        result = self.rules.bulk_write(ops, ordered=False)
        return {
            "matched": result.matched_count,
            "upserted": len(result.upserted_ids) if result.upserted_ids else 0,
            "modified": result.modified_count,
            "skipped": skipped,
        }

    # ------------------------------------------------------------ 查询
    def find_by_item(self, item_name, rule_type=None, accident_type=None, limit=200):
        """按产品名查规则。工具的第一查询键。

        :param item_name: 产品名称，与 insurance_clauses.item_name 同源
        :param rule_type: 规则类型（required_docs / claim_timeline），None 表示不限
        :param accident_type: 事故类型关键词，做**子串匹配**而非精确匹配 ——
               用户问「身故」，规则里存的是「身故保险金」，精确匹配会全部落空
        :return: 规则文档列表（已去掉 _id 以外的 Mongo 内部字段）
        """
        if not item_name:
            return []
        query = {"item_name": item_name}
        if rule_type:
            query["rule_type"] = rule_type
        cond = _accident_filter(accident_type)
        if cond:
            query.update(cond)
        return list(self.rules.find(query, {"_id": 0}).limit(limit))

    def find_by_insurance_type(self, insurance_type, rule_type=None, accident_type=None, limit=200):
        """按险种查规则。产品级查不到时的降级路径。

        降级是**有代价的**：拿到的是同类产品的通用要求，不等于该产品的约定。
        调用方必须把「这是降级结果」透传到答案里，不能让它看起来像精确命中。
        """
        if not insurance_type:
            return []
        query = {"insurance_type": insurance_type}
        if rule_type:
            query["rule_type"] = rule_type
        cond = _accident_filter(accident_type)
        if cond:
            query.update(cond)
        return list(self.rules.find(query, {"_id": 0}).limit(limit))

    def find_accident_types(self, item_name, rule_type="required_docs"):
        """列出某产品在规则表里覆盖的全部事故类型。

        用途：用户只说「理赔要什么材料」而没指明事故类型时，
        工具需要把可选范围交给模型，让答案能反问而不是瞎猜。
        """
        if not item_name:
            return []
        cursor = self.rules.find(
            {"item_name": item_name, "rule_type": rule_type},
            {"_id": 0, "accident_type": 1},
        )
        seen, ordered = set(), []
        for row in cursor:
            value = row.get("accident_type")
            if value and value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def find_product_names(self, rule_type=None, limit=500):
        """列出规则表覆盖的产品名。供评测脚本与人工核对覆盖面使用。"""
        query = {"rule_type": rule_type} if rule_type else {}
        return sorted(self.rules.distinct("item_name", query))[:limit]

    # ------------------------------------------------------------ 统计
    def stats(self):
        """规则表体检：总条数与按 rule_type / extract_method / verified 的分组计数。

        拿去回答「规则表到底建起来没有、有没有漏抽、有没有没通过校验的」。
        """
        total = self.rules.count_documents({})
        by_type = {
            row["_id"]: row["n"]
            for row in self.rules.aggregate([{"$group": {"_id": "$rule_type", "n": {"$sum": 1}}}])
        }
        by_method = {
            row["_id"]: row["n"]
            for row in self.rules.aggregate([{"$group": {"_id": "$extract_method", "n": {"$sum": 1}}}])
        }
        unverified = self.rules.count_documents({"verified": False})
        return {
            "collection": rule_config.collection,
            "db": self.db_name,
            "total": total,
            "by_rule_type": by_type,
            "by_extract_method": by_method,
            "unverified": unverified,
        }


def _accident_filter(accident_type):
    """把事故类型关键词转成 Mongo 过滤条件，支持单个或多个关键词。

    为什么要支持多个：条款里写的是「身故保险金」，用户口语说「死亡」或「人没了」。
    工具层先把口语扩展成一组候选词（见 `app/tool/insurance_rule_tools.py` 的
    别名表），这里负责把它们变成「或」关系。

    用子串匹配而不是精确匹配：用户说「身故」，规则里存的是「身故保险金」，
    精确匹配会全部落空。

    :return: 可直接 `query.update()` 的 dict；无有效关键词时返回 None
    """
    if not accident_type:
        return None
    keys = accident_type if isinstance(accident_type, (list, tuple, set)) else [accident_type]
    keys = [str(k).strip() for k in keys if str(k).strip()]
    if not keys:
        return None
    if len(keys) == 1:
        return {"accident_type": {"$regex": _escape_regex(keys[0])}}
    return {"$or": [{"accident_type": {"$regex": _escape_regex(k)}} for k in keys]}


def _escape_regex(text):
    """把用户输入转义成安全的正则片段。

    不做这步的话，用户问「理赔材料(身故)」里那对括号会被当成正则分组，
    轻则查不到，重则抛正则语法错。
    """
    import re

    return re.escape(str(text))


# ---------------------------------------------------------------- 单例
_rule_mongo_tool = None

try:
    _rule_mongo_tool = RuleMongoTool()
except Exception as e:
    # 模块加载阶段连不上不该炸掉整个服务 —— 保留懒加载兜底，
    # 与 mongo_history_utils.py 的处理方式一致。
    logger.warning(f"规则库在模块加载阶段未就绪，将在首次调用时重试：{e}")


def get_rule_mongo_tool() -> RuleMongoTool:
    """获取规则库工具单例。模块加载时未连上则在此重试一次。"""
    global _rule_mongo_tool
    if _rule_mongo_tool is None:
        _rule_mongo_tool = RuleMongoTool()
    return _rule_mongo_tool


if __name__ == "__main__":
    import json

    tool = get_rule_mongo_tool()
    print(json.dumps(tool.stats(), ensure_ascii=False, indent=2))
