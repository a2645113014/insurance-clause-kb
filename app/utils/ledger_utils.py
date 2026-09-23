"""语料台账（data/clauses/sources.csv）读取。

台账是产品级元数据的**单一事实来源**，目前有两个消费者：

- ``node_import_milvus.step_0_load_product_meta``
  取 company / product_name / product_code / filing_no / insurance_type，写到每个切片上。
- ``node_item_name_recognition.step_3_resolve_item_name``
  取「产品名称」当 item_name —— 它同时是检索过滤键（``expr = item_name in [...]``）
  和 MongoDB 规则表的查询键。

两者必须用**同一套 doc_id ↔ 台账行匹配规则**。分散实现迟早会漂移，而一旦漂移，
症状是「条款库里的 item_name 在产品名库和规则库里都查不到」——
表现为服务不报错但答不出任何东西，很难从异常里反查。

为什么 item_name 要用台账而不是让大模型识别
--------------------------------------------
item_name 是**键**，必须与条款库切片、产品名库、规则库三处逐字一致。
大模型输出格式不受控：实测换模型后同一份文档开始返回带成对引号的空串 ``""``，
5 份文档的 item_name 因此塌缩成同一个键，265 个切片挤在一起、跨产品互相召回。
台账的「产品名称」列是监管备案名，实测与历史大模型产出 17/17 逐字一致，
改用它不改变任何已有数据，却让 item_name 彻底摆脱对模型行为的依赖。

``sanitize_item_name`` 保留给「台账缺行」的兜底路径（新采语料尚未登记台账时），
它对大模型输出做清洗，避免同一类污染再次进入库。
"""
import csv
import re
from pathlib import Path

from app.core.logger import logger, PROJECT_ROOT

# 语料台账：产品级元数据的单一来源（由 data/collect_clauses.py 幂等维护）
SOURCES_CSV_PATH = Path(PROJECT_ROOT) / "data" / "clauses" / "sources.csv"

# 台账里承载「对外产品名」的列名
PRODUCT_NAME_COLUMN = "产品名称"

# 成对包裹的引号/括号：大模型偶尔把答案包一层再返回，不剥掉就会把引号写进 item_name。
# 实测新模型对部分文档返回 `""`（带引号的空串），`if not item_name` 这种真值判断拦不住它
_QUOTE_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("\u201c", "\u201d"),  # 中文双引号
    ("\u2018", "\u2019"),  # 中文单引号
    ("\u300c", "\u300d"),  # 「」
    ("\u300e", "\u300f"),  # 『』
    ("\u300a", "\u300b"),  # 《》
    ("\u3010", "\u3011"),  # 【】
)

_WHITESPACE_RE = re.compile(r"\s+")


def load_ledger_row(doc_id: str):
    """按 doc_id 取台账原始行（中文列名 → 单元格值），取不到返回 None。

    doc_id 是**文件 stem**（不含 .pdf 后缀），而台账「文件名」列带后缀，所以先 ``Path(...).stem``
    归一后再比 —— 这一条不一致曾导致台账永远匹配不上。

    刻意不做进程级缓存：台账由外部脚本维护，缓存会让更新在服务重启前不生效，
    而单次读取只有 18 行，代价可忽略。
    """
    if not doc_id:
        return None
    if not SOURCES_CSV_PATH.exists():
        logger.warning(f"未找到语料台账 {SOURCES_CSV_PATH}")
        return None
    try:
        # utf-8-sig：台账由 Excel 维护，可能带 BOM，直接 utf-8 读会把首个列名读成 "\ufeff序号"
        with open(SOURCES_CSV_PATH, "r", encoding="utf-8-sig", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                if Path((row.get("文件名") or "").strip()).stem == doc_id:
                    return row
    except Exception as e:
        logger.error(f"读取语料台账失败：{e}", exc_info=True)
    return None


def load_product_name(doc_id: str) -> str:
    """从台账取该文档对应的产品备案名，取不到返回空串。"""
    row = load_ledger_row(doc_id)
    if not row:
        return ""
    return (row.get(PRODUCT_NAME_COLUMN) or "").strip()


def sanitize_item_name(raw: str, fallback: str = "") -> str:
    """清洗大模型产出的 item_name，清洗后为空则回退 fallback。

    三步：
    1. 剥掉成对包裹的引号（可叠多层，如 ``"「产品名」"``）；
    2. 去掉所有空白 —— 保险产品名是连续中文，不含空格语义，而模型偶发插入的空格
       （实测 ``（H2025 互联网）`` vs 台账 ``（H2025互联网）``）会让键与台账、规则库失配；
    3. 空串回退，避免 ``""`` 这种「非空但无意义」的值被写进库里当键。
    """
    text = (raw or "").strip()
    for _ in range(5):
        peeled = False
        for left, right in _QUOTE_PAIRS:
            if len(text) >= 2 and text.startswith(left) and text.endswith(right):
                text = text[1:-1].strip()
                peeled = True
        if not peeled:
            break

    text = _WHITESPACE_RE.sub("", text)
    if text:
        return text
    return (fallback or "").strip()
