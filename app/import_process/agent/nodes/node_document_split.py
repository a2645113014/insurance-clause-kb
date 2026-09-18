import hashlib
import json
import re
from pathlib import Path

from app.core.logger import logger, node_log, step_log, PROJECT_ROOT
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

"""
本节点职责：把一份条款纯文本切成「一个条款一个 chunk」的结构化切片，并写入 state["chunks"]。

为什么不能沿用按 Markdown 标题（`^#`）切分：条款 PDF 直抽出的纯文本里没有 Markdown 标题层级，
原有切分会退化成「全文一个 section + 按 200 字硬切」，把条款从中间斩断。条款库的核心设计是
**以条款号为主键**，因此切分必须以「条款号锚点」为切分依据。

三级切分（对应 Milvus 字段 chunk_level）：
  L1 条款号级 —— 按 `N.M` 锚点切分，一个条款一个 chunk（默认路径，占绝大多数）
  L2 分项级   —— 条款超过 CHUNK_SIZE 时，在其内部子项（`1.` / `（一）`）边界做贪心合并，
                 目的不是切碎而是「保证不切断子项的前提下贴近目标长度」
  L3 兜底级   —— L2 之后仍超长时，用递归字符切分按标点边界兜底

处理的两家保司排版差异（实测）：
  太平洋人寿：顶层 `1．`（全角点）独占一行 → 标题另起一行 → 正文，且标题可能跨行（`合同成立与生` + `效`）
  中英人寿  ：顶层 `第1 章`（章式） + 二级 `1.1 合同标题`（同行） → 正文，标题可能被列宽截断成两行

正文区识别：每份 PDF 开头都有「封面 + 阅读指引 + 条款目录」，目录会把全部条款号按顺序列一遍。
这些行对条款检索无增益（信息在正文条款里都有且更权威），全部丢弃。
丢弃不影响可回溯性 —— 原始 MD 文件完整保留，本节点只决定"哪些行进 chunk"。
"""

# ==================== 切分参数 ====================
# 单个切片目标长度。800 字对应的是一到两个完整条款的体量
CHUNK_SIZE = 800
# L3 兜底切分时的重叠长度
CHUNK_OVERLAP = 80
# 标题续行的最大长度（超过此长度视为正文，不再并入标题）
MAX_TITLE_EXT_LEN = 12
# 标题最多吸收的续行数，防止把连续短句误并入标题
MAX_TITLE_EXT_LINES = 3
# 正文起点判定：锚点后最多允许连续几行短行（即标题区）
MAX_SHORT_RUN = 2
# 正文起点判定：多长的行算「正文行」
MIN_BODY_LINE_LEN = 25
# 正文起点判定：需要连续几行正文行才认定进入正文区
MIN_BODY_RUN = 2
# L2 切分的长度硬上限（优先不切断子项，故允许适度超过 CHUNK_SIZE）
L2_HARD_CAP = int(CHUNK_SIZE * 1.6)

# ==================== 条款号正则 ====================
# 顶层（太平洋式）：全角句点，且后面不是数字（排除 `1.1`）
# 只认全角句点是有意的 —— 中英条款内部的子项 `1.` 用半角句点，若一并认作顶层会被大量误判
RE_CLAUSE_TOP = re.compile(r"^\s*(\d{1,2})\s*．\s*(?![\d])\s*(.*)$")
# 顶层（中英式）：第N章
RE_CLAUSE_CHAPTER = re.compile(r"^\s*第\s*(\d{1,2})\s*章\s*(.*)$")
# 二级：数字.数字（半角或全角）。三重排除 ——
#   (?!\d)                 后面不能再跟数字（排除 `1.10` 被截成 `1.1`）
#   (?!\s*[．.]\s*\d)      不能再跟 `.数字`（排除三级的 `2.4.1`）
#   (?!\s*[标点%‰])        不能紧跟右括号或百分号（排除正文里的「13.4）」引用与「8.4%」等百分比）
RE_CLAUSE_SUB = re.compile(
    r"^\s*(\d{1,2})\s*[．.]\s*(\d{1,2})(?!\d)(?!\s*[．.]\s*\d)"
    r"(?!\s*[）)】\]，,。、；;：%‰])\s*(.*)$"
)
# 子项锚点（用于 L2）：`1.` / `2、`
RE_SUB_ITEM = re.compile(r"^\s*(\d{1,2})\s*[．.、]\s*(?![\d．.])(?=\S)")
# 子项锚点（用于 L2）：`（一）` / `(1)`
RE_SUB_ITEM_CN = re.compile(r"^\s*[（(]\s*[一二三四五六七八九十百\d]{1,3}\s*[）)]\s*(?=\S)")
# 以句末标点结尾的行 —— 不可能是标题
RE_SENT_END = re.compile(r"[。；：，,;]$")
# 以标点或百分号开头的「标题」—— 实为正文残留，不算锚点
RE_BAD_TITLE_HEAD = re.compile(r"^[）)】\]，,。、；;：:%‰]")

# ==================== 条款类型关键词 ====================
# 兜底表：当 insurance_kb/02_clause_schema.json 不可读时使用
_FALLBACK_TYPE_KEYWORDS = {
    "保障责任": ["保险责任", "保障责任", "保险金给付责任", "我们承担的保险责任"],
    "责任免除": ["责任免除", "除外责任", "不承担给付保险金责任", "免除保险人责任"],
    "等待期": ["等待期", "观察期"],
    "保险期间": ["保险期间", "保障期间", "合同效力"],
    "犹豫期": ["犹豫期", "冷静期"],
    "保险金申请": ["保险金申请", "申请与给付", "索赔", "理赔材料", "申请人须提供"],
    "保险金给付": ["保险金给付", "给付比例", "给付限额", "免赔额", "赔付比例"],
    "释义": ["释义", "名词解释", "本条款下列名词定义"],
    "合同构成与效力": ["合同构成", "合同成立", "合同生效", "合同变更", "合同终止"],
    "保险费": ["保险费", "交费方式", "交费期间", "宽限期", "保险费的交纳"],
    "现金价值与退保": ["现金价值", "退保", "解除合同", "未满期净保险费"],
    "保单贷款": ["保单贷款", "借款", "贷款金额"],
    "受益人": ["受益人", "受益人的指定", "受益人的变更", "保险金作为遗产"],
    "如实告知": ["如实告知", "明确说明", "告知义务", "询问"],
    "合同解除与变更": ["合同解除", "合同变更", "解除合同的权利"],
    "争议处理与法律适用": ["争议处理", "管辖", "诉讼", "法律适用", "司法管辖"],
}


def load_clause_type_keywords():
    """
    读取条款类型关键词表。
    单一事实来源是 insurance_kb/02_clause_schema.json（与 SFT 输出契约共用同一份定义），
    读不到时退回内置兜底表，保证节点不会因外部文件缺失而中断。
    """
    schema_path_obj = Path(PROJECT_ROOT) / "insurance_kb" / "02_clause_schema.json"
    try:
        schema = json.loads(schema_path_obj.read_text(encoding="utf-8"))
        keywords = schema["enums"]["clause_type"]["match_keywords"]
        # 去掉「其他」，它由匹配失败兜底产生，不参与正向匹配
        return {k: v for k, v in keywords.items() if k != "其他" and v}
    except Exception as e:
        logger.warning(f"读取 clause_type 关键词表失败，使用内置兜底表：{e}")
        return _FALLBACK_TYPE_KEYWORDS


@step_log("step_1_get_content")
def step_1_get_content(state: ImportGraphState):
    """步骤1：取内容并统一换行符"""
    # 获取md_content
    md_content = state["md_content"]
    # 判断md_content是否为空
    if not md_content:
        logger.error("md文档内容获取失败，无法完成切分")
        raise RuntimeError("md文档内容获取失败，无法完成切分")
    # 统一将md_content中的\r\n和\r替换为\n
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    # 获取file_title
    file_title = state["file_title"]
    return md_content, file_title


def is_clause_anchor(line: str):
    """
    判断一行是否是条款锚点。
    :return: (anchor_level, clause_no, inline_title) 或 None
             anchor_level 取值 chapter / sub / top
    """
    stripped_line = line.strip()
    if not stripped_line:
        return None
    # 章式顶层优先（第N章）
    match = RE_CLAUSE_CHAPTER.match(stripped_line)
    if match:
        return "chapter", f"第{match.group(1)}章", match.group(2).strip()
    # 二级 N.M
    match = RE_CLAUSE_SUB.match(stripped_line)
    if match:
        # 标题以标点开头 → 说明这行是正文残片（如被换行截断的「13.4）」），不是条款锚点
        if RE_BAD_TITLE_HEAD.match(match.group(3).strip()):
            return None
        return "sub", f"{match.group(1)}.{match.group(2)}", match.group(3).strip()
    # 数字式顶层 N．
    match = RE_CLAUSE_TOP.match(stripped_line)
    if match:
        if RE_BAD_TITLE_HEAD.match(match.group(2).strip()):
            return None
        return "top", f"{match.group(1)}.", match.group(2).strip()
    return None


def normalize_clause_no(anchor_level: str, clause_no: str) -> str:
    """
    归一化条款号，供排序与引用拼接使用。
    统一目标：两家保司的顶层都收敛成纯数字（太平洋 `1．` → `1`，中英 `第1章` → `1`）
    """
    if anchor_level == "sub":
        major, minor = clause_no.split(".")
        return f"{int(major)}.{int(minor)}"
    if anchor_level == "chapter":
        return str(int(clause_no.replace("第", "").replace("章", "")))
    return str(int(clause_no.rstrip("．.")))


@step_log("step_2_locate_body_start")
def step_2_locate_body_start(md_content: str):
    """
    步骤2：定位正文起点，丢弃开头的封面 / 阅读指引 / 条款目录

    判据（两家通用）：找第一个「条款锚点行 + 其后最多 2 行短行（标题区） + 连续 2 行长文本」的位置。
      太平洋：`1.1` → `合同构成`(4字) → `"太保…合同"（以下简称本合同）由`(38字) → `保险条款…`(33字)
      中英  ：`1.1 合同构成` → `我们与您订立的《中英人寿…》合同（以下简称本`(35字) → `合同）由本保险条款…`(33字)
      目录区：`1.1 合同构成` → 下一行即另一个条款锚点 → 直接中断，不会被误判

    叠加「编号回落」约束：目录区的最后一项后面可能紧跟封面长行，但它相对前一个二级锚点是递增的，
    而正文的第一个二级锚点必然小于目录里的最后一个编号（目录列到 16.34，正文从 1.1 重新开始）。

    命中后再向前回溯，把紧邻的顶层锚点（章标题）纳入正文区 —— 否则 `1．您与我们订立的合同`
    这一行会被留在丢弃区，导致条款缺失父级路径。
    """
    lines = md_content.split("\n")
    line_count = len(lines)
    prev_sub_no = None

    for idx in range(line_count):
        anchor = is_clause_anchor(lines[idx])
        if not anchor:
            continue
        current_sub_no = (
            tuple(int(x) for x in anchor[1].split(".")) if anchor[0] == "sub" else None
        )

        # 判定「锚点 + 连续短行(<=MAX_SHORT_RUN) + 连续长行(>=MIN_BODY_RUN)」
        short_run = 0
        long_run = 0
        is_body_start = False
        for probe in range(idx + 1, min(idx + 12, line_count)):
            probe_line = lines[probe].strip()
            if not probe_line:
                continue
            if is_clause_anchor(probe_line):
                break
            if len(probe_line) >= MIN_BODY_LINE_LEN:
                long_run += 1
                if long_run >= MIN_BODY_RUN:
                    is_body_start = True
                    break
            else:
                short_run += 1
                long_run = 0
                if short_run > MAX_SHORT_RUN:
                    break

        if is_body_start:
            # 回落约束只对二级锚点生效
            if current_sub_no is None or prev_sub_no is None or current_sub_no < prev_sub_no:
                body_start = idx
                # 向前回溯，纳入紧邻的顶层锚点（章标题）
                back = idx - 1
                while back >= 0 and idx - back <= 6:
                    back_line = lines[back].strip()
                    back_anchor = is_clause_anchor(back_line)
                    if back_anchor and back_anchor[0] in ("top", "chapter"):
                        body_start = back
                        break
                    if back_line and len(back_line) > MAX_TITLE_EXT_LEN:
                        break
                    back -= 1
                logger.info(
                    f"正文起点定位成功：第 {body_start} 行，丢弃前置区 {body_start} 行"
                    f"（封面 / 阅读指引 / 条款目录）"
                )
                return lines[body_start:]

        if current_sub_no is not None:
            prev_sub_no = current_sub_no

    logger.warning("未识别出正文起点，全文参与切分")
    return lines


@step_log("step_3_split_by_clause")
def step_3_split_by_clause(body_lines, file_title: str):
    """
    步骤3：按条款锚点切分，并做标题跨行合并

    标题合并规则：锚点行若没有同行标题（太平洋式），或标题被列宽截断（中英式），
    吸收后续的短行作为标题续行。遇到子项锚点（`1.` / `（1）`）立即停止，
    避免把条款正文的第一条子项误并入标题。
    """
    # 先收集全部锚点位置
    anchor_marks = []
    for line_idx, line in enumerate(body_lines):
        anchor = is_clause_anchor(line)
        if anchor:
            anchor_marks.append((line_idx, anchor[0], anchor[1], anchor[2]))

    if not anchor_marks:
        logger.warning(f"{file_title} 未识别出任何条款锚点")
        return []

    sections = []
    for pos, (line_idx, anchor_level, clause_no, inline_title) in enumerate(anchor_marks):
        # 本条款的结束位置 = 下一个锚点的行号
        section_end = (
            anchor_marks[pos + 1][0] if pos + 1 < len(anchor_marks) else len(body_lines)
        )
        # 标题
        title_parts = [inline_title] if inline_title else []
        cursor = line_idx + 1
        ext_count = 0
        while cursor < section_end:
            cursor_line = body_lines[cursor].strip()
            if not cursor_line:
                cursor += 1
                continue
            # 子项锚点 → 正文开始，停止吸收
            if RE_SUB_ITEM.match(cursor_line) or RE_SUB_ITEM_CN.match(cursor_line):
                break
            if (
                len(cursor_line) <= MAX_TITLE_EXT_LEN
                and not RE_SENT_END.search(cursor_line)
                and ext_count < MAX_TITLE_EXT_LINES
            ):
                title_parts.append(cursor_line)
                ext_count += 1
                cursor += 1
            else:
                break
        # 正文
        body_text = "\n".join(
            line.strip() for line in body_lines[cursor:section_end] if line.strip()
        )
        sections.append(
            {
                "anchor_level": anchor_level,
                "clause_no": clause_no,
                "clause_no_norm": normalize_clause_no(anchor_level, clause_no),
                "clause_title": "".join(title_parts).strip(),
                "body": body_text,
                "file_title": file_title,
            }
        )
    return sections


def match_clause_type(clause_title: str, clause_type_keywords: dict) -> str:
    """
    按条款标题匹配 clause_type。
    只匹配标题、不匹配正文 —— 正文提到别的术语太常见（如「基本保险金额」条款里提到保险费），
    标错比标成「其他」危害更大：检索时按 clause_type 过滤会漏召回或召错。
    """
    for clause_type, keywords in clause_type_keywords.items():
        for keyword in keywords:
            if keyword in clause_title:
                return clause_type
    return "其他"


@step_log("step_4_refine_chunks")
def step_4_refine_chunks(sections, file_title: str):
    """
    步骤4：组装最终切片（L1 直通 / L2 分项 / L3 兜底），并补齐元数据

    关键点：章级条款（有子条款的 `1．`/`第1章`）本身不产生切片，只作为子条款的 clause_path 前缀；
    无子条款的一级条款（如中英的「第5章 现金价值」）则自身产生一个切片，否则内容会整段丢失。
    """
    clause_type_keywords = load_clause_type_keywords()

    # 统计哪些顶层锚点下存在子条款
    top_with_children = set()
    current_top_no = None
    for section in sections:
        if section["anchor_level"] in ("top", "chapter"):
            current_top_no = section["clause_no"]
        elif section["anchor_level"] == "sub" and current_top_no:
            top_with_children.add(current_top_no)

    final_chunks = []
    current_top_section = None
    for section in sections:
        # 顶层锚点：记录上下文；若无子条款则自身成切片
        if section["anchor_level"] in ("top", "chapter"):
            current_top_section = section
            if section["clause_no"] not in top_with_children:
                full_text = (
                    section["clause_title"] + "\n" + section["body"]
                ).strip()
                if full_text:
                    final_chunks.extend(
                        build_chunk(
                            section, full_text, file_title, None, clause_type_keywords
                        )
                    )
            continue

        # 二级条款
        full_text = (section["clause_title"] + "\n" + section["body"]).strip()
        if not full_text:
            logger.warning(f"{file_title} 条款 {section['clause_no']} 正文为空，已跳过")
            continue
        final_chunks.extend(
            build_chunk(
                section, full_text, file_title, current_top_section, clause_type_keywords
            )
        )
    return final_chunks


def build_chunk(section, full_text, file_title, top_section, clause_type_keywords):
    """对单个条款做 L1/L2/L3 分级切分，产出符合 02_clause_schema.json 的切片字典"""
    pieces = split_long_text(full_text)

    # 构造层级路径：`1 您与我们订立的合同 > 1.1 合同构成`
    if top_section:
        clause_path = (
            f'{top_section["clause_no_norm"]} {top_section["clause_title"]}'
            f' > {section["clause_no"]} {section["clause_title"]}'
        ).strip()
    else:
        clause_path = f'{section["clause_no_norm"]} {section["clause_title"]}'.strip()

    clause_type = match_clause_type(section["clause_title"], clause_type_keywords)

    chunk_list = []
    for chunk_seq, (chunk_level, piece_text) in enumerate(pieces, start=1):
        chunk_list.append(
            {
                "clause_no": section["clause_no"],
                "clause_no_norm": section["clause_no_norm"],
                "clause_title": section["clause_title"],
                "clause_path": clause_path,
                "clause_type": clause_type,
                # 字段名与 02_clause_schema.json 对齐：正文字段为 text
                "text": piece_text,
                "chunk_seq": chunk_seq,
                "chunk_level": chunk_level,
                "char_len": len(piece_text),
                # text 的 sha256，作为「先删后插」幂等导入的校验依据
                "content_hash": hashlib.sha256(piece_text.encode("utf-8")).hexdigest(),
                "file_title": file_title,
            }
        )
    return chunk_list


def split_long_text(full_text: str):
    """
    三级切分核心：返回 [(chunk_level, text), ...]

    L1 长度达标 → 整条一个切片
    L2 超长     → 在子项边界做「贪心合并」，累加到接近 CHUNK_SIZE 才断开。
                  注意这里的策略是「合并」而不是「切碎」：直接把每个子项各切一片会产出
                  大量 100 字以下的碎片（实测 03 号重疾条款会从 143 片暴涨到 866 片），
                  且同一个条款被拆散后无法还原上下文。
    L3 仍超长   → 用递归字符切分按标点边界兜底
    """
    if len(full_text) <= CHUNK_SIZE:
        return [("L1", full_text)]

    # 收集子项锚点行号
    lines = full_text.split("\n")
    item_index_list = [
        i
        for i, line in enumerate(lines)
        if RE_SUB_ITEM.match(line.strip()) or RE_SUB_ITEM_CN.match(line.strip())
    ]

    if len(item_index_list) >= 2:
        bounds = item_index_list + [len(lines)]

        def segments_len(segments):
            return sum(len(x) + 1 for x in segments)

        groups = []
        current_group = lines[: item_index_list[0]] if item_index_list[0] > 0 else []
        for k in range(len(item_index_list)):
            segment = lines[bounds[k]: bounds[k + 1]]
            if current_group and segments_len(current_group) + segments_len(segment) > CHUNK_SIZE:
                merged = "\n".join(current_group).strip()
                if merged:
                    groups.append(merged)
                current_group = segment
            else:
                current_group = current_group + segment
        merged = "\n".join(current_group).strip()
        if merged:
            groups.append(merged)

        groups = [g for g in groups if g]
        if len(groups) >= 2 and all(len(g) <= L2_HARD_CAP for g in groups):
            return [("L2", g) for g in groups]

    # L3 兜底
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " "],
    )
    return [("L3", piece) for piece in splitter.split_text(full_text)]


@step_log("step_5_backup_chunks")
def step_5_backup_chunks(final_chunks, state):
    """步骤5：把切片写出到 json 便于排查（不参与后续流程）"""
    chunks_backup_path = Path(state["md_path"]).parent / "chunks.json"
    with open(chunks_backup_path, "w", encoding="utf-8") as f:
        json.dump(
            final_chunks,
            f,
            ensure_ascii=False,  # 中文直接原文存储
            indent=4,  # 表示json结构的缩进
        )
    logger.info(f"切片备份已写出：{chunks_backup_path}")


@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    实现要点:
    1. 丢弃封面 / 阅读指引 / 条款目录，定位正文区。
    2. 以条款号为主键做三级切分（L1 条款号 / L2 分项 / L3 兜底）。
    3. 生成含 clause_no / clause_no_norm / clause_type / clause_path 的元数据切片。
    """
    # 记录节点的状态为运行中
    add_running_task(state["task_id"], "node_document_split")
    # 步骤1：获取与清洗内容
    md_content, file_title = step_1_get_content(state)
    # 步骤2：定位正文起点，丢弃封面/阅读指引/条款目录
    body_lines = step_2_locate_body_start(md_content)
    # 步骤3：按条款号锚点切分并合并跨行标题
    sections = step_3_split_by_clause(body_lines, file_title)
    # 步骤4：三级切分并补齐元数据
    final_chunks = step_4_refine_chunks(sections, file_title)
    # 步骤5：更新状态并备份切片数据
    state["chunks"] = final_chunks
    step_5_backup_chunks(final_chunks, state)
    # 记录节点的状态为已完成
    add_done_task(state["task_id"], "node_document_split")

    clause_count = len({c["clause_no_norm"] for c in final_chunks})
    level_stat = {}
    for c in final_chunks:
        level_stat[c["chunk_level"]] = level_stat.get(c["chunk_level"], 0) + 1
    logger.info(
        f"{file_title} 切分完成：覆盖 {clause_count} 个条款，"
        f"产出 {len(final_chunks)} 个切片，层级分布={level_stat}"
    )
    return state


if __name__ == "__main__":
    """
    单元测试：批量跑 output 下已解析出的条款 MD，验证切分质量
    前置条件：先执行过 node_pdf_to_md，output/<stem>/<stem>.md 已存在
    """
    import os

    logger.info("===== 开始 node_document_split 切分质量批量测试 =====")

    output_dir_obj = Path(PROJECT_ROOT) / "output"
    md_path_list = sorted(
        p for p in output_dir_obj.glob("*/*.md") if not p.name.endswith(".REJECTED.md")
    )

    if not md_path_list:
        logger.error(f"未找到可切分的 MD 文件，请先执行 node_pdf_to_md：{output_dir_obj}")
    else:
        logger.info(f"待切分文件数：{len(md_path_list)}")
        summary_rows = []
        for md_path_obj in md_path_list:
            test_state = {
                "task_id": f"test_split_{md_path_obj.stem}",
                "md_path": str(md_path_obj),
                "md_content": md_path_obj.read_text(encoding="utf-8"),
                "file_title": md_path_obj.stem,
            }
            try:
                result_state = node_document_split(test_state)
                chunks = result_state["chunks"]
                lengths = [c["char_len"] for c in chunks] or [0]
                summary_rows.append(
                    f"{md_path_obj.stem[:32]:<34} | 切片={len(chunks):>4} "
                    f"| 条款={len({c['clause_no_norm'] for c in chunks}):>3} "
                    f"| 最长={max(lengths):>4} 均={sum(lengths) // len(lengths):>3} "
                    f"| 空={sum(1 for c in chunks if not c['text'].strip())}"
                )
            except Exception as e:
                summary_rows.append(f"{md_path_obj.stem[:32]:<34} | FAIL | {e}")

        logger.info("===== 切分汇总 =====")
        for row in summary_rows:
            logger.info(row)

    logger.info("===== 结束 node_document_split 切分质量批量测试 =====")
