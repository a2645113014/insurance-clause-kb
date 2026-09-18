import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import fitz  # PyMuPDF：本地读取 PDF 文本层，不依赖外部服务
import requests

from app.conf.mineru_config import mineru_config
from app.core.logger import logger, node_log, step_log, PROJECT_ROOT
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task

"""
本节点职责：把 PDF 变成纯文本并写入 state["md_content"]，对下游保持出口契约不变
（state["md_path"] + state["md_content"]）。

执行策略为「本地优先、云端兜底」两路：

【主路】PyMuPDF(fitz) 本地直抽 —— 零成本、毫秒级、全角标点原样保留、条款号在行首
  原因：保险条款 PDF 绝大多数由排版软件导出，自带干净文本层；用 OCR 类服务反而会把
  正文误判为表格、把全角逗号转成半角，破坏「value 逐字摘录原文」的可验证性。

【兜底】MinerU 云端 OCR —— 仅当主路判定「无文本层」（纯扫描件/图片型 PDF）时启用

质量分级三档（见 assess_text_quality）：
  ok               → 用 fitz 结果直接返回
  no_text_layer    → 返回 None，由调用方转 MinerU OCR 兜底
  broken_encoding  → 抛错拒收（字体缺 ToUnicode 映射，两路都拿不到正确字符）

1.  **准备参数**: 获取 PDF 路径和输出目录。
2.  **本地直抽**: PyMuPDF 抽取全文并对文本层质量分级。
3.  **兜底判定**: 无文本层时调用 MinerU 在线 API (`/file-urls/batch`) 获取上传链接。
4.  **上传文件**: 将 PDF 文件 PUT 到签名 URL。
5.  **轮询结果**: 循环查询任务状态 (`/extract-results/batch/{batch_id}`)，直到完成。
6.  **获取结果**: 下载生成的 ZIP 包，解压并读取 `.md` 文件内容到 state。
"""

# ==================== 文本层质量判定阈值 ====================
# 每页平均字符数低于此值 → 判定为「无文本层」（纯扫描件），转 MinerU OCR 兜底
MIN_CHARS_PER_PAGE = 50
# CJK 字符占非空白字符比例低于此值 → 判定为「字体缺 ToUnicode 映射」，任何解析器都无法还原
MIN_CJK_RATIO = 0.05
# fitz 文本块是否按纵向坐标重排。
# 【2026-09-18 实测 17 份真语料】必须置 False：
#   本批条款 PDF 多为「左标题列 + 右正文列」的表格式排版，且内容流本身已是正确的阅读
#   顺序。置 True 会按 y 坐标把左右两列的内容拼到同一行，实测产生 14~23 行跨栏错拼
#   （如把「(1)保险合同；(2)申请人及…」错拼成「(1)保险合同；请所需的证明(…」），
#   条款号落在行首的比例从 ~82% 跌到 ~40%。
#   两种模式下「去除全部空白后的字符数」完全相等（差值 0），说明不丢内容，差异仅在排列。
PYMUPDF_SORT_BLOCKS = False

@step_log("step_1_validate_paths")
def step_1_validate_paths(state: ImportGraphState):
    """
    步骤1：路径校验与初始化
    校验PDF输入文件与输出目录的有效性，遵循「输入严格校验、输出自动修复」的鲁棒性设计原则：
    1. 校验PDF路径非空且文件真实存在，不存在则直接抛出异常（快速失败）
    2. 校验输出目录，为空则赋予默认值，不存在则自动创建（自动容错）
    3. 统一转换为Path对象处理，保证路径操作的规范性与跨平台兼容性
    :param state: 流程状态字典，包含pdf_path、local_dir
    :return: pdf_path所对应的Path对象，local_dir所对应的Path对象
    """
    # 分别获取pdf_path和local_dir
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")
    # 判断pdf_path是否为空
    if not pdf_path:
        # pdf_path为空，直接抛异常
        logger.error(f"pdf_path为空，请重新上传文件")
        raise ValueError("pdf_path为空，请重新上传文件")
    # 判断local_dir是否为空
    if not local_dir:
        # local_dir为空，赋值默认值
        local_dir = PROJECT_ROOT / "output"
        # 更新state中local_dir为默认值
        state["local_dir"] = local_dir
        logger.warning(f"local_dir为空，使用默认值:{local_dir}")
    # 分别获取pdf_path和local_dir所对应的Path对象
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    # 判断pdf_path_obj所对应的文件是否存在
    if not pdf_path_obj.exists():
        # pdf_path_obj所对应的文件不存在，直接抛异常
        logger.error(f"{pdf_path}所对应的文件不存在，请检查文件来源")
        raise ValueError(f"{pdf_path}所对应的文件不存在，请检查文件来源")
    # 判断local_dir_obj所对应的目录是否存在
    if not local_dir_obj.exists():
        # local_dir_obj所对应的目录不存在，创建
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    # 返回pdf_path_obj和local_dir_obj
    return pdf_path_obj, local_dir_obj

def assess_text_quality(text: str, page_count: int):
    """
    对抽取出的文本做质量分级，决定是否可以信任本地抽取结果。
    :param text: PyMuPDF 抽取出的全文
    :param page_count: PDF 页数
    :return: (level, chars_per_page, cjk_ratio)
             level 取值 ok / no_text_layer / broken_encoding
    """
    # 去掉首尾空白后统计字符数，避免空白页把均值抬高
    stripped_text = text.strip()
    total_chars = len(stripped_text)
    # 每页平均字符数：纯扫描件的文本层通常为空或只有零星页码
    chars_per_page = total_chars / page_count if page_count > 0 else 0.0
    # 统计非空白字符数作为 CJK 占比的分母（排除换行/空格对比例的干扰）
    non_space_chars = sum(1 for ch in stripped_text if not ch.isspace())
    # 统计 CJK 统一表意文字数量（\u4e00-\u9fff 覆盖常用汉字区）
    cjk_chars = sum(1 for ch in stripped_text if "\u4e00" <= ch <= "\u9fff")
    # CJK 占比：中文条款正常应在 60% 以上，低于 5% 说明取到的全是错误码位
    cjk_ratio = cjk_chars / non_space_chars if non_space_chars > 0 else 0.0

    if chars_per_page < MIN_CHARS_PER_PAGE:
        return "no_text_layer", chars_per_page, cjk_ratio
    if cjk_ratio < MIN_CJK_RATIO:
        return "broken_encoding", chars_per_page, cjk_ratio
    return "ok", chars_per_page, cjk_ratio

@step_log("step_2_extract_with_pymupdf")
def step_2_extract_with_pymupdf(pdf_path_obj: Path, local_dir_obj: Path, stem: str):
    """
    步骤2（主路）：本地 PyMuPDF 直抽文本 + 文本层质量分级
    :param pdf_path_obj: 已校验的 PDF Path 对象
    :param local_dir_obj: 输出目录 Path 对象
    :param stem: PDF 无后缀纯名称，用于命名输出文件
    :return: 质量合格时返回 MD 文件绝对路径；判定无文本层时返回 None（交调用方走 MinerU）
    :raise ValueError: 判定为 broken_encoding（字体缺 ToUnicode 映射，不可用）
    """
    # 打开 PDF 文档
    pdf_doc = fitz.open(str(pdf_path_obj))
    try:
        # 获取总页数
        page_count = pdf_doc.page_count
        # 逐页抽取纯文本。
        # sort=PYMUPDF_SORT_BLOCKS：按文本块纵向坐标重排，修正双栏排版的跨栏错序。
        page_text_list = [
            page.get_text("text", sort=PYMUPDF_SORT_BLOCKS) for page in pdf_doc
        ]
    finally:
        # 无论抽取是否异常都要释放文档句柄
        pdf_doc.close()

    # 拼接全文（页间用空行分隔，保留页面边界信息便于后续定位页码）
    raw_text = "\n\n".join(page_text_list)
    # 质量分级
    level, chars_per_page, cjk_ratio = assess_text_quality(raw_text, page_count)
    logger.info(
        f"PyMuPDF 抽取完成：页数={page_count}，每页字符数={chars_per_page:.1f}，"
        f"CJK占比={cjk_ratio:.1%}，判定={level}"
    )

    # 无文本层：不落盘，直接交调用方走 MinerU OCR 兜底
    if level == "no_text_layer":
        logger.warning(
            f"{pdf_path_obj.name} 每页字符数仅 {chars_per_page:.1f}（阈值 {MIN_CHARS_PER_PAGE}），"
            f"判定为无文本层，转 MinerU OCR 兜底"
        )
        return None

    # 写出 MD 文件，目录结构与 MinerU 输出保持一致：local_dir/<stem>/<stem>.md
    target_dir_obj = local_dir_obj / stem
    target_dir_obj.mkdir(parents=True, exist_ok=True)
    md_path_obj = target_dir_obj / f"{stem}.md"
    md_path_obj.write_text(raw_text, encoding="utf-8")

    # 编码损坏：留痕后抛错，绝不静默产出垃圾切片进库
    if level == "broken_encoding":
        rejected_path_obj = target_dir_obj / f"{stem}.REJECTED.md"
        rejected_path_obj.write_text(raw_text, encoding="utf-8")
        md_path_obj.unlink(missing_ok=True)
        raise ValueError(
            f"{pdf_path_obj.name} 文本层 CJK 占比仅 {cjk_ratio:.1%}（阈值 {MIN_CJK_RATIO:.0%}），"
            f"判定为字体缺少 ToUnicode 映射表，PyMuPDF 与 MinerU 均无法还原正确字符。"
            f"坏样本已写出至 {rejected_path_obj}，请更换该产品其他版本或从语料中剔除"
        )

    return str(md_path_obj.resolve())

@step_log("step_3_upload_and_poll")
def step_3_upload_and_poll(pdf_path_obj, local_dir_obj):
    """
    步骤3（兜底）：上传PDF至MinerU并轮询解析任务状态
    核心流程：配置校验 → 获取上传链接 → 文件上传（含重试） → 任务轮询（直至完成/失败/超时）
    参数：pdf_path_obj-已校验的PDF Path对象；output_dir_obj-输出目录Path对象
    返回：解析结果ZIP包下载链接full_zip_url
    异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
    """
    # 配置校验
    if not mineru_config.api_key or not mineru_config.base_url:
        logger.error("mineru的配置为空，请检查配置文件")
        raise ValueError("mineru的配置为空，请检查配置文件")
    # 准备访问MinerU的相关数据
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    # 第一次请求的请求体，model_version必须添加，且值必须是vlm
    data = {
        "files": [
            {"name": "demo.pdf", "data_id": "abcd"}
        ],
        "model_version": "vlm"
    }

    # 发送第一次请求，作用是检测MinerU服务器是否能够正常连接
    response = requests.post(url, headers=header, json=data)
    # 获取此次请求的响应状态码，若不是200，则说明MinerU无法正常连接
    if response.status_code != 200:
        logger.error(f"连接MinerU服务器失败，响应状态码:{response.status_code}")
        raise RuntimeError(f"连接MinerU服务器失败，响应状态码:{response.status_code}")
    # 获取此次请求的响应体
    result = response.json()
    # 判断此次请求的接口调用状态
    if result["code"] != 0:
        logger.error(f"MinerU服务器端接口调用失败，接口状态码:{result['code']}，接口处理信息:{result['msg']}")
        raise RuntimeError(f"MinerU服务器端接口调用失败，接口状态码:{result['code']}，接口处理信息:{result['msg']}")
    # 说明第一次请求成功，获取任务id和上传pdf的链接地址
    batch_id = result["data"]["batch_id"]
    file_upload_url = result["data"]["file_urls"][0]

    # 发送第二次请求，将pdf文件中的内容上传到MinerU所返回的上传链接地址
    # 读取pdf文件中的内容
    pdf_file_data = pdf_path_obj.read_bytes()
    # 发送请求，使用Session.put()上传文件，可以关闭系统环境变量，避免OSS预签名URL校验失败
    with requests.Session() as session:
        session.trust_env = False
        upload_response = session.put(file_upload_url, data=pdf_file_data)
        # 判断响应状态码是否为200
        if upload_response.status_code != 200:
            raise RuntimeError(f"pdf文件上传失败，状态码:{upload_response.status_code}，请重试")

    # 发送第三次请求，通过batch_id获取解析结果，即pdf转换md之后的压缩文件的地址
    batch_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    # 最大超时时间10分钟
    timeout_seconds = 600
    # 轮询间隔3秒
    poll_interval = 3
    # 第一次轮询的时间
    start_time = time.time()
    while True:
        # 判断轮询的时间是否超过了最大的超时时间
        if time.time() - start_time > timeout_seconds:
            logger.error("获取解析结果超时")
            raise TimeoutError("获取解析结果超时")
        # 发送请求获取解析结果
        try:
            poll_response = requests.get(batch_url, headers=header)
        except Exception:
            # 发送请求过程中出现了异常，等待3秒，重新发送请求
            logger.warning("获取解析结果时出现异常")
            time.sleep(poll_interval)
            continue
        # 判断此次请求的响应状态码
        status_code = poll_response.status_code
        # 判断status_code是否为200
        if status_code != 200:
            # 判断status_code是否在500和600之间，若是则表示MinerU服务器端出现了问题，则重试
            if 500 <= status_code < 600:
                logger.warning(f"MinerU服务器端出现了问题，状态码:{status_code}")
                time.sleep(poll_interval)
                continue
            else:
                # 表示访问MinerU服务器出现问题，则抛异常
                raise RuntimeError("访问MinerU服务器出现问题")
        # 获取此次服务器响应的响应体
        poll_result = poll_response.json()
        # 判断接口状态码是否为0
        if poll_result["code"] != 0:
            logger.warning(f"MinerU服务器端接口调用出现问题，状态码:{poll_result['code']}，接口处理信息:{poll_result['msg']}")
            time.sleep(poll_interval)
            continue
        # 获取服务器响应的解析结果
        extract_result = poll_result["data"]["extract_result"][0]
        # 判断extract_result是否为空
        if not extract_result:
            logger.warning(f"未获取到解析结果，请重试")
            time.sleep(poll_interval)
            continue
        # 判断解析结果的状态是否为done
        if extract_result["state"] == "done":
            # 表示任务已完成，获取压缩文件的地址
            full_zip_url = extract_result["full_zip_url"]
            # 判断full_zip_url是否为空
            if not full_zip_url:
                # 表示解析任务已完成，但是没有有效的压缩文件下载地址
                raise RuntimeError("解析任务已完成，但是没有有效的压缩文件下载地址")
            return full_zip_url
        elif extract_result["state"] == "failed":
            # 表示解析失败
            raise RuntimeError("MinerU解析pdf文件失败")
        else:
            # 表示任务进行中，等待3秒，重新发送请求
            time.sleep(poll_interval)

@step_log("step_4_download_and_extract")
def step_4_download_and_extract(zip_url: str, local_dir_obj: Path, stem: str):
    """
    步骤4（兜底）：下载MinerU解析结果ZIP包并解压，提取目标MD文件（重命名统一规范）
    核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件（按优先级） → 重命名统一为PDF同名
    参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
    返回：最终MD文件的字符串格式绝对路径
    异常：RuntimeError(下载失败)、FileNotFoundError(无MD文件)
    """
    # 请求zip_url，下载压缩文件
    response = requests.get(zip_url, timeout=120)
    # 判断响应状态码是否为200
    if response.status_code != 200:
        raise ValueError(f"下载压缩文件失败，状态码:{response.status_code}")
    # 设置保存压缩文件的路径，并保存压缩文件
    zip_save_path = local_dir_obj / f"{stem}_result.zip"
    zip_save_path.write_bytes(response.content)
    # 设置保存压缩文件解压之后的文件的路径
    extract_target_dir = local_dir_obj / stem
    # 判断extract_target_dir是否存在，若存在，先将该文件之前相关的文件上传
    if extract_target_dir.exists():
        shutil.rmtree(extract_target_dir)
    # 创建extract_target_dir所对应的目录
    # mkdir(parents=True, exist_ok=True)
    # parents=True表示可以创建多层目录
    # exist_ok=True表示若目录存在也不会报错
    extract_target_dir.mkdir(parents=True, exist_ok=True)
    # 解压压缩文件
    with zipfile.ZipFile(zip_save_path, "r") as zip_ref:
        zip_ref.extractall(extract_target_dir)
    # 获取extract_target_dir目录中所有的md文件
    md_file_list = list(extract_target_dir.rglob("*.md"))
    # 判断md_file_list是否为空
    if not md_file_list:
        raise RuntimeError("解压之后的结果中没有任何的md文件")
    # 获取pdf转换的md文件
    target_md_file = None
    # 先获取extract_target_dir中和原pdf文件标题一致的md文件
    for md_file in md_file_list:
        if md_file.stem == stem:
            target_md_file = md_file
            break
    # 若extract_target_dir中没有和原pdf文件标题一致的md文件，再找full.md
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name == "full.md":
                target_md_file = md_file
                break
    # 若extract_target_dir中没有和原pdf文件标题一致的md文件，也没有full.md，直接获取md文件列表中的第一个
    if not target_md_file:
        target_md_file = md_file_list[0]
    # 判断最终获取的md文件的标题是否是stem，即和原pdf文件标题一致，若不一致，则修改
    if target_md_file.stem != stem:
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))
    # 返回最终md文件的绝对路径
    return str(target_md_file.resolve())


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
    """
    # 记录当前的节点状态为运行中
    add_running_task(state["task_id"], "node_pdf_to_md")
    # 步骤1：路径校验
    pdf_path_obj, local_dir_obj = step_1_validate_paths(state)
    # 取PDF无后缀名，供输出文件命名与目录命名使用
    stem = pdf_path_obj.stem
    # 步骤2：主路 —— 本地 PyMuPDF 直抽（零成本、毫秒级、标点与条款号原样保留）
    # 返回 None 表示文本层为空，判定为纯扫描件，需走云端 OCR 兜底
    final_md_path = step_2_extract_with_pymupdf(pdf_path_obj, local_dir_obj, stem)
    # 步骤3、4：兜底 —— 仅当主路判定无文本层时，才调用 MinerU 云端 OCR
    if not final_md_path:
        logger.info("主路判定为无文本层，启动 MinerU OCR 兜底")
        zip_url = step_3_upload_and_poll(pdf_path_obj, local_dir_obj)
        final_md_path = step_4_download_and_extract(zip_url, local_dir_obj, stem)
    # 更新状态中的md_path
    state["md_path"] = final_md_path
    # 将md文件中内容保存到状态的md_content中
    with open(final_md_path, "r", encoding="utf-8") as f:
        state["md_content"] = f.read()
    # 记录当前的节点状态为已完成
    add_done_task(state["task_id"], "node_pdf_to_md")
    return state

if __name__ == "__main__":

    # 单元测试：批量验证 data/clauses/pdf 下全部语料的解析质量
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    # 语料目录
    test_pdf_dir_obj = Path(PROJECT_ROOT) / "data" / "clauses" / "pdf"
    test_pdf_list = sorted(test_pdf_dir_obj.glob("*.pdf"))

    if not test_pdf_list:
        logger.error(f"语料目录下没有 PDF：{test_pdf_dir_obj}")
    else:
        logger.info(f"待解析语料数量：{len(test_pdf_list)}")
        # 汇总每个文件的解析结果
        summary_list = []
        for pdf_file_obj in test_pdf_list:
            test_state = create_default_state(
                task_id=f"test_pdf2md_{pdf_file_obj.stem}",
                pdf_path=str(pdf_file_obj),
                local_dir=str(Path(PROJECT_ROOT) / "output"),
            )
            try:
                result_state = node_pdf_to_md(test_state)
                summary_list.append(
                    (pdf_file_obj.name, "OK", len(result_state.get("md_content") or ""), "")
                )
            except Exception as e:
                summary_list.append((pdf_file_obj.name, "FAIL", 0, str(e)))
        # 打印汇总表
        logger.info("===== 解析汇总 =====")
        for file_name, status, char_count, err_msg in summary_list:
            logger.info(f"{status} | {char_count} 字符 | {file_name} | {err_msg}")

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")