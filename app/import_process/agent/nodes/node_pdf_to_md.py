import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from app.conf.mineru_config import mineru_config
from app.core.logger import logger, node_log, step_log, PROJECT_ROOT
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task

"""
1.  **准备参数**: 获取 PDF 路径和输出目录。
2.  **请求上传**: 调用 MinerU 在线 API (`/file-urls/batch`) 获取上传链接。
3.  **上传文件**: 将 PDF 文件 PUT 到签名 URL。
4.  **轮询结果**: 循环查询任务状态 (`/extract-results/batch/{batch_id}`)，直到完成。
5.  **获取结果**: 下载生成的 ZIP 包，解压并读取 `.md` 文件内容到 state。
"""

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

@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj, local_dir_obj):
    """
    步骤2：上传PDF至MinerU并轮询解析任务状态
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

@step_log("step_3_download_and_extract")
def step_3_download_and_extract(zip_url: str, local_dir_obj: Path, stem: str):
    """
    步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件（重命名统一规范）
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
    # 步骤2：通过MinerU将pdf转换为md
    zip_url = step_2_upload_and_poll(pdf_path_obj, local_dir_obj)
    # 步骤3：下载压缩包并解压
    final_md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)
    # 更新状态中的md_path
    state["md_path"] = final_md_path
    # 将md文件中内容保存到状态的md_content中
    with open(final_md_path, "r", encoding="utf-8") as f:
        state["md_content"] = f.read()
    # 记录当前的节点状态为已完成
    add_done_task(state["task_id"], "node_pdf_to_md")
    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    print(node_pdf_to_md(test_state))

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")