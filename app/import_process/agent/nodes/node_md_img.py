import base64
import os
import re
import sys
from collections import deque
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from app.clients.minio_utils import get_minio_client
from app.conf.lm_config import lm_config
from app.conf.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.utils.task_utils import add_running_task, add_done_task

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# 判断md文件中所引用的图片是否是IMAGE_EXTENSIONS中所支持的图片
def is_supported_image(file_name: str) -> bool:
    return Path(file_name).suffix.lower() in IMAGE_EXTENSIONS

@step_log("step_1_get_content")
def step_1_get_content(state: ImportGraphState):
    # 获取md_path并判断是否为空
    md_path = state["md_path"]
    if not md_path:
        raise ValueError("md_path为空，请检查流程")
    # 获取md_path所对应的Path对象
    md_path_obj = Path(md_path)
    # 判断md_path_obj是否存在
    if not md_path_obj.exists():
        raise FileNotFoundError(f"{md_path}文件不存在")
    # 判断md_content是否为空
    # 若为空，说明当前流程中，上传的文件是md，节点的执行node_entry-->node_md_img
    # 若不为空，说明前流程中，是将上传的pdf转换为md文件
    if not state["md_content"]:
        state["md_content"] = md_path_obj.read_text(encoding="utf-8")
    # 获取存储md文件所对应的图片的目录路径
    images_dir_obj = md_path_obj.parent / "images"
    return state["md_content"], md_path_obj, images_dir_obj

@step_log("step_2_scan_images")
def step_2_scan_images(md_content: str, images_dir_obj: Path):
    # 创建存储最终结果的列表，list[tuple(图片名,图片路径,tuple(上文, 下文))]
    image_targets = []
    # 遍历images_dir_obj中所有的图片文件
    for image_file in images_dir_obj.iterdir():
        # 获取每个图片文件的文件名
        image_name = image_file.name
        # 判断文件名是否是支持的图片类型
        if not is_supported_image(image_name):
            logger.warning(f"{image_name}不是支持的图片类型")
            continue
        # 创建匹配md_content中图片的正则表达式
        # re.escape()将字符串中所有的特殊符号进行转义
        # re.escape(a.jpg)==>a\.jpg
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) +".*?\)")
        # 在md_content中匹配正则
        match_results = list(pattern.finditer(md_content))
        # 判断match_results是否为空
        if not match_results:
            logger.warning(f"图片{image_name}未在md文件中引用")
            continue
        # 获取image_name在md_content中第一次出现的开始索引和结束索引
        start, end = match_results[0].span()
        # 分别获取开始索引前100个字符和结束索引后100个字符作为上下文
        pre_text = md_content[max(0, start - 100):start]
        post_text = md_content[end:min(len(md_content), end + 100)]
        # 将上文和下文组合成上下文
        context = (pre_text, post_text)
        # 将遍历的图片的信息追加到image_targets中
        image_targets.append((image_name, str(image_file), context))
    return image_targets

@step_log("step_3_image_summary")
def step_3_image_summary(image_targets, stem):
    # 创建存储最终结果的字典
    summaries = {}
    # 创建一个双端队列
    requests_limiter = deque()
    # 遍历image_targets，获取md文件中每个图片的摘要信息
    for image_name, image_path, context in image_targets:
        # 实现限流
        apply_api_rate_limit(requests_limiter, max_requests=100)
        # 获取提示词
        prompt = load_prompt("image_summary", root_folder=stem, image_content=context)
        # 获取视觉模型
        vl_model = get_llm_client(lm_config.lv_model)
        # 判断图片路径image_path是否是字符串
        if isinstance(image_path, str):
            image_path = Path(image_path)
        # 将图片文件中内容转换为base64格式的字符串
        image_base64 = base64.b64encode(image_path.read_bytes()).decode(encoding="utf-8")
        # 准备用户提示词，以下是访问视觉模型解析图片的提示词的固定结构
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        )
        # 创建链对象
        chain = vl_model | StrOutputParser()
        # 调用视觉模型
        summary = chain.invoke([message])
        # 存储图片和所对应的摘要信息
        summaries[image_name] = summary
    return summaries

@step_log("step_4_upload_images_replace")
def step_4_upload_images_replace(image_summaries, image_targets, md_content, stem):
    # 获取minio的客户端对象
    minio_client = get_minio_client()
    # 将之前md文件中的图片查询
    object_list = minio_client.list_objects(
        bucket_name=minio_config.bucket_name, # 设置桶名
        prefix=f"{minio_config.minio_img_dir[1:]} / {stem}", # 设置获取的图片的前缀（即所在的目录）
        recursive=True, # 是否递归获取所有的子目录中的文件
    )
    # 将object_list中的图片转换为DeleteObject对象
    delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
    # 将object_list中的图片删除
    delete_errors = minio_client.remove_objects(
        bucket_name=minio_config.bucket_name,
        delete_object_list=delete_object_list
    )
    # 遍历删除的结果
    for error in delete_errors:
        logger.warning(f"图片删除失败,{error}")
    # 创建存储图片上传到minio之后的url的字典
    image_urls = {}
    # 遍历image_targets，将图片上传到minio
    for image_name, image_path, _ in image_targets:
        try:
            # 上传图片
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir}/{stem}/{image_name}",
                file_path=image_path,
                content_type="image/jpeg"
            )
            # 获取图片在minio中的地址
            image_urls[image_name] = f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_name}"
        except Exception as e:
            logger.error(f"{image_name}上传失败,{e}")
    # 创建存储图片所对应的摘要信息和minio中url的变量
    image_infos = {}
    # 遍历image_summaries
    for image_name, summary in image_summaries.items():
        image_infos[image_name] = (summary, image_urls[image_name])
    # 遍历image_infos，将md_content中的![]()-->![summary](url)
    for image_name, (summary, url) in image_infos.items():
        # 创建正则表达式，匹配md_content中的图片
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + ".*?\)")
        # 替换md_content中的![]()-->![summary](url)
        md_content = pattern.sub(lambda _: f"![{summary}]({url})", md_content)
    return md_content

@step_log("step_5_backup_md_file")
def step_5_backup_md_file(md_path_obj: Path, new_md_content: str):
    # 创建新的md文件的路径
    new_md_path_obj = md_path_obj.parent / f"{md_path_obj.stem}_new{md_path_obj.suffix}"
    # 向new_md_path_obj所对应的文件中写入new_md_content
    new_md_path_obj.write_text(new_md_content, encoding="utf-8")
    return str(new_md_path_obj)


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    # 记录任务状态为运行中
    add_running_task(state["task_id"], "node_md_img")
    # 步骤1：进行核心参数校验 [校验md_path/md_content/返回images的文件夹地址]
    md_content, md_path_obj, images_dir_obj = step_1_get_content(state)
    # 步骤2：提取md文件中的图片
    image_targets = step_2_scan_images(md_content, images_dir_obj)
    # 步骤3：进行图片内容总结和处理[调用多模态模型,总结图片内容,最终返回 图片名/总结
    image_summaries = step_3_image_summary(image_targets, md_path_obj.stem)
    # 步骤4：上传图片到minio服务器,替换图片的本地地址和描述!返回替换后的md_content内容
    new_md_content = step_4_upload_images_replace(image_summaries, image_targets , md_content, md_path_obj.stem)
    # 步骤5：备份新的md内容,改为原名称_new.md
    new_md_file_path_str = step_5_backup_md_file(md_path_obj, new_md_content)
    # 更新状态
    state["md_content"] = new_md_content
    state["md_path"] = new_md_file_path_str
    # 记录任务状态为已完成
    add_done_task(state["task_id"], "node_md_img")
    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")