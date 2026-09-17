import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

import anyio
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from app.core.logger import logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, add_done_task, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED, add_running_task, get_task_status, get_running_task_list, get_done_task_list


def prewarm_vector_model():
    """
    预热向量模型（在独立线程中串行执行）

    模型首次加载需要十几秒，且 transformers 的 from_pretrained 依赖进程级全局状态，
    并发执行会互相破坏并让权重停在 meta device。这里在服务启动阶段串行加载一次，
    把这段耗时从「首个请求内」提前到「启动期」，同时彻底排除多请求并发触发的首次加载竞争。
    """
    from app.lm.embedding_utils import get_bge_m3_ef
    get_bge_m3_ef()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 启动预热：放进线程池执行，避免阻塞事件循环导致服务迟迟不可用
    # 预热失败不阻断启动——首次请求时仍会走带锁的懒加载并重试
    try:
        await anyio.to_thread.run_sync(prewarm_vector_model)
    except Exception as e:  # noqa: BLE001
        logger.error(f"启动预热向量模型失败，将在首次请求时重试加载：{e}")
    yield


app = FastAPI(title="import service", description="掌柜智库导入服务！", lifespan=lifespan)

# 解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/import.html")
def import_file():
    # 获取import.html文件的路径
    import_file_path = PROJECT_ROOT / "app" / "import_process" / "page" / "import.html"
    # 判断import_file_path所对应的文件是否存在
    if not import_file_path.exists():
        raise HTTPException(status_code=404, detail="import.html not found")
    # 直接将import.html响应到浏览器
    return FileResponse(str(import_file_path))

# 创建后台任务，即执行LangGraph的图对象完成RAG的导入流程
def run_graph_task(task_id, local_file_path, local_dir):
    try:
        logger.info("导入流程开始")
        # 记录后台任务的执行状态为执行中
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        # 创建状态
        init_state = create_default_state(
            task_id=task_id,
            local_file_path=local_file_path,
            local_dir=local_dir,
        )
        # 流式执行图
        results = kb_import_app.stream(init_state)
        for result in results:
            for node_name, node_result in result.items():
                add_done_task(task_id, node_name)
        # 记录后台任务的执行状态为已完成
        logger.info("导入流程成功")
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        # 记录后台任务的执行状态为失败
        logger.error(f"导入流程失败，{e}")
        update_task_status(task_id, TASK_STATUS_FAILED)

@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    # 获取今天的日期，格式化为年月日格式的字符串
    today_str = datetime.now().strftime("%Y%m%d")
    # 设置today_str所对应的Path对象
    today_str_dir = PROJECT_ROOT / "output" / today_str
    # 创建存储每个上传文件所对应的task_id的列表
    task_ids = []
    # 遍历files，上传文件
    for file in files:
        # 使用uuid获取随机序列作为task_id
        task_id = str(uuid.uuid4())
        # 存储task_id
        task_ids.append(task_id)
        # 记录当前任务的状态为进行中
        add_running_task(task_id, "upload_file")
        # 获取上传文件的位置
        upload_dir_path = today_str_dir / task_id
        upload_dir_path.mkdir(parents=True, exist_ok=True)
        upload_file_path = upload_dir_path / file.filename
        # with open(upload_file_path, "wb") as f:
        with upload_file_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        # 记录当前任务的状态为已完成
        add_done_task(task_id, "upload_file")
        # 执行后台任务，即执行图
        background_tasks.add_task(
            run_graph_task,
            task_id,
            str(upload_file_path),
            str(upload_dir_path)
        )
    return {
        "code": 200,
        "message": "upload success",
        "task_ids": task_ids
    }

# 根据task_id查询当前任务的状态信息
@app.get("/status/{task_id}")
async def get_status(task_id: str):
    return {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "running_list": get_running_task_list(task_id),
        "done_list": get_done_task_list(task_id)
    }

if __name__ == "__main__":
    uvicorn.run("file_import_service:app", host="127.0.0.1", port=8000, reload=True)
