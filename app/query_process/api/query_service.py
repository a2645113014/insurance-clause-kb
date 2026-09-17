from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.clients.mongo_history_utils import get_recent_messages, clear_history
from app.core.logger import logger
from app.query_process.agent.state import create_query_default_state

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
# from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import kb_query_app


# 定义fastapi对象
app = FastAPI(title="query service", description="掌柜智库查询服务！")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="查询内容")
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")

# 访问chat.html
@app.get("/chat.html")
def chat():
    # 获取chat.html的路径
    chat_file_path = Path(__file__).parent.parent / "page" / "chat.html"
    # 判断chat_file_path是否存在
    if not chat_file_path.exists():
        raise HTTPException(status_code=404, detail="chat.html页面不存在")
    return FileResponse(str(chat_file_path))

# 创建后台任务，通过图对象处理以后的问题query
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    # 创建初始化状态
    init_state = create_query_default_state(session_id=session_id, original_query=user_query, is_stream=is_stream)
    try:
        # 执行图对象
        kb_query_app.invoke(init_state)
        # 更新当前后台任务的状态为已完成
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        logger.error(f"{session_id}后台任务失败，{e}")
        # 更新当前后台任务的状态为失败
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        # 判断是否是流式调用
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})

# 处理用户的问题
@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    # 获取session_id,user_query,is_stream
    session_id = request.session_id or str(uuid.uuid4())
    user_query = request.query
    is_stream = request.is_stream
    # 判断是否为流式调用
    if is_stream:
        # 创建队列
        create_sse_queue(session_id)
    # 修改当前后台任务的状态为运行中
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
    # 判断是否为流式调用
    if is_stream:
        # 执行后台任务
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        return {
            "message": "结果正在处理中...",
            "session_id": session_id
        }
    else:
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")
        return {
            "message": "处理完成！",
            "session_id": session_id,
            "answer": answer,
            # 图片与引用来源同样要给到非流式请求：此前只返回 answer，
            # 非流式模式下的来源展示只能靠前端从正文里猜，与流式模式表现不一致（需求 §8）
            "image_urls": get_task_result(session_id, "image_urls", []),
            "citations": get_task_result(session_id, "citations", []),
            # 返回真实完成的节点列表，便于前端显示阶段进度
            "done_list": get_done_task_list(session_id)
        }

# 创建处理sse请求的路径处理函数
@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# 健康检查
@app.get("/health")
def health():
    return {"ok": True}

# 查询最近的10条历史对话
@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 10):
    try:
        history_list = get_recent_messages(session_id, limit)
        for history in history_list:
            history["_id"] = str(history["_id"])
        return {"session_id": session_id, "items": history_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")

# 清空历史对话
@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    count =  clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
