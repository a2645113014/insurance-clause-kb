import asyncio
import uuid
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()
# 跨域配置（保持不变）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class QueryRequest(BaseModel):
    query: str
    session_id: str = None

task_queues = {}

async def test_task(session_id: str, query: str):
    task_queues[session_id] = asyncio.Queue()
    queue = task_queues[session_id]
    for i in range(5):
        await queue.put(
            {
                "event": "progress",
                "data": f"这是progress事件的数据{i+1}"
            }
        )
        await asyncio.sleep(1)
    await queue.put(
        {
            "event": "complete",
            "data": "任务完成"
        }
    )
    await queue.put(None)

@app.post("/submit_query")
async def submit(request: QueryRequest, background_tasks: BackgroundTasks):
    session_id = request.session_id or str(uuid.uuid4())
    background_tasks.add_task(test_task, session_id, request.query)
    return {
        "message": "submit success",
        "session_id": session_id
    }

@app.get("/stream/{session_id}")
async def stream(session_id: str):
    async def generate_data():
        while session_id not in task_queues:
            await asyncio.sleep(0.2)
        # 获取session_id所对应的队列
        queue = task_queues[session_id]
        # 获取队列中的数据
        while True:
            msg = await queue.get()
            if msg is None:
                break
            yield f"event: {msg['event']}\n"
            yield f"data: {msg['data']}\n\n"
    return StreamingResponse(generate_data(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)