import asyncio

import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 创建存储不同会话所对应的队列的字典
# session_id: queue
task_queues = {}

async def test_task(session_id: str):
    if session_id not in task_queues:
        # asyncio.Queue()创建队列
        task_queues[session_id] = asyncio.Queue()
    # 通过session_id获取所对应的队列
    queue = task_queues[session_id]
    for i in range(5):
        await queue.put(f"这是{session_id}所对应队列中的第{i+1}条数据")
        await asyncio.sleep(1)
    # 表示任务已完成，即不再向队列中存储数据
    await queue.put(None)



@app.get("/submit/{session_id}")
async def submit(session_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(test_task, session_id)
    return {
        "message": "submit success",
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
            yield f"data: {msg}\n\n"
    return StreamingResponse(generate_data(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)