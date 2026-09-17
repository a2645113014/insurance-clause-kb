"""
    1、固定的代码
    1> 生成器生成数据（yield）
    2> 必须使用StreamingResponse作为路径处理函数的返回值
    3> 每次服务器推送的数据必须满足以下的格式
    event: xxx\n        事件，可选
    data: xxx\n\n       数据，必须
    4> 服务器推送的数据的媒体类型必须为media_type="text/event-stream"
"""

import asyncio

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# 1. 初始化+跨域（最基础配置）
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/stream/{sessionId}")
async def simple_stream(sessionId: str):
    async def generate_data():
        for i in range(5):
            yield f"data: Hello, {sessionId}, {i}!\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(generate_data(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)