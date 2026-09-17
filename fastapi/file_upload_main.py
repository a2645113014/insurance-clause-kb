import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks

app = FastAPI()

# 测试文件上传
@app.post("/upload/file")
async def upload_file(
        file: UploadFile = File(...),
        remark: str = None
):
    ALLOWED_TYPES = ["image/jpeg", "image/png", "image/gif"]
    # file.content_type表示上传的文件的类型
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"上传的文件类型不允许，只允许{', '.join(ALLOWED_TYPES)}"
        )
    upload_dir_path = Path(__file__).parent / "upload"
    if not upload_dir_path.exists():
        upload_dir_path.mkdir(parents=True, exist_ok=True)
    # file.filename表示上传的文件的文件名
    upload_file_path = upload_dir_path / file.filename
    # 文件复制
    with open(upload_file_path, "wb") as f:
        while True:
            content = await file.read(1024 * 1024)
            if not content:
                break
            f.write(content)
    return {
        "code": 0,
        "msg": "文件上传成功",
        "data": {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": f"{file.size} 字节",  # 文件大小
            "save_path": upload_file_path,
            "remark": remark or "无备注"
        }
    }

# 测试：后台任务
# 创建任务所对应的函数
def test_task(message: str):
    asyncio.sleep(5)
    with open("test.txt", "w") as f:
        f.write(message)

# 创建路径处理函数
@app.get("/test/background/task")
def test_background_task(background_tasks: BackgroundTasks):
    background_tasks.add_task(test_task, "helloworld")
    return {
        "message":"任务正在后台执行"
    }

if __name__ == "__main__":
    uvicorn.run("file_upload_main:app", host="127.0.0.1", port=8000, reload=True)