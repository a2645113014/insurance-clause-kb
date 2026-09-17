import asyncio

import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse, FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, \
    StreamingResponse

app = FastAPI()

# 测试1：路径处理函数直接返回简单数据类型(字符串、数值、布尔)
# 结果：返回简单数据类型会直接作为响应体响应到浏览器端，其中布尔会自动转换，True-->true,False-->false
@app.get("/response/str")
def test_response_str():
    # return "helloworld"
    return True

# 测试2：路径处理函数返回列表或字典
# 结果：路径处理函数返回列表或字典会被转换为json（数组或对象），再作为响应体响应到浏览器端
@app.get("/response/dict")
def test_response_dict():
    # return {"name":"张三", "age":18, "gender":"male"}
    return ["张三", "李四", "王五"]

# 测试3：返回JSONResponse对象
@app.get("/response/json")
def test_response_json():
    # return JSONResponse(
    #     content={"name":"张三", "gender":"男"},
    #     status_code=200,
    #     headers={
    #         "test-response-header": "helloworld"
    #     }
    # )
    return {"name":"张三", "gender":"男"}

# 测试4：返回FileResponse对象
@app.get("/response/file")
def test_response_file():
    # 下载的效果
    # return FileResponse(
    #     path="./test/工作记录.xlsx",
    #     filename="工作记录.xlsx"
    # )
    # 响应的html文件会直接显示在页面中
    return FileResponse(
        path="./test/hello.html",
        media_type="text/html"
    )

# 测试5：返回HTMLResponse对象
@app.get("/response/html")
def test_response_html():
    html_str = """
    <ol>
        <li>张三</li>
        <li>李四</li>
        <li>王五</li>
        <li>张六</li>
    </ol>
    """
    return HTMLResponse(
        content=html_str,
    )

# 测试6：返回PlainTextResponse对象
@app.get("/response/text")
def test_response_text():
    # return PlainTextResponse(
    #     content="helloworld"
    # )

    # return "helloworld"

    html_str = """
        <ol>
            <li>张三</li>
            <li>李四</li>
            <li>王五</li>
            <li>张六</li>
        </ol>
        """
    return PlainTextResponse(
        content=html_str,
    )

# 测试7：返回RedirectResponse对象
# RedirectResponse：重定向，指定一个新的url响应到浏览器，由浏览器再次访问
@app.get("/response/redirect")
def test_response_redirect():
    return RedirectResponse("/response/json")

# 测试8：返回StreamingResponse对象
async def generate_stream():
    words = ["你", "好", "，", "这", "是", "流", "式", "响", "应"]
    for word in words:
        await asyncio.sleep(0.5)
        yield word
@app.get("/response/stream")
def test_response_stream():
    return StreamingResponse(
        content=generate_stream(),
        media_type="text/html",
    )

# 测试9：处理请求的过程中，抛出HTTPException
@app.get("/response/exception")
def test_response_exception():
    raise HTTPException(status_code=444, detail="报错了")
    return {"message": "helloworld"}

if __name__ == "__main__":
    uvicorn.run("response_main:app", host="127.0.0.1", port=8000, reload=True)