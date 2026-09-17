"""
    1、@app.get("/")
    用于标记路径处理函数，app为FastAPI的核心对象名，get表示当前允许的请求方式
    因此当以get请求方式，访问/时，就会映射到该装饰器所装饰的函数的执行
    其中/表示服务器端的绝对路径，/-->协议://ip：port
    /-->http://127.0.0.1:8000
    /test/path-->http://127.0.0.1:8000/test/path
    2、请求参数
    路径参数：请求参数作为路径的一部分进行传输，/test/path/param/admin/123456
    查询参数：请求参数以参数名=参数值的方式传输，/test/query/param?username=admin&password=123456
    请求体参数：以json格式传输请求参数，必须在请求体中传输，请求方式一定不能是get
"""

import uvicorn
from fastapi import FastAPI, Header
from pydantic import BaseModel

# 创建FastAPI的核心对象
app = FastAPI()

# 案例1：/-->{"message":"helloworld"}
@app.get("/")
def test_read_root():
    return {"message":"helloworld"}

# 测试路径参数
@app.get("/test/path/param/{username}/{password}")
def test_path_param(username: str = None, password: str = None):
    # 获取路径参数，只需要路径处理函数的参数位置设置参数，保证参数名和路径中占位符的名字保持一致，就可以获取路径参数
    return {"username":username,"password":password}

# 测试查询参数
@app.get("/test/query/param")
def test_query_param(username: str = None, password: str = None):
    # 获取查询参数，只需要路径处理函数的参数位置设置参数，保证参数名和传输的查询参数的参数名保持一致，就可以获取查询参数
    return {"username":username,"password":password}

class UserInfo(BaseModel):
    username: str = None
    password: str = None
    age: int = None


# 测试请求体参数，{username:"admin",password:"123456"}
@app.post("/test/body/param")
def test_body_param(user_info: UserInfo):
    # 获取请求体参数，需要创建一个模型类，该类中的属性和请求体中json格式的参数的键保持一致
    # 此时只需要在路径处理函数的参数位置设置模型类类型的参数，就可以获取请求体参数
    return {"username": user_info.username, "password": user_info.password, "age": user_info.age}

# 测试获取请求报文中的请求头信息
@app.get("/test/header")
def test_header(user_agent: str = Header(None)):
    # 获取请求报文中的某个请求头信息，在路径处理函数的参数位置设置参数
    # 保证参数名和请求头中的键保持一致（将键中-替换为_），同时设置参数的类型为str，默认值为Header(None)
    return {"user_agent": user_agent}



if __name__ == "__main__":
    uvicorn.run("request_main:app", host="127.0.0.1", port=8000, reload=True)