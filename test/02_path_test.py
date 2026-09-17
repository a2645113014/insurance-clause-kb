"""
    测试：Path
"""
import os
import sys
from pathlib import Path

# 获取项目的根路径
# print(Path(__file__).parent.parent)
# print(Path(__file__).parents[1])

# 获取文件的标题
# print(Path(__file__).stem)
# print(os.path.basename(__file__).split(".")[0])

# 测试os.path.exists()
# print(os.path.exists(r"D:\workspace\WH260713\WH260713_knowledge_base\doc\hak180产品安全手册.md"))
# path = Path(r"D:\workspace\WH260713\WH260713_knowledge_base\doc\hak180产品安全手册.pdf")
# print(path.exists())

# 测试路径的拼接
# print(os.path.join("aaa", "bbb"))
# path = Path("aaa")
# print(path / "bbb")

# 测试Path对象的常用属性
# path = Path(r"D:\workspace\WH260713\WH260713_knowledge_base\doc\hak180产品安全手册.pdf")
# print(path.parent) # 获取当前path对象所对应文件或目录的父目录（返回Path对象）
# print(list(path.parents)) # 获取当前path对象所对应文件或目录的所有的父目录（返回Path对象列表，按由近到远的顺序存储数据）
# print(path.name) # 获取当前path对象所对应文件或目录的文件名（xx.后缀）
# print(path.stem) # 获取当前path对象所对应文件或目录的标题（不包含后缀）
# print(path.suffix) # 获取当前path对象所对应文件的后缀

# 测试Path对象的常用方法
# path = Path("02_path_test.py")
# print(path.absolute()) # 获取绝对路径
# print(path.exists()) # 判断文件是否存在
# print(path.is_file()) # 判断是否是文件
# print(path.is_dir()) # 判断是否是目录
# path = Path("demo")
# path.mkdir(parents=True, exist_ok=True) # 创建目录

# path = Path(__file__).parent # D:\workspace\WH260713\WH260713_knowledge_base\test
# for file in path.iterdir(): # 遍历path所对应目录中所有的文件或目录
#     print(file)
# for file in path.rglob("*.py"): # 递归获取path对象所对应的目录中所有的文件或指定类型的文件
#     print(file)

# 获取某个路径中文件名
# print(os.path.basename(__file__))
# print(Path(__file__).name)

# 获取某个文件所在目录的路径
# print(os.path.dirname(__file__))
# print(Path(__file__).parent)
