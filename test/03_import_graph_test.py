"""
    测试：导入模块图对象的执行
"""
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state

# 创建初始状态
# state = {
#     "local_file_path": "hak180产品安全手册.pdf"
# }
state = create_default_state(local_file_path="hak180产品安全手册.md")

result = kb_import_app.invoke(state)

print(result)

# 输出图的结构
# print(kb_import_app.get_graph().draw_ascii())
kb_import_app.get_graph().print_ascii()