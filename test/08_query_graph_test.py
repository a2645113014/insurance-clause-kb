from app.query_process.agent.main_graph import kb_query_app
from app.query_process.agent.state import create_query_default_state

init_state = create_query_default_state(
    session_id="test_session_id",
    original_query="你好"
)

result = kb_query_app.invoke(init_state)
print("最终的状态：")
print(result)

print("图的结构：")
kb_query_app.get_graph().print_ascii()