from langgraph.constants import END
from langgraph.graph import StateGraph

from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_rule_tool import is_rule_query, node_rule_tool
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.state import QueryGraphState

builder = StateGraph(QueryGraphState)

# 添加节点
builder.add_node(node_item_name_confirm)
builder.add_node(node_search_embedding)
builder.add_node(node_search_embedding_hyde)
builder.add_node(node_web_search_mcp)
builder.add_node(node_rrf)
builder.add_node(node_rerank)
builder.add_node(node_rule_tool)
builder.add_node(node_answer_output)

# 创建条件边的路径函数
# 判断state中answer，若有answer有值，则指向node_answer_output；
# 理赔材料/时限类问题指向 node_rule_tool（走 MongoDB 确定性键值查询，跳过检索）；
# 其余问题指向三路检索
def condition_fun(state: QueryGraphState):
    if state["answer"]:
        return "node_answer_output"
    if is_rule_query(state):
        return "node_rule_tool"
    return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"


# 添加边
# 设置初始节点
builder.set_entry_point("node_item_name_confirm")
# 添加条件边
builder.add_conditional_edges(
    "node_item_name_confirm",
    condition_fun,
    {
        "node_search_embedding": "node_search_embedding",
        "node_search_embedding_hyde": "node_search_embedding_hyde",
        "node_web_search_mcp": "node_web_search_mcp",
        "node_rule_tool": "node_rule_tool",
        "node_answer_output": "node_answer_output",
    }
)
builder.add_edge("node_search_embedding", "node_rrf")
builder.add_edge("node_search_embedding_hyde", "node_rrf")
builder.add_edge("node_web_search_mcp", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_answer_output")
# 规则工具节点直接汇入生成节点：规则问题的答案已在 rule_context 里，
# 不需要（也不应该）再叠一层检索结果
builder.add_edge("node_rule_tool", "node_answer_output")
builder.add_edge("node_answer_output", END)

kb_query_app = builder.compile()
