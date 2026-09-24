from langgraph.constants import END
from langgraph.graph import StateGraph

from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_compliance_check import (
    extract_clause_text,
    is_compliance_query,
    node_compliance_check,
)
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
builder.add_node(node_compliance_check)
builder.add_node(node_answer_output)

# 创建条件边的路径函数
# 判断state中answer，若有answer有值，则指向node_answer_output；
# 合规审查意图指向 node_compliance_check（Phase 3.3 Planner 分流）；
# 理赔材料/时限类问题指向 node_rule_tool（走 MongoDB 确定性键值查询，跳过检索）；
# 其余问题指向三路检索
def condition_fun(state: QueryGraphState):
    # answer 优先：node_item_name_confirm 在产品名歧义时会写一条「您是想问哪个产品」
    # 的澄清答复，那比合规节点的「请提供待审表述」更具体，应当放行。
    # （产品名确实找不到时的「未找到相关产品」兜底，在 confirm 节点里已对
    #  合规类问题跳过，不会走到这里。）
    if state["answer"]:
        return "node_answer_output"
    if is_compliance_query(state):
        query = state.get("rewritten_query") or state.get("original_query") or ""
        # 用户直接给了条款原文 → 不必检索，直接审
        if extract_clause_text(query):
            return "node_compliance_check"
        # 只指了产品/条款号 → 先走三路检索把原文捞回来，rerank 后再转合规节点
        if state.get("item_names"):
            return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"
        # 既没有原文、也没有可定位的产品 → 交给合规节点显式说明缺什么
        return "node_compliance_check"
    if is_rule_query(state):
        return "node_rule_tool"
    return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"


def after_rerank(state: QueryGraphState):
    """rerank 之后的分流：合规意图转合规节点，其余照旧生成答案。

    能走到这里说明是「合规意图但没内联原文」那条路径（有原文的在
    condition_fun 里就直奔合规节点了），待审表述由 rerank 结果提供。
    """
    if is_compliance_query(state):
        return "node_compliance_check"
    return "node_answer_output"


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
        "node_compliance_check": "node_compliance_check",
        "node_answer_output": "node_answer_output",
    }
)
builder.add_edge("node_search_embedding", "node_rrf")
builder.add_edge("node_search_embedding_hyde", "node_rrf")
builder.add_edge("node_web_search_mcp", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")
# rerank 之后不再直连生成节点：合规路径要在拿到条款原文后转合规节点判定
builder.add_conditional_edges(
    "node_rerank",
    after_rerank,
    {
        "node_compliance_check": "node_compliance_check",
        "node_answer_output": "node_answer_output",
    }
)
# 规则工具节点直接汇入生成节点：规则问题的答案已在 rule_context 里，
# 不需要（也不应该）再叠一层检索结果
builder.add_edge("node_rule_tool", "node_answer_output")
# 合规节点把判定结果写进 answer，再交给生成节点统一做 SSE 推送与历史落库
# （node_answer_output 的 step_1 检测到 answer 已存在就不会重复调模型）
builder.add_edge("node_compliance_check", "node_answer_output")
builder.add_edge("node_answer_output", END)

kb_query_app = builder.compile()
