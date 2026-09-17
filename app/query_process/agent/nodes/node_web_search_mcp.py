import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from app.conf.bailian_mcp_config import mcp_config
from app.core.logger import node_log
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 调用MCP工具
async def mcp_call_streamable(query):
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": mcp_config.api_key},
            "timeout": 300,
            "sse_read_timeout": 300,
            "terminate_on_close": True,
        },
        max_retry_attempts=2,
    )
    try:
        await search_mcp.connect()
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": query, "count": 5},
        )
        return result
    finally:
        await search_mcp.cleanup()

@node_log("node_web_search_mcp")
def node_web_search_mcp(state: QueryGraphState):
    # 记录当前任务的状态为进行中
    add_running_task(state["session_id"], "node_web_search_mcp", state["is_stream"])
    # 获取状态中的rewritten_query
    rewritten_query = state["rewritten_query"]
    # 创建存储最终结果的列表
    results = []
    # 判断rewritten_query是否为空
    if rewritten_query:
        # 异步调用MCP服务
        result = asyncio.run(mcp_call_streamable(rewritten_query))
        # 获取网络搜索的结果主要数据
        text = result.content[0].text
        # 将json格式的字符串text转换为字典
        text_dict = json.loads(text)
        # 遍历text_dict中pages所对应的数据
        for page in text_dict.get("pages", []):
            results.append(
                {
                    "title": page.get("title", "").strip(),
                    "url": page.get("url", "").strip(),
                    "snippet": page.get("snippet", "").strip(),
                }
            )
    # 记录当前任务的状态为已完成
    add_done_task(state["session_id"], "node_web_search_mcp", state["is_stream"])
    return {"web_search_docs": results}

if __name__ == "__main__":
    init_state = {
        "session_id": "abc",
        "is_stream": False,
        "rewritten_query": "HAK 180 烫金机如何安全使用",
    }
    result = node_web_search_mcp(init_state)
    print(result)