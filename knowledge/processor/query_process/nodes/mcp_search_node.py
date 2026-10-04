import asyncio      # ← 新增
import json
import logging
import os           # ← 新增


from knowledge.processor.query_process.prompts.item_name_extract_prompt import USER_HYDE_PROMPT_TEMPLATE
from langchain_core.messages import  HumanMessage,SystemMessage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model, generate_hybrid_embeddings
from knowledge.utils.milvus_util import get_milvus_client,create_hybrid_search_requests,execute_hybrid_search_query
from knowledge.utils.llm_client import get_llm_client
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model, generate_hybrid_embeddings
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query


class McpSearchNode(BaseNode):
    name = "mcp_search_node"
    """
    负责从网络上查询当前的问题[整个知识库都找不到这个问题]
    mcp形式调用网络第三方各种通用的搜索工具
    """


    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 参数校验
        validated_rewritten_query, validate_item_names = self._validate_query_inputs(state)

        # 2. 创建mcp_client 并且让客户端执行工具 bailian_web_search
        mcp_result = self._create_execute_web_search(validated_rewritten_query)

        # 3. 更新state
        #    只返回本节点修改的字段：本节点与 vector/hyde/kg 三路并行，
        #    返回整个 state 会触发 InvalidUpdateError("At key 'session_id': ...")
        return {"web_search_docs": mcp_result}

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:  # 1 usage
        # 1. 获取state的rewritten_query
        rewritten_query = state.get('rewritten_query', "")

        # 2. 获取state的item_names
        item_names = state.get('item_names', "")

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 4. 返回
        return rewritten_query, item_names

    def _create_execute_web_search(self, validated_rewritten_query):
        from langchain_mcp_adapters.client import MultiServerMCPClient

        # 1、构造 LangChain MultiMcpServerClient
        multi_server_client = MultiServerMCPClient(
            connections={
                "bing-cn-mcp-server": {
                    "transport": "streamable_http",
                    "url": "https://mcp.api-inference.modelscope.net/619ec146c2a043/mcp",
                    "headers": {
                        "Authorization": "Bearer ms-9d19ab8c-ed18-4698-a7ac-ce0413d87b56"}
                }
            }
        )

        tools = asyncio.run(multi_server_client.get_tools())
        print(tools)

        from langchain.agents import create_agent
        from langchain_deepseek import ChatDeepSeek

        llm = ChatDeepSeek(model="deepseek-v4-flash")
        agent = create_agent(
            model=llm,
            tools=tools
        )
        res = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": validated_rewritten_query}]}))
        msgs = res["messages"]

        final_answer = next((m.content for m in reversed(msgs) if m.type == "ai"), "")

        return final_answer

