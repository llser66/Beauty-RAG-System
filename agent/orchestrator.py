"""
Agent 编排器 — 基于 LangChain 官方 create_agent()

参考文档:
  https://docs.langchain.com/oss/python/langchain/agents
  https://docs.langchain.com/oss/python/langchain/tools

create_agent() 内部自动处理:
  - ReAct 循环  (思考 → 调用工具 → 观察结果 → 继续思考)
  - 工具调用解析与执行
  - 停止条件判断
"""

from typing import AsyncGenerator, List, Dict, Any, Optional

from langchain.agents import create_agent
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from agent.tools import create_tools


# ── System Prompt ───────────────────────────────────────
AGENT_SYSTEM_PROMPT = """你是一位专业的美妆护肤顾问助手「Beauty Agent」。你拥有丰富的护肤、彩妆、成分知识库，可以回答用户的美妆相关问题。

## 你的能力
你可以使用以下工具来获取信息：
1. **search_knowledge_base** — 从内部知识库检索美妆护肤专业知识
2. **search_images** — 根据描述匹配相关的美妆产品图、效果图
3. **get_knowledge_base_stats** — 查看知识库有什么资料
4. **analyze_skin** — 分析皮肤照片（当用户上传了皮肤照片时调用此工具）

## 工作方式
对于每个用户问题，你应该：
1. **思考** → 需要什么信息才能回答
2. **行动** → 调用合适的工具获取信息
3. **观察** → 看工具返回了什么
4. **综合** → 综合所有信息给出最终回答

## 行为准则
- 始终用中文回答，语气专业、温和、有亲和力
- 回答时标注信息来源（如从哪个文档找到的）
- 如果知识库中没有相关信息，诚实告知，不要编造
- 如果用户问题模糊，主动询问澄清细节
- 回答格式清晰有条理，适当使用标题、列表
- 如果用户上传了皮肤照片，请务必调用 analyze_skin 工具分析，不要只看文字描述
"""


# ── 消息构建 ───────────────────────────────────────────
def _build_messages(
    chat_history: Optional[List[Dict]] = None,
    user_query: str = "",
) -> list:
    """
    将多轮对话历史 + 当前问题拼装成消息列表，
    传给 create_agent().invoke() 的 "messages" 键。
    """
    messages = []

    if chat_history:
        for msg in chat_history[-8:]:  # 最多保留 8 轮历史
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))

    if user_query:
        messages.append(HumanMessage(content=user_query))

    return messages


# ── 结果解析 ───────────────────────────────────────────
def _parse_result(result: dict) -> Dict[str, Any]:
    """
    从 agent.invoke() 返回的 StateGraph 结果中提取:
      - answer: 最终回答文本
      - steps:  中间步骤 (tool + args + summary)
      - images: 图片列表 (从工具调用中收集)
    """
    all_messages = result.get("messages", [])

    # 最终回答 = 最后一条 AIMessage 且无 tool_calls
    answer = ""
    for m in reversed(all_messages):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            answer = str(m.content)
            break
    if not answer:
        answer = "抱歉，我没有获取到足够的信息来回答这个问题。"

    # 提取步骤
    steps = []
    for m in all_messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                steps.append({
                    "tool": tc.get("name", "unknown"),
                    "args": tc.get("args", {}),
                    "result_summary": "已完成",
                })

    # 图片（search_images 工具返回的文本中可能包含图片信息）
    images = []
    for m in all_messages:
        if isinstance(m, ToolMessage) and m.name == "search_images":
            content = str(m.content)
            if "找到以下图片" in content:
                images.append({"tool_result": content})

    return {"answer": answer, "steps": steps, "images": images}


# ═════════════════════════════════════════════════════════
# run_agent — 非流式 Agent
# ═════════════════════════════════════════════════════════
async def run_agent(
    user_query: str,
    llm,
    text_store=None,
    rerank_func=None,
    milvus_uri: str = "http://127.0.0.1:19530",
    image_collection: str = "image_embeddings",
    stats_func=None,
    vision_llm=None,
    chat_history: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """运行 Agent 自主推理（非流式）。

    参数:
        user_query:      用户输入
        llm:             ChatOpenAI 实例
        text_store:      Milvus 向量存储
        rerank_func:     BGE 重排函数
        milvus_uri:      Milvus 地址
        image_collection: 图片集合名
        stats_func:      统计函数
        vision_llm:      视觉模型
        chat_history:    多轮历史 [{"role":"user"|"assistant", "content":...}]

    返回:
        {"answer": str, "steps": [...], "images": [...]}
    """
    # 1. 创建工具（闭包注入依赖）
    tools = create_tools(
        text_store=text_store,
        rerank_func=rerank_func,
        milvus_uri=milvus_uri,
        image_collection=image_collection,
        stats_func=stats_func,
        vision_llm=vision_llm,
    )

    # 2. 构建初始消息
    messages = _build_messages(chat_history=chat_history, user_query=user_query)

    # 3. 创建 Agent（LangChain 官方 API）
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

    # 4. 运行
    result = agent.invoke({"messages": messages})

    # 5. 解析返回
    return _parse_result(result)


# ═════════════════════════════════════════════════════════
# run_agent_stream — 流式 Agent
# ═════════════════════════════════════════════════════════
async def run_agent_stream(
    user_query: str,
    llm,
    text_store=None,
    rerank_func=None,
    milvus_uri: str = "http://127.0.0.1:19530",
    image_collection: str = "image_embeddings",
    stats_func=None,
    vision_llm=None,
    chat_history: Optional[List[Dict]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """运行 Agent 自主推理（流式），逐步 yield 事件。

    yield 事件类型:
        {"type": "step", "tool": str, "args": dict}
        {"type": "text",  "content": str}
        {"type": "done"}
        {"type": "error", "content": str}
    """
    # 1. 创建工具
    tools = create_tools(
        text_store=text_store,
        rerank_func=rerank_func,
        milvus_uri=milvus_uri,
        image_collection=image_collection,
        stats_func=stats_func,
        vision_llm=vision_llm,
    )

    # 2. 构建初始消息
    messages = _build_messages(chat_history=chat_history, user_query=user_query)

    # 3. 创建 Agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

    # 4. 流式运行 — stream_mode="updates" 返回每步增量
    seen_tool_calls = set()
    try:
        async for chunk in agent.astream(
            {"messages": messages},
            stream_mode="updates",
        ):
            # chunk 结构: {"model": {...}} 或 {"tools": {...}}
            for node_name, node_output in chunk.items():
                msgs = node_output.get("messages", [])

                for m in msgs:
                    # 工具调用 → 通知前端
                    if isinstance(m, AIMessage) and m.tool_calls:
                        for tc in m.tool_calls:
                            if tc["id"] not in seen_tool_calls:
                                seen_tool_calls.add(tc["id"])
                                yield {
                                    "type": "step",
                                    "tool": tc.get("name", "unknown"),
                                    "args": tc.get("args", {}),
                                }

                    # 最终文本 → 推送内容
                    elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
                        yield {"type": "text", "content": str(m.content)}

        yield {"type": "done"}

    except Exception as e:
        yield {"type": "error", "content": str(e)}
