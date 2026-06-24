"""
Agent 工具定义层 — 使用 LangChain 官方 @tool 装饰器

参考文档:
  https://docs.langchain.com/oss/python/langchain/tools
  https://docs.langchain.com/oss/python/langchain/agents

模式: 工厂函数 create_tools(deps) → List[BaseTool]
  闭包捕获依赖 (text_store, rerank_func, vision_llm 等),
  LLM 只看到工具签名参数 (query, top_k 等), 不会看到内部依赖。
"""

import os
import base64
from typing import List, Optional

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# ═══════════════════════════════════════════════════════════════
# create_tools — 工具工厂（依赖注入通过闭包完成）
# ═══════════════════════════════════════════════════════════════

def create_tools(
    text_store=None,
    rerank_func=None,
    milvus_uri: str = "http://127.0.0.1:19530",
    image_collection: str = "image_embeddings",
    stats_func=None,
    vision_llm=None,
) -> list:
    """
    创建 Agent 工具列表。

    参数:
        text_store:   Milvus 向量存储实例
        rerank_func:  BGE 重排函数
        milvus_uri:   Milvus 连接地址
        image_collection: 图片集合名称
        stats_func:   get_kb_stats 函数引用
        vision_llm:   视觉模型 ChatOpenAI 实例
    """

    # ── 1. search_knowledge_base ──────────────────────────
    @tool
    def search_knowledge_base(query: str, top_k: int = 5) -> str:
        """从美妆护肤知识库中检索与问题相关的文本资料，返回最相关的知识片段。支持中文查询。

        Args:
            query: 搜索查询关键词，应准确提炼用户问题的核心语义
            top_k: 返回结果数量 (1-10)，默认 5
        """
        if text_store is None:
            return "错误：知识库引擎未初始化"

        # 粗召回 top 10
        raw_docs = text_store.similarity_search(query, k=10)

        # BGE 重排精排
        if rerank_func:
            try:
                reranked = rerank_func(query, raw_docs, top_k=min(top_k, 5))
                top_docs = [item[0] for item in reranked]
            except Exception:
                top_docs = raw_docs[: min(top_k, 5)]
        else:
            top_docs = raw_docs[: min(top_k, 5)]

        # 拼装结果
        fragments = []
        sources = []
        seen = set()
        for doc in top_docs:
            fragments.append(doc.page_content)
            src = doc.metadata.get("source", "")
            if src and src not in seen:
                sources.append(src)
                seen.add(src)

        return (
            f"找到 {len(fragments)} 条相关知识：\n"
            + "\n---\n".join(fragments)
            + f"\n\n来源文件: {', '.join(s.split('/')[-1] for s in sources)}"
        )

    # ── 2. search_images ──────────────────────────────────
    @tool
    def search_images(query: str, limit: int = 5) -> str:
        """搜索美妆相关的图片素材，输入文字描述即可匹配最相关的图片。
        用于找产品图、效果图、素材图。

        Args:
            query: 图片描述关键字，如"控油面膜"、"口红试色"
            limit: 返回图片数量 (1-10)，默认 5
        """
        from pymilvus import connections, MilvusClient
        from utils.multimodal import embed_text_for_image_search

        try:
            connections.connect(alias="agent_img_search", uri=milvus_uri)
            ic = MilvusClient(alias="agent_img_search")

            query_vec = embed_text_for_image_search(query)
            img_results = ic.search(
                collection_name=image_collection,
                data=[query_vec.tolist()],
                limit=limit,
                output_fields=["filepath", "filename", "tags"],
            )

            matched = []
            if img_results and img_results[0]:
                for hit in img_results[0]:
                    score = round(hit["distance"], 4)
                    if score >= 0.15:
                        matched.append(
                            f"  - {hit['entity']['filename']} (相关度: {score})"
                        )

            ic.close()
            connections.disconnect("agent_img_search")

            if not matched:
                return "未找到匹配的图片"
            return "找到以下图片：\n" + "\n".join(matched)
        except Exception as e:
            return f"图片搜索异常: {e}"

    # ── 3. get_knowledge_base_stats ───────────────────────
    @tool
    def get_knowledge_base_stats() -> str:
        """查看当前知识库的统计数据，包括文本切片数、图片数、源文档数。
        当用户询问"知识库有什么"、"你们有什么资料"时调用此工具。
        """
        if stats_func is None:
            return "统计功能不可用"
        try:
            s = stats_func()
            return (
                f"知识库统计：\n"
                f"  - 文本切片: {s.get('text_chunks', 0)} 条\n"
                f"  - 图片素材: {s.get('images', 0)} 张\n"
                f"  - 源文档: {s.get('documents', 0)} 个"
            )
        except Exception as e:
            return f"获取统计失败: {e}"

    # ── 4. analyze_skin ───────────────────────────────────
    @tool
    def analyze_skin(image_path: str) -> str:
        """分析用户上传的皮肤照片，识别肤质类型、皮肤问题，并给出护肤建议。
        当用户上传了面部/皮肤照片并希望分析时调用此工具。

        Args:
            image_path: 皮肤照片的本地文件路径
        """
        if vision_llm is None:
            return "错误：视觉分析模型未初始化"
        if not image_path or not os.path.exists(image_path):
            return f"错误：图片不存在 — {image_path}"

        try:
            with open(image_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(image_path)[1].lower()
            mime_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp",
            }
            mime = mime_map.get(ext, "image/jpeg")

            msg = HumanMessage(content=[
                {
                    "type": "text",
                    "text": (
                        "你是一位专业皮肤科顾问。请分析这张皮肤照片，给出以下内容：\n"
                        "1. 肤质类型（干性/油性/混合性/中性/敏感肌）\n"
                        "2. 皮肤问题（痘痘/黑头/泛红/暗沉/细纹/毛孔粗大等）\n"
                        "3. 护肤建议（包括成分推荐和注意事项）\n"
                        "请用中文回答，专业但易懂。"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64_img}"},
                },
            ])
            response = vision_llm.invoke([msg])
            return response.content or "分析失败"
        except Exception as e:
            return f"皮肤分析异常: {e}"

    return [
        search_knowledge_base,
        search_images,
        get_knowledge_base_stats,
        analyze_skin,
    ]
