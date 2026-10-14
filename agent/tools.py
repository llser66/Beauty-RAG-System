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
import json
import logging
from typing import List, Optional

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


logger = logging.getLogger(__name__)


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
    image_path: Optional[str] = None,
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
        image_path:   当前用户上传的皮肤图片（由服务端校验后注入）
    """

    image_path_from_request = image_path

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

        query = str(query or "").strip()[:500]
        try:
            top_k = max(1, min(int(top_k), 10))
        except (TypeError, ValueError):
            top_k = 5

        # 粗召回 top 10
        try:
            raw_docs = text_store.similarity_search(query, k=10)
        except Exception:
            logger.exception("知识库检索失败")
            return "知识库检索暂时不可用，请稍后重试"

        # BGE 重排精排
        if rerank_func:
            try:
                reranked = rerank_func(query, raw_docs, top_k=min(top_k, 5))
                top_docs = [item[0] for item in reranked]
            except Exception:
                logger.exception("知识库重排失败")
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
        if os.getenv("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
            return json.dumps(
                {
                    "message": "公网演示模式未启用图片向量库，可继续体验知识库问答和 Agent 对话。",
                    "images": [],
                },
                ensure_ascii=False,
            )

        from pymilvus import MilvusClient
        from utils.multimodal import embed_text_for_image_search

        limit = max(1, min(int(limit), 10))
        ic = None
        try:
            ic = MilvusClient(uri=milvus_uri)

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
                            {
                                "filename": hit["entity"]["filename"],
                                "score": score,
                            }
                        )

            return json.dumps(
                {
                    "message": f"找到 {len(matched)} 张匹配图片" if matched else "未找到匹配的图片",
                    "images": matched,
                },
                ensure_ascii=False,
            )
        except Exception:
            logger.exception("图片搜索失败")
            return "图片搜索暂时不可用，请稍后重试"
        finally:
            if ic is not None:
                try:
                    ic.close()
                except Exception:
                    pass

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
        except Exception:
            logger.exception("知识库统计失败")
            return "知识库统计暂时不可用，请稍后重试"

    # ── 4. analyze_skin ───────────────────────────────────
    @tool
    def analyze_skin() -> str:
        """分析用户上传的皮肤照片，识别肤质类型、皮肤问题，并给出护肤建议。
        当用户上传了面部/皮肤照片并希望分析时调用此工具。
        """
        if vision_llm is None:
            return "错误：视觉分析模型未初始化"
        # 路径只允许使用 API 校验后注入的文件，不能让模型自行读取本地任意路径。
        selected_path = image_path_from_request
        if not selected_path or not os.path.exists(selected_path):
            return "错误：当前对话没有可用的皮肤照片"

        try:
            with open(selected_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(selected_path)[1].lower()
            mime_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp",
            }
            mime = mime_map.get(ext, "image/jpeg")

            msg = HumanMessage(content=[
                {
                    "type": "text",
                    "text": (
                        "你是一位谨慎的护肤信息助手。请仅描述照片中可观察到的外观特征，给出以下内容：\n"
                        "1. 可能的肤质表现（干燥、油光、混合等），避免把观察结果当成诊断\n"
                        "2. 可见的皮肤表现（如痘痘、黑头、泛红、暗沉、细纹、毛孔等）\n"
                        "3. 一般性的护肤建议（包括成分注意事项，不给处方或药物剂量）\n"
                        "不要根据照片推断身份、年龄、种族、健康状态或美丑，只讨论与护肤有关的可观察表现。\n"
                        "请用中文回答，专业但易懂。明确说明这不是医疗诊断；若有严重、持续或不确定的症状，请咨询医生。"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64_img}"},
                },
            ])
            response = vision_llm.invoke([msg])
            return response.content or "分析失败"
        except Exception:
            logger.exception("皮肤分析失败")
            return "皮肤分析暂时不可用，请稍后重试"

    return [
        search_knowledge_base,
        search_images,
        get_knowledge_base_stats,
        analyze_skin,
    ]
