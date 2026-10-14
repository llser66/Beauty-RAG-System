"""
美妆护肤企业级 RAG 引擎 — Milvus 高能版（生产优化）
多文档 · 多模态 · BGE 重排 · Token 鉴权 · 连接池 · 健康检查
"""

import asyncio
import hashlib
import json
import logging
import os
import mimetypes
import time
import traceback
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncGenerator, Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pymilvus import MilvusClient, DataType

from utils.document_loader import scan_knowledge_base, load_single_file
from utils.multimodal import embed_image, embed_text_for_image_search, CLIP_DIM
from utils.reranker import rerank_documents
from utils.auth import verify_token, get_api_token
from utils.validation import (
    UploadValidationError,
    clean_question,
    ensure_within_directory,
    sanitize_filename,
)
from utils.privacy import cleanup_expired_files, normalize_image_bytes
from utils.safety import assess_question
from agent.orchestrator import run_agent, run_agent_stream

# ── 日志初始化 ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("beauty_rag")

load_dotenv()

# ── 配置模型 ────────────────────────────────────
class Settings:
    """集中管理所有配置，支持环境变量覆盖"""
    SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_BASE_URL: str = os.getenv("SILICONFLOW_BASE_URL", "")
    MILVUS_URI: str = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
    TEXT_COLLECTION: str = "text_chunks"
    IMAGE_COLLECTION: str = "image_embeddings"
    KNOWLEDGE_DIR: str = os.getenv("KNOWLEDGE_DIR", os.path.join(os.path.dirname(__file__), "knowledge_base"))
    IMAGE_DIR: str = os.path.join(os.path.dirname(__file__), "images")
    CHAT_UPLOAD_DIR: str = os.path.join(os.path.dirname(__file__), "chat_uploads")
    LLM_MODEL: str = "deepseek-ai/DeepSeek-V3"
    VISION_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct"
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-8B"
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "3"))
    IMAGE_SEARCH_LIMIT: int = int(os.getenv("IMAGE_SEARCH_LIMIT", "5"))
    IMAGE_MIN_SCORE: float = float(os.getenv("IMAGE_MIN_SCORE", "0.18"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    CHAT_UPLOAD_RETENTION_HOURS: int = max(
        1, int(os.getenv("CHAT_UPLOAD_RETENTION_HOURS", "24"))
    )
    ENV: str = os.getenv("ENV", "production")
    API_PORT: int = int(os.getenv("API_PORT", "8877"))


settings = Settings()

if not settings.SILICONFLOW_API_KEY or not settings.SILICONFLOW_BASE_URL:
    raise RuntimeError("请在 .env 中配置 SILICONFLOW_API_KEY 和 SILICONFLOW_BASE_URL")

# 确保目录存在
for path in [settings.KNOWLEDGE_DIR, settings.IMAGE_DIR, settings.CHAT_UPLOAD_DIR]:
    os.makedirs(path, exist_ok=True)

# ── 全局状态与锁 ─────────────────────────────────
text_store: Optional[Milvus] = None
milvus_client: Optional[MilvusClient] = None
rebuild_lock = asyncio.Lock()
ready = False  # 服务就绪标志

# ── LLM & Embeddings ──────────────────────────────
llm = ChatOpenAI(
    api_key=settings.SILICONFLOW_API_KEY,
    base_url=settings.SILICONFLOW_BASE_URL,
    model=settings.LLM_MODEL,
    temperature=0.1,
)

vision_llm = ChatOpenAI(
    api_key=settings.SILICONFLOW_API_KEY,
    base_url=settings.SILICONFLOW_BASE_URL,
    model=settings.VISION_MODEL,
    temperature=0.1,
    max_tokens=1024,
)

embeddings = OpenAIEmbeddings(
    api_key=settings.SILICONFLOW_API_KEY,
    openai_api_base=settings.SILICONFLOW_BASE_URL,
    model=settings.EMBEDDING_MODEL,
)

# ── RAG 提示模板 ─────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""
你是一位专业、克制、可信的美妆护肤顾问。请严格根据下方提供的上下文回答用户问题。
如果上下文没有足够依据，请明确说明，不要编造品牌、价格或功效；涉及皮肤疾病、孕期或严重过敏时，提醒用户咨询专业医生。
安全边界：{safety_instruction}

上下文：
{context}

用户问题：
{question}
""")


# ── 工具函数 ─────────────────────────────────────
def compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


async def _retrieve_rerank_context(question: str) -> tuple[List[Document], str]:
    """
    文本检索 + BGE 重排，返回 (top_docs, context_str)
    """
    global text_store
    if text_store is None:
        raise HTTPException(status_code=503, detail="文本检索引擎未就绪")

    # 粗召回
    raw_docs = await asyncio.to_thread(
        text_store.similarity_search, question, k=settings.RERANK_TOP_K * 3 + 2
    )

    # 精排（异步化，避免阻塞）
    try:
        loop = asyncio.get_running_loop()
        reranked = await loop.run_in_executor(
            None, rerank_documents, question, raw_docs, settings.RERANK_TOP_K
        )
        top_docs = [item[0] for item in reranked]
    except Exception as e:
        logger.warning(f"重排异常，回退原始排序: {e}")
        top_docs = raw_docs[: settings.RERANK_TOP_K]

    context = "\n\n".join(doc.page_content for doc in top_docs)
    return top_docs, context


async def _search_similar_images(question: str) -> List[Dict[str, Any]]:
    """
    基于文本搜索相关图片，返回图片信息列表
    """
    global milvus_client
    if milvus_client is None:
        logger.error("图片检索客户端未初始化")
        return []

    matched_images = []
    try:
        query_vec = await asyncio.to_thread(embed_text_for_image_search, question)
        img_results = await asyncio.to_thread(
            milvus_client.search,
            collection_name=settings.IMAGE_COLLECTION,
            data=[query_vec.tolist()],
            limit=settings.IMAGE_SEARCH_LIMIT,
            output_fields=["filepath", "filename", "tags"],
        )
        if img_results and img_results[0]:
            for hit in img_results[0]:
                score = round(hit["distance"], 4)
                if score < settings.IMAGE_MIN_SCORE:
                    continue
                matched_images.append({
                    "path": f"/static/images/{hit['entity']['filename']}",
                    "filename": hit['entity']['filename'],
                    "score": score,
                    "tags": hit['entity'].get("tags", ""),
                })
    except Exception as e:
        logger.error(f"图片检索异常: {e}\n{traceback.format_exc()}")

    return matched_images


# ── 核心 RAG 流水线（同步 + 流式）──────────────
async def chat_with_rerank(question: str) -> dict:
    """
    完整 RAG：检索 → 重排 → LLM 生成 + 图片检索
    """
    top_docs, context = await _retrieve_rerank_context(question)

    # LLM 生成与图片搜索并行
    assessment = assess_question(question)
    messages = RAG_PROMPT.format_messages(
        context=context,
        question=question,
        safety_instruction=assessment.instruction,
    )
    llm_task = asyncio.create_task(llm.ainvoke(messages))
    img_task = asyncio.create_task(_search_similar_images(question))

    result = await llm_task
    matched_images = await img_task

    sources = list(set(d.metadata.get("source", "") for d in top_docs))
    return {
        "answer": result.content,
        "images": matched_images,
        "context_sources": sources,
        "safety_notice": assessment.notice or None,
    }


async def chat_with_rerank_stream(question: str) -> AsyncGenerator[dict, None]:
    """
    流式 RAG：检索 → 重排 → 流式 LLM + 并行图片检索
    """
    top_docs, context = await _retrieve_rerank_context(question)
    assessment = assess_question(question)
    if assessment.notice:
        yield {"type": "safety", "content": assessment.notice}
    messages = RAG_PROMPT.format_messages(
        context=context,
        question=question,
        safety_instruction=assessment.instruction,
    )

    img_task = asyncio.create_task(_search_similar_images(question))

    # 直接转发模型流，避免先把所有 token 收集完才返回。
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield {"type": "text", "content": chunk.content}

    # 再推送图片和元数据
    matched_images = await img_task
    sources = list(set(d.metadata.get("source", "") for d in top_docs))
    yield {"type": "metadata", "images": matched_images, "sources": sources}


# ── 索引重建（带锁保护）────────────────────────
async def rebuild_text_index():
    global text_store, milvus_client
    async with rebuild_lock:
        logger.info("🔄 重建文本索引...")
        all_docs = scan_knowledge_base(settings.KNOWLEDGE_DIR)
        if not all_docs:
            raise RuntimeError("未扫描到任何文档！")
        chunked = chunk_documents(all_docs)
        logger.info(f"切片后共 {len(chunked)} 个文本块")

        # 删除旧集合
        if milvus_client and milvus_client.has_collection(settings.TEXT_COLLECTION):
            milvus_client.drop_collection(settings.TEXT_COLLECTION)

        # 重建
        text_store = Milvus.from_documents(
            documents=chunked,
            embedding=embeddings,
            connection_args={"uri": settings.MILVUS_URI},
            collection_name=settings.TEXT_COLLECTION,
            drop_old=True,
        )
        # 确保落盘
        try:
            milvus_client.flush(collection_name=settings.TEXT_COLLECTION)
        except Exception as e:
            logger.warning(f"flush 文本集合失败: {e}")
        logger.info(f"✅ 文本索引重建完成，共 {len(chunked)} 个向量")


async def rebuild_image_index():
    global milvus_client
    async with rebuild_lock:
        logger.info("🔄 重建图片索引...")
        if milvus_client is None:
            raise RuntimeError("Milvus 客户端未初始化")

        # 删除旧集合
        if milvus_client.has_collection(settings.IMAGE_COLLECTION):
            milvus_client.drop_collection(settings.IMAGE_COLLECTION)

        # 创建集合
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=CLIP_DIM)
        schema.add_field("filepath", DataType.VARCHAR, max_length=500)
        schema.add_field("filename", DataType.VARCHAR, max_length=255)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index("vector", metric_type="IP", index_type="IVF_FLAT", params={"nlist": 128})

        milvus_client.create_collection(
            collection_name=settings.IMAGE_COLLECTION,
            schema=schema,
            index_params=index_params,
        )

        # 扫描图片
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_files = []
        for ext in image_extensions:
            for p in Path(settings.IMAGE_DIR).glob(f"*{ext}"):
                image_files.append(p)
            for p in Path(settings.IMAGE_DIR).glob(f"*{ext.upper()}"):
                image_files.append(p)

        if not image_files:
            logger.warning("⚠️ images/ 目录没有图片")
            milvus_client.load_collection(settings.IMAGE_COLLECTION)
            return

        inserted = 0
        for img_path in image_files:
            try:
                vec = embed_image(str(img_path))
                tags = "美妆产品,护肤品,美容"
                milvus_client.insert(
                    collection_name=settings.IMAGE_COLLECTION,
                    data=[{
                        "vector": vec.tolist(),
                        "filepath": str(img_path),
                        "filename": img_path.name,
                        "tags": tags,
                    }]
                )
                inserted += 1
                logger.info(f"🖼️ {img_path.name} → 已入库")
            except Exception as e:
                logger.error(f"❌ {img_path.name} 处理失败: {e}")

        milvus_client.flush(collection_name=settings.IMAGE_COLLECTION)
        milvus_client.load_collection(settings.IMAGE_COLLECTION)
        logger.info(f"✅ 图片索引重建完成，共 {inserted} 张")


def get_kb_stats() -> dict:
    """获取知识库统计（复用全局客户端）"""
    stats = {"text_chunks": 0, "images": 0, "documents": 0}
    if milvus_client is None:
        return stats
    try:
        if milvus_client.has_collection(settings.TEXT_COLLECTION):
            milvus_client.flush(collection_name=settings.TEXT_COLLECTION)
            s = milvus_client.get_collection_stats(settings.TEXT_COLLECTION)
            stats["text_chunks"] = s.get("row_count", 0)
        if milvus_client.has_collection(settings.IMAGE_COLLECTION):
            milvus_client.flush(collection_name=settings.IMAGE_COLLECTION)
            s = milvus_client.get_collection_stats(settings.IMAGE_COLLECTION)
            stats["images"] = s.get("row_count", 0)
    except Exception as e:
        logger.error(f"统计获取异常: {e}")

    if os.path.isdir(settings.KNOWLEDGE_DIR):
        stats["documents"] = len([
            f for f in os.listdir(settings.KNOWLEDGE_DIR)
            if os.path.isfile(os.path.join(settings.KNOWLEDGE_DIR, f))
        ])
    return stats


# ── Lifespan：启动/关闭 ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_store, milvus_client, ready
    logger.info("🚀 美妆 RAG 引擎启动中...")

    try:
        # 启动时初始化 Token，前端可从同一份 .env 读取，避免公开返回密钥。
        get_api_token()
        deleted = await asyncio.to_thread(
            cleanup_expired_files,
            settings.CHAT_UPLOAD_DIR,
            settings.CHAT_UPLOAD_RETENTION_HOURS,
        )
        if deleted:
            logger.info(f"🧹 已清理 {deleted} 个过期聊天图片")
        # 初始化全局 Milvus 客户端
        milvus_client = MilvusClient(uri=settings.MILVUS_URI)
        # 测试连接
        milvus_client.list_collections()
        logger.info("✅ Milvus 连接成功")

        # 文本索引处理
        if milvus_client.has_collection(settings.TEXT_COLLECTION):
            stats = milvus_client.get_collection_stats(settings.TEXT_COLLECTION)
            count = stats.get("row_count", 0)
            if count == 0:
                logger.info("文本集合为空，触发重建...")
                await rebuild_text_index()
            else:
                text_store = Milvus(
                    embedding_function=embeddings,
                    collection_name=settings.TEXT_COLLECTION,
                    connection_args={"uri": settings.MILVUS_URI},
                )
                logger.info(f"✅ 加载已有文本索引 ({count} 条)")
        else:
            await rebuild_text_index()

        # 图片索引处理
        if not milvus_client.has_collection(settings.IMAGE_COLLECTION):
            await rebuild_image_index()
        else:
            milvus_client.load_collection(settings.IMAGE_COLLECTION)
            logger.info(f"✅ 加载已有图片索引: {settings.IMAGE_COLLECTION}")

        stats = get_kb_stats()
        logger.info(f"📊 知识库状态: {stats.get('text_chunks', 0)} 文本块, {stats.get('images', 0)} 张图片")
        ready = True

    except Exception as e:
        logger.error(f"⚠️ 初始化失败: {e}\n{traceback.format_exc()}")
        ready = False

    yield

    logger.info("🛑 服务关闭，释放资源...")
    if milvus_client:
        try:
            milvus_client.close()
        except Exception:
            pass
    ready = False


# ── FastAPI App ──────────────────────────────────
app = FastAPI(
    title="Beauty Agent — 美妆护肤 AI 助手 (企业版)",
    description="RAG 知识库 + Agent 自主推理 + 多模态检索 + Token 鉴权 + 生产级优化",
    version="3.1.0",
    lifespan=lifespan,
)

# 静态文件
app.mount("/static/images", StaticFiles(directory=settings.IMAGE_DIR), name="images")

# ── 请求/响应模型 ──────────────────────────────
class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    image_filename: Optional[str] = Field(default=None, max_length=255)
    history: List[ChatHistoryMessage] = Field(default_factory=list, max_length=8)


class ChatResponse(BaseModel):
    answer: str
    images: Optional[List[Dict[str, Any]]] = None
    context_sources: Optional[List[str]] = None
    safety_notice: Optional[str] = None

class AgentStep(BaseModel):
    tool: str = ""
    args: Dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""

class AgentResponse(BaseModel):
    answer: str = ""
    steps: List[AgentStep] = Field(default_factory=list)
    images: List[dict] = Field(default_factory=list)
    safety_notice: Optional[str] = None


def _require_ready() -> None:
    if not ready or text_store is None:
        raise HTTPException(status_code=503, detail="AI 服务正在启动，请稍后重试")


def _validated_question(req: ChatRequest) -> str:
    try:
        return clean_question(req.question)
    except UploadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _resolve_chat_image(image_filename: Optional[str]) -> Optional[str]:
    if not image_filename:
        return None
    try:
        safe_name = sanitize_filename(
            image_filename,
            {".jpg", ".jpeg", ".png", ".webp", ".bmp"},
        )
        if safe_name != image_filename or not safe_name.startswith("chat_"):
            raise UploadValidationError("图片引用无效")
        path = ensure_within_directory(settings.CHAT_UPLOAD_DIR, safe_name)
    except UploadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="上传的图片不存在或已过期")
    return str(path)


def _history_for_agent(req: ChatRequest) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for item in req.history[-8:]:
        role = item.role
        content = item.content
        if content.strip():
            history.append({"role": role, "content": content.strip()[:4000]})
    return history


def _normalize_uploaded_image(file_bytes: bytes, extension: str) -> bytes:
    """验证图片并移除 EXIF/GPS 等可能泄露隐私的元数据。"""
    try:
        normalized = normalize_image_bytes(file_bytes, extension)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="上传内容不是有效图片") from exc
    if not normalized:
        raise HTTPException(status_code=400, detail="上传内容不是有效图片")
    return normalized


# ── 健康检查 ────────────────────────────────────
@app.get("/health")
async def health():
    status = {
        "status": "ok" if ready else "not_ready",
        "milvus": "connected" if milvus_client else "disconnected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return JSONResponse(content=status, status_code=200 if ready else 503)


@app.get("/api/v1/privacy")
async def privacy_policy():
    """公开展示数据用途和安全边界，供用户在上传前查看。"""
    return {
        "image_retention_hours": settings.CHAT_UPLOAD_RETENTION_HOURS,
        "image_access": "仅允许携带 Bearer Token 的 API 请求读取",
        "image_metadata": "上传图片会重新编码并移除 EXIF/GPS 元数据",
        "external_processing": "聊天文本和皮肤图片可能发送到配置的外部模型服务",
        "medical_scope": "仅提供一般护肤信息，不进行疾病诊断，不替代医生",
        "user_control": "用户可以删除当前聊天图片，系统也会自动清理过期图片",
    }


# ── RAG 端点（鉴权）──────────────────────────────
@app.post("/api/v1/rag/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, _=Depends(verify_token)):
    _require_ready()
    try:
        result = await chat_with_rerank(_validated_question(req))
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("RAG 对话异常")
        raise HTTPException(status_code=500, detail="RAG 服务暂时不可用，请稍后重试")


@app.post("/api/v1/rag/chat/stream")
async def chat_stream_endpoint(req: ChatRequest, _=Depends(verify_token)):
    _require_ready()
    question = _validated_question(req)

    async def event_generator():
        try:
            async for chunk in chat_with_rerank_stream(question):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("流式对话异常")
            yield f"data: {json.dumps({'type': 'error', 'content': '暂时无法完成回答，请稍后重试'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ── Agent 端点 ─────────────────────────────────
@app.post("/api/v1/agent/chat", response_model=AgentResponse)
async def agent_chat_endpoint(req: ChatRequest, _=Depends(verify_token)):
    _require_ready()
    try:
        question = _validated_question(req)
        result = await run_agent(
            user_query=question,
            llm=llm,
            text_store=text_store,
            rerank_func=rerank_documents,
            milvus_uri=settings.MILVUS_URI,
            image_collection=settings.IMAGE_COLLECTION,
            stats_func=get_kb_stats,
            vision_llm=vision_llm,
            chat_history=_history_for_agent(req),
            image_path=_resolve_chat_image(req.image_filename),
        )
        return AgentResponse(
            answer=result["answer"],
            steps=[AgentStep(**s) for s in result["steps"]],
            images=result["images"],
            safety_notice=result.get("safety_notice"),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Agent 对话异常")
        raise HTTPException(status_code=500, detail="Agent 服务暂时不可用，请稍后重试")


@app.post("/api/v1/agent/chat/stream")
async def agent_chat_stream_endpoint(req: ChatRequest, _=Depends(verify_token)):
    _require_ready()
    question = _validated_question(req)
    image_path = _resolve_chat_image(req.image_filename)
    history = _history_for_agent(req)

    async def event_generator():
        try:
            async for event in run_agent_stream(
                user_query=question,
                llm=llm,
                text_store=text_store,
                rerank_func=rerank_documents,
                milvus_uri=settings.MILVUS_URI,
                image_collection=settings.IMAGE_COLLECTION,
                stats_func=get_kb_stats,
                vision_llm=vision_llm,
                chat_history=history,
                image_path=image_path,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") == "done":
                    break
            yield "data: [DONE]\n\n"
        except Exception:
            logger.exception("Agent 流式异常")
            yield f"data: {json.dumps({'type': 'error', 'content': 'Agent 暂时无法完成回答，请稍后重试'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ── 知识库管理（全部加鉴权）────────────────────
@app.post("/api/v1/knowledge/upload")
async def upload_knowledge(
    file: UploadFile = File(...),
    _=Depends(verify_token),
):
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    doc_exts = {".txt", ".pdf", ".docx", ".xlsx"}
    try:
        filename = sanitize_filename(file.filename or "", image_exts | doc_exts)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    file_bytes = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if not file_bytes:
        raise HTTPException(status_code=400, detail="上传文件不能为空")
    if len(file_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"文件大小不能超过 {settings.MAX_UPLOAD_SIZE_MB}MB")

    ext = Path(filename).suffix.lower()
    file_md5 = compute_md5(file_bytes)

    if ext in image_exts:
        file_bytes = _normalize_uploaded_image(file_bytes, ext)
        file_md5 = compute_md5(file_bytes)
        stored_name = f"{Path(filename).stem}_{file_md5[:10]}{ext}"
        save_path = ensure_within_directory(settings.IMAGE_DIR, stored_name)
        if save_path.exists():
            logger.info(f"⏭️ 图片已存在，跳过: {stored_name}")
            return {"status": "ok", "type": "image", "filename": stored_name, "dedup": True}

        await asyncio.to_thread(save_path.write_bytes, file_bytes)

        try:
            vec = await asyncio.to_thread(embed_image, str(save_path))
            if milvus_client is None:
                raise RuntimeError("Milvus 客户端未初始化")
            await asyncio.to_thread(
                milvus_client.insert,
                collection_name=settings.IMAGE_COLLECTION,
                data=[{
                    "vector": vec.tolist(),
                    "filepath": str(save_path),
                    "filename": stored_name,
                }],
            )
            await asyncio.to_thread(milvus_client.flush, collection_name=settings.IMAGE_COLLECTION)
            logger.info(f"✅ 图片已入库: {stored_name}")
            return {"status": "ok", "type": "image", "filename": stored_name}
        except Exception:
            logger.exception("图片处理失败")
            raise HTTPException(status_code=500, detail="图片处理失败，请稍后重试")

    elif ext in doc_exts:
        save_path = ensure_within_directory(settings.KNOWLEDGE_DIR, filename)
        was_existing = save_path.exists()
        if os.path.exists(save_path):
            existing_md5 = compute_md5(await asyncio.to_thread(save_path.read_bytes))
            if existing_md5 == file_md5:
                logger.info(f"⏭️ 文档已存在，跳过: {filename}")
                return {"status": "ok", "type": "document", "filename": filename, "dedup": True}

        await asyncio.to_thread(save_path.write_bytes, file_bytes)

        try:
            new_docs = await asyncio.to_thread(load_single_file, str(save_path))
            if not new_docs:
                raise RuntimeError(f"文件 {filename} 解析后内容为空")
            chunked = chunk_documents(new_docs)
            if was_existing:
                # 替换同名文件时全量重建，避免旧内容残留在向量库中。
                await rebuild_text_index()
            elif text_store is not None:
                async with rebuild_lock:  # 添加文档时不希望并发重建
                    await asyncio.to_thread(text_store.add_documents, chunked)
                # 强制 flush 确保实时可见
                if milvus_client:
                    await asyncio.to_thread(milvus_client.flush, collection_name=settings.TEXT_COLLECTION)
                logger.info(f"✅ 文档增量入库: {filename} → {len(chunked)} 个切片")
            else:
                await rebuild_text_index()
            return {"status": "ok", "type": "document", "filename": filename}
        except Exception:
            logger.exception("文档处理失败")
            raise HTTPException(status_code=500, detail="文档处理失败，请检查文件内容后重试")
    else:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")


@app.post("/api/v1/knowledge/rebuild")
async def rebuild_knowledge(_=Depends(verify_token)):
    try:
        await rebuild_text_index()
        await rebuild_image_index()
        return {"status": "ok", "message": "全量索引重建完成"}
    except Exception:
        logger.exception("重建索引失败")
        raise HTTPException(status_code=500, detail="重建索引失败，请检查 Milvus 和知识库文件")


@app.get("/api/v1/knowledge/stats")
async def knowledge_stats(_=Depends(verify_token)):
    return get_kb_stats()


@app.get("/api/v1/auth/token")
async def show_token(_=Depends(verify_token)):
    """兼容旧客户端，但不再通过公开接口返回 Token。"""
    return {"configured": bool(get_api_token())}


# ── 聊天图片上传 ─────────────────────────────────
@app.post("/api/v1/chat/upload_image")
async def chat_upload_image(
    file: UploadFile = File(...),
    _=Depends(verify_token),
):
    await asyncio.to_thread(
        cleanup_expired_files,
        settings.CHAT_UPLOAD_DIR,
        settings.CHAT_UPLOAD_RETENTION_HOURS,
    )
    try:
        filename = sanitize_filename(
            file.filename or f"chat_{int(time.time())}.jpg",
            {".jpg", ".jpeg", ".png", ".webp", ".bmp"},
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ext = Path(filename).suffix.lower()

    file_bytes = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if not file_bytes:
        raise HTTPException(status_code=400, detail="上传图片不能为空")
    if len(file_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"文件大小不能超过 {settings.MAX_UPLOAD_SIZE_MB}MB")

    file_bytes = _normalize_uploaded_image(file_bytes, ext)
    if len(file_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"文件大小不能超过 {settings.MAX_UPLOAD_SIZE_MB}MB")
    safe_name = f"chat_{uuid4().hex}{ext}"
    save_path = ensure_within_directory(settings.CHAT_UPLOAD_DIR, safe_name)
    await asyncio.to_thread(save_path.write_bytes, file_bytes)

    return {
        "status": "ok",
        "file_id": safe_name,
        "url": f"/api/v1/chat/uploads/{safe_name}",
        "filename": safe_name,
    }


@app.get("/api/v1/chat/uploads/{file_id}")
async def get_chat_upload(file_id: str, _=Depends(verify_token)):
    """受鉴权保护的聊天图片读取接口，不再通过公开静态目录暴露。"""
    path = _resolve_chat_image(file_id)
    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@app.delete("/api/v1/chat/uploads/{file_id}")
async def delete_chat_upload(file_id: str, _=Depends(verify_token)):
    """允许用户主动删除本次聊天上传的图片。"""
    path = _resolve_chat_image(file_id)
    await asyncio.to_thread(path.unlink)
    return {"status": "deleted", "file_id": file_id}


# ── 兼容旧版 ─────────────────────────────────────
@app.post("/chat")
async def legacy_chat(req: ChatRequest, _=Depends(verify_token)):
    """兼容旧版 /chat 端点，但沿用鉴权和输入校验。"""
    _require_ready()
    try:
        result = await chat_with_rerank(_validated_question(req))
        return {"answer": result["answer"]}
    except Exception:
        logger.exception("旧版对话异常")
        raise HTTPException(status_code=500, detail="服务暂时不可用，请稍后重试")


# ── 启动入口 ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.API_PORT, reload=False)
