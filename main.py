"""
美妆护肤企业级 RAG 引擎 — Milvus 高能版
多文档 · 多模态 · BGE 重排 · Token 鉴权
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pymilvus import MilvusClient, DataType, connections

from utils.document_loader import scan_knowledge_base
from utils.multimodal import embed_image, embed_text_for_image_search, CLIP_DIM
from utils.reranker import rerank_documents
from utils.auth import verify_token, get_api_token
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import hashlib, json, asyncio, time

# ── Agent ──────────────────────────────────────────────
from agent.orchestrator import run_agent, run_agent_stream

load_dotenv()

# ── 配置 ──────────────────────────────────────────
sf_key = os.getenv("SILICONFLOW_API_KEY")
sf_url = os.getenv("SILICONFLOW_BASE_URL")
if not sf_key or not sf_url:
    raise RuntimeError("请在 .env 中配置 SILICONFLOW_API_KEY 和 SILICONFLOW_BASE_URL")

MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "image_embeddings"
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", os.path.join(os.path.dirname(__file__), "knowledge_base"))
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "images")

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
CHAT_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "chat_uploads")
os.makedirs(CHAT_UPLOAD_DIR, exist_ok=True)

# ── LLM & Embeddings ──────────────────────────────
llm = ChatOpenAI(
    api_key=sf_key, base_url=sf_url,
    model="deepseek-ai/DeepSeek-V3", temperature=0.1
)
# 视觉模型（用于分析皮肤照片）
vision_llm = ChatOpenAI(
    api_key=sf_key, base_url=sf_url,
    model="Qwen/Qwen3-VL-8B-Instruct", temperature=0.1,
    max_tokens=1024,
)
embeddings = OpenAIEmbeddings(
    api_key=sf_key, openai_api_base=sf_url,
    model="Qwen/Qwen3-Embedding-8B"
)

# ── 全局状态 ───────────────────────────────────────
text_store: Optional[Milvus] = None


# ── MD5 文件指纹（内容去重）───────────────────────
def compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ── 文本切片 ───────────────────────────────────────
def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


# ── 重排 RAG 对话 ──────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""
你是一位精通小红书美妆爆文逻辑的文案专家。请严格根据下方提供的上下文回答用户问题。

上下文：
{context}

用户问题：
{question}
""")


async def chat_with_rerank(question: str) -> dict:
    """
    完整 RAG 流水线：粗搜 → 重排 → LLM 生成 + 图片检索
    """
    global text_store

    if text_store is None:
        raise HTTPException(status_code=500, detail="文本引擎未初始化")

    # 1. 文本粗召回 top 10
    raw_docs = text_store.similarity_search(question, k=10)

    # 2. BGE-Reranker 精排 → top 3（带容错）
    try:
        reranked = rerank_documents(question, raw_docs, top_k=3)
        top_docs = [item[0] for item in reranked]
        top_scores = [round(item[1], 4) for item in reranked]
    except Exception as e:
        print(f"⚠️ 重排异常，使用原始排序: {e}")
        top_docs = raw_docs[:3]
        top_scores = []

    # 3. 构建上下文
    context = "\n\n".join(doc.page_content for doc in top_docs)

    # 4. LLM 生成
    messages = RAG_PROMPT.format_messages(context=context, question=question)
    result = await llm.ainvoke(messages)

    # 5. 图片检索（独立连接，避免冲突）
    matched_images = []
    try:
        connections.connect(alias="image_conn", uri=MILVUS_URI)
        ic = MilvusClient(alias="image_conn")
        query_vec = embed_text_for_image_search(question)
        img_results = ic.search(
            collection_name=IMAGE_COLLECTION,
            data=[query_vec.tolist()],
            limit=5,
            output_fields=["filepath", "filename", "tags"],
        )
        if img_results and img_results[0]:
            MIN_SCORE = 0.18
            for hit in img_results[0]:
                score = round(hit["distance"], 4)
                if score < MIN_SCORE:
                    continue
                matched_images.append({
                    "path": f"/static/images/{hit['entity']['filename']}",
                    "filename": hit["entity"]["filename"],
                    "score": score,
                    "tags": hit["entity"].get("tags", ""),
                })
        ic.close()
        connections.disconnect("image_conn")
    except Exception as e:
        print(f"⚠️ 图片检索异常: {e}")

    return {
        "answer": result.content,
        "rerank_scores": [{"index": i, "score": s} for i, s in enumerate(top_scores)],
        "images": matched_images,
        "context_sources": list(set(d.metadata.get("source", "") for d in top_docs)),
    }


# ── 流式 RAG 流水线（SSE 打字机）────────────────
async def chat_with_rerank_stream(question: str):
    """
    流式 RAG：检索 → 重排 → 流式 LLM 生成 → 图片检索
    用于 SSE /api/v1/rag/chat/stream 端点
    """
    global text_store
    if text_store is None:
        yield {"type": "error", "content": "文本引擎未初始化"}
        return

    # 1. 文本粗召回 top 10
    raw_docs = text_store.similarity_search(question, k=10)

    # 2. BGE-Reranker 精排 → top 3
    try:
        reranked = rerank_documents(question, raw_docs, top_k=3)
        top_docs = [item[0] for item in reranked]
    except Exception as e:
        print(f"⚠️ 重排异常，使用原始排序: {e}")
        top_docs = raw_docs[:3]

    # 3. 构建上下文
    context = "\n\n".join(doc.page_content for doc in top_docs)
    messages = RAG_PROMPT.format_messages(context=context, question=question)

    # 4. 流式 LLM 生成（逐 chunk 推送）
    async for chunk in llm.astream(messages):
        yield {"type": "text", "content": chunk.content}

    # 5. 图片检索
    matched_images = []
    try:
        connections.connect(alias="image_conn", uri=MILVUS_URI)
        ic = MilvusClient(alias="image_conn")
        query_vec = embed_text_for_image_search(question)
        img_results = ic.search(
            collection_name=IMAGE_COLLECTION,
            data=[query_vec.tolist()],
            limit=5,
            output_fields=["filepath", "filename", "tags"],
        )
        if img_results and img_results[0]:
            MIN_SCORE = 0.18
            for hit in img_results[0]:
                score = round(hit["distance"], 4)
                if score < MIN_SCORE:
                    continue
                matched_images.append({
                    "path": f"/static/images/{hit['entity']['filename']}",
                    "filename": hit["entity"]["filename"],
                    "score": score,
                    "tags": hit["entity"].get("tags", ""),
                })
        ic.close()
        connections.disconnect("image_conn")
    except Exception as e:
        print(f"⚠️ 图片检索异常: {e}")

    sources = list(set(d.metadata.get("source", "") for d in top_docs))
    yield {"type": "metadata", "images": matched_images, "sources": sources}


# ── 重建 Milvus 索引 ──────────────────────────────
def rebuild_text_index():
    """全量重建文本向量索引"""
    global text_store
    print("🔄 重建文本索引...")

    # 扫描文档
    all_docs = scan_knowledge_base(KNOWLEDGE_DIR)
    if not all_docs:
        raise RuntimeError("未扫描到任何文档！")

    # 切片
    chunked = chunk_documents(all_docs)
    print(f"  切片后共 {len(chunked)} 个文本块")

    # 删除旧集合并重建
    connections.connect(alias="rebuild_conn", uri=MILVUS_URI)
    rc = MilvusClient(alias="rebuild_conn")
    if rc.has_collection(TEXT_COLLECTION):
        rc.drop_collection(TEXT_COLLECTION)
    rc.close()
    connections.disconnect("rebuild_conn")

    text_store = Milvus.from_documents(
        documents=chunked,
        embedding=embeddings,
        connection_args={"uri": MILVUS_URI},
        collection_name=TEXT_COLLECTION,
        drop_old=True,
    )
    # 确保数据落盘
    try:
        connections.connect(alias="flush_conn", uri=MILVUS_URI)
        fc = MilvusClient(alias="flush_conn")
        fc.flush(collection_name=TEXT_COLLECTION)
        fc.close()
        connections.disconnect("flush_conn")
    except Exception:
        pass
    print(f"✅ 文本索引重建完成，共 {len(chunked)} 个向量")


def rebuild_image_index():
    """全量重建图片向量索引"""
    print("🔄 重建图片索引...")

    connections.connect(alias="image_conn", uri=MILVUS_URI)
    ic = MilvusClient(alias="image_conn")

    # 删除旧集合
    if ic.has_collection(IMAGE_COLLECTION):
        ic.drop_collection(IMAGE_COLLECTION)

    # 创建新集合：512维 CLIP 向量
    schema = ic.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=CLIP_DIM)
    schema.add_field("filepath", DataType.VARCHAR, max_length=500)
    schema.add_field("filename", DataType.VARCHAR, max_length=255)

    index_params = ic.prepare_index_params()
    index_params.add_index("vector", metric_type="IP", index_type="IVF_FLAT", params={"nlist": 128})

    ic.create_collection(
        collection_name=IMAGE_COLLECTION,
        schema=schema,
        index_params=index_params,
    )

    # 扫描 images/ 目录，逐张嵌入
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = []
    for ext in image_extensions:
        for p in Path(IMAGE_DIR).glob(f"*{ext}"):
            image_files.append(p)
        for p in Path(IMAGE_DIR).glob(f"*{ext.upper()}"):
            image_files.append(p)

    if not image_files:
        print("⚠️ images/ 目录没有图片")
        ic.close()
        connections.disconnect("image_conn")
        return

    inserted = 0
    for img_path in image_files:
        try:
            vec = embed_image(str(img_path))
            # 用文件名生成描述标签
            tags = f"美妆产品,护肤品,美容"
            ic.insert(
                collection_name=IMAGE_COLLECTION,
                data=[{
                    "vector": vec.tolist(),
                    "filepath": str(img_path),
                    "filename": img_path.name,
                    "tags": tags,
                }]
            )
            inserted += 1
            print(f"  🖼️  {img_path.name} → 已入库")
        except Exception as e:
            print(f"  ❌ {img_path.name} 处理失败: {e}")

    # 创建索引
    ic.flush(collection_name=IMAGE_COLLECTION)
    ic.load_collection(IMAGE_COLLECTION)
    ic.close()
    connections.disconnect("image_conn")
    print(f"✅ 图片索引重建完成，共 {inserted} 张")


# ── 统计 ──────────────────────────────────────────
def get_kb_stats() -> dict:
    """获取知识库统计"""
    stats = {
        "text_chunks": 0,
        "images": 0,
        "documents": 0,
    }
    try:
        connections.connect(alias="image_conn", uri=MILVUS_URI)
        client = MilvusClient(alias="image_conn")
        # Flush 确保持久化，保证 row_count 准确
        try:
            if client.has_collection(TEXT_COLLECTION):
                client.flush(collection_name=TEXT_COLLECTION)
            if client.has_collection(IMAGE_COLLECTION):
                client.flush(collection_name=IMAGE_COLLECTION)
        except Exception:
            pass
        if client.has_collection(TEXT_COLLECTION):
            s = client.get_collection_stats(TEXT_COLLECTION)
            stats["text_chunks"] = s.get("row_count", 0)
        if client.has_collection(IMAGE_COLLECTION):
            s = client.get_collection_stats(IMAGE_COLLECTION)
            stats["images"] = s.get("row_count", 0)
        client.close()
        connections.disconnect("image_conn")
    except Exception as e:
        print(f"统计读取异常: {e}")

    if os.path.isdir(KNOWLEDGE_DIR):
        stats["documents"] = len([f for f in os.listdir(KNOWLEDGE_DIR)
                                  if os.path.isfile(os.path.join(KNOWLEDGE_DIR, f))])
    return stats


# ── Lifespan ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_store
    print("🚀 美妆 RAG 引擎启动中...")

    try:
        # 为图片操作创建独立连接（避免与 text_store 的默认连接冲突）
        connections.connect(alias="image_conn", uri=MILVUS_URI)
        image_client = MilvusClient(alias="image_conn")

        # 检查 Milvus 是否已有数据
        has_text = image_client.has_collection(TEXT_COLLECTION)
        has_image = image_client.has_collection(IMAGE_COLLECTION)

        if has_text:
            # 检查是否为空（使用 get_collection_stats）
            try:
                stats = image_client.get_collection_stats(TEXT_COLLECTION)
                text_count = stats.get("row_count", 0)
            except Exception:
                text_count = 0
            if text_count == 0:
                print("  文本索引为空，触发重建...")
                image_client.close()
                connections.disconnect("image_conn")
                rebuild_text_index()
            else:
                text_store = Milvus(
                    embedding_function=embeddings,
                    collection_name=TEXT_COLLECTION,
                    connection_args={"uri": MILVUS_URI},
                )
                print(f"✅ 加载已有文本索引 ({text_count} 条)")
        else:
            # 首次启动，重建
            image_client.close()
            connections.disconnect("image_conn")
            rebuild_text_index()

        if not has_image:
            rebuild_image_index()
        else:
            # 重新连接并加载
            connections.connect(alias="image_conn", uri=MILVUS_URI)
            image_client = MilvusClient(alias="image_conn")
            image_client.load_collection(IMAGE_COLLECTION)
            print(f"✅ 加载已有图片索引: {IMAGE_COLLECTION}")

        stats = get_kb_stats()
        print(f"📊 知识库状态: {stats.get('text_chunks', 0)} 文本块, {stats.get('images', 0)} 张图片")

    except Exception as e:
        print(f"⚠️ 初始化异常: {e}")
        print("⚠️ 请确保 Milvus 服务已启动 (docker compose up -d)")

    yield

    print("🛑 服务关闭，释放资源...")
    try:
        connections.disconnect("image_conn")
    except Exception:
        pass


# ── FastAPI App ──────────────────────────────────
app = FastAPI(
    title="Beauty Agent — 美妆护肤 AI 助手",
    description="RAG 知识库 + Agent 自主推理 + 多模态检索 + Token 鉴权",
    version="3.0.0",
    lifespan=lifespan,
)


# ── 静态资源 ──────────────────────────────────────
# 将 images/ 目录暴露为 HTTP 可访问的静态资源，解决前端 file:// 跨域问题
app.mount("/static/images", StaticFiles(directory=IMAGE_DIR), name="images")
app.mount("/static/chat_uploads", StaticFiles(directory=CHAT_UPLOAD_DIR), name="chat_uploads")


# ── 请求模型 ──────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    rerank_scores: Optional[List[dict]] = None
    images: Optional[List[dict]] = None
    context_sources: Optional[List[str]] = None


# ── Agent 请求模型 ────────────────────────────────────
class AgentStep(BaseModel):
    tool: str = ""
    args: dict = {}
    result_summary: str = ""

class AgentResponse(BaseModel):
    answer: str = ""
    steps: List[AgentStep] = []
    images: List[dict] = []


@app.post("/api/v1/rag/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    _=Depends(verify_token),
):
    """RAG 对话（含 BGE 重排 + 多模态检索）"""
    try:
        return await chat_with_rerank(req.question)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 异常: {str(e)}")


@app.post("/api/v1/rag/chat/stream")
async def chat_stream_endpoint(req: ChatRequest, _=Depends(verify_token)):
    """流式 RAG 对话（SSE 打字机效果）"""
    async def event_generator():
        try:
            async for chunk in chat_with_rerank_stream(req.question):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ── Agent 端点 ─────────────────────────────────────
@app.post("/api/v1/agent/chat", response_model=AgentResponse)
async def agent_chat_endpoint(
    req: ChatRequest,
    _=Depends(verify_token),
):
    """Agent 对话（自主推理 + 工具调用）— 基于 LangChain create_agent()"""
    try:
        result = await run_agent(
            user_query=req.question,
            llm=llm,
            text_store=text_store,
            rerank_func=rerank_documents,
            milvus_uri=MILVUS_URI,
            image_collection=IMAGE_COLLECTION,
            stats_func=get_kb_stats,
            vision_llm=vision_llm,
        )
        return AgentResponse(
            answer=result["answer"],
            steps=[AgentStep(**s) for s in result["steps"]],
            images=result["images"],
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent 异常: {str(e)}")


@app.post("/api/v1/agent/chat/stream")
async def agent_chat_stream_endpoint(req: ChatRequest, _=Depends(verify_token)):
    """Agent 流式对话（SSE，逐步推送思考过程 + 答案）— 基于 LangChain create_agent()"""
    async def event_generator():
        try:
            async for event in run_agent_stream(
                user_query=req.question,
                llm=llm,
                text_store=text_store,
                rerank_func=rerank_documents,
                milvus_uri=MILVUS_URI,
                image_collection=IMAGE_COLLECTION,
                stats_func=get_kb_stats,
                vision_llm=vision_llm,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") == "done":
                    break
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/v1/knowledge/upload")
async def upload_knowledge(
    file: UploadFile = File(...),
):
    """上传文档或图片到知识库（MD5 去重 + 增量入库）"""
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    doc_exts = {".txt", ".pdf", ".docx", ".xlsx"}

    file_bytes = await file.read()
    file_md5 = compute_md5(file_bytes)

    if ext in image_exts:
        save_path = os.path.join(IMAGE_DIR, filename)

        # MD5 去重：内容相同跳过
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                if compute_md5(f.read()) == file_md5:
                    print(f"⏭️ 图片已存在，跳过: {filename}")
                    return {"status": "ok", "type": "image", "filename": filename, "dedup": True}

        with open(save_path, "wb") as f:
            f.write(file_bytes)

        try:
            vec = embed_image(save_path)
            connections.connect(alias="image_conn", uri=MILVUS_URI)
            ic = MilvusClient(alias="image_conn")
            ic.insert(
                collection_name=IMAGE_COLLECTION,
                data=[{
                    "vector": vec.tolist(),
                    "filepath": save_path,
                    "filename": filename,
                }]
            )
            ic.close()
            connections.disconnect("image_conn")
            print(f"✅ 图片已入库: {filename}")
            return {"status": "ok", "type": "image", "filename": filename}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"图片处理失败: {str(e)}")

    elif ext in doc_exts:
        save_path = os.path.join(KNOWLEDGE_DIR, filename)

        # MD5 去重：内容相同跳过
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                if compute_md5(f.read()) == file_md5:
                    print(f"⏭️ 文档已存在，跳过: {filename}")
                    return {"status": "ok", "type": "document", "filename": filename, "dedup": True}

        with open(save_path, "wb") as f:
            f.write(file_bytes)

        # 增量添加（无需全量重建）
        try:
            from utils.document_loader import load_single_file
            new_docs = load_single_file(save_path)
            if not new_docs:
                raise RuntimeError(f"文件 {filename} 解析后内容为空")
            chunked = chunk_documents(new_docs)
            if text_store is not None:
                text_store.add_documents(chunked)
                print(f"✅ 文档增量入库: {filename} → {len(chunked)} 个切片")
            else:
                rebuild_text_index()
            return {"status": "ok", "type": "document", "filename": filename}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")


@app.post("/api/v1/knowledge/rebuild")
async def rebuild_knowledge():
    """全量重建所有索引"""
    try:
        rebuild_text_index()
        rebuild_image_index()
        return {"status": "ok", "message": "全量索引重建完成"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建失败: {str(e)}")


@app.get("/api/v1/knowledge/stats")
async def knowledge_stats():
    """知识库统计（公开）"""
    return get_kb_stats()


@app.get("/api/v1/auth/token")
async def show_token():
    """查看当前 API Token（仅首次启动时查看用）"""
    return {"token": get_api_token()}


# ── 聊天图片上传 ─────────────────────────────────
@app.post("/api/v1/chat/upload_image")
async def chat_upload_image(file: UploadFile = File(...)):
    """上传聊天中的图片，返回可访问的路径"""
    filename = file.filename or f"chat_{int(time.time())}.jpg"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=400, detail=f"不支持的图片格式: {ext}")

    # 用时间戳防重名
    safe_name = f"chat_{int(time.time())}_{hashlib.md5(filename.encode()).hexdigest()[:8]}{ext}"
    save_path = os.path.join(CHAT_UPLOAD_DIR, safe_name)

    file_bytes = await file.read()
    with open(save_path, "wb") as f:
        f.write(file_bytes)

    return {
        "status": "ok",
        "path": save_path,
        "url": f"/static/chat_uploads/{safe_name}",
        "filename": safe_name,
    }


# ── 兼容旧版端点 ────────────────────────────────
class LegacyChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def legacy_chat(req: LegacyChatRequest):
    """兼容旧版 /chat 端点（无鉴权）"""
    try:
        result = await chat_with_rerank(req.question)
        return {"answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"集群或大模型异常: {str(e)}")


# ── 启动入口 ──────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)
