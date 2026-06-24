"""
🎯 BGE-Reranker 精细化重排引擎
对 Milvus 粗召回结果进行深度相关性打分，剔除噪声

注意：首次使用需要下载模型（~500MB~1.2GB），HF 在中国可能被墙，
系统会自动尝试 hf-mirror.com 镜像。若网络不通，重排降级为原始排序，
不影响核心 RAG 流程。
"""

import os
from typing import List, Tuple
from langchain_core.documents import Document

# 优先使用 HuggingFace 镜像（中国大陆加速）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")

# 全局缓存重排模型
_reranker_pipeline = None

# ── 配置 ──────────────────────────────────────────
# BAAI/bge-reranker-v2-m3 质量最高（~1.2GB，已下载后加载极快）
# BAAI/bge-reranker-base   适中（~1.1GB）
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def get_reranker():
    """懒加载 BGE-Reranker 交叉编码模型

    优先读缓存；未缓存时尝试在线下载（镜像站加速）。
    所有网络异常均抛出并交由上游容错。
    """
    global _reranker_pipeline
    if _reranker_pipeline is not None:
        return _reranker_pipeline

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    # 1) 尝试本地缓存（无网络请求）
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL, local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_MODEL, local_files_only=True
        )
        print(f"✅ 重排模型 {RERANKER_MODEL} 加载完成（缓存命中）")
        model.eval()
        _reranker_pipeline = (model, tokenizer)
        return _reranker_pipeline
    except OSError:
        print(f"  本地未缓存 {RERANKER_MODEL}，尝试在线下载...")

    # 2) 在线下载（通过镜像站，带超时保护）
    import requests
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    test_url = f"{endpoint}/BAAI/bge-reranker-v2-m3/resolve/main/config.json"
    try:
        resp = requests.head(test_url, timeout=5)
        if resp.status_code != 200:
            raise ConnectionError(f"镜像 {endpoint} 不可达 ({resp.status_code})")
    except Exception as e:
        print(f"  ⚠️ HF 镜像不可达 ({type(e).__name__})，跳过重排模型加载")
        raise RuntimeError(f"无法连接 HF 镜像: {e}")

    # 镜像可达 → 下载
    try:
        print(f"  🌐 正在从 {endpoint} 下载 {RERANKER_MODEL}...")
        model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        print(f"✅ 重排模型 {RERANKER_MODEL} 下载完成")
        model.eval()
        _reranker_pipeline = (model, tokenizer)
        return _reranker_pipeline
    except Exception as e:
        raise RuntimeError(f"模型下载失败: {e}")


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 3
) -> List[Tuple[Document, float]]:
    """
    对 Document 列表用 BGE-Reranker 交叉编码打分
    返回 (doc, score) 列表，按分数降序
    """
    if not documents:
        return []

    model, tokenizer = get_reranker()
    import torch

    pairs = [[query, doc.page_content] for doc in documents]
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    with torch.no_grad():
        scores = model(**inputs).logits.view(-1).float().numpy()

    # 打包并排序
    scored = list(zip(documents, scores.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def rerank_texts(
    query: str,
    texts: List[str],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """直接对文本列表重排（不经过 Document 包装）"""
    docs = [Document(page_content=t) for t in texts]
    results = rerank_documents(query, docs, top_k)
    return [(r[0].page_content, r[1]) for r in results]
