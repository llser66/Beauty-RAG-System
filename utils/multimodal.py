"""
多模态向量引擎 — CLIP 图片向量化
支持中英文图片-文本双路检索
"""

import os
import numpy as np
from typing import List, Optional, Union
from PIL import Image as PILImage

# 全局缓存 CLIP 模型（首次加载后复用）
_clip_model = None
_clip_processor = None


def get_clip_model():
    """懒加载 CLIP 模型（优先读缓存，HF 被墙时切镜像）"""
    global _clip_model, _clip_processor
    if _clip_model is None:
        print("🔄 正在加载 CLIP 多模态模型（首次需下载 ~600MB）...")
        from transformers import CLIPModel, CLIPProcessor
        model_name = "openai/clip-vit-base-patch32"

        # 优先离线加载
        try:
            _clip_model = CLIPModel.from_pretrained(model_name, local_files_only=True)
            _clip_processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        except OSError:
            import os
            os.environ["HF_ENDPOINT"] = os.getenv(
                "HF_ENDPOINT", "https://hf-mirror.com"
            )
            print("  本地未缓存，从 HuggingFace 下载...")
            _clip_model = CLIPModel.from_pretrained(model_name)
            _clip_processor = CLIPProcessor.from_pretrained(model_name)

        print("✅ CLIP 模型加载完成")
    return _clip_model, _clip_processor


def embed_image(image_path: str) -> np.ndarray:
    """
    将图片文件转为 CLIP 向量 (512维)
    返回归一化的 numpy 数组
    """
    import torch
    model, processor = get_clip_model()
    image = PILImage.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    # 兼容 transformers 新版（返回 BaseModelOutputWithPooling）
    if isinstance(outputs, torch.Tensor):
        vector = outputs
    else:
        vector = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0]
    vector = vector.detach().numpy().flatten()
    norm = np.linalg.norm(vector)
    return (vector / norm).astype(np.float32) if norm > 0 else vector.astype(np.float32)


def embed_text_for_image_search(text: str) -> np.ndarray:
    """
    将查询文本用 CLIP 文本编码器转为向量 (512维)
    用于在 image_collection 中搜图
    """
    import torch
    model, processor = get_clip_model()
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    outputs = model.get_text_features(**inputs)
    # 兼容 transformers 新版
    if isinstance(outputs, torch.Tensor):
        vector = outputs
    else:
        vector = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0]
    vector = vector.detach().numpy().flatten()
    norm = np.linalg.norm(vector)
    return (vector / norm).astype(np.float32) if norm > 0 else vector.astype(np.float32)


CLIP_DIM = 512  # CLIP ViT-B/32 输出维度
