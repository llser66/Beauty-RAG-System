"""Agent 工具结果的结构化解析，供后端和 Streamlit 前端共用。"""

import json
import math
import re
from pathlib import PurePosixPath
from typing import Any


_IMAGE_NAME = re.compile(r"^[\w\u4e00-\u9fff.-]+\.(?:jpg|jpeg|png|webp|bmp)$", re.IGNORECASE)


def parse_image_results(content: str, static_prefix: str = "/static/images") -> list[dict[str, Any]]:
    """从 search_images 工具的 JSON 结果中提取安全的前端图片对象。"""

    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return []

    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        return []

    results: list[dict[str, Any]] = []
    for item in payload["images"]:
        if not isinstance(item, dict):
            continue
        raw_name = str(item.get("filename", "")).replace("\\", "/")
        name = PurePosixPath(raw_name).name
        if name != raw_name or not _IMAGE_NAME.fullmatch(name):
            continue
        try:
            score = float(item.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        results.append(
            {
                "filename": name,
                "score": round(score, 4),
                "path": f"{static_prefix.rstrip('/')}/{name}",
            }
        )
    return results


def summarize_tool_result(content: str, max_length: int = 120) -> str:
    """把工具原始返回压缩成适合思考面板显示的一行摘要。"""

    images = parse_image_results(content)
    if images:
        return f"找到 {len(images)} 张相关图片"

    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("message"), str):
        text = payload["message"]
    else:
        text = str(content or "")

    text = " ".join(text.split())
    if len(text) > max_length:
        return text[: max_length - 1] + "…"
    return text
