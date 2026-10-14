"""输入、文件名和本地上传路径校验。"""

from pathlib import Path
import re
from typing import Iterable


class UploadValidationError(ValueError):
    """用户上传内容不符合应用约束时抛出的可预期异常。"""


_SAFE_STEM = re.compile(r"[^\w\u4e00-\u9fff.-]+", re.UNICODE)


def clean_question(question: str, max_length: int = 2000) -> str:
    """清理并校验聊天问题，避免空输入、控制字符和超长请求。"""

    if not isinstance(question, str):
        raise UploadValidationError("问题必须是文本")
    if "\x00" in question:
        raise UploadValidationError("问题包含非法控制字符")

    value = question.strip()
    if not value:
        raise UploadValidationError("问题不能为空")
    if len(value) > max_length:
        raise UploadValidationError(f"问题长度不能超过 {max_length} 个字符")
    return value


def sanitize_filename(filename: str, allowed_extensions: Iterable[str]) -> str:
    """返回不含路径组件的安全文件名，并严格限制扩展名。"""

    if not isinstance(filename, str) or not filename.strip():
        raise UploadValidationError("文件名不能为空")

    # 同时处理 Windows 和 POSIX 路径分隔符，防止路径穿越。
    normalized = filename.replace("\\", "/")
    name = Path(normalized).name
    if not name or name in {".", ".."}:
        raise UploadValidationError("文件名无效")

    extension = Path(name).suffix.lower()
    allowed = {str(ext).lower() for ext in allowed_extensions}
    if extension not in allowed:
        raise UploadValidationError(f"不支持的文件格式: {extension or '[无扩展名]'}")

    stem = _SAFE_STEM.sub("_", Path(name).stem).strip("._-")
    if not stem:
        stem = "upload"
    return f"{stem}{extension}"


def ensure_within_directory(base_dir: str | Path, relative_name: str) -> Path:
    """解析路径并确保结果仍在指定目录内。"""

    base = Path(base_dir).resolve()
    candidate = (base / relative_name).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise UploadValidationError("文件路径超出允许目录") from exc
    return candidate
