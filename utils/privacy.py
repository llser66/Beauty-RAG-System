"""上传图片的隐私保护工具：去除元数据并清理过期临时文件。"""

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def normalize_image_bytes(file_bytes: bytes, extension: str) -> bytes:
    """重新编码图片，避免把 EXIF、GPS 等原始元数据带入临时存储。"""

    suffix = str(extension or "").lower()
    if suffix not in _IMAGE_EXTENSIONS:
        raise ValueError(f"不支持的图片格式: {suffix}")

    try:
        with Image.open(BytesIO(file_bytes)) as source:
            source.load()
            output = BytesIO()
            if suffix in {".jpg", ".jpeg"}:
                source.convert("RGB").save(
                    output,
                    format="JPEG",
                    quality=92,
                    optimize=True,
                    exif=b"",
                )
            elif suffix == ".png":
                source.save(output, format="PNG", optimize=True)
            elif suffix == ".webp":
                source.convert("RGBA").save(output, format="WEBP", quality=92)
            else:
                source.convert("RGB").save(output, format="BMP")
            return output.getvalue()
    except Exception as exc:
        raise ValueError("上传内容不是有效图片") from exc


def cleanup_expired_files(
    directory: str | Path,
    retention_hours: int,
    now: datetime | None = None,
) -> int:
    """只删除指定临时目录中的过期普通文件，不跟随符号链接。"""

    if retention_hours <= 0:
        raise ValueError("retention_hours 必须大于 0")

    base = Path(directory)
    if not base.exists():
        return 0

    cutoff = (now or datetime.now(timezone.utc)).timestamp() - retention_hours * 3600
    deleted = 0
    for candidate in base.iterdir():
        if candidate.is_symlink() or not candidate.is_file():
            continue
        try:
            if candidate.stat().st_mtime < cutoff:
                candidate.unlink()
                deleted += 1
        except FileNotFoundError:
            continue
    return deleted

