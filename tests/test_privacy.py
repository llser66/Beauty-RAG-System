import os
from datetime import datetime, timedelta, timezone
from io import BytesIO

from PIL import Image
import pytest

from utils.privacy import cleanup_expired_files, normalize_image_bytes


def _jpeg_bytes_with_metadata() -> bytes:
    image = Image.new("RGB", (4, 4), (220, 180, 190))
    exif = image.getexif()
    exif[0x010E] = "private test metadata"
    output = BytesIO()
    image.save(output, format="JPEG", exif=exif)
    return output.getvalue()


def test_normalize_image_bytes_removes_metadata():
    normalized = normalize_image_bytes(_jpeg_bytes_with_metadata(), ".jpg")

    with Image.open(BytesIO(normalized)) as image:
        assert image.format == "JPEG"
        assert not image.getexif()


def test_cleanup_expired_files_removes_old_uploads_only(tmp_path):
    old_file = tmp_path / "chat_old.jpg"
    fresh_file = tmp_path / "chat_fresh.jpg"
    old_file.write_bytes(b"old")
    fresh_file.write_bytes(b"fresh")
    old_time = (datetime.now(timezone.utc) - timedelta(hours=30)).timestamp()
    os.utime(old_file, (old_time, old_time))

    deleted = cleanup_expired_files(tmp_path, retention_hours=24)

    assert deleted == 1
    assert not old_file.exists()
    assert fresh_file.exists()


def test_cleanup_requires_positive_retention(tmp_path):
    with pytest.raises(ValueError):
        cleanup_expired_files(tmp_path, retention_hours=0)
