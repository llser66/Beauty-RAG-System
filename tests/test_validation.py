import pytest

from utils.validation import (
    UploadValidationError,
    clean_question,
    ensure_within_directory,
    sanitize_filename,
)


def test_clean_question_strips_outer_whitespace_without_destroying_content():
    assert clean_question("  推荐适合油皮的防晒\n") == "推荐适合油皮的防晒"


def test_clean_question_rejects_blank_and_oversized_input():
    with pytest.raises(UploadValidationError):
        clean_question(" \n\t")

    with pytest.raises(UploadValidationError):
        clean_question("a" * 2001)


def test_sanitize_filename_removes_path_components_and_keeps_chinese_name():
    assert sanitize_filename(r"C:\fake\面霜 配方.pdf", {".pdf"}) == "面霜_配方.pdf"


def test_sanitize_filename_rejects_unknown_extension_and_empty_name():
    with pytest.raises(UploadValidationError):
        sanitize_filename("payload.exe", {".pdf"})

    with pytest.raises(UploadValidationError):
        sanitize_filename("../", {".pdf"})


def test_ensure_within_directory_blocks_directory_escape(tmp_path):
    safe = ensure_within_directory(tmp_path, "uploads/photo.jpg")
    assert safe.parent == (tmp_path / "uploads").resolve()

    with pytest.raises(UploadValidationError):
        ensure_within_directory(tmp_path, "../secret.txt")

