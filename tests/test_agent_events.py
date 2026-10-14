import json

from utils.agent_events import parse_image_results, summarize_tool_result


def test_parse_image_results_builds_static_paths():
    content = json.dumps(
        {
            "message": "找到图片",
            "images": [
                {"filename": "beauty_001.jpg", "score": 0.8123},
            ],
        },
        ensure_ascii=False,
    )

    assert parse_image_results(content) == [
        {
            "filename": "beauty_001.jpg",
            "score": 0.8123,
            "path": "/static/images/beauty_001.jpg",
        }
    ]


def test_parse_image_results_ignores_malformed_or_unsafe_items():
    content = json.dumps(
        {
            "images": [
                {"filename": "../secret.jpg", "score": 0.9},
                {"filename": "beauty_002.jpg", "score": "0.7"},
            ]
        }
    )

    assert parse_image_results(content) == [
        {
            "filename": "beauty_002.jpg",
            "score": 0.7,
            "path": "/static/images/beauty_002.jpg",
        }
    ]


def test_summarize_tool_result_is_short_and_user_friendly():
    content = json.dumps({"images": [{"filename": "a.jpg"}, {"filename": "b.jpg"}]})
    assert summarize_tool_result(content) == "找到 2 张相关图片"

    assert summarize_tool_result("  知识库检索完成\n包含成分与使用建议。  ") == "知识库检索完成 包含成分与使用建议。"
