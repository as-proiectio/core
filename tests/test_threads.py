import os
import sys
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.threads_generator import generate_fallback_threads, generate_threads_content
from src.threads_publisher import ThreadsPublisher, publish_structured_report_to_threads


def test_fallback_threads_generation():
    date_str = "2026-08-14"
    cio_points = ["S&P 500 신기록 달성", "엔비디아 블랙웰 양산"]
    categories_data = {
        "Semiconductor": [
            {
                "title_ko": "엔비디아 블랙웰 양산 돌입",
                "tickers": ["NVDA", "TSM"],
            }
        ]
    }
    report_url = "https://alphasignals.cloud/report/2026-08-14"

    res = generate_fallback_threads(date_str, cio_points, categories_data, report_url)

    assert "root_post" in res
    assert "2026-08-14" in res["root_post"]
    assert len(res["thread_replies"]) == 1
    assert "$NVDA" in res["thread_replies"][0]
    assert "cta_reply" in res
    assert report_url in res["cta_reply"]


@patch("src.threads_generator.call_gemini_for_threads")
def test_generate_threads_content_with_gemini(mock_gemini):
    mock_gemini.return_value = {
        "root_post": "🚨 엔비디아 실적 발표 전야 3줄 요약",
        "thread_replies": ["(1/10) 🔹 [반도체] $NVDA"],
        "cta_reply": "웹에서 확인하세요: https://alphasignals.cloud"
    }

    structured_data = {
        "date": "2026-08-14",
        "cio_points": ["미 증시 호조"],
        "articles": [
            {
                "category": "Semiconductor",
                "title_ko": "엔비디아 상승",
                "tickers": ["NVDA"]
            }
        ]
    }

    res = generate_threads_content(structured_data, api_key="dummy_key")
    assert res["root_post"] == "🚨 엔비디아 실적 발표 전야 3줄 요약"
    assert len(res["thread_replies"]) == 1


def test_threads_publisher_dry_run():
    publisher = ThreadsPublisher(dry_run=True)
    thread_data = {
        "root_post": "Root post text",
        "thread_replies": ["Reply 1 text", "Reply 2 text"],
        "cta_reply": "CTA link text",
    }

    published_ids = publisher.publish_thread(thread_data)
    assert len(published_ids) == 4
    for pid in published_ids:
        assert pid.startswith("mock_published_mock_container_")


@patch("requests.post")
def test_threads_publisher_live_mock(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"id": "123456789"}
    )

    with patch.dict(os.environ, {"ENABLE_THREADS_POST": "true"}):
        publisher = ThreadsPublisher(
            user_id="user_123",
            access_token="token_abc",
            dry_run=False
        )
        thread_data = {
            "root_post": "Root post",
            "thread_replies": ["Reply 1"],
            "cta_reply": "CTA post"
        }
        published_ids = publisher.publish_thread(thread_data)
        assert len(published_ids) == 3
        assert published_ids == ["123456789", "123456789", "123456789"]
