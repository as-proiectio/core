import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.threads_generator import generate_fallback_threads, generate_threads_content
from src.threads_publisher import ThreadsPublisher


def test_fallback_threads_generation():
    date_str = "2026-08-14"
    categories_data = {
        "Semiconductor": [
            {
                "title_ko": "엔비디아 블랙웰 양산 돌입",
                "tickers": ["NVDA", "TSM"],
            }
        ]
    }
    report_url = "https://alphasignals.cloud"

    res = generate_fallback_threads(date_str, categories_data, 249, report_url)

    assert "root_post" in res
    assert "1/4" in res["root_post"]
    assert len(res["thread_replies"]) == 3
    assert "2/4" in res["thread_replies"][0]
    assert "3/4" in res["thread_replies"][1]
    assert "249개" in res["thread_replies"][2]
    assert "4/4" in res["thread_replies"][2]


@patch("src.threads_generator.call_gemini_for_threads")
def test_generate_threads_content_with_gemini(mock_gemini):
    mock_gemini.return_value = {
        "root_post": "1. 오늘 밤 미국장에서 제일 주목해야 할 이슈가 있음. 1/4",
        "thread_replies": [
            "5. 빅테크들이 자체 칩으로 돌아서는 속도가 훨씬 빠름. 2/4",
            "8. 공급망 전반에서 수혜 기업들로 차별화 장세가 예상됨. 3/4",
            "11. 오늘 밤 주요 지수와 거래량을 필수 체크하길. 4/4",
        ],
    }

    structured_data = {
        "date": "2026-08-14",
        "total_articles": 249,
        "articles": [
            {
                "category": "Semiconductor",
                "title_ko": "엔비디아 상승",
                "tickers": ["NVDA"],
            }
        ],
    }

    res = generate_threads_content(structured_data, api_key="dummy_key")
    assert "1/4" in res["root_post"]
    assert len(res["thread_replies"]) == 3
    assert "249개" in res["thread_replies"][2]


def test_threads_publisher_dry_run():
    publisher = ThreadsPublisher(dry_run=True)
    thread_data = {
        "root_post": "Root post text",
        "thread_replies": ["Reply 1 text", "Reply 2 text", "Reply 3 with CTA"],
    }

    published_ids = publisher.publish_thread(thread_data)
    assert len(published_ids) == 4
    for pid in published_ids:
        assert pid.startswith("mock_published_mock_container_")


@patch("requests.post")
def test_threads_publisher_live_mock(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200, json=lambda: {"id": "123456789"}
    )

    with patch.dict(os.environ, {"ENABLE_THREADS_POST": "true"}):
        publisher = ThreadsPublisher(
            user_id="user_123", access_token="token_abc", dry_run=False
        )
        thread_data = {
            "root_post": "Root post",
            "thread_replies": ["Reply 1", "Reply 2", "Reply 3 with CTA"],
        }
        published_ids = publisher.publish_thread(thread_data)
        assert len(published_ids) == 4
        assert published_ids == ["123456789", "123456789", "123456789", "123456789"]
