import unittest
from unittest.mock import patch, MagicMock
import os
import json
import tempfile
import shutil
import sys
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from translator import (
    translate_new_articles,
    translate_missing_report_articles,
    TranslationError,
)


class TestTranslatorFailFast(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.test_dir, "state.json")
        self.cache_file = os.path.join(self.test_dir, "cache.json")
        self.report_file = os.path.join(self.test_dir, "report.txt")

        # Create a valid state file with some articles
        state_data = {
            "category_normal_outputs": {
                "General": [
                    "[Apple Sinks 6%](https://apple.com)\nApple shares fell on Friday."
                ]
            }
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f)

        # Create a valid report file with missing translations
        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write("[Tesla Rallies 5%](https://tesla.com)\nTesla stock surged.")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("translator.call_gemini_translator_api")
    def test_translate_new_articles_raises_runtime_error(self, mock_api):
        mock_api.side_effect = Exception("API connection timeout")

        with self.assertRaises(TranslationError) as context:
            translate_new_articles(self.state_file, self.cache_file)

        self.assertIn(
            "Translation pipeline failed during batch 1", str(context.exception)
        )

    @patch("translator.call_gemini_translator_api")
    def test_translate_missing_report_articles_raises_runtime_error(self, mock_api):
        mock_api.side_effect = Exception("Quota exceeded")

        with self.assertRaises(TranslationError) as context:
            translate_missing_report_articles(self.report_file, self.cache_file)

        self.assertIn("Retry translation failed during batch 1", str(context.exception))


class TestTranslatorSplitAndRetry(unittest.TestCase):
    @patch("translator.requests.post")
    @patch("translator.time.sleep")
    def test_split_and_retry_on_timeout(self, mock_sleep, mock_post):
        articles = [
            {"url": "https://apple.com", "title": "Apple Title", "body": "Apple Body"},
            {"url": "https://tesla.com", "title": "Tesla Title", "body": "Tesla Body"},
        ]

        def side_effect(url, **kwargs):
            json_payload = kwargs.get("json")
            assert json_payload is not None
            payload_text = json_payload["contents"][0]["parts"][0]["text"]
            payload_data = json.loads(payload_text)

            if len(payload_data) == 2:
                raise requests.exceptions.Timeout("Read timeout")
            elif len(payload_data) == 1:
                art = payload_data[0]
                url_placeholder = art["url"]
                resp = MagicMock()
                resp.status_code = 200

                translated_art = {
                    "url": url_placeholder,
                    "title": f"Translated {art['title']}",
                    "body": f"Translated {art['body']}",
                }
                resp.json.return_value = {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps(
                                            {"translations": [translated_art]}
                                        )
                                    }
                                ]
                            }
                        }
                    ]
                }
                return resp
            else:
                raise ValueError("Unexpected batch size in mock")

        mock_post.side_effect = side_effect

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            from translator import call_gemini_translator_api, COOLDOWN_SLEEP_SECONDS

            results = call_gemini_translator_api(
                articles, retries_per_model=1, backoff_factor=1
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            results["https://apple.com"]["title"], "Translated Apple Title"
        )
        self.assertEqual(
            results["https://tesla.com"]["title"], "Translated Tesla Title"
        )

        # Verify cooldown sleep was called between batch runs
        mock_sleep.assert_any_call(COOLDOWN_SLEEP_SECONDS)

    @patch("translator.requests.post")
    @patch("translator.time.sleep")
    def test_split_and_retry_on_json_error(self, mock_sleep, mock_post):
        articles = [
            {"url": "https://apple.com", "title": "Apple Title", "body": "Apple Body"},
            {"url": "https://tesla.com", "title": "Tesla Title", "body": "Tesla Body"},
        ]

        def side_effect(url, **kwargs):
            json_payload = kwargs.get("json")
            assert json_payload is not None
            payload_text = json_payload["contents"][0]["parts"][0]["text"]
            payload_data = json.loads(payload_text)

            resp = MagicMock()
            resp.status_code = 200

            if len(payload_data) == 2:
                resp.json.return_value = {
                    "candidates": [{"content": {"parts": [{"text": "invalid_json{"}]}}]
                }
            elif len(payload_data) == 1:
                art = payload_data[0]
                url_placeholder = art["url"]
                translated_art = {
                    "url": url_placeholder,
                    "title": f"Translated {art['title']}",
                    "body": f"Translated {art['body']}",
                }
                resp.json.return_value = {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps(
                                            {"translations": [translated_art]}
                                        )
                                    }
                                ]
                            }
                        }
                    ]
                }
            return resp

        mock_post.side_effect = side_effect

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            from translator import call_gemini_translator_api, COOLDOWN_SLEEP_SECONDS

            results = call_gemini_translator_api(
                articles, retries_per_model=1, backoff_factor=1
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            results["https://apple.com"]["title"], "Translated Apple Title"
        )
        self.assertEqual(
            results["https://tesla.com"]["title"], "Translated Tesla Title"
        )

        # Verify cooldown sleep was called between batch runs
        mock_sleep.assert_any_call(COOLDOWN_SLEEP_SECONDS)

    @patch("translator.requests.post")
    @patch("translator.time.sleep")
    def test_no_split_on_single_article(self, mock_sleep, mock_post):
        articles = [
            {"url": "https://apple.com", "title": "Apple Title", "body": "Apple Body"}
        ]

        mock_post.side_effect = requests.exceptions.Timeout("Read timeout")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            from translator import call_gemini_translator_api

            with self.assertRaises(TranslationError):
                call_gemini_translator_api(
                    articles, retries_per_model=1, backoff_factor=1
                )


class TestGenerateKoreanFullDraft(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.en_report = os.path.join(self.test_dir, "en_report.txt")
        self.ko_draft = os.path.join(self.test_dir, "ko_draft.txt")
        self.cache_file = os.path.join(self.test_dir, "cache.json")

        # Create a mock English report
        self.report_content = (
            "### Daily Point\n"
            "_ Dow Jones 40,000 (+0.5%)\n"
            "_ S&P 500 5,000 (+1.0%)\n"
            "**Topline Signals**\n"
            "- **Nvidia**: NVIDIA shares rose.\n"
            "Good day.\n"
            "IGNORING short term noise.\n\n"
            "### Weekly Schedule\n"
            "13 Jul (Monday)\n"
            "FOMC Bowman Speaks\n"
        )
        with open(self.en_report, "w", encoding="utf-8") as f:
            f.write(self.report_content)

        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump({}, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("translator.call_gemini_translator_api")
    def test_generate_korean_full_draft_with_daily_point(self, mock_translate_api):
        # mock translation response
        mock_translate_api.return_value = {
            "daily_point_signals": {
                "title": "Daily Point Signals",
                "body": "- **엔비디아**: 엔비디아 주가 상승.",
            },
            "daily_point_commentary": {
                "title": "Daily Point Commentary",
                "body": "단기 소음을 무시해야 합니다.",
            },
        }

        from translator import generate_korean_full_draft

        success = generate_korean_full_draft(
            self.en_report, self.ko_draft, self.cache_file
        )
        self.assertTrue(success)

        # Verify arguments passed to call_gemini_translator_api
        mock_translate_api.assert_called_once()
        call_kwargs = mock_translate_api.call_args[1]
        self.assertTrue(call_kwargs.get("preserve_newlines"))
        self.assertIsNotNone(call_kwargs.get("custom_system_prompt"))
        self.assertIn(
            "Chief Investment Officer (CIO)", call_kwargs.get("custom_system_prompt")
        )

        # Verify content of the generated draft
        with open(self.ko_draft, "r", encoding="utf-8") as f:
            ko_content = f.read()

        self.assertIn("### Daily Point", ko_content)
        self.assertIn("_ Dow Jones 40,000 (+0.5%)", ko_content)
        self.assertIn("**Topline Signals**", ko_content)
        self.assertIn("- **엔비디아**: 엔비디아 주가 상승.", ko_content)
        self.assertIn("안녕하세요.", ko_content)
        self.assertNotIn("Good day.", ko_content)
        self.assertIn("단기 소음을 무시해야 합니다.", ko_content)
        self.assertIn("### 주간 일정", ko_content)

        # Verify cache file structure
        with open(self.cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        self.assertIn("__daily_point_commentary__", cache_data)
        self.assertEqual(
            cache_data["__daily_point_commentary__"]["body"],
            "- **엔비디아**: 엔비디아 주가 상승.\n\n안녕하세요.\n\n단기 소음을 무시해야 합니다.",
        )


class TestTranslationCleaner(unittest.TestCase):
    def test_normalize_br_tags(self):
        from translation_cleaner import TranslationCleaner

        # Test cases for various br tags
        cases = [
            ("안녕하세요.<br/ >반갑습니다.", "안녕하세요.<br />반갑습니다."),
            ("안녕하세요.<br / >반갑습니다.", "안녕하세요.<br />반갑습니다."),
            ("안녕하세요.<br>반갑습니다.", "안녕하세요.<br />반갑습니다."),
            ("안녕하세요.<br/>반갑습니다.", "안녕하세요.<br />반갑습니다."),
            ("안녕하세요.< br />반갑습니다.", "안녕하세요.<br />반갑습니다."),
            ("안녕하세요.< br/>반갑습니다.", "안녕하세요.<br />반갑습니다."),
            ("안녕하세요.<BR>반갑습니다.", "안녕하세요.<br />반갑습니다."),
            ("안녕하세요.<BR />반갑습니다.", "안녕하세요.<br />반갑습니다."),
        ]
        for input_text, expected in cases:
            with self.subTest(input_text=input_text):
                self.assertEqual(TranslationCleaner.clean(input_text), expected)


class TestMacroTranslationCache(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.news_file = os.path.join(self.test_dir, "daily_news_20260721.json")
        self.project_root_patch = patch(
            "translator.os.path.dirname", return_value=self.test_dir
        )
        self.project_root_patch.start()

        # Create directories
        os.makedirs(os.path.join(self.test_dir, "config"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "data"), exist_ok=True)

        self.map_file = os.path.join(
            self.test_dir, "config", "macro_translation_map.json"
        )
        self.cache_file = os.path.join(
            self.test_dir, "data", "macro_translation_cache.json"
        )

    def tearDown(self):
        self.project_root_patch.stop()
        shutil.rmtree(self.test_dir)

    @patch("translator.translate_macro_events")
    def test_translate_and_cache_weekly_schedule(self, mock_translate):
        # Setup static map and cache
        with open(self.map_file, "w", encoding="utf-8") as f:
            json.dump({"CPI (YoY)": "소비자물가지수 (YoY)"}, f)

        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "2026-07-20_Unemployment Rate": "실업률",
                    "2026-07-19_Old Event": "오래된 이벤트",
                },
                f,
            )

        # Setup news file
        news_data = {
            "weekly_schedule": [
                {
                    "name": "CPI (YoY)",
                    "importance": "high",
                    "korean_name": "",
                    "utc_time": "2026-07-22T14:30:00Z",
                },
                {
                    "name": "Unemployment Rate",
                    "importance": "high",
                    "korean_name": "",
                    "utc_time": "2026-07-20T14:30:00Z",
                },
                {
                    "name": "New Indicator",
                    "importance": "medium",
                    "korean_name": "",
                    "utc_time": "2026-07-22T15:00:00Z",
                },
                {
                    "name": "Some Earnings",
                    "importance": "earnings",
                    "korean_name": "",
                    "utc_time": "2026-07-22T00:00:00Z",
                },
            ]
        }
        with open(self.news_file, "w", encoding="utf-8") as f:
            json.dump(news_data, f)

        mock_translate.return_value = {"New Indicator": "새로운 지표"}

        from translator import translate_and_cache_weekly_schedule_events

        translate_and_cache_weekly_schedule_events(self.news_file, "20260721")

        # Verify news file in-place update
        with open(self.news_file, "r", encoding="utf-8") as f:
            updated_news = json.load(f)

        events = updated_news["weekly_schedule"]
        self.assertEqual(
            events[0]["korean_name"], "소비자물가지수 (YoY)"
        )  # From static map
        self.assertEqual(events[1]["korean_name"], "실업률")  # From cache
        self.assertEqual(events[2]["korean_name"], "새로운 지표")  # Translated by mock
        self.assertEqual(events[3]["korean_name"], "")  # Earnings call untouched

        # Verify cache cleanup and updates (today is 2026-07-21, yesterday is 2026-07-20. 2026-07-19 is D-2, so deleted)
        with open(self.cache_file, "r", encoding="utf-8") as f:
            updated_cache = json.load(f)

        self.assertIn("2026-07-20_Unemployment Rate", updated_cache)
        self.assertIn("2026-07-22_New Indicator", updated_cache)
        self.assertEqual(updated_cache["2026-07-22_New Indicator"], "새로운 지표")
        self.assertNotIn(
            "2026-07-19_Old Event", updated_cache
        )  # Cleans up old cache entry


class TestPostProcessTranslationRules(unittest.TestCase):
    def test_default_rule_replacements(self):
        from translator import load_translation_rules, post_process_translation

        rules = load_translation_rules()
        title = "미국 달러화화 가치 전망 및 100 bps 상승"
        body = "금리는 25 %p 인상되었으며 미국 달러화화 강세입니다."

        processed_title, processed_body = post_process_translation(
            title, body, rules, preserve_newlines=True
        )

        self.assertIn("미국 달러화", processed_title)
        self.assertNotIn("미국 달러화화", processed_title)
        self.assertIn("100bp", processed_title)

        self.assertIn("미국 달러화", processed_body)
        self.assertNotIn("미국 달러화화", processed_body)
        self.assertIn("25%p", processed_body)


if __name__ == "__main__":
    unittest.main()
