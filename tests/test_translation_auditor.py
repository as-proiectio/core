import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from translation_cleaner import TranslationCleaner
from translation_auditor import TranslationAuditor


class TestTranslationCleanerEnhanced(unittest.TestCase):
    def test_chinese_numerals(self):
        text = "보유량은 84万3,775개 및 3亿 달러입니다."
        cleaned = TranslationCleaner.clean_chinese_numerals(text)
        self.assertEqual(cleaned, "보유량은 84만3,775개 및 3억 달러입니다.")

    def test_malformed_korean_numbers(self):
        text = "현금 4,15억 달러 조달 및 7.5억 달러 투입."
        cleaned = TranslationCleaner.clean_malformed_korean_numbers(text)
        self.assertIn("4억 1,500만 달러", cleaned)
        self.assertIn("7억 5,000만 달러", cleaned)

    def test_clean_full_pipeline(self):
        text = "오픈에이아이 모델이 84万3,775개 데이터를 4,15억 달러로 처리합니다.<br>반갑습니다."
        cleaned = TranslationCleaner.clean(text)
        self.assertIn("84만3,775개", cleaned)
        self.assertIn("4억 1,500만 달러", cleaned)
        self.assertIn("<br />", cleaned)


class TestTranslationAuditor(unittest.TestCase):
    def test_audit_text_detects_issues(self):
        sample_text = (
            "오픈에이아이 모델이 84万3,775개 데이터를 4,15억 달러로 분석합니다.\n"
            "바이낸스는 시장 점유율을 유지하며 순유입을 기록했습니다.\n"
            "바이낸스는 시장 점유율을 유지하며 순유입을 기록했습니다.\n"
        )

        issues = TranslationAuditor.audit_text(sample_text, source_label="unit_test")
        categories = [issue["category"] for issue in issues]

        self.assertIn("chinese_numeral_leak", categories)
        self.assertIn("malformed_number_format", categories)
        self.assertIn("known_misspelling", categories)
        self.assertIn("duplicate_sentence_leak", categories)


if __name__ == "__main__":
    unittest.main()
