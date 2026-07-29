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

    def test_chinese_numerals_multi_digit(self):
        text = "매출 18亿 달러 및 3.5亿 달러 순이익."
        cleaned = TranslationCleaner.clean_chinese_numerals(text)
        self.assertEqual(cleaned, "매출 18억 달러 및 3.5억 달러 순이익.")

    def test_malformed_korean_numbers(self):
        text = "현금 4,15억 달러 조달 및 7.5억 달러 투입."
        cleaned = TranslationCleaner.clean_malformed_korean_numbers(text)
        self.assertIn("4억 1,500만 달러", cleaned)
        self.assertIn("7억 5,000만 달러", cleaned)

    def test_clean_redundant_parentheses(self):
        text = "xAI(xAI) 인프라 구축 및 TSMC(TSMC) 실적 발표. Nvidia(Nvidia) 주가 상승."
        cleaned = TranslationCleaner.clean_redundant_parentheses(text)
        self.assertEqual(
            cleaned, "xAI 인프라 구축 및 TSMC 실적 발표. Nvidia 주가 상승."
        )

    def test_clean_full_pipeline(self):
        text = "오픈에이아이 모델이 84万3,775개 데이터를 4,15억 달러로 처리합니다.<br>반갑습니다."
        cleaned = TranslationCleaner.clean(text)
        self.assertIn("84만3,775개", cleaned)
        self.assertIn("4억 1,500만 달러", cleaned)
        self.assertIn("<br />", cleaned)


class TestTranslationAuditorSystemicRules(unittest.TestCase):
    def test_audit_text_detects_systemic_issues(self):
        sample_text = (
            "오픈에이아이 모델이 84万3,775개 데이터를 4,15억 달러로 분석합니다.\n"
            "캐릭터닷에이아이 및 볼츠닷에프와이아이 서비스 공시.\n"
            "시큐리티즈(Securitize) 주가는 상승했습니다.\n"
            "$AAPL 및 $NVDA 목표가 상향 조정함.\n"
            "바이낸스는 시장 점유율을 유지하며 순유입을 기록했습니다.\n"
            "바이낸스는 시장 점유율을 유지하며 순유입을 기록했습니다.\n"
        )

        issues = TranslationAuditor.audit_text(sample_text, source_label="unit_test")
        categories = [issue["category"] for issue in issues]

        self.assertIn("chinese_numeral_leak", categories)
        self.assertIn("malformed_number_format", categories)
        self.assertIn("phonetic_ai_slug_anomaly", categories)
        self.assertIn("phonetic_tld_spelling", categories)
        self.assertIn("phonetic_suffix_transliteration", categories)
        self.assertIn("unlocalized_ticker_leak", categories)
        self.assertIn("duplicate_sentence_leak", categories)


if __name__ == "__main__":
    unittest.main()
