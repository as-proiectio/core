"""
Translation Auditor module for Alpha Signals Core.

Scans Korean translations for systemic quality issues using pattern-based rules:
- Chinese character leaks (e.g. 84万, 3亿)
- Malformed Korean decimal number notations (e.g. 4,15억)
- Phonetic AI brand slugs (e.g. 오픈에이아이, 캐릭터닷에이아이, 씨쓰리에이아이, 하이퍼스케러)
- Phonetic web domain TLD spellings (e.g. 볼츠닷에프와이아이, 헬스케어닷고브)
- Phonetic proper noun suffix misspellings (e.g. 시큐리티즈(Securitize) -> 시큐리타이즈(Securitize))
- Unlocalized raw ticker symbols (e.g. $AAPL, $NVDA in body text)
- Custom rules loaded dynamically from config/custom_translation_rules.json
- Duplicate sentence leaks and informal sentence endings (~함, ~음)

Logs audit candidates to date-stamped JSON log files (e.g. logs/translation_audit_YYYYMMDD.json)
for manual review and rule refinement. No auto-modification is performed on report content.
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional


class TranslationAuditor:
    # Baseline fallback mappings for known common misspellings
    BASELINE_RULE_MAPPINGS = {
        "오픈에이아이": "오픈AI",
        "하이퍼스케러": "하이퍼스케일러",
        "쿠리노스": "큐리노스",
        "시큐리티즈": "시큐리타이즈",
        "볼츠닷에프와이아이": "Vaults.fyi",
        "레이오프스닷에프와이아이": "Layoffs.fyi",
        "헬스케어닷고브": "Healthcare.gov",
        "피스칼닷에이아이": "Fiscal.ai",
        "대만반도체제조": "TSMC",
        "대만적체전로": "TSMC",
        "대만적층회로제조": "TSMC",
    }

    @classmethod
    def get_custom_rules(cls) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        """Loads literal & regex replacements from config/custom_translation_rules.json."""
        literal_map = dict(cls.BASELINE_RULE_MAPPINGS)
        regex_list = []

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        custom_path = os.path.join(project_root, "config", "custom_translation_rules.json")
        if os.path.exists(custom_path):
            try:
                with open(custom_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "literal_replacements" in data and isinstance(data["literal_replacements"], dict):
                        literal_map.update(data["literal_replacements"])
                    if "regex_replacements" in data and isinstance(data["regex_replacements"], list):
                        regex_list.extend(data["regex_replacements"])
            except Exception:
                pass
        return literal_map, regex_list

    @classmethod
    def audit_text(cls, text: str, source_label: str = "text") -> List[Dict[str, Any]]:
        """
        Audits a block of Korean translated text using systemic pattern rules.
        """
        if not text:
            return []

        issues: List[Dict[str, Any]] = []
        literal_map, custom_regexes = cls.get_custom_rules()

        lines = text.split("\n")
        for line_num, line in enumerate(lines, 1):
            line_str = line.strip()
            if not line_str:
                continue

            # Rule 1. Chinese numeral leaks (e.g. 84万, 3亿, 18亿)
            chinese_num_match = re.search(r"([\d,.]+)\s*([万亿])", line_str)
            if chinese_num_match:
                num_str = chinese_num_match.group(1)
                found_char = chinese_num_match.group(2)
                repl = "만" if found_char == "万" else "억"
                issues.append(
                    {
                        "source": source_label,
                        "line_number": line_num,
                        "category": "chinese_numeral_leak",
                        "severity": "high",
                        "detected_text": chinese_num_match.group(0),
                        "suggested_fix": f"{num_str}{repl}",
                        "line_snippet": line_str[:120],
                    }
                )

            # Rule 2. Malformed Korean number format (e.g. 4,15억, 4.15억 - 1 or 2 digits after comma/dot)
            malformed_num_match = re.search(r"(\d+)[,\.](\d{1,2})\s*억", line_str)
            if malformed_num_match:
                issues.append(
                    {
                        "source": source_label,
                        "line_number": line_num,
                        "category": "malformed_number_format",
                        "severity": "medium",
                        "detected_text": malformed_num_match.group(0),
                        "suggested_fix": f"{malformed_num_match.group(1)}억 {int(malformed_num_match.group(2).ljust(4, '0')[:4]):,}만",
                        "line_snippet": line_str[:120],
                    }
                )

            # Rule 3. Phonetic AI brand slugs (e.g. ~에이아이, ~스케러)
            phonetic_ai_match = re.search(r"([가-힣]{2,}(?:에이아이|스케러))", line_str)
            if phonetic_ai_match:
                raw_slug = phonetic_ai_match.group(0)
                if raw_slug.endswith("에이아이"):
                    prefix = raw_slug[:-4]
                    fix_suggestion = f"{prefix}AI" if prefix else "AI"
                else:
                    fix_suggestion = raw_slug.replace("스케러", "스케일러")

                issues.append(
                    {
                        "source": source_label,
                        "line_number": line_num,
                        "category": "phonetic_ai_slug_anomaly",
                        "severity": "high",
                        "detected_text": raw_slug,
                        "suggested_fix": fix_suggestion,
                        "line_snippet": line_str[:120],
                    }
                )

            # Rule 4. Phonetic web domain TLD spellings (e.g. ~닷컴, ~닷고브, ~닷에프와이아이, ~닷아이오)
            phonetic_domain_match = re.search(r"([가-힣]{2,}(?:닷컴|닷고브|닷에프와이아이|닷아이오|닷오알지))", line_str)
            if phonetic_domain_match:
                issues.append(
                    {
                        "source": source_label,
                        "line_number": line_num,
                        "category": "phonetic_tld_spelling",
                        "severity": "medium",
                        "detected_text": phonetic_domain_match.group(0),
                        "suggested_fix": "Preserve original ASCII web domain (e.g. Vaults.fyi, Healthcare.gov)",
                        "line_snippet": line_str[:120],
                    }
                )

            # Rule 5. Phonetic proper noun suffix misspellings (e.g. 시큐리티즈(Securitize), 쿠리노스(Curinos))
            suffix_match = re.search(r"([가-힣]+(?:리티즈|쿠리노스))\s*\(([A-Za-z]+)\)", line_str)
            if suffix_match:
                raw_term = suffix_match.group(1)
                en_term = suffix_match.group(2)
                if raw_term.endswith("리티즈") and en_term.lower().endswith("ize"):
                    fix_suggestion = f"{raw_term[:-3]}타이즈({en_term})"
                elif "쿠리노스" in raw_term:
                    fix_suggestion = f"큐리노스({en_term})"
                else:
                    fix_suggestion = f"Standardize loanword suffix for {en_term}"

                issues.append(
                    {
                        "source": source_label,
                        "line_number": line_num,
                        "category": "phonetic_suffix_transliteration",
                        "severity": "medium",
                        "detected_text": suffix_match.group(0),
                        "suggested_fix": fix_suggestion,
                        "line_snippet": line_str[:120],
                    }
                )

            # Rule 6. User-configured custom literal rules
            for wrong, correct in literal_map.items():
                if wrong in line_str:
                    issues.append(
                        {
                            "source": source_label,
                            "line_number": line_num,
                            "category": "custom_rule_violation",
                            "severity": "high",
                            "detected_text": wrong,
                            "suggested_fix": correct,
                            "line_snippet": line_str[:120],
                        }
                    )

            # Rule 7. User-configured custom regex rules
            for reg_rule in custom_regexes:
                pat = reg_rule.get("pattern")
                repl = reg_rule.get("replacement", "")
                if pat:
                    reg_match = re.search(pat, line_str)
                    if reg_match:
                        try:
                            eval_fix = re.sub(pat, repl, reg_match.group(0))
                        except Exception:
                            eval_fix = repl

                        issues.append(
                            {
                                "source": source_label,
                                "line_number": line_num,
                                "category": "custom_regex_violation",
                                "severity": "medium",
                                "detected_text": reg_match.group(0),
                                "suggested_fix": eval_fix,
                                "line_snippet": line_str[:120],
                            }
                        )

            # Rule 8. Unlocalized raw ticker symbols in body text (e.g. $AAPL, $NVDA)
            ticker_match = re.search(r"\$([A-Z]{2,5})\b", line_str)
            if ticker_match and not line_str.startswith("_"):
                issues.append(
                    {
                        "source": source_label,
                        "line_number": line_num,
                        "category": "unlocalized_ticker_leak",
                        "severity": "low",
                        "detected_text": ticker_match.group(0),
                        "suggested_fix": f"Replace raw ticker {ticker_match.group(0)} with Korean company name",
                        "line_snippet": line_str[:120],
                    }
                )

            # Rule 9. Informal sentence endings (~함, ~음, ~임)
            if re.search(r"[가-힣](?:함|음|임)\.\s*$", line_str) and not line_str.startswith("-"):
                issues.append(
                    {
                        "source": source_label,
                        "line_number": line_num,
                        "category": "informal_sentence_ending",
                        "severity": "low",
                        "detected_text": line_str[-10:],
                        "suggested_fix": "End sentence with formal Korean ending (~습니다 / ~합니다)",
                        "line_snippet": line_str[:120],
                    }
                )

        # Rule 10. Duplicate consecutive sentences across paragraphs
        sentences = re.split(r"(?<=[.!?])\s+", text)
        prev_s = ""
        for s in sentences:
            s_clean = s.strip()
            if not s_clean or len(s_clean) < 15:
                continue
            if s_clean == prev_s:
                issues.append(
                    {
                        "source": source_label,
                        "line_number": -1,
                        "category": "duplicate_sentence_leak",
                        "severity": "high",
                        "detected_text": s_clean[:80],
                        "suggested_fix": "Deduplicate repeated sentence",
                        "line_snippet": s_clean[:120],
                    }
                )
            prev_s = s_clean

        return issues

    @classmethod
    def audit_file(cls, file_path: str, output_log_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Audits a markdown or text report file and logs any detected issues.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        issues = cls.audit_text(content, source_label=file_path)

        if output_log_path:
            os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
            with open(output_log_path, "w", encoding="utf-8") as f:
                json.dump(issues, f, ensure_ascii=False, indent=2)

        return issues


def main():
    parser = argparse.ArgumentParser(description="Audit Korean translation output for quality issues.")
    parser.add_argument("--file", type=str, required=True, help="Path to translated report file to audit")
    parser.add_argument(
        "--output-log",
        type=str,
        default=None,
        help="Path to save audit candidates JSON log (default: logs/translation_audit_YYYYMMDD.json)",
    )
    args = parser.parse_args()

    print(f"🔍 Auditing translation file: {args.file}...")
    issues = TranslationAuditor.audit_file(args.file, output_log_path=args.output_log)

    print(f"\n📊 Audit Complete. Found {len(issues)} candidate issue(s):")
    for idx, issue in enumerate(issues, 1):
        print(f"  [{idx}] Category: {issue['category']} (Severity: {issue['severity']})")
        print(f"      Detected: '{issue['detected_text']}' -> Suggested Fix: '{issue['suggested_fix']}'")
        print(f"      Snippet: {issue['line_snippet']}\n")

    if args.output_log:
        print(f"📁 Logged issues to: {args.output_log}")


if __name__ == "__main__":
    main()
