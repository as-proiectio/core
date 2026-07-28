"""
Translation Auditor module for Alpha Signals Core.

Scans Korean translations for quality issues (e.g. Chinese character leaks,
malformed number notations, phonetic domain names, known misspellings, and duplicate sentences),
and logs audit candidates to date-stamped JSON log files (e.g. logs/translation_audit_YYYYMMDD.json)
for manual review and rule refinement. No auto-modification is performed on report content.
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional


class TranslationAuditor:
    @classmethod
    def get_known_misspellings_from_rules(cls) -> Dict[str, str]:
        """Loads literal replacements from config/custom_translation_rules.json and default rules."""
        mapping = {
            "오픈에이아이": "오픈AI",
            "하이퍼스케러": "하이퍼스케일러",
            "쿠리노스": "큐리노스",
            "시큐리티즈": "시큐리타이즈",
            "볼츠닷에프와이아이": "Vaults.fyi",
            "레이오프스닷에프와이아이": "Layoffs.fyi",
            "헬스케어닷고브": "Healthcare.gov",
        }
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        custom_path = os.path.join(project_root, "config", "custom_translation_rules.json")
        if os.path.exists(custom_path):
            try:
                with open(custom_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "literal_replacements" in data and isinstance(
                        data["literal_replacements"], dict
                    ):
                        mapping.update(data["literal_replacements"])
            except Exception:
                pass
        return mapping

    @classmethod
    def audit_text(cls, text: str, source_label: str = "text") -> List[Dict[str, Any]]:
        """
        Audits a block of text and returns a list of detected issue objects.
        """
        if not text:
            return []

        issues: List[Dict[str, Any]] = []
        misspelling_map = cls.get_known_misspellings_from_rules()

        lines = text.split("\n")
        for line_num, line in enumerate(lines, 1):
            line_str = line.strip()
            if not line_str:
                continue

            # 1. Check Chinese numeral leak (e.g. 84万, 3亿)
            chinese_num_match = re.search(r"(\d+)\s*([万亿])", line_str)
            if chinese_num_match:
                found_char = chinese_num_match.group(2)
                repl = "만" if found_char == "万" else "억"
                issues.append(
                    {
                        "source": source_label,
                        "line_number": line_num,
                        "category": "chinese_numeral_leak",
                        "severity": "high",
                        "detected_text": chinese_num_match.group(0),
                        "suggested_fix": re.sub(r"(\d+)\s*[万亿]", rf"\1{repl}", chinese_num_match.group(0)),
                        "line_snippet": line_str[:120],
                    }
                )

            # 2. Check malformed Korean number format (e.g. 4,15억, 4.15억 - 1 or 2 digits after comma/dot)
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

            # 3. Check phonetic web domain spellings (e.g. ~닷에프와이아이, ~닷컴 inside text)
            phonetic_domain_match = re.search(r"([가-힣]+닷[가-힣]+)", line_str)
            if phonetic_domain_match:
                issues.append(
                    {
                        "source": source_label,
                        "line_number": line_num,
                        "category": "phonetic_domain_spelling",
                        "severity": "medium",
                        "detected_text": phonetic_domain_match.group(0),
                        "suggested_fix": "Keep original English web domain (e.g. Vaults.fyi)",
                        "line_snippet": line_str[:120],
                    }
                )

            # 4. Check known misspellings
            for wrong, correct in misspelling_map.items():
                if wrong in line_str:
                    issues.append(
                        {
                            "source": source_label,
                            "line_number": line_num,
                            "category": "known_misspelling",
                            "severity": "high",
                            "detected_text": wrong,
                            "suggested_fix": correct,
                            "line_snippet": line_str[:120],
                        }
                    )

        # 5. Check duplicate consecutive sentences across paragraphs
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
