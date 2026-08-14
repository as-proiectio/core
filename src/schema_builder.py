"""
Schema Builder module for Alpha Signals Core.

Synthesizes daily raw facts, categorized news, Korean translations, CIO insights,
and Aho-Corasick extracted stock tickers into a single canonical structured JSON report.
Uses the unique URL as the primary key to merge English and Korean article data.
Saves the artifact to `data/structured/signal_{YYYYMMDD}.json` (Full report only).
"""

import glob
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional
import pytz
from datetime import datetime

from shared.shared_logger import setup_logger
from src.ticker_matcher import extract_tickers_from_text

logger = setup_logger("logs/schema_builder.log", __name__)


def parse_cio_points_from_report(report_text: str) -> List[str]:
    """Parses topline bullet points or daily points from CIO text."""
    if not report_text:
        return []

    points = []
    # Try finding Topline Signals section
    topline_match = re.search(
        r"\*\*Topline Signals\*\*\s*\n\n(.*?)(?=\n\n|\n[#A-Za-z]|\Z)",
        report_text,
        re.DOTALL,
    )
    if topline_match:
        raw_bullets = re.split(r"<br\s*/?>", topline_match.group(1))
        for b in raw_bullets:
            clean = re.sub(r"^[\s\-_*]+", "", b).strip()
            if clean:
                points.append(clean)

    # Fallback to Daily Point section lines if topline not found
    if not points:
        daily_match = re.search(
            r"### Daily Point\s*\n(.*?)(?=\n\n|\n###|\Z)", report_text, re.DOTALL
        )
        if daily_match:
            lines = daily_match.group(1).split("\n")
            for line in lines:
                clean = re.sub(r"^[\s\-_*]+", "", line).strip()
                if clean:
                    points.append(clean)

    return points


def parse_market_indices(report_text: str) -> Dict[str, str]:
    """Extracts market index summary lines from Daily Point header."""
    indices = {}
    if not report_text:
        return indices

    for match in re.finditer(
        r"[_*\-]\s*([A-Za-z0-9\s&]+)\s+([\d,.]+)\s*\(([-+\d.]+%)\)", report_text
    ):
        name = match.group(1).strip()
        val = match.group(2).strip()
        pct = match.group(3).strip()
        indices[name] = f"{val} ({pct})"
    return indices


def generate_article_id(url: str, index: int, date_str: str) -> str:
    """Generates a deterministic unique ID for an article."""
    if not url:
        return f"art_{date_str}_{index:03d}"
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"art_{date_str}_{url_hash}"


def parse_markdown_articles(md_text: str) -> Dict[str, Dict[str, str]]:
    """
    Parses categorized markdown report into a dictionary keyed by URL.
    Returns: { url: { 'category': str, 'title': str, 'content': str } }
    """
    if not md_text:
        return {}

    articles: Dict[str, Dict[str, str]] = {}
    sections = re.split(r"\n###\s+", md_text)

    for sec in sections[1:]:
        lines = sec.split("\n")
        cat_header = lines[0].strip()
        if cat_header in ["Daily Point", "Weekly Schedule", "주간 일정"]:
            continue

        cat_body = "\n".join(lines[1:])
        # Split into article blocks
        blocks = re.split(r"(?=\[(?:.*?)\]\(https?://(?:.*?)\))", cat_body)
        for b in blocks:
            b = b.strip()
            m = re.match(
                r"^\[(.*?)\]\((https?://.*?)\)(?:\s*<br\s*/?>)?\s*(.*)", b, re.DOTALL
            )
            if m:
                title = m.group(1).strip()
                url = m.group(2).strip()
                content = m.group(3).strip()
                # Clean any trailing HTML breaks
                content = re.sub(r"<br\s*/?>", "", content).strip()
                articles[url] = {
                    "category": cat_header,
                    "title": title,
                    "content": content,
                }

    return articles


def build_structured_report(
    report_type: str = "full",
    target_date: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Synthesizes translated articles, categories, tickers, and CIO commentary
    into data/structured/signal_{target_date}.json.
    (Full report only).
    """
    # Skip premarket as requested
    if report_type != "full":
        logger.info(
            f"Skipping structured JSON report for non-full report type: {report_type}"
        )
        return None

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_dir is None:
        data_dir = os.path.join(project_root, "data")

    if not target_date:
        us_tz = pytz.timezone("America/New_York")
        target_date = datetime.now(us_tz).strftime("%Y%m%d")

    try:
        formatted_date = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}"
    except Exception:
        formatted_date = target_date

    logger.info(f"Building structured JSON report for {formatted_date}...")

    # 1. Load English & Korean Reports / Text
    en_md_path = os.path.join(data_dir, "report", f"alpha_signal_{target_date}.md")
    ko_md_path = os.path.join(data_dir, "report", f"alpha_signal_{target_date}_ko.md")
    en_txt_path = os.path.join(data_dir, f"final_report_{target_date}.txt")
    ko_txt_path = os.path.join(data_dir, f"final_report_ko_{target_date}.txt")

    en_text = ""
    if os.path.exists(en_md_path):
        with open(en_md_path, "r", encoding="utf-8") as f:
            en_text = f.read()
    elif os.path.exists(en_txt_path):
        with open(en_txt_path, "r", encoding="utf-8") as f:
            en_text = f.read()

    ko_text = ""
    if os.path.exists(ko_md_path):
        with open(ko_md_path, "r", encoding="utf-8") as f:
            ko_text = f.read()
    elif os.path.exists(ko_txt_path):
        with open(ko_txt_path, "r", encoding="utf-8") as f:
            ko_text = f.read()

    # 2. Parse Articles from Markdown / Text using URL as Primary Key
    en_articles = parse_markdown_articles(en_text)
    ko_articles = parse_markdown_articles(ko_text)

    # Fallback to translated_state_{target_date}.json if ko_articles empty
    if not ko_articles:
        trans_file = os.path.join(data_dir, f"translated_state_{target_date}.json")
        if os.path.exists(trans_file):
            try:
                with open(trans_file, "r", encoding="utf-8") as f:
                    trans_map = json.load(f)
                for url, val in trans_map.items():
                    ko_articles[url] = {
                        "category": "",
                        "title": val.get("title", ""),
                        "content": val.get("body", ""),
                    }
            except Exception as e:
                logger.warning(f"Failed to read fallback translated_state: {e}")

    # Fallback to sorted category JSON files if en_articles empty
    if not en_articles:
        category_files = glob.glob(
            os.path.join(data_dir, f"*_sorted_{target_date}.json")
        )
        for cat_file in category_files:
            cat_name = (
                os.path.basename(cat_file)
                .replace(f"_sorted_{target_date}.json", "")
                .replace("_", " ")
            )
            try:
                with open(cat_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
                if isinstance(items, list):
                    for item in items:
                        url = item.get("url", "").strip()
                        if url:
                            en_articles[url] = {
                                "category": cat_name,
                                "title": item.get("title", "").strip(),
                                "content": item.get("content", "").strip(),
                            }
            except Exception as e:
                logger.warning(f"Error reading category file {cat_file}: {e}")

    # 3. Synthesize Articles using URL matching
    synthesized_articles: List[Dict[str, Any]] = []
    all_urls = list(dict.fromkeys(list(en_articles.keys()) + list(ko_articles.keys())))

    for idx, url in enumerate(all_urls):
        en_item = en_articles.get(url, {})
        ko_item = ko_articles.get(url, {})

        cat = en_item.get("category") or ko_item.get("category") or "General"
        title_en = en_item.get("title", "")
        content_en = en_item.get("content", "")
        title_ko = ko_item.get("title", "")
        content_ko = ko_item.get("content", "")

        # Extract tickers with Aho-Corasick across all text fields
        full_text = f"{title_en} {title_ko} {content_en} {content_ko}"
        tickers = extract_tickers_from_text(full_text)

        art_id = generate_article_id(url, idx, target_date)
        synthesized_articles.append(
            {
                "id": art_id,
                "category": cat,
                "tickers": tickers,
                "title": title_en or title_ko,
                "title_ko": title_ko or title_en,
                "content": content_en,
                "content_ko": content_ko,
                "url": url,
            }
        )

    # 4. Build Final Document (100% Pure Article & Ticker Asset)
    structured_doc = {
        "date": formatted_date,
        "type": "full",
        "total_articles": len(synthesized_articles),
        "articles": synthesized_articles,
    }

    # 6. Save to data/structured/
    structured_dir = os.path.join(data_dir, "structured")
    os.makedirs(structured_dir, exist_ok=True)
    out_filepath = os.path.join(structured_dir, f"signal_{target_date}.json")

    with open(out_filepath, "w", encoding="utf-8") as f:
        json.dump(structured_doc, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Successfully generated structured report: {out_filepath} ({len(synthesized_articles)} articles)"
    )
    return out_filepath
