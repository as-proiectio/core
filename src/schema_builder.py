"""
Schema Builder module for Alpha Signals Core.

Synthesizes daily raw facts, categorized news, Korean translations, CIO insights,
and Aho-Corasick extracted stock tickers into a single canonical structured JSON report.
Saves the artifact to `data/structured/` for long-term programmatic SEO and downstream pipelines.
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
        raw_bullets = topline_match.group(1).split("<br />")
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


def build_structured_report(
    report_type: str = "full",
    target_date: str = None,
    data_dir: str = None,
) -> Optional[str]:
    """
    Synthesizes translated articles, categories, tickers, and CIO commentary
    into data/structured/signal_{target_date}.json.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_dir is None:
        data_dir = os.path.join(project_root, "data")

    if not target_date:
        us_tz = pytz.timezone("America/New_York")
        target_date = datetime.now(us_tz).strftime("%Y%m%d")

    # Format date YYYY-MM-DD
    try:
        formatted_date = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}"
    except Exception:
        formatted_date = target_date

    logger.info(
        f"Building structured JSON report for {formatted_date} (Type: {report_type})..."
    )

    # 1. Load CIO report text (Prefer Korean, fallback English)
    cio_ko_file = os.path.join(
        data_dir,
        (
            f"premarket_report_ko_{target_date}.txt"
            if report_type == "premarket"
            else f"final_report_ko_{target_date}.txt"
        ),
    )
    cio_en_file = os.path.join(
        data_dir,
        (
            f"premarket_report_{target_date}.txt"
            if report_type == "premarket"
            else f"final_report_{target_date}.txt"
        ),
    )

    cio_text = ""
    if os.path.exists(cio_ko_file):
        with open(cio_ko_file, "r", encoding="utf-8") as f:
            cio_text = f.read()
    elif os.path.exists(cio_en_file):
        with open(cio_en_file, "r", encoding="utf-8") as f:
            cio_text = f.read()

    cio_points = parse_cio_points_from_report(cio_text)
    market_indices = parse_market_indices(cio_text)

    # 2. Load Translated State
    trans_file = os.path.join(
        data_dir,
        (
            "translated_state_pre.json"
            if report_type == "premarket"
            else f"translated_state_{target_date}.json"
        ),
    )
    translated_map: Dict[str, Dict[str, str]] = {}
    if os.path.exists(trans_file):
        try:
            with open(trans_file, "r", encoding="utf-8") as f:
                translated_map = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load translated state: {e}")

    # 3. Load Article Categories and English Contents
    # Collect from sorted category files: *_sorted_{target_date}.json
    category_files = glob.glob(os.path.join(data_dir, f"*_sorted_{target_date}.json"))
    article_by_url: Dict[str, Dict[str, Any]] = {}

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
                        article_by_url[url] = {
                            "category": cat_name,
                            "title": item.get("title", "").strip(),
                            "content": item.get("content", "").strip(),
                            "url": url,
                        }
        except Exception as e:
            logger.warning(f"Error reading category file {cat_file}: {e}")

    # Fallback to daily_news_*.json if sorted files are empty
    if not article_by_url:
        daily_news_file = os.path.join(
            data_dir,
            (
                f"premarket_news_{target_date}.json"
                if report_type == "premarket"
                else f"daily_news_{target_date}.json"
            ),
        )
        if os.path.exists(daily_news_file):
            try:
                with open(daily_news_file, "r", encoding="utf-8") as f:
                    news_data = json.load(f)
                news_list = (
                    news_data.get("news", [])
                    if isinstance(news_data, dict)
                    else news_data
                )
                for item in news_list:
                    url = item.get("url", "").strip()
                    if url:
                        article_by_url[url] = {
                            "category": item.get("category", "General"),
                            "title": item.get("title", "").strip(),
                            "content": item.get("content", "").strip(),
                            "url": url,
                        }
            except Exception as e:
                logger.warning(f"Error reading daily news file: {e}")

    # 4. Synthesize Articles
    # If translated_map exists, iterate translated items for highest accuracy
    synthesized_articles = []
    seen_urls = set()

    # Priority 1: Translated articles
    for idx, (url, trans_item) in enumerate(translated_map.items()):
        seen_urls.add(url)
        orig_item = article_by_url.get(url, {})
        cat = orig_item.get("category", "General")
        title_en = orig_item.get("title", "")
        content_en = orig_item.get("content", "")

        title_ko = trans_item.get("title", "").strip()
        content_ko = trans_item.get("body", "").strip()

        # Extract tickers with Aho-Corasick on all available text fields
        full_search_text = f"{title_en} {title_ko} {content_en} {content_ko}"
        tickers = extract_tickers_from_text(full_search_text)

        art_id = generate_article_id(url, idx, target_date)
        synthesized_articles.append(
            {
                "id": art_id,
                "category": cat,
                "tickers": tickers,
                "title": title_en or title_ko,
                "title_ko": title_ko,
                "content": content_en,
                "content_ko": content_ko,
                "url": url,
            }
        )

    # Priority 2: Any remaining articles in article_by_url (if not translated)
    for url, orig_item in article_by_url.items():
        if url in seen_urls:
            continue
        seen_urls.add(url)
        title_en = orig_item.get("title", "").strip()
        content_en = orig_item.get("content", "").strip()
        cat = orig_item.get("category", "General")
        tickers = extract_tickers_from_text(f"{title_en} {content_en}")
        art_id = generate_article_id(url, len(synthesized_articles), target_date)
        synthesized_articles.append(
            {
                "id": art_id,
                "category": cat,
                "tickers": tickers,
                "title": title_en,
                "title_ko": "",
                "content": content_en,
                "content_ko": "",
                "url": url,
            }
        )

    # 5. Build Final Document
    structured_doc = {
        "date": formatted_date,
        "type": report_type,
        "cio_points": cio_points,
        "market_indices": market_indices,
        "total_articles": len(synthesized_articles),
        "articles": synthesized_articles,
    }

    # 6. Save to data/structured/
    structured_dir = os.path.join(data_dir, "structured")
    os.makedirs(structured_dir, exist_ok=True)

    out_filename = (
        f"signal_premarket_{target_date}.json"
        if report_type == "premarket"
        else f"signal_{target_date}.json"
    )
    out_filepath = os.path.join(structured_dir, out_filename)

    with open(out_filepath, "w", encoding="utf-8") as f:
        json.dump(structured_doc, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Successfully generated structured report: {out_filepath} ({len(synthesized_articles)} articles)"
    )
    return out_filepath
