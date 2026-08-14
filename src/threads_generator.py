"""
Threads Content Generator module for Alpha Signals Core.

Uses a single Google AI Studio Gemini API call to synthesize a viral, high-converting
Threads thread (3-second hook root post + keyword/ticker-centric category replies + web CTA).
Includes a deterministic rule-based fallback if the API is unreachable.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional
import requests

from shared.env_utils import load_env_file
from shared.shared_logger import setup_logger

load_env_file()
logger = setup_logger("logs/threads_generator.log", __name__)

THREADS_SYSTEM_PROMPT = """You are an elite financial journalist and viral social media strategist on Meta Threads.
Transform the provided market intelligence report into a high-engagement, scannable Threads thread.

Formatting & Tone Rules:
1. Root Post:
   - MUST feature an irresistible 3-second hook (curiosity + high user gain / key market catalysts).
   - Summarize the top 3 market drivers concisely in bullet points.
   - End with an invitation to read the sector breakdown below (e.g. "👇 지금 시장 주도하는 10대 테마 & 핵심 종목 타래 정리").
   - DO NOT include external URLs in the root post (avoids reach penalty).

2. Thread Replies (One per category or bundled top categories, max 8-10 replies):
   - Keep it ultra-scannable for busy mobile users.
   - Format:
     (N/10) 🔹 [카테고리명]
     • 핵심 키워드: [키워드 2-3개]
     • 핵심 종목: $TICKER(종목명), $TICKER(종목명)
     • 1줄 핵심: [가장 중요한 팩트 한 문장]

3. Final CTA Reply:
   - Provide a compelling reason to visit the web app:
     "📊 오늘 분석된 300개 전체 기사 요약본과 종목별 타임라인 검색은 웹에서 무료로 확인하세요.\n🔗 {report_url}"

Return ONLY valid JSON matching this schema:
{
  "root_post": "string",
  "thread_replies": ["string", "string"],
  "cta_reply": "string"
}
"""


def call_gemini_for_threads(
    user_prompt: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Calls Gemini API with JSON response format to generate threads content."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not defined.")

    models_to_try = [
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
    ]

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": THREADS_SYSTEM_PROMPT}]},
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
        },
    }

    last_err = None
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        try:
            logger.info(f"Calling Gemini API for Threads with model: {model}...")
            resp = requests.post(url, json=payload, headers=headers, timeout=45)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            res_data = resp.json()

            candidates = res_data.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates returned from Gemini")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError("No parts found in Gemini candidate")

            text = ""
            for part in parts:
                if not part.get("thought"):
                    text = part.get("text", "").strip()
                    break
            if not text and parts:
                text = parts[0].get("text", "").strip()

            # Clean JSON markdown fences if present
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\n", "", text, flags=re.IGNORECASE)
                text = re.sub(r"\n```$", "", text)

            return json.loads(text)
        except Exception as e:
            logger.warning(f"Model {model} failed: {e}")
            last_err = e
            continue

    raise last_err or ValueError("All Gemini models failed")


def generate_fallback_threads(
    date_str: str,
    cio_points: List[str],
    categories_data: Dict[str, List[Dict[str, Any]]],
    report_url: str,
) -> Dict[str, Any]:
    """Deterministic fallback thread generator when LLM is unavailable."""
    # Root Post
    cio_summary = "\n".join([f"• {p}" for p in cio_points[:3]]) or "• 글로벌 시장 주요 지표 및 핵심 섹터 분석"
    root_post = (
        f"🚨 [{date_str} 장전 핵심 시그널]\n\n"
        f"오늘 미 증시 3줄 핵심 요약:\n{cio_summary}\n\n"
        f"👇 시장 주도 섹터 및 핵심 종목 타래 정리 🧵"
    )

    # Replies per category
    replies = []
    cat_items = list(categories_data.items())[:10]
    for idx, (cat_name, articles) in enumerate(cat_items, start=1):
        # Aggregate unique tickers
        tickers = []
        for art in articles:
            tickers.extend(art.get("tickers", []))
        tickers_unique = sorted(list(set(tickers)))[:4]
        ticker_str = ", ".join([f"${t}" for t in tickers_unique]) if tickers_unique else "관련 주요 종목"

        top_art = articles[0] if articles else {}
        top_title = top_art.get("title_ko") or top_art.get("title") or "섹터 주요 동향 분석"

        reply_text = (
            f"({idx}/{len(cat_items)}) 🔹 [{cat_name}]\n"
            f"• 핵심 종목: {ticker_str}\n"
            f"• 주요 이슈: {top_title}"
        )
        replies.append(reply_text)

    # CTA
    cta_reply = (
        f"📊 오늘 분석된 300개 전체 기사 요약본과 종목별 타임라인 검색은 웹에서 무료로 확인하세요.\n"
        f"🔗 {report_url}"
    )

    return {
        "root_post": root_post,
        "thread_replies": replies,
        "cta_reply": cta_reply,
    }


def generate_threads_content(
    structured_data: Dict[str, Any],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry point: Synthesizes Threads thread JSON from structured report data.
    Uses Gemini API with deterministic rule-based fallback.
    """
    date_str = structured_data.get("date", "")
    cio_points = structured_data.get("cio_points", [])
    articles = structured_data.get("articles", [])
    report_url = f"https://alphasignals.cloud/report/{date_str}"

    # Group articles by category
    categories_data: Dict[str, List[Dict[str, Any]]] = {}
    for art in articles:
        cat = art.get("category", "General")
        if cat not in categories_data:
            categories_data[cat] = []
        categories_data[cat].append(art)

    # Prepare concise summary text for LLM prompt
    summary_lines = []
    summary_lines.append(f"Date: {date_str}")
    summary_lines.append("CIO Points:")
    for p in cio_points:
        summary_lines.append(f"- {p}")

    summary_lines.append("\nTop Categories & Sample Articles:")
    for cat_name, cat_arts in list(categories_data.items())[:10]:
        all_tickers = []
        sample_titles = []
        for art in cat_arts[:3]:
            all_tickers.extend(art.get("tickers", []))
            t = art.get("title_ko") or art.get("title")
            if t:
                sample_titles.append(t)
        tickers_str = ", ".join(sorted(list(set(all_tickers)))[:5])
        summary_lines.append(
            f"Category [{cat_name}] (Tickers: {tickers_str}):\n"
            + "\n".join([f"  * {t}" for t in sample_titles])
        )

    user_prompt = (
        f"Generate a Threads thread based on this market report:\n\n"
        + "\n".join(summary_lines)
        + f"\n\nReport URL: {report_url}"
    )

    try:
        content = call_gemini_for_threads(user_prompt, api_key=api_key)
        if (
            isinstance(content, dict)
            and "root_post" in content
            and "thread_replies" in content
        ):
            # Ensure CTA is present
            if "cta_reply" not in content or not content["cta_reply"]:
                content["cta_reply"] = (
                    f"📊 오늘 분석된 300개 전체 기사 요약본과 종목별 타임라인 검색은 웹에서 무료로 확인하세요.\n🔗 {report_url}"
                )
            logger.info("Successfully generated Threads content via Gemini API.")
            return content
    except Exception as e:
        logger.warning(f"Gemini Threads generation failed, falling back to template: {e}")

    return generate_fallback_threads(
        date_str, cio_points, categories_data, report_url
    )
