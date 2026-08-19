"""
Threads Content Generator module for Alpha Signals Core.

Synthesizes high-engagement, single-topic deep dive narrative Threads (1 root post + 3 chained replies)
optimized for Korean retail investors before US market open (6 PM ~ 10 PM KST).
Includes an inline Thread-native CTA and deterministic fallback.
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

THREADS_SYSTEM_PROMPT = """You are an elite Wall Street financial analyst and viral social media strategist on Meta Threads writing for Korean retail investors before US market open (6 PM ~ 10 PM KST).
Transform the provided market intelligence report into a high-engagement, single-topic deep dive narrative thread (4 posts total: 1 root post + 3 chained replies).

Target Audience & Mindset:
- Korean investors preparing for tonight's US stock market open.
- They want a deep, compelling narrative on the #1 most impactful catalyst/theme from today's data, not a shallow catalog of 10 categories.

Formatting & Tone Rules:
1. Narrative Structure (Single Theme Deep-Dive, 4 posts total):
   - Select the SINGLE MOST CRITICAL catalyst/company/theme in today's report (e.g. Big Tech AI chip deal, major macro divergence, breakthrough earnings/SEC filing).
   - Post 1 (Root Post, 1/4):
     • 1~4 numbered concise Korean "음슴체" sentences (~임, ~했음, ~함).
     • Line 1: Provocative 3-second hook (counter-intuitive angle / curiosity gap).
     • Line 2: Concrete, verified factual catalyst (deal, contract, earnings, SEC filing).
     • Line 3: Deeper macro/industry reason why this is happening.
     • Line 4: Thread preview ending with "1/4" (e.g. "오늘 밤 개장 전 알아야 할 핵심 시그널 정리함. 1/4").
     • DO NOT include URLs in root post (avoids reach penalty).
   - Post 2 (Reply 1, 2/4 - 맥락 & 배경):
     • 5~7 numbered sentences. Why Big Tech/institutions are changing behavior and the underlying driver. Ending with "2/4".
   - Post 3 (Reply 2, 3/4 - 공급망 파급 효과):
     • 8~10 numbered sentences. Upstream/downstream winners/losers and supply chain shifts. Ending with "3/4".
   - Post 4 (Reply 3, 4/4 - 오늘 밤 실전 체크포인트 & 인라인 CTA):
     • 11~12 numbered sentences. What to monitor when the market opens tonight.
     • Inline 2-line Thread-native CTA:
       "유익했다면 하트 & 저장 누르고 오늘 밤 장 열릴 때 참고하길.\\n오늘 분석된 {total_articles}개 전체 시그널과 종목별 타임라인은 프로필 링크에서 무료로 확인 가능함. 4/4"

2. Strict Prohibitions:
   - NO Dollar Ticker Symbols: NEVER write $NVDA, $MRVL, $AAPL, etc. Use 100% natural Korean company names (엔비디아, 마벨 테크놀로지, 애플, 구글, 브로드컴).
   - NO Price/Quote Hallucination: DO NOT fabricate real-time stock prices or pre-market percentage swings (e.g., "장외 +14% 급등"). Rely ONLY on verified news facts, deal metrics, or SEC filings in the report.
   - NO Artificial Hashtags: Do not include compound tags like #LG엔비디아.
   - Tone MUST be concise, authoritative, natural Korean "음슴체" throughout.

Return ONLY valid JSON matching this schema:
{
  "root_post": "string (lines 1-4, ending with 1/4)",
  "thread_replies": [
    "string (lines 5-7, ending with 2/4)",
    "string (lines 8-10, ending with 3/4)",
    "string (lines 11-12 + inline CTA, ending with 4/4)"
  ]
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
    categories_data: Dict[str, List[Dict[str, Any]]],
    total_articles: int,
    report_url: str = "https://alphasignals.cloud",
) -> Dict[str, Any]:
    """Deterministic fallback thread generator in 4-post narrative format."""
    top_cat = next(iter(categories_data.keys()), "시장 동향")
    top_arts = categories_data.get(top_cat, [])
    top_title = (
        top_arts[0].get("title_ko") or top_arts[0].get("title")
        if top_arts
        else "미국 증시 주요 변동성"
    )
    second_title = (
        top_arts[1].get("title_ko") or top_arts[1].get("title")
        if len(top_arts) > 1
        else "빅테크 공급망 수급 변화"
    )

    root_post = (
        f"1. 오늘 밤 미국장에서 가장 주목해야 할 핵심 이슈가 있음.\n\n"
        f"2. {top_title}.\n\n"
        f"3. 단순 일회성 이슈가 아니라 관련 공급망 전반에 자금이 이동하는 구조적 신호임.\n\n"
        f"4. 오늘 밤 개장 전 알아야 할 핵심 시그널 정리함. 1/4"
    )

    reply_1 = (
        f"5. {second_title}.\n\n"
        f"6. 기관 투자자들과 주요 기업들이 리스크 관리와 포트폴리오 재편에 나서는 배경임.\n\n"
        f"7. 장 시작 전 글로벌 매크로와 자금 흐름을 선제적으로 읽는 것이 중요함. 2/4"
    )

    reply_2 = (
        "8. 관련 산업군 전반에서 실적과 수주 모멘텀이 있는 핵심 기업들로 차별화 장세가 예상됨.\n\n"
        "9. 단기 변동성보다는 구조적인 성장성과 수급 방향성에 집중할 필요가 있음.\n\n"
        "10. 개장 직후 초기 수급 쏠림 현상을 주시해야 함. 3/4"
    )

    reply_3 = (
        f"11. 오늘 밤 미국장에서는 주요 지수와 함께 해당 테마 선도 기업들의 거래량을 필수 체크하길.\n\n"
        f"12. 장중 발표될 경제 지표와 기업 코멘트에 따라 변동성이 확대될 수 있음.\n\n"
        f"유익했다면 하트 & 저장 누르고 오늘 밤 장 열릴 때 참고하길.\n"
        f"오늘 분석된 {total_articles}개 전체 시그널과 종목별 타임라인은 프로필 링크에서 무료로 확인 가능함. 4/4"
    )

    return {
        "root_post": root_post,
        "thread_replies": [reply_1, reply_2, reply_3],
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
    articles = structured_data.get("articles", [])
    total_articles = structured_data.get("total_articles", len(articles))
    report_url = "https://alphasignals.cloud"

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
    summary_lines.append(f"Total Articles Analyzed: {total_articles}")

    summary_lines.append("\nTop Categories & Sample Headlines:")
    for cat_name, cat_arts in list(categories_data.items())[:10]:
        sample_titles = []
        for art in cat_arts[:3]:
            t = art.get("title_ko") or art.get("title")
            content = art.get("content_ko") or art.get("content") or ""
            if t:
                snippet = f"{t}: {content[:100]}" if content else t
                sample_titles.append(snippet)
        summary_lines.append(
            f"Category [{cat_name}]:\n" + "\n".join([f"  * {t}" for t in sample_titles])
        )

    expected_cta = f"유익했다면 하트 & 저장 누르고 오늘 밤 장 열릴 때 참고하길.\n오늘 분석된 {total_articles}개 전체 시그널과 종목별 타임라인은 프로필 링크에서 무료로 확인 가능함. 4/4"

    user_prompt = (
        "Generate a 4-post narrative Threads thread for Korean investors preparing for tonight's US market open based on this market report:\n\n"
        + "\n".join(summary_lines)
        + f"\n\nTotal Articles: {total_articles}"
        + f"\nRequired Inline CTA for Post 4:\n{expected_cta}"
    )

    try:
        content = call_gemini_for_threads(user_prompt, api_key=api_key)
        if (
            isinstance(content, dict)
            and "root_post" in content
            and "thread_replies" in content
            and len(content["thread_replies"]) >= 1
        ):
            # Ensure inline CTA is present in the final reply
            last_reply = content["thread_replies"][-1]
            if (
                f"{total_articles}개" not in last_reply
                and "프로필 링크" not in last_reply
            ):
                content["thread_replies"][-1] = (
                    f"{last_reply}\n\n"
                    f"유익했다면 하트 & 저장 누르고 오늘 밤 장 열릴 때 참고하길.\n"
                    f"오늘 분석된 {total_articles}개 전체 시그널과 종목별 타임라인은 프로필 링크에서 무료로 확인 가능함. 4/4"
                )
            logger.info("Successfully generated Threads content via Gemini API.")
            return content
    except Exception as e:
        logger.warning(
            f"Gemini Threads generation failed, falling back to template: {e}"
        )

    return generate_fallback_threads(
        date_str, categories_data, total_articles, report_url
    )
