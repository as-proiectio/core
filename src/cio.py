"""
CIO module for Alpha Signals Core.

This module synthesizes daily reports with a professional, billionaire-mentor styled narrative
(Daily Point) and executes the Premarket 3D news selection system (Macro, Surprise, Trend).
It leverages Google AI Studio's Gemini 3.5 Flash API (with Ollama llama3.1 fallbacks)
for premarket scoring and floor/ceiling news limits.
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Optional
import pytz
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.env_utils import load_env_file
from shared.shared_logger import setup_logger
from shared.time_utils import parse_utc_time

# Populate os.environ with local config prior to bootstrapping dependencies.
load_env_file()

LOG_FILE = "logs/cio.log"

logger = setup_logger(LOG_FILE, __name__)

# ==============================================================================
# 1. AI Prompts & Templates (Dynamic & Fallback Configurations)
# ==============================================================================

# Minimal fallback prompts for the open-source release.
# These provide basic commentary output without our proprietary billionaire-mentor style rules.
DAILY_COMMENTARY_SYSTEM_PROMPT = """You are a financial commentary generator.
Provide a clear, brief analysis of the market indicators, weekly schedules, and extracted facts."""

DAILY_COMMENTARY_USER_PROMPT_TEMPLATE = """
Write a daily market commentary report based on:
Market Indicators: {market_indicators_text}
Weekly Schedule: {weekly_schedule_text}
Extracted Facts: {extracted_facts_text}
"""

PREMARKET_SELECTION_SYSTEM_PROMPT = """You are a Chief Investment Officer selecting the most critical premarket news articles.
Filter out noise and return the article IDs that are most likely to move sectors or the general market today."""

PREMARKET_SELECTION_USER_PROMPT_GEMINI = """
Evaluate the following news articles and select the top IDs:
{articles_text_for_prompt}

Return a JSON object: {{"selected_ids": [1, 2, 3]}}
"""

PREMARKET_SELECTION_USER_PROMPT_OLLAMA = """
Select the top critical news from:
{extracted_facts_text}
"""

# Load the proprietary billionaire-mentor style prompts dynamically if they exist.
# The 'config/' directory is gitignored to protect commercial intellectual property (IP).
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cio_config_path = os.path.join(
        project_root, "config", "prompts", "cio_prompts.json"
    )
    if os.path.exists(cio_config_path):
        with open(cio_config_path, "r", encoding="utf-8") as f:
            custom_cio = json.load(f)
            if custom_cio and isinstance(custom_cio, dict):
                DAILY_COMMENTARY_SYSTEM_PROMPT = custom_cio.get(
                    "DAILY_COMMENTARY_SYSTEM_PROMPT", DAILY_COMMENTARY_SYSTEM_PROMPT
                )
                DAILY_COMMENTARY_USER_PROMPT_TEMPLATE = custom_cio.get(
                    "DAILY_COMMENTARY_USER_PROMPT_TEMPLATE",
                    DAILY_COMMENTARY_USER_PROMPT_TEMPLATE,
                )
                PREMARKET_SELECTION_SYSTEM_PROMPT = custom_cio.get(
                    "PREMARKET_SELECTION_SYSTEM_PROMPT",
                    PREMARKET_SELECTION_SYSTEM_PROMPT,
                )
                PREMARKET_SELECTION_USER_PROMPT_GEMINI = custom_cio.get(
                    "PREMARKET_SELECTION_USER_PROMPT_GEMINI",
                    PREMARKET_SELECTION_USER_PROMPT_GEMINI,
                )
                PREMARKET_SELECTION_USER_PROMPT_OLLAMA = custom_cio.get(
                    "PREMARKET_SELECTION_USER_PROMPT_OLLAMA",
                    PREMARKET_SELECTION_USER_PROMPT_OLLAMA,
                )
except Exception:
    # Fail silently to maintain engine robustness
    pass

# ==============================================================================
# 2. Helper Functions
# ==============================================================================


def format_market_map(data: dict, display_only: bool = False) -> str:
    """
    Format the market map. If display_only is True, return only Dow, S&P, Nasdaq, Bitcoin.
    Otherwise, return a comprehensive text representation of sectors and industries for the LLM.
    """
    if not data or "market_map" not in data:
        if "market_indicators" in data:
            market_map = {"Indices": data["market_indicators"], "Sectors": {}}
        else:
            return ""
    else:
        market_map = data["market_map"]

    lines = []

    # 1. Major Indices
    indices = market_map.get("Indices", {})
    display_keys = ["Dow Jones", "S&P 500", "Nasdaq", "Bitcoin"]

    for name in display_keys:
        if name in indices:
            info = indices[name]
            price = info.get("price", "0")
            change = info.get("change", "0")

            try:
                change_float = float(str(change).replace("%", "").strip())
                change_str = (
                    f"+{change_float}%" if change_float > 0 else f"{change_float}%"
                )
            except ValueError:
                change_str = f"{change}"
                if not change_str.endswith("%"):
                    change_str += "%"

            lines.append(f"_ {name} {price} ({change_str})")

    if display_only:
        return "\n".join(lines)

    # 2. Complete Sector/Industry Heatmap for AI Context
    lines.append("\n[Detailed Market Map (S&P 500 + Target Tickers)]")
    sectors = market_map.get("Sectors", {})
    for sec_name, sec_data in sectors.items():
        lines.append(f"\nSector: {sec_name} (Avg: {sec_data.get('sector_avg')})")
        for ind_name, ind_data in sec_data.get("industries", {}).items():
            lines.append(
                f"  Industry: {ind_name} (Avg: {ind_data.get('industry_avg')})"
            )
            stocks_line = []
            for t_name, t_data in ind_data.get("details", {}).items():
                stocks_line.append(f"{t_name}: {t_data.get('change')}")
            lines.append(f"    Stocks: {', '.join(stocks_line)}")

    return "\n".join(lines)


def format_weekly_schedule(data: dict, today_str: Optional[str] = None) -> str:
    """
    Format weekly schedule into a strict 7-day rolling calendar starting from today_str.
    For the English report, events are grouped and dates formatted using America/New_York timezone.
    All-day UTC midnight (00:00:00) events keep their date untouched to prevent timezone shifts.
    """
    events = []
    if data and "weekly_schedule" in data:
        events = data["weekly_schedule"] or []

    ny_tz = pytz.timezone("America/New_York")

    # Determine base date
    if today_str:
        try:
            base_date = datetime.strptime(today_str, "%Y%m%d").date()
        except ValueError:
            base_date = datetime.now(ny_tz).date()
    else:
        base_date = datetime.now(ny_tz).date()

    # Generate exactly 7 consecutive dates
    target_dates = [base_date + timedelta(days=i) for i in range(7)]
    grouped_events = {d: [] for d in target_dates}

    for event in events:
        utc_time_str = event.get("utc_time")
        if not utc_time_str:
            continue

        utc_dt = parse_utc_time(utc_time_str)

        # TIMEZONE MIDNIGHT WORKAROUND:
        # All-day macro events or market holidays are often scheduled at exactly UTC 00:00:00.
        # Converting them to New York timezone (EST/EDT) shifts the timestamp back by 4-5 hours,
        # which incorrectly places the event on the previous day. To prevent this date distortion,
        # we bypass the timezone shift for midnight events and retain the raw UTC date directly.
        if utc_dt.hour == 0 and utc_dt.minute == 0 and utc_dt.second == 0:
            local_date = utc_dt.date()
        else:
            local_dt = utc_dt.astimezone(ny_tz)
            local_date = local_dt.date()

        # Format event description
        currency = event.get("currency", "USD")
        importance = event.get("importance", "medium")
        name = event.get("name", "").strip()

        if importance == "holiday":
            evt_str = f"Holiday - {name}"
        elif importance == "earnings":
            evt_str = f"{name}"
        else:
            evt_str = f"({currency}) {name}"

        # Only group if it falls within the 7-day target range
        if local_date in grouped_events:
            grouped_events[local_date].append(evt_str)

    # Reconstruct lines sorted by local date chronologically
    lines = []
    for d in target_dates:
        d_str = d.strftime("%d %b (%A)")
        lines.append(d_str)
        for evt in grouped_events[d]:
            lines.append(evt)
        lines.append("")

    return "\n".join(lines).strip()


def call_gemini_api(
    sys_prompt: str, user_prompt: str, response_mime_type: str = "text/plain"
) -> str:
    """
    Calls the Gemini API (REST endpoint) using the provided system and user prompts.
    Tries early access gemini-3.5-flash first, then falls back gracefully to previous generations.
    """
    # Fetch the Gemini API Key from environment variables.
    # If the key is not defined, we raise a ValueError to prevent empty/failing API calls.
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not defined.")

    models_to_try = [
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    last_err = None
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "systemInstruction": {"parts": [{"text": sys_prompt}]},
            "generationConfig": {
                "temperature": 0.0 if response_mime_type == "application/json" else 0.4,
                "responseMimeType": response_mime_type,
            },
        }

        try:
            logger.info(f"Calling Gemini API with model: {model} ...")
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            if response.status_code == 404:
                logger.warning(f"Model {model} returned 404. Trying fallback...")
                continue
            response.raise_for_status()
            res_data = response.json()

            candidates = res_data.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates returned from Gemini API")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError("No parts found in the first candidate")

            content_text = ""
            for part in parts:
                if not part.get("thought"):
                    content_text = part.get("text", "").strip()
                    break
            if not content_text and parts:
                content_text = parts[0].get("text", "").strip()
            return content_text
        except Exception as e:
            logger.error(f"Error calling Gemini API with model {model}: {e}")
            last_err = e
            continue

    if last_err:
        raise last_err
    else:
        raise ValueError("Failed to call Gemini API: All models failed.")


def clean_json_response(response_text: str) -> str:
    """
    Cleans any markdown blocks (```json ... ```) from the LLM response.
    """
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def parse_facts_into_articles(facts_text: str) -> list[dict]:
    """
    Parses facts_text (extracted_facts_YYYYMMDD.txt) into a list of articles.
    """
    blocks = re.split(r"\n\n+", facts_text)
    articles = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith("### "):
            continue

        match = re.match(r"^\[(.*?)\]\((.*?)\)(?:\n([\s\S]*))?$", block)
        if match:
            title = match.group(1).strip()
            url = match.group(2).strip()
            body = match.group(3).strip() if match.group(3) else ""
            articles.append(
                {"title": title, "url": url, "body": body, "raw_block": block}
            )

    return articles


def generate_daily_commentary_gemini(
    market_text: str,
    schedule_text: str,
    facts_text: str,
) -> str:
    """
    Generate Daily Point narrative commentary using Gemini 3.5 Flash.
    """
    sys_prompt = DAILY_COMMENTARY_SYSTEM_PROMPT
    user_prompt = DAILY_COMMENTARY_USER_PROMPT_TEMPLATE.format(
        market_indicators_text=market_text,
        weekly_schedule_text=schedule_text,
        extracted_facts_text=facts_text,
    )

    return call_gemini_api(sys_prompt, user_prompt, response_mime_type="text/plain")


def generate_daily_commentary_ollama(
    market_text: str,
    schedule_text: str,
    facts_text: str,
) -> str:
    """
    Generate Daily Point narrative commentary using local Ollama.
    """
    sys_prompt = DAILY_COMMENTARY_SYSTEM_PROMPT
    user_prompt = DAILY_COMMENTARY_USER_PROMPT_TEMPLATE.format(
        market_indicators_text=market_text,
        weekly_schedule_text=schedule_text,
        extracted_facts_text=facts_text,
    )

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.1",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "num_ctx": 16384,
            "num_predict": 3000,
            "temperature": 0.4,
            "top_p": 0.9,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=None)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.error(f"Failed to communicate with local Ollama: {e}")
        raise e


def select_premarket_news_gemini(facts_text: str) -> str:
    """
    Selects the top 5 to 12 most critical news items for the premarket briefing using Gemini 3.5 Flash.
    Parses facts into articles, assigns IDs, calls Gemini API in JSON Mode, and reconstructs the output in Python.

    The model evaluates news based on a 3-Dimension Scoring System: Macro Impact, Surprise Factor, and Catalyst Urgency.
    It targets articles scoring 12+ points, padding the selection to a minimum (floor) of 5 items if too few exist,
    or capping it at a maximum (ceiling) of 12 items to maintain report density.
    """
    articles = parse_facts_into_articles(facts_text)
    if not articles:
        logger.warning("No parsed articles found in facts text.")
        return "No articles available for selection."

    formatted_articles = []
    for idx, art in enumerate(articles, 1):
        formatted_articles.append(
            f"--- ARTICLE ID: {idx} ---\n"
            f"Title: {art['title']}\n"
            f"URL: {art['url']}\n"
            f"Body: {art['body']}\n"
        )
    articles_text_for_prompt = "\n".join(formatted_articles)

    user_prompt = PREMARKET_SELECTION_USER_PROMPT_GEMINI.format(
        articles_text_for_prompt=articles_text_for_prompt
    )

    gemini_response = call_gemini_api(
        sys_prompt=PREMARKET_SELECTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_mime_type="application/json",
    )

    cleaned_res = clean_json_response(gemini_response)
    logger.info(f"Received premarket response from Gemini: {cleaned_res}")
    res_data = json.loads(cleaned_res)
    selected_ids = res_data.get("selected_ids", [])

    selected_blocks = []
    for s_id in selected_ids:
        try:
            s_idx = int(s_id) - 1
            if 0 <= s_idx < len(articles):
                selected_blocks.append(articles[s_idx]["raw_block"])
        except (ValueError, TypeError):
            continue

    if not selected_blocks:
        raise ValueError("No valid article IDs were returned or resolved from Gemini.")

    selected_news = "\n\n".join(selected_blocks)
    logger.info(
        f"Successfully selected and merged {len(selected_blocks)} premarket news items using Gemini."
    )
    return selected_news


def select_premarket_news_ollama(facts_text: str) -> str:
    """
    Selects the top 5 to 12 most critical news items for the premarket briefing using Ollama.
    """
    sys_prompt = PREMARKET_SELECTION_SYSTEM_PROMPT
    user_prompt = PREMARKET_SELECTION_USER_PROMPT_OLLAMA.format(
        extracted_facts_text=facts_text
    )

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.1",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "num_ctx": 16384,
            "num_predict": 2000,
            "temperature": 0.0,
            "top_p": 0.1,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=None)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.error(f"Failed to communicate with local Ollama: {e}")
        raise e


# ==============================================================================
# 3. Main Executable Pipelines
# ==============================================================================


def run_full_cio(today_str: str, data_dir: str):
    news_file = os.path.join(data_dir, f"daily_news_{today_str}.json")
    output_file = os.path.join(data_dir, f"final_report_{today_str}.txt")
    facts_file = os.path.join(data_dir, f"extracted_facts_{today_str}.txt")

    # Fallback to finding the latest files if today's don't exist
    if not os.path.exists(news_file):
        json_files = sorted(glob.glob(os.path.join(data_dir, "daily_news_*.json")))
        if json_files:
            news_file = json_files[-1]

    if not os.path.exists(facts_file):
        txt_files = sorted(glob.glob(os.path.join(data_dir, "extracted_facts_*.txt")))
        if txt_files:
            facts_file = txt_files[-1]

    # Load market indicators and weekly calendar feed
    data = {}
    if os.path.exists(news_file):
        logger.info(f"Loading market data from {news_file}")
        with open(news_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from {news_file}")
    else:
        logger.warning(
            f"{news_file} not found. Proceeding with empty market/schedule data."
        )

    # Load semantic extracted facts
    facts_text = ""
    if os.path.exists(facts_file):
        logger.info(f"Loading extracted facts from {facts_file}")
        with open(facts_file, "r", encoding="utf-8") as f:
            facts_text = f.read()
    else:
        logger.warning(f"{facts_file} not found. Proceeding with empty facts.")

    # Format components
    market_text_for_prompt = format_market_map(data, display_only=False)
    market_text_for_report = format_market_map(data, display_only=True)
    schedule_text = format_weekly_schedule(data, today_str)

    # Generate Narrative Commentary
    logger.info("Generating full report AI commentary from Gemini 3.5 Flash...")
    try:
        commentary = generate_daily_commentary_gemini(
            market_text_for_prompt,
            schedule_text,
            facts_text,
        )
        logger.info("Successfully generated AI commentary using Gemini.")
    except Exception as e:
        logger.warning(
            f"Failed to generate AI commentary using Gemini API: {e}. Falling back to local Ollama..."
        )
        try:
            commentary = generate_daily_commentary_ollama(
                market_text_for_prompt,
                schedule_text,
                facts_text,
            )
            logger.info(
                "Successfully generated AI commentary using local Ollama fallback."
            )
        except Exception as fallback_err:
            logger.error(f"Full report commentary fallback also failed: {fallback_err}")
            commentary = "Error generating commentary."

    # Merge content into final report format
    logger.info("Merging content into final report format...")

    # Normalize: ensure exactly one newline (\n) between the [title](url) link and its body text
    facts_text = re.sub(
        r"(\[.*?\]\(https?://.*?\))\s*\n+(?=[^\[])",
        r"\1\n",
        facts_text,
    )

    report = (
        f"## {today_str}\n\n"
        "### Daily Point\n"
        f"{market_text_for_report}\n\n"
        f"{commentary}\n\n"
        "### Weekly Schedule\n"
        f"{schedule_text}\n\n"
        f"{facts_text}"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Report successfully generated and saved to {output_file}")


def run_premarket_cio(today_str: str, data_dir: str):
    output_file = os.path.join(data_dir, f"premarket_report_{today_str}.txt")
    facts_file = os.path.join(data_dir, f"extracted_facts_{today_str}.txt")

    if not os.path.exists(facts_file):
        txt_files = sorted(glob.glob(os.path.join(data_dir, "extracted_facts_*.txt")))
        if txt_files:
            facts_file = txt_files[-1]

    facts_text = ""
    if os.path.exists(facts_file):
        logger.info(f"Loading extracted facts from {facts_file}")
        with open(facts_file, "r", encoding="utf-8") as f:
            facts_text = f.read()
    else:
        logger.warning(f"{facts_file} not found. Proceeding with empty facts.")

    # Select critical Premarket news items
    logger.info("Selecting premarket news using Gemini...")
    try:
        selected_news = select_premarket_news_gemini(facts_text)
    except Exception as e:
        logger.warning(
            f"Failed to select premarket news using Gemini API: {e}. Falling back to local Ollama..."
        )
        try:
            selected_news = select_premarket_news_ollama(facts_text)
            logger.info(
                "Successfully selected premarket news using local Ollama fallback."
            )
        except Exception as fallback_err:
            logger.error(f"Premarket selection fallback also failed: {fallback_err}")
            selected_news = "Error selecting news."

    # Compile premarket report
    logger.info("Merging content into premarket report format...")

    # Normalize: ensure exactly one newline (\n) between the [title](url) link and its body text
    selected_news = re.sub(
        r"(\[.*?\]\(https?://.*?\))\s*\n+(?=[^\[])",
        r"\1\n",
        selected_news,
    )

    report = f"## {today_str} Premarket\n\n{selected_news}"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Premarket report successfully generated and saved to {output_file}")


def run_cio(report_type: str = "full", target_date: Optional[str] = None):
    try:
        us_tz = pytz.timezone("America/New_York")
        if target_date:
            today_str = target_date
        else:
            today_str = datetime.now(us_tz).strftime("%Y%m%d")

        logger.info(f"Starting CIO Pipeline for {today_str} (Type: {report_type})")

        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        os.makedirs(data_dir, exist_ok=True)

        if report_type == "premarket":
            run_premarket_cio(today_str, data_dir)
        else:
            run_full_cio(today_str, data_dir)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Process terminating.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIO Report Generator")
    parser.add_argument(
        "--type",
        choices=["full", "premarket"],
        default="full",
        help="Type of report to generate",
    )
    args = parser.parse_args()
    run_cio(report_type=args.type)
