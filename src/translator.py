"""
Translator module for Alpha Signals Core.

This module translates English markdown reports to publication-ready Korean reports.
It leverages Google AI Studio's Gemini and Gemma model chain (defaulting to gemini-3-1-flash-lite) in JSON mode,
maintains translation caches to minimize redundant translation costs, and localizes schedules.
"""

import argparse
import glob
import json
import os
import re
import signal
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import pytz
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formatter import run_formatter
from translation_cleaner import TranslationCleaner
from shared.env_utils import load_env_file
from shared.shared_logger import setup_logger


class TranslationError(Exception):
    """Raised when translation API or translation logic fails."""

    pass


# Populate os.environ with local config prior to bootstrapping dependencies.
load_env_file()

LOG_FILE = "logs/translator.log"
logger = setup_logger(LOG_FILE, __name__)

# Model fallback chain configuration
# Can be overridden via TRANSLATOR_MODEL env var (e.g. TRANSLATOR_MODEL="gemma-4-31b,gemma-4-26b")
env_model = os.environ.get("TRANSLATOR_MODEL", "")
if env_model:
    MODEL_CHAIN = [m.strip() for m in env_model.split(",") if m.strip()]
else:
    MODEL_CHAIN = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemma-4-31b-it",
        "gemma-4-26b-a4b-it",
    ]

# Cooldown sleep between split batch runs to respect rate limits
COOLDOWN_SLEEP_SECONDS = 2.0

# Category translation mapping: Translates English section headers to their official Korean equivalent
# used for publish-ready capital market reports in South Korea (e.g. "### General" -> "### 경제 일반").
CATEGORY_MAPPING = {
    "### General": "### 경제 일반",
    "### Bitcoin": "### 비트코인",
    "### Semiconductor": "### 반도체",
    "### AI/Robotics/EV": "### AI / 로봇 / EV",
    "### Power/Grid": "### 전력 / 인프라",
    "### Software": "### 소프트웨어",
    "### Aerospace": "### 우주 항공",
    "### Bio": "### 바이오",
    "### Consumer/Retail": "### 소비재 / 리테일",
    "### Others": "### 기타",
}


def normalize_category_header(header: str) -> str:
    """Normalize headers by removing whitespace and slashes for stable lookup."""
    return header.replace(" ", "").replace("/", "").strip()


# Precompute a normalized lookup mapping to resolve raw spaces and forward slashes safely.
NORM_MAPPING = {normalize_category_header(k): v for k, v in CATEGORY_MAPPING.items()}


def parse_article_block(block: str) -> tuple[str, str, str]:
    """
    Parses an article block into (url, title, body).
    Example: [Apple Sinks 6%...](https://...)\nBody text here...
    """
    match = re.match(r"^\[(.*?)\]\((.*?)\)(?:\s*<br\s*/?>)?(.*)", block, re.DOTALL)
    if match:
        title = match.group(1).strip()
        url = match.group(2).strip()
        body = match.group(3).strip()
        return url, title, body
    return "", "", ""


# Minimal fallback system prompt for the open-source release.
# Instructs the model to return JSON translations without complex capital-market noise filters.
DEFAULT_TRANSLATION_SYSTEM_PROMPT = """You are a professional financial translator specializing in translating US financial news and articles into high-quality, professional Korean.
You MUST output your response strictly as a JSON object containing a "translations" key mapped to an array of translated articles. Each item in the array must preserve the "url" key and contain the translated "title" and "body" keys."""

# Load the proprietary translation system prompt dynamically if it exists.
# Since config/ is gitignored, this allows local runs to keep proprietary translation IP.
TRANSLATION_SYSTEM_PROMPT = DEFAULT_TRANSLATION_SYSTEM_PROMPT

DAILY_POINT_TRANSLATION_SYSTEM_PROMPT = """You are a professional financial translator specializing in translating Chief Investment Officer (CIO) market commentaries and topline signals into high-quality, professional Korean.
You MUST preserve all markdown formatting, including bold text (**text**), bullet points (- ), line breaks, paragraph structure, and HTML breaks (<br />). Do NOT merge paragraphs or remove bullet points.
You MUST output your response strictly as a JSON object containing a "translations" key mapped to an array of translated articles. Each item in the array must preserve the "url" key and contain the translated "title" and "body" keys."""

try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_file_path = os.path.join(
        project_root, "config", "prompts", "translation_system_prompt.txt"
    )
    if os.path.exists(prompt_file_path):
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            custom_prompt = f.read().strip()
            if custom_prompt:
                TRANSLATION_SYSTEM_PROMPT = custom_prompt
except Exception:
    # Fail silently to maintain execution flow with the default prompt
    pass


def clean_body_newlines(body: str) -> str:
    """
    Replaces newlines and HTML breaks inside the article body with periods
    and joins them into a single continuous paragraph.
    """
    if not body:
        return ""
    # Replace '<br />', '<br>', '<br/>' with newlines first to normalize
    body_clean = re.sub(r"<br\s*/?>", "\n", body)

    # Split the body by newlines
    lines = body_clean.split("\n")
    cleaned_parts = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        # If it doesn't end with a sentence ending punctuation, append a period
        if not line_str.endswith((".", "!", "?", "。")):
            line_str += "."

        cleaned_parts.append(line_str)

    # Join with a single space
    result = " ".join(cleaned_parts)

    # Clean up any consecutive periods like ".. " or "..." to a single period "."
    # Also clean up consecutive spaces
    result = re.sub(r"\.+", ".", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def load_translation_rules() -> dict:
    """
    Loads translation rules (dynamic guidance, literal replacements, regex replacements, pre-processing rules)
    from shared/default_translation_rules.json and merges them with config/custom_translation_rules.json if present.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_path = os.path.join(
        project_root, "shared", "default_translation_rules.json"
    )
    custom_path = os.path.join(project_root, "config", "custom_translation_rules.json")

    rules = {
        "dynamic_guidance": [],
        "literal_replacements": {},
        "regex_replacements": [],
        "pre_processing_rules": [],
    }

    # 1. Load defaults
    if os.path.exists(default_path):
        try:
            with open(default_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "dynamic_guidance" in data and isinstance(
                    data["dynamic_guidance"], list
                ):
                    rules["dynamic_guidance"].extend(data["dynamic_guidance"])
                if "pre_processing_rules" in data and isinstance(
                    data["pre_processing_rules"], list
                ):
                    rules["pre_processing_rules"].extend(data["pre_processing_rules"])
                if "regex_replacements" in data and isinstance(
                    data["regex_replacements"], list
                ):
                    rules["regex_replacements"].extend(data["regex_replacements"])
                if "literal_replacements" in data and isinstance(
                    data["literal_replacements"], dict
                ):
                    rules["literal_replacements"].update(data["literal_replacements"])
        except Exception as e:
            logger.error(f"Failed to load default translation rules: {e}")

    # 2. Load custom/proprietary overrides and merge
    if os.path.exists(custom_path):
        try:
            with open(custom_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "dynamic_guidance" in data and isinstance(
                    data["dynamic_guidance"], list
                ):
                    rules["dynamic_guidance"].extend(data["dynamic_guidance"])
                if "pre_processing_rules" in data and isinstance(
                    data["pre_processing_rules"], list
                ):
                    rules["pre_processing_rules"].extend(data["pre_processing_rules"])
                if "regex_replacements" in data and isinstance(
                    data["regex_replacements"], list
                ):
                    rules["regex_replacements"].extend(data["regex_replacements"])
                if "literal_replacements" in data and isinstance(
                    data["literal_replacements"], dict
                ):
                    rules["literal_replacements"].update(data["literal_replacements"])
        except Exception as e:
            logger.error(f"Failed to load custom translation rules: {e}")

    return rules


def pre_process_articles(
    articles: List[Dict[str, str]], rules: dict
) -> List[Dict[str, str]]:
    """
    Applies pre-processing rules (e.g. normalizing bps, bp, %p) to English titles and bodies.
    """
    processed = []
    pre_rules = rules.get("pre_processing_rules", [])

    for art in articles:
        title = art.get("title", "")
        body = art.get("body", "")

        # Apply configured pre-processing rules
        for rule in pre_rules:
            pattern = rule.get("pattern")
            repl = rule.get("replacement")
            if pattern and repl is not None:
                title = re.sub(pattern, repl, title, flags=re.IGNORECASE)
                body = re.sub(pattern, repl, body, flags=re.IGNORECASE)

        processed.append({"url": art["url"], "title": title, "body": body})
    return processed


def post_process_translation(
    ko_title: str, ko_body: str, rules: dict, preserve_newlines: bool = False
) -> tuple[str, str]:
    """
    Applies post-processing rules (newline cleaning, literal & regex replacements) to Korean titles and bodies.
    """
    # 1. Clean body newlines (Requirement 7)
    if not preserve_newlines:
        ko_body = clean_body_newlines(ko_body)

    # 2. Apply literal replacements
    literal_repls = rules.get("literal_replacements", {})
    for target, repl in literal_repls.items():
        ko_title = ko_title.replace(target, repl)
        ko_body = ko_body.replace(target, repl)

    # 3. Apply regex replacements
    regex_repls = rules.get("regex_replacements", [])
    for r in regex_repls:
        pattern = r.get("pattern")
        repl = r.get("replacement")
        if pattern and repl is not None:
            ko_title = re.sub(pattern, repl, ko_title)
            ko_body = re.sub(pattern, repl, ko_body)

    # 4. Apply modular TranslationCleaner filter
    ko_title = TranslationCleaner.clean(ko_title)
    ko_body = TranslationCleaner.clean(ko_body)

    return ko_title.strip(), ko_body.strip()


def detect_dynamic_guidelines(articles: List[Dict[str, str]], rules: dict) -> str:
    """
    Scans a batch of articles to dynamically construct translation guidance
    to prevent common LLM financial translation errors.
    """
    guidelines = []

    # Concatenate all titles and bodies for scanning
    combined_text = ""
    for art in articles:
        combined_text += f"\n{art.get('title', '')}\n{art.get('body', '')}"
    combined_text_lower = combined_text.lower()

    # Check configured guidelines
    dynamic_rules = rules.get("dynamic_guidance", [])
    for r in dynamic_rules:
        keywords = r.get("keywords", [])
        rule_text = r.get("rule", "")
        if any(kw.lower() in combined_text_lower for kw in keywords):
            guidelines.append(rule_text)

    # Always check general economic context as fallback / backup
    if any(
        keyword in combined_text_lower
        for keyword in [
            "household",
            "consumer",
            "spending",
            "inflation",
            "gdp",
            "fed",
            "rate",
            "economy",
        ]
    ):
        guidelines.append(
            "- General Economy/Consumer Context: For macroeconomic, retail, or general household data, "
            "carefully analyze the sentence structure to understand who is spending, what they are spending on, and the overall context. "
            "Avoid literal translations that result in nonsensical Korean sentences."
        )

    if guidelines:
        guidelines_str = (
            "\n[Dynamic Translation Guidance for this Batch]\n" + "\n".join(guidelines)
        )
        return guidelines_str
    return ""


def build_payload(
    user_prompt: str, model: str, system_prompt: Optional[str] = None
) -> dict:
    """Builds the API request payload dynamically based on model capabilities."""
    generation_config: Dict[str, Any] = {
        "temperature": 0.1,
        "responseMimeType": "application/json",
    }

    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {
            "parts": [{"text": system_prompt or TRANSLATION_SYSTEM_PROMPT}]
        },
        "generationConfig": generation_config,
    }

    return payload


def call_gemini_translator_api(
    articles: List[Dict[str, str]],
    retries_per_model: int = 1,
    backoff_factor: int = 5,
    custom_system_prompt: Optional[str] = None,
    preserve_newlines: bool = False,
    *args,
    **kwargs,
) -> Dict[str, Dict[str, str]]:
    """
    Calls Google AI Studio's API to translate a batch of articles in JSON mode.
    Tries each model in MODEL_CHAIN sequentially. If a model fails, is rate-limited,
    or is not found, falls back to the next model in the chain.
    Uses exponential backoff: 5s -> 15s -> 45s between retries.
    """
    # Fetch the Gemini API Key from environment variables.
    # If the key is not defined, we raise a ValueError to prevent empty/failing API calls.
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise TranslationError("GEMINI_API_KEY environment variable is not defined.")
    headers = {"Content-Type": "application/json"}

    # Load translation rules (default and custom)
    rules = load_translation_rules()

    # Map original URLs to short placeholders (id-0, id-1, etc.) to prevent LLM URL typos/mismatches
    placeholder_to_original_url = {}
    temp_articles = []
    for idx, art in enumerate(articles):
        placeholder = f"id-{idx}"
        placeholder_to_original_url[placeholder] = art["url"]
        temp_articles.append(
            {"url": placeholder, "title": art["title"], "body": art["body"]}
        )

    # Pre-process the English articles with temporary placeholder URLs
    processed_articles = pre_process_articles(temp_articles, rules)
    user_prompt = json.dumps(processed_articles, ensure_ascii=False)

    # Detect dynamic guidelines based on the pre-processed articles
    dynamic_guidelines = detect_dynamic_guidelines(processed_articles, rules)
    batch_system_prompt = custom_system_prompt or TRANSLATION_SYSTEM_PROMPT
    if dynamic_guidelines and not custom_system_prompt:
        batch_system_prompt = batch_system_prompt + "\n" + dynamic_guidelines
        logger.info(
            f"Appended dynamic guidelines to translation system prompt:\n{dynamic_guidelines}"
        )

    last_err = None
    for model in MODEL_CHAIN:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        for attempt in range(retries_per_model):
            start_time = time.time()
            try:
                # Build payload dynamically based on caches
                payload = build_payload(user_prompt, model, batch_system_prompt)

                logger.info(
                    f"Calling API for translation batch of size {len(articles)} (Attempt {attempt + 1}/{retries_per_model}) using model {model}..."
                )
                resp = requests.post(url, json=payload, headers=headers, timeout=60)

                elapsed = time.time() - start_time
                logger.info(
                    f"Translation API response for {model} received in {elapsed:.2f}s with status {resp.status_code}"
                )

                # Check for Rate Limit (HTTP 429)
                if resp.status_code == 429:
                    sleep_time = backoff_factor * (3**attempt)
                    logger.warning(
                        f"Rate limit (429) hit for model {model}. Sleeping for {sleep_time} seconds before retry..."
                    )
                    time.sleep(sleep_time)
                    continue

                # If it's a 404 (model not found) or 403 (permission / quota issue), skip this model immediately
                if resp.status_code in (403, 404):
                    logger.warning(
                        f"Model {model} returned HTTP {resp.status_code}. Skipping this model..."
                    )
                    break

                # Server errors (500, 503): wait longer before retrying
                if resp.status_code in (500, 503):
                    sleep_time = backoff_factor * (3**attempt) * 2
                    logger.warning(
                        f"Server error ({resp.status_code}) from model {model}. Sleeping {sleep_time}s before retry..."
                    )
                    time.sleep(sleep_time)
                    continue

                resp.raise_for_status()
                res_data = resp.json()

                candidates = res_data.get("candidates", [])
                if not candidates:
                    raise ValueError(f"No candidates returned from model {model}")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise ValueError(f"No parts found in candidate from model {model}")

                content_text = ""
                for part in parts:
                    if not part.get("thought"):
                        content_text = part.get("text", "").strip()
                        break
                if not content_text and parts:
                    content_text = parts[0].get("text", "").strip()

                if content_text.startswith("```"):
                    content_text = re.sub(
                        r"^```(?:json)?\n", "", content_text, flags=re.IGNORECASE
                    )
                    content_text = re.sub(r"\n```$", "", content_text)
                content_text = content_text.strip()

                result = json.loads(content_text)
                translations_list = result.get("translations", [])

                mapped_results = {}
                for idx_in_list, item in enumerate(translations_list):
                    url_key = item.get("url")
                    idx = None
                    if url_key:
                        m = re.search(r"(\d+)", str(url_key))
                        if m:
                            idx = int(m.group(1))

                    # Fallback to list order index if url_key index is not resolved
                    if idx is None or not (0 <= idx < len(articles)):
                        idx = idx_in_list

                    if 0 <= idx < len(articles):
                        original_url = articles[idx]["url"]
                        raw_title = item.get("title", "").strip()
                        raw_body = item.get("body", "").strip()

                        # Apply post-processing (including line breaks removal)
                        ko_title, ko_body = post_process_translation(
                            raw_title,
                            raw_body,
                            rules,
                            preserve_newlines=preserve_newlines,
                        )

                        mapped_results[original_url] = {
                            "title": ko_title,
                            "body": ko_body,
                        }
                logger.info(
                    f"Successfully translated batch of {len(translations_list)} articles with {model} in {elapsed:.2f}s"
                )
                return mapped_results

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Error calling translation API with model {model} after {elapsed:.2f}s: {e}"
                )
                last_err = e
                if attempt < retries_per_model - 1:
                    sleep_time = backoff_factor * (3**attempt)
                    logger.info(f"Retrying model {model} in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.warning(
                        f"All attempts for model {model} failed. Falling back to next model..."
                    )

    if last_err:
        # Check if the error is a timeout or JSON parsing error, and we have multiple articles to split
        is_timeout_or_json_error = (
            isinstance(last_err, (requests.exceptions.Timeout, json.JSONDecodeError))
            or "read timeout" in str(last_err).lower()
        )

        if is_timeout_or_json_error and len(articles) > 1:
            mid = len(articles) // 2
            left_batch = articles[:mid]
            right_batch = articles[mid:]
            logger.warning(
                f"Translation failed due to timeout or JSON error ({last_err}). "
                f"Splitting batch of size {len(articles)} into {len(left_batch)} and {len(right_batch)}..."
            )

            # Recursive call for left half
            left_results = call_gemini_translator_api(
                left_batch, retries_per_model, backoff_factor, *args, **kwargs
            )

            # Cooldown sleep to respect rate limits
            logger.info(
                f"Sleeping {COOLDOWN_SLEEP_SECONDS} seconds between split batch runs..."
            )
            time.sleep(COOLDOWN_SLEEP_SECONDS)

            # Recursive call for right half
            right_results = call_gemini_translator_api(
                right_batch, retries_per_model, backoff_factor, *args, **kwargs
            )

            # Merge and return results
            return {**left_results, **right_results}

        raise TranslationError(f"Translation API call failed: {last_err}") from last_err
    else:
        raise TranslationError(
            "Failed to translate batch: All fallback models in the chain failed."
        )


def translate_macro_events(event_names: List[str]) -> Dict[str, str]:
    """
    Translates a list of unique English macroeconomic indicator names or holidays into Korean.
    Uses Gemini API in JSON mode and handles model fallbacks.
    """
    if not event_names:
        return {}

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error(
            "GEMINI_API_KEY environment variable is not defined. Skipping macro translation."
        )
        return {}

    headers = {"Content-Type": "application/json"}

    system_prompt = (
        "You are a professional financial translator. Translate the given macroeconomic indicator names or holidays into standard Korean.\n"
        "Keep standard formatting clean. Translate suffixes like '(YoY)' to '(전년 대비)', '(MoM)' to '(전월 대비)', '(QoQ)' to '(전분기 대비)' if appropriate.\n"
        "You MUST output your response strictly as a JSON object containing a 'translations' key mapped to a dictionary/object of original English name to Korean translation.\n"
        "Example output:\n"
        "{\n"
        '  "translations": {\n'
        '    "Marine Day": "바다의 날",\n'
        '    "Pending Home Sales (MoM)": "펜딩 주택판매지수 (전월 대비)"\n'
        "  }\n"
        "}"
    )

    user_prompt = json.dumps({"events": event_names}, ensure_ascii=False)

    last_err = None
    for model in MODEL_CHAIN:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        try:
            payload = build_payload(user_prompt, model, system_prompt)
            logger.info(
                f"Calling API for macro events translation of size {len(event_names)} using model {model}..."
            )
            resp = requests.post(url, json=payload, headers=headers, timeout=30)

            if resp.status_code == 429:
                logger.warning(
                    f"Rate limit (429) hit for model {model}. Skipping to next model..."
                )
                continue
            if resp.status_code in (403, 404):
                logger.warning(
                    f"Model {model} returned HTTP {resp.status_code}. Skipping this model..."
                )
                continue
            if resp.status_code in (500, 503):
                logger.warning(
                    f"Server error ({resp.status_code}) from model {model}. Skipping this model..."
                )
                continue

            resp.raise_for_status()
            res_data = resp.json()

            candidates = res_data.get("candidates", [])
            if not candidates:
                raise ValueError(f"No candidates returned from model {model}")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError(f"No parts found in candidate from model {model}")

            content_text = ""
            for part in parts:
                if not part.get("thought"):
                    content_text = part.get("text", "").strip()
                    break
            if not content_text and parts:
                content_text = parts[0].get("text", "").strip()

            if content_text.startswith("```"):
                content_text = re.sub(
                    r"^```(?:json)?\n", "", content_text, flags=re.IGNORECASE
                )
                content_text = re.sub(r"\n```$", "", content_text)
            content_text = content_text.strip()

            result = json.loads(content_text)
            translations = result.get("translations", {})
            logger.info(
                f"Successfully translated {len(translations)} macro events with {model}"
            )
            return translations

        except Exception as e:
            logger.warning(f"Model {model} failed to translate macro events: {e}")
            last_err = e
            continue

    logger.error(
        f"Failed to translate macro events: All models in chain failed. Last error: {last_err}"
    )
    return {}


def translate_and_cache_weekly_schedule_events(news_file: str, today_str: str) -> None:
    """
    Scans weekly schedule in news_file, translates untranslated macro indicators,
    saves translations to data/macro_translation_cache.json, cleans old cache entries,
    and updates news_file in-place.
    """
    if not os.path.exists(news_file):
        logger.warning(
            f"News file {news_file} does not exist. Skipping weekly schedule translation."
        )
        return

    try:
        with open(news_file, "r", encoding="utf-8") as f:
            news_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read news file for weekly schedule translation: {e}")
        return

    weekly_schedule = news_data.get("weekly_schedule", [])
    if not weekly_schedule:
        logger.info("No weekly schedule events found in news file.")
        return

    # Load static map
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    map_file = os.path.join(project_root, "config", "macro_translation_map.json")
    macro_map = {}
    if os.path.exists(map_file):
        try:
            with open(map_file, "r", encoding="utf-8") as f:
                macro_map = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load macro_translation_map.json: {e}")

    # Load dynamic cache
    cache_file = os.path.join(project_root, "data", "macro_translation_cache.json")
    macro_cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                macro_cache = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load macro_translation_cache.json: {e}")

    # Find events needing translation (importance not in earnings/holiday, and korean_name is empty)
    to_translate_events = []
    to_translate_names = set()

    for event in weekly_schedule:
        importance = event.get("importance", "medium")
        if importance in ("earnings", "holiday"):
            continue

        korean_name = event.get("korean_name", "").strip()
        name = event.get("name", "").strip()
        if not name:
            continue

        utc_time_str = event.get("utc_time") or ""
        date_part = utc_time_str[:10]
        cache_key = f"{date_part}_{name}"

        # 1. Check if already has a korean_name
        if korean_name:
            continue

        # 2. Check static map
        if name in macro_map:
            event["korean_name"] = macro_map[name]
            continue

        # 3. Check dynamic cache
        if cache_key in macro_cache:
            event["korean_name"] = macro_cache[cache_key]
            continue

        # 4. Otherwise, queue for translation
        to_translate_events.append((event, name, cache_key))
        to_translate_names.add(name)

    # Call translation if there are new items
    if to_translate_names:
        translations = translate_macro_events(list(to_translate_names))
        for event, name, cache_key in to_translate_events:
            translated_val = translations.get(name)
            if translated_val:
                event["korean_name"] = translated_val
                macro_cache[cache_key] = translated_val
            else:
                logger.warning(
                    f"Could not translate macro event '{name}', falling back to English."
                )

    # Clean up cache: remove entries older than D-1 (yesterday)
    try:
        from datetime import timedelta

        today_date = datetime.strptime(today_str, "%Y%m%d").date()
        yesterday_date = today_date - timedelta(days=1)

        cleaned_cache = {}
        for key, val in macro_cache.items():
            if "_" in key:
                date_part_str, _ = key.split("_", 1)
                try:
                    entry_date = datetime.strptime(date_part_str, "%Y-%m-%d").date()
                    if entry_date >= yesterday_date:
                        cleaned_cache[key] = val
                except ValueError:
                    cleaned_cache[key] = val
            else:
                cleaned_cache[key] = val
        macro_cache = cleaned_cache
    except Exception as e:
        logger.error(f"Failed to clean up macro translation cache: {e}")

    # Write updated cache
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(macro_cache, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Updated macro translation cache. Active cache size: {len(macro_cache)}"
        )
    except Exception as e:
        logger.error(f"Failed to write macro translation cache: {e}")

    # Write updated news file in-place
    try:
        news_data["weekly_schedule"] = weekly_schedule
        with open(news_file, "w", encoding="utf-8") as f:
            json.dump(news_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully updated weekly schedule translations in {news_file}")
    except Exception as e:
        logger.error(
            f"Failed to update news file with weekly schedule translations: {e}"
        )


def translate_new_articles(
    state_file: str, cache_file: str, limit_urls: Optional[List[str]] = None
) -> None:
    """
    Load state_file and cache_file, translate any new articles that are NOT in the cache.
    If limit_urls is provided, we only translate those URLs.
    """
    if not os.path.exists(state_file):
        logger.warning(f"State file {state_file} does not exist. Skipping translation.")
        return

    with open(state_file, "r", encoding="utf-8") as f:
        state_data = json.load(f)

    cache: Dict[str, Dict[str, str]] = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load translation cache: {e}")

    # Gather all blocks from normal and sec outputs
    blocks = []
    for cat, items in state_data.get("category_normal_outputs", {}).items():
        blocks.extend(items)
    for cat, items in state_data.get("category_sec_outputs", {}).items():
        blocks.extend(items)

    # Gather items to translate
    to_translate = []
    updated = False
    seen_urls = set()

    for block in blocks:
        url, title, body = parse_article_block(block)
        if not url:
            continue
        if limit_urls is not None and url not in limit_urls:
            continue
        if url in cache:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Rule for SEC filings (8-K, 10-K, 10-Q, SEC Filing): do not translate body, cache immediately
        if re.search(r"\b(8-K|10-K|10-Q|SEC Filing)\b", title, re.IGNORECASE):
            logger.info(f"Skipping translation for SEC filing marker: {title[:50]}...")
            cache[url] = {"title": title.strip(), "body": ""}
            updated = True
            continue

        to_translate.append({"url": url, "title": title, "body": body})

    if not to_translate:
        if updated:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.info("No new translations needed.")
        return

    logger.info(f"Translating {len(to_translate)} articles in batches...")

    # Use environment variable for batch size, default to 4 to optimize API calls
    batch_size = int(os.environ.get("TRANSLATOR_BATCH_SIZE", "4"))
    batches = [
        to_translate[i : i + batch_size]
        for i in range(0, len(to_translate), batch_size)
    ]

    for batch_idx, batch in enumerate(batches):
        try:
            # Enforce 2-second cooldown delay between calls to respect RPM 15
            if batch_idx > 0:
                logger.info(
                    "Sleeping 2 seconds to avoid exceeding RPM 15 rate limit..."
                )
                time.sleep(2.0)

            translations = call_gemini_translator_api(batch)

            batch_updated = False
            for article in batch:
                url = article["url"]
                if url in translations:
                    cache[url] = translations[url]
                    batch_updated = True
                    updated = True
                else:
                    logger.warning(f"Translation output missing for url: {url}")

            if batch_updated:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
                logger.info(
                    f"Incremental translation cache saved for batch {batch_idx + 1}/{len(batches)}."
                )

        except Exception as e:
            logger.error(f"Error during translation of batch {batch_idx + 1}: {e}")
            raise TranslationError(
                f"Translation pipeline failed during batch {batch_idx + 1}: {e}"
            ) from e

    if updated:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.info(f"Final translation cache saved containing {len(cache)} items.")


def extract_urls_from_report(report_file: str) -> List[str]:
    """Extract all article URLs from a formatted report."""
    if not os.path.exists(report_file):
        return []
    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()
    return re.findall(r"\[.*?\]\((https?://.*?)\)", content)


def extract_articles_from_report(report_file: str) -> List[Dict[str, str]]:
    """Extract all articles (url, title, body) from a report file."""
    if not os.path.exists(report_file):
        return []
    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split report by potential article blocks starting with [Title](Url)
    blocks = re.split(r"(?=\[.*?\]\(https?://.*?\))", content)
    articles = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        url, title, body = parse_article_block(block)
        if url:
            articles.append({"url": url, "title": title, "body": body})
    return articles


def translate_missing_report_articles(en_report_file: str, cache_file: str) -> None:
    """
    Check for any URLs in the English report that are missing from the cache,
    and translate them before generating the final report.
    """
    if not os.path.exists(en_report_file):
        logger.warning(
            f"English report file {en_report_file} does not exist. Cannot check for missing translations."
        )
        return

    # Load cache
    cache: Dict[str, Dict[str, str]] = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache in missing articles translation: {e}")

    # Extract all articles from the report file
    report_articles = extract_articles_from_report(en_report_file)

    # Find which ones are missing from the cache
    missing_articles = []
    seen_urls = set()
    for art in report_articles:
        url = art["url"]
        title = art["title"]
        if url not in cache and url not in seen_urls:
            seen_urls.add(url)
            # Rule for SEC filings (8-K, 10-K, 10-Q, SEC Filing): do not translate body, cache immediately
            if re.search(r"\b(8-K|10-K|10-Q|SEC Filing)\b", title, re.IGNORECASE):
                logger.info(
                    f"Skipping translation for SEC filing marker in missing check: {title[:50]}..."
                )
                cache[url] = {"title": title.strip(), "body": ""}
                continue
            missing_articles.append(art)

    if not missing_articles:
        logger.info("No missing translations detected in the report.")
        return

    logger.warning(
        f"Detected {len(missing_articles)} missing translations in the report! Retrying translation..."
    )

    # Translate the missing articles in batches
    batch_size = int(os.environ.get("TRANSLATOR_BATCH_SIZE", "4"))
    batches = [
        missing_articles[i : i + batch_size]
        for i in range(0, len(missing_articles), batch_size)
    ]

    updated = False
    for batch_idx, batch in enumerate(batches):
        try:
            # Enforce cooldown delay between calls
            if batch_idx > 0:
                logger.info("Sleeping 2 seconds to avoid rate limits during retry...")
                time.sleep(2.0)

            translations = call_gemini_translator_api(batch)
            for article in batch:
                url = article["url"]
                if url in translations:
                    cache[url] = translations[url]
                    updated = True
                else:
                    logger.warning(f"Retry translation missing for url: {url}")
        except Exception as e:
            logger.error(
                f"Error during retry translation of batch {batch_idx + 1}: {e}"
            )
            raise TranslationError(
                f"Retry translation failed during batch {batch_idx + 1}: {e}"
            ) from e

    if updated:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Updated translation cache with retried missing articles. Total items: {len(cache)}"
        )


def sync_premarket_cache_to_delta(
    cache_file: str, pre_state_file: str, en_report_file: str
) -> None:
    """
    Filters translated_state_pre.json (cache_file) so that it only retains URLs
    that are present in extracted_state_pre.json (pre_state_file) or the current premarket report.
    This ensures that translated_state_pre.json only contains the delta for the current premarket session.
    """
    if not os.path.exists(cache_file):
        return

    # Load cache
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load cache in sync_premarket_cache: {e}")
        return

    # Collect allowed URLs
    allowed_urls = set()

    # 1. From extracted_state_pre.json
    if os.path.exists(pre_state_file):
        try:
            with open(pre_state_file, "r", encoding="utf-8") as f:
                pre_state = json.load(f)
            # Find all URLs in category_normal_outputs and category_sec_outputs
            for cat, items in pre_state.get("category_normal_outputs", {}).items():
                for item in items:
                    url, _, _ = parse_article_block(item)
                    if url:
                        allowed_urls.add(url)
            for cat, items in pre_state.get("category_sec_outputs", {}).items():
                for item in items:
                    url, _, _ = parse_article_block(item)
                    if url:
                        allowed_urls.add(url)
            # Also check extracted_urls field if any
            for url in pre_state.get("extracted_urls", []):
                allowed_urls.add(url)
        except Exception as e:
            logger.error(f"Failed to load pre_state in sync_premarket_cache: {e}")

    # Filter cache
    filtered_cache = {url: val for url, val in cache.items() if url in allowed_urls}

    # Save cache back
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(filtered_cache, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Synced {cache_file} to match premarket delta. Kept {len(filtered_cache)} of {len(cache)} items."
        )
    except Exception as e:
        logger.error(f"Failed to save synced cache: {e}")


def generate_korean_premarket_draft(
    en_report_file: str, ko_draft_file: str, cache_file: str
) -> bool:
    """Generates the Korean premarket report draft from the English report and cache."""
    if not os.path.exists(en_report_file):
        logger.error(f"English premarket report {en_report_file} does not exist.")
        return False

    with open(en_report_file, "r", encoding="utf-8") as f:
        content = f.read()

    cache: Dict[str, Dict[str, str]] = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)

    header_match = re.match(
        r"^##\s+(.*?)\s+Premarket\s*\n*(.*)", content, re.DOTALL | re.IGNORECASE
    )
    if not header_match:
        logger.error("Failed to parse English premarket report header.")
        return False

    date_str = header_match.group(1).strip()
    body_content = header_match.group(2).strip()

    blocks = re.split(r"(?=\[.*?\]\(https?://.*?\))", body_content)
    ko_blocks = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        url, title, body = parse_article_block(block)
        if url and url in cache:
            ko_title = cache[url]["title"]
            ko_body = cache[url]["body"]
            if ko_body:
                ko_blocks.append(f"[{ko_title}]({url})<br />{ko_body}")
            else:
                ko_blocks.append(f"[{ko_title}]({url})")
        else:
            if url:
                logger.warning(
                    f"Excluding untranslated premarket article from draft: {url}"
                )
            else:
                ko_blocks.append(block)

    ko_content = f"## {date_str} Premarket\n\n" + "\n\n".join(ko_blocks)
    with open(ko_draft_file, "w", encoding="utf-8") as f:
        f.write(ko_content)
    logger.info(f"Generated Korean premarket draft at {ko_draft_file}")
    return True


def generate_korean_full_draft(
    en_report_file: str, ko_draft_file: str, cache_file: str
) -> bool:
    """Generates the Korean full report draft from the English report and cache."""
    if not os.path.exists(en_report_file):
        logger.error(f"English full report {en_report_file} does not exist.")
        return False

    with open(en_report_file, "r", encoding="utf-8") as f:
        content = f.read()

    cache: Dict[str, Dict[str, str]] = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)

    # 1. Translate & Extract ### Daily Point
    dp_match = re.search(r"###\s+Daily\s+Point", content, re.IGNORECASE)
    ko_daily_point = ""

    if dp_match:
        start_dp = dp_match.start()
        # Since we parse the English report, search only for the English header "Weekly Schedule".
        next_sec = re.search(
            r"\n###\s+Weekly\s+Schedule",
            content[dp_match.end() :],
            re.IGNORECASE,
        )
        if next_sec:
            end_dp = dp_match.end() + next_sec.start()
            en_dp_content = content[start_dp:end_dp].strip()

            # Split into non-translated parts (header + index metrics + Topline Signals title) and translated parts (the rest).
            topline_match = re.search(
                r"\*\*Topline\s+Signals\*\*", en_dp_content, re.IGNORECASE
            )
            if topline_match:
                non_trans_part = en_dp_content[: topline_match.end()].strip()
                trans_part = en_dp_content[topline_match.end() :].strip()
            else:
                # Fallback path if the Topline Signals header is missing.
                lines = en_dp_content.split("\n")
                last_index_line = -1
                for idx, line in enumerate(lines):
                    if line.strip().startswith("_ "):
                        last_index_line = idx
                if last_index_line != -1:
                    non_trans_part = "\n".join(lines[: last_index_line + 1]).strip()
                    trans_part = "\n".join(lines[last_index_line + 1 :]).strip()
                else:
                    non_trans_part = "### Daily Point"
                    trans_part = en_dp_content.replace("### Daily Point", "").strip()

            # Check translation cache or invoke translation API.
            cache_key = "__daily_point_commentary__"
            if cache_key in cache:
                ko_dp_body = cache[cache_key]["body"]
                logger.info("Daily Point block translation loaded from cache.")
            else:
                try:
                    logger.info(
                        "Daily Point translation cache missing. Requesting LLM translation..."
                    )

                    # Pre-process: split trans_part by 'Good day' / 'Good day.' to avoid LLM placeholder issues
                    # and guarantee precise positioning of '안녕하세요.'.
                    parts = re.split(
                        r"\bGood\s+day\.?\s*\n*",
                        trans_part,
                        maxsplit=1,
                        flags=re.IGNORECASE,
                    )
                    if len(parts) == 2:
                        dp_articles = [
                            {
                                "url": "daily_point_signals",
                                "title": "Daily Point Signals",
                                "body": parts[0].strip(),
                            },
                            {
                                "url": "daily_point_commentary",
                                "title": "Daily Point Commentary",
                                "body": parts[1].strip(),
                            },
                        ]
                    else:
                        dp_articles = [
                            {
                                "url": "daily_point",
                                "title": "Daily Point",
                                "body": trans_part.strip(),
                            }
                        ]

                    translations = call_gemini_translator_api(
                        dp_articles,
                        custom_system_prompt=DAILY_POINT_TRANSLATION_SYSTEM_PROMPT,
                        preserve_newlines=True,
                    )

                    if len(parts) == 2:
                        if (
                            "daily_point_signals" in translations
                            and "daily_point_commentary" in translations
                        ):
                            ko_signals = translations["daily_point_signals"][
                                "body"
                            ].strip()
                            ko_commentary = translations["daily_point_commentary"][
                                "body"
                            ].strip()
                            ko_dp_body = (
                                f"{ko_signals}\n\n안녕하세요.\n\n{ko_commentary}"
                            )
                        else:
                            raise TranslationError(
                                "Daily Point split translation missing in API response."
                            )
                    else:
                        if "daily_point" in translations:
                            ko_dp_body = translations["daily_point"]["body"].strip()
                            # Fallback replacement if 'Good day' is somehow still inside body
                            ko_dp_body = re.sub(
                                r"\bGood\s+day\.?\s*\n*",
                                "안녕하세요.\n\n",
                                ko_dp_body,
                                flags=re.IGNORECASE,
                            )
                            ko_dp_body = re.sub(
                                r"(안녕하십니까|좋은\s*하루\s*되세요|좋은\s*하루입니다)\.?\s*\n*",
                                "안녕하세요.\n\n",
                                ko_dp_body,
                            )
                        else:
                            raise TranslationError(
                                "Daily Point translation missing in API response."
                            )

                    cache[cache_key] = {
                        "title": "Daily Point Commentary",
                        "body": ko_dp_body,
                    }
                    # Save updated translation cache.
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error(
                        f"Failed to translate Daily Point: {e}. Falling back to English."
                    )
                    ko_dp_body = trans_part

            ko_daily_point = f"{non_trans_part}\n\n{ko_dp_body}"
            # Shrink content boundaries to prevent duplication during final assembly.
            content = content[:start_dp].rstrip() + "\n\n" + content[end_dp:].lstrip()
        else:
            logger.warning("Could not find next section after Daily Point to delete.")

    # 2. Map Category Headers and replace article blocks
    sections = re.split(r"(?=\n###\s+)", content)
    ko_sections = []

    # Prepend the translated Daily Point block to the top of the Korean report.
    if ko_daily_point:
        ko_sections.append(ko_daily_point)

    for sec in sections:
        sec_strip = sec.strip()
        if not sec_strip:
            continue

        lines = sec_strip.split("\n")
        header = lines[0].strip()

        norm_header = normalize_category_header(header)
        if norm_header in NORM_MAPPING:
            ko_header = NORM_MAPPING[norm_header]
            body_text = "\n".join(lines[1:]).strip()
            blocks = re.split(r"(?=\[.*?\]\(https?://.*?\))", body_text)

            ko_blocks = []
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                url, title, body = parse_article_block(block)
                if url and url in cache:
                    ko_title = cache[url]["title"]
                    ko_body = cache[url]["body"]
                    if ko_body:
                        ko_blocks.append(f"[{ko_title}]({url})<br />{ko_body}")
                    else:
                        ko_blocks.append(f"[{ko_title}]({url})")
                else:
                    if url:
                        logger.warning(
                            f"Excluding untranslated article from draft: {url}"
                        )
                    else:
                        ko_blocks.append(block)

            ko_sec = f"{ko_header}\n\n" + "\n\n".join(ko_blocks)
            ko_sections.append(ko_sec)
        elif norm_header == normalize_category_header(
            "### Weekly Schedule"
        ) or norm_header == normalize_category_header("### 주간 일정"):
            ko_sections.append("### 주간 일정\n" + "\n".join(lines[1:]))
        else:
            ko_sections.append(sec_strip)

    ko_content = "\n\n".join(ko_sections)
    with open(ko_draft_file, "w", encoding="utf-8") as f:
        f.write(ko_content)
    logger.info(f"Generated Korean full report draft at {ko_draft_file}")
    return True


def _acquire_preemption_lock(lock_file: str) -> None:
    """Check for stale translation process, terminate it, and acquire lock for current process."""
    if os.path.exists(lock_file):
        try:
            with open(lock_file, "r") as lf:
                old_pid = int(lf.read().strip())
            if os.getpid() != old_pid:
                try:
                    os.kill(old_pid, 0)  # Check if process exists
                    logger.info(
                        f"Stale translation process {old_pid} detected. Terminating it..."
                    )
                    os.kill(old_pid, signal.SIGTERM)
                    time.sleep(1)
                except OSError:
                    logger.info(
                        f"Previous translation process {old_pid} is no longer running."
                    )
        except Exception as e:
            logger.warning(f"Error checking/killing old translation process: {e}")

    # Acquire lock for current process
    os.makedirs(os.path.dirname(lock_file), exist_ok=True)
    try:
        with open(lock_file, "w") as lf:
            lf.write(str(os.getpid()))
        logger.info(f"Acquired translation.lock (PID {os.getpid()}).")
    except Exception as e:
        logger.error(f"Failed to write translation.lock: {e}")


def run_translator(
    report_type: str = "full",
    target_date: Optional[str] = None,
    on_ko_report_ready: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Main entry point for translation pipeline.
    Acquires translation.lock with PID-based preemption.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")

    if target_date:
        today_str = target_date
    else:
        us_tz = pytz.timezone("America/New_York")
        today_str = datetime.now(us_tz).strftime("%Y%m%d")

        # Check if a custom json file is explicitly passed to capture testing
        custom_date = None
        for arg in sys.argv[1:]:
            m = re.search(r"(\d{8})", arg)
            if m:
                custom_date = m.group(1)
                break

        if custom_date:
            today_str = custom_date

    state_file = os.path.join(data_dir, f"extracted_state_{today_str}.json")

    if report_type == "premarket":
        cache_file = os.path.join(data_dir, "translated_state_pre.json")
    else:
        cache_file = os.path.join(data_dir, f"translated_state_{today_str}.json")

        # Merge premarket translation cache if it exists from previous premarket runs
        pre_cache_file = os.path.join(data_dir, "translated_state_pre.json")
        if os.path.exists(pre_cache_file):
            try:
                with open(pre_cache_file, "r", encoding="utf-8") as f:
                    pre_cache = json.load(f)

                standard_cache = {}
                if os.path.exists(cache_file):
                    with open(cache_file, "r", encoding="utf-8") as f:
                        standard_cache = json.load(f)

                merged_cache = {**pre_cache, **standard_cache}
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(merged_cache, f, ensure_ascii=False, indent=2)

                logger.info(
                    f"Merged {len(pre_cache)} translation keys from translated_state_pre.json into today's cache."
                )
            except Exception as e:
                logger.error(f"Failed to merge translated_state_pre.json: {e}")

    # Preemption Lock Check
    trans_lock_file = os.path.join(project_root, "logs", "translation.lock")
    _acquire_preemption_lock(trans_lock_file)

    try:
        logger.info(f"Running Translator (Type: {report_type}) for date {today_str}")

        # Translate and cache macro events in weekly schedule
        news_file = os.path.join(data_dir, f"daily_news_{today_str}.json")
        translate_and_cache_weekly_schedule_events(news_file, today_str)

        if report_type == "incremental":
            try:
                translate_new_articles(state_file, cache_file)
            except TranslationError as te:
                logger.error(
                    f"Translation pipeline failed for incremental, proceeding to draft generation: {te}"
                )

        elif report_type == "premarket":
            pre_state_file = os.path.join(data_dir, "extracted_state_pre.json")

            # 1. Prioritize selected premarket articles from delta pre file
            en_report = os.path.join(data_dir, f"premarket_report_{today_str}.txt")

            # Fallback to latest premarket file if not exists
            if not os.path.exists(en_report):
                files = sorted(
                    glob.glob(os.path.join(data_dir, "premarket_report_*.txt"))
                )
                if files:
                    en_report = files[-1]

            premarket_urls = extract_urls_from_report(en_report)
            logger.info(
                f"Premarket report selected {len(premarket_urls)} articles for translation."
            )

            # Translate only selected premarket articles from delta pre state
            try:
                translate_new_articles(
                    pre_state_file, cache_file, limit_urls=premarket_urls
                )
            except TranslationError as te:
                logger.error(f"Premarket delta translation failed, proceeding: {te}")

            # Check and translate any missing articles from the report before generating draft
            try:
                translate_missing_report_articles(en_report, cache_file)
            except TranslationError as te:
                logger.error(
                    f"Premarket missing articles translation failed, proceeding: {te}"
                )

            # Generate KST premarket draft and format it immediately
            ko_draft = os.path.join(data_dir, f"premarket_report_ko_{today_str}.txt")
            generate_korean_premarket_draft(en_report, ko_draft, cache_file)

            ko_output_dir = os.path.join(data_dir, "premarket")
            os.makedirs(ko_output_dir, exist_ok=True)
            ko_final_report = os.path.join(
                ko_output_dir, f"alpha_signal_premarket_{today_str}_ko.md"
            )

            run_formatter(ko_draft, ko_final_report, lang="ko")
            logger.info(
                f"Successfully generated final Korean premarket report at {ko_final_report}"
            )

            # Audit final premarket report and log candidate issues with date in filename
            try:
                from translation_auditor import TranslationAuditor

                audit_log_path = os.path.join(
                    project_root, "logs", f"translation_audit_{today_str}.json"
                )
                audit_issues = TranslationAuditor.audit_file(
                    ko_final_report, output_log_path=audit_log_path
                )
                logger.info(
                    f"Translation Audit completed. Logged {len(audit_issues)} candidate(s) to {audit_log_path}"
                )
            except Exception as audit_err:
                logger.error(
                    f"Failed to execute TranslationAuditor audit step: {audit_err}"
                )

            # Trigger immediate push callback if provided
            if on_ko_report_ready:
                try:
                    on_ko_report_ready(ko_final_report)
                except Exception as cb_err:
                    logger.error(
                        f"Failed to execute on_ko_report_ready callback: {cb_err}"
                    )

            # 2. Translate all remaining articles from delta pre state
            logger.info(
                "Proceeding to translate all remaining articles in premarket state..."
            )
            try:
                translate_new_articles(pre_state_file, cache_file)
            except TranslationError as te:
                logger.error(
                    f"Premarket remaining articles translation failed, proceeding: {te}"
                )

            # Sync the premarket translation cache to the premarket delta
            sync_premarket_cache_to_delta(cache_file, pre_state_file, en_report)

        elif report_type == "full":
            # 1. Translate all articles to complete the cache
            try:
                translate_new_articles(state_file, cache_file)
            except TranslationError as te:
                logger.error(
                    f"Full pipeline translation failed, proceeding to draft generation: {te}"
                )

            # 2. Read full English report
            en_report = os.path.join(data_dir, f"final_report_{today_str}.txt")
            if not os.path.exists(en_report):
                files = sorted(glob.glob(os.path.join(data_dir, "final_report_*.txt")))
                if files:
                    en_report = files[-1]

            # Check and translate any missing articles from the report before generating draft
            try:
                translate_missing_report_articles(en_report, cache_file)
            except TranslationError as te:
                logger.error(
                    f"Full pipeline missing articles translation failed, proceeding: {te}"
                )

            ko_draft = os.path.join(data_dir, f"final_report_ko_{today_str}.txt")
            generate_korean_full_draft(en_report, ko_draft, cache_file)

            ko_output_dir = os.path.join(data_dir, "report")
            os.makedirs(ko_output_dir, exist_ok=True)
            ko_final_report = os.path.join(
                ko_output_dir, f"alpha_signal_{today_str}_ko.md"
            )

            run_formatter(ko_draft, ko_final_report, lang="ko")
            logger.info(
                f"Successfully generated final Korean full report at {ko_final_report}"
            )

            # Audit final full report and log candidate issues with date in filename
            try:
                from translation_auditor import TranslationAuditor

                audit_log_path = os.path.join(
                    project_root, "logs", f"translation_audit_{today_str}.json"
                )
                audit_issues = TranslationAuditor.audit_file(
                    ko_final_report, output_log_path=audit_log_path
                )
                logger.info(
                    f"Translation Audit completed. Logged {len(audit_issues)} candidate(s) to {audit_log_path}"
                )
            except Exception as audit_err:
                logger.error(
                    f"Failed to execute TranslationAuditor audit step: {audit_err}"
                )

            # Trigger immediate push callback if provided
            if on_ko_report_ready:
                try:
                    on_ko_report_ready(ko_final_report)
                except Exception as cb_err:
                    logger.error(
                        f"Failed to execute on_ko_report_ready callback: {cb_err}"
                    )

    finally:
        # Remove translation.lock
        if os.path.exists(trans_lock_file):
            try:
                os.remove(trans_lock_file)
                logger.info("translation.lock removed.")
            except Exception as e:
                logger.warning(f"Failed to remove translation.lock: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Alpha Signals Core Translation Pipeline"
    )
    parser.add_argument(
        "--type",
        choices=["full", "premarket", "incremental"],
        default="full",
        help="Type of report to translate",
    )
    args, _ = parser.parse_known_args()
    run_translator(report_type=args.type)
