import os
import io
import json
import time
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from tenacity import retry, stop_after_attempt, wait_exponential
import pdfplumber

# Import all read/write operations from db_handler
# This eliminates circular imports with strategy_tester.py
from experts.db_handler import (
    save_rules, load_rules,
    save_raw_transcript, load_raw_transcript,
    save_raw_book_text, load_raw_book_text,
    save_query, load_query,
    log_conflicts, query_path,
)

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

# Use .strip() to handle any accidental trailing/leading spaces in .env
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "").strip()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()

genai.configure(api_key=GEMINI_API_KEY)

# Flash is free tier — handles text AND vision (PDFs, images)
MODEL = "gemini-1.5-flash"

# =========================================================
# CACHE PATHS
# =========================================================
# knowledge/
#   raw/
#     transcripts/   <- raw YouTube transcript text
#     books/         <- raw PDF text
#   extracted/       <- Gemini rule extraction results
#   queries/         <- answers to one-off questions
#   conflicts_log.json
# All paths and read/write functions managed by db_handler.py

# =========================================================
# CHANNEL DATABASE
# =========================================================

YOUTUBE_CHANNELS = {
    "the_trading_channel": {
        "name": "The Trading Channel",
        "handle": "@TheTradingChannel",
        "channel_id": "UCGL9ubdGcvZh_dvSV2z1hoQ",
        "focus": ["forex", "price_action", "risk_management"]
    },
    "rayner_teo": {
        "name": "Rayner Teo",
        "handle": "@tradingwithrayner",
        "channel_id": "UCFSn-h8wTnhpKJMteN76Abg",
        "focus": ["trend_following", "swing_trading", "price_action"]
    },
    "ict": {
        "name": "ICT - Inner Circle Trader",
        "handle": "@ICT",
        # NOTE: @ICT handle points to an unrelated Indian education
        # channel. This ID is the real ICT trading channel by
        # Michael Huddleston.
        "channel_id": "UCtjxa77NqamhVC8atV85Rog",
        "focus": [
            "smart_money", "liquidity", "market_structure",
            "order_blocks", "fair_value_gaps", "kill_zones"
        ]
    },
    "warrior_trading": {
        "name": "Warrior Trading",
        "handle": "@WarriorTrading",
        "channel_id": "UCBayuhgYpKNbhJxfExYkPfA",
        "focus": ["momentum_trading", "day_trading", "stock_scalping"]
    },
    "adam_khoo": {
        "name": "Adam Khoo",
        "handle": "@AdamKhoo",
        "channel_id": "UCK-aOjEvZNJl3HINja0gAiQ",
        "focus": [
            "stock_investing", "trading_psychology",
            "options", "macro_analysis"
        ]
    },
    "quantreo": {
        "name": "Quantreo",
        "handle": "@Quantreo",
        "channel_id": "UCp7jckfiEglNf_Gj62VR0pw",
        "focus": ["algorithmic_trading", "python", "backtesting"]
    },
    "trader_tom": {
        "name": "Trader Tom",
        "handle": "@TraderTom",
        "channel_id": "UC4C43bjs7bwwasNPecJm8bw",
        "focus": ["discretionary_trading", "trader_psychology"]
    },
    "al_brooks_trading": {
        "name": "Al Brooks Trading",
        "handle": "@AlBrooksTrading",
        "channel_id": "UCgkcoiJK7e33vMUbM5E-OQw",
        "focus": [
            "price_action", "market_structure",
            "advanced_analysis", "trend_bars", "reversal_patterns"
        ]
    },
}

# Channel-specific extra schema fields
CHANNEL_SCHEMAS = {
    "ict": {
        "extra_fields": [
            "liquidity_levels", "order_blocks",
            "fair_value_gaps", "kill_zones",
            "market_maker_patterns"
        ]
    },
    "warrior_trading": {
        "extra_fields": [
            "momentum_triggers", "float_analysis", "halt_patterns"
        ]
    },
    "rayner_teo": {
        "extra_fields": [
            "trend_structure", "pullback_zones",
            "moving_average_rules"
        ]
    },
    "al_brooks_trading": {
        "extra_fields": [
            "bar_patterns", "two_legged_pullbacks", "measured_moves"
        ]
    }
}

# =========================================================
# BOOK DATABASE
# =========================================================

BOOK_DATABASE = {

    # ── Books you have downloaded ─────────────────────────

    "technical_analysis_murphy": {
        "title": "Technical Analysis of the Financial Markets",
        "author": "John Murphy",
        # scanned = no text layer, Gemini reads PDF visually
        # text    = has a text layer, extract with pdfplumber
        "type": "scanned",
        "topics": [
            "chart_patterns", "indicators", "dow_theory",
            "volume", "intermarket_analysis"
        ],
        # Exact filename from your books/ folder
        "path": "books/John_J._Murphy_-_Technical_Analysis_Of_The_Financial_Markets.PDF"
    },

    "ict_trading_strategy": {
        "title": "ICT Trading Strategy",
        "author": "ICT",
        "type": "scanned",
        "topics": [
            "smart_money", "liquidity", "order_blocks",
            "fair_value_gaps", "market_structure",
            "kill_zones", "institutional_trading"
        ],
        "path": "books/ICT-Trading-Strategy-1.PDF"
    },

    "identifying_chart_patterns": {
        "title": "Identifying Chart Patterns",
        "author": "Unknown",
        "type": "scanned",
        "topics": [
            "chart_patterns", "technical_analysis",
            "reversals", "continuations", "breakouts"
        ],
        "path": "books/Idenitfying-Chart-Patterns.PDF"
    },

    "liquidity_sweep_trading": {
        "title": "Liquidity Sweep in Trading",
        "author": "Unknown",
        "type": "scanned",
        "topics": [
            "liquidity", "smart_money",
            "liquidity_sweeps", "stop_hunts",
            "market_structure"
        ],
        "path": "books/Liquidity-Sweep-in-Trading.PDF"
    },

    "smart_money_concept_strategy": {
        "title": "Smart Money Concept Strategy",
        "author": "Unknown",
        "type": "scanned",
        "topics": [
            "smart_money", "institutional_trading",
            "order_blocks", "market_structure",
            "liquidity_zones"
        ],
        "path": "books/Smart-Money-Concept-Strategy-PDF.PDF"
    },

    "technical_analysis_kanu_jain": {
        "title": "Technical Analysis",
        "author": "Dr Kanu Jain",
        "type": "scanned",
        "topics": [
            "technical_analysis", "indicators",
            "chart_patterns", "trend_analysis"
        ],
        "path": "books/B.Com(Hons)_IIIy_DrKanuJain.PDF"
    },

    # ── Books to add when you get the PDFs ───────────────
    # Drop the PDF in your books/ folder and uncomment

    # "trading_in_the_zone": {
    #     "title": "Trading in the Zone",
    #     "author": "Mark Douglas",
    #     "type": "text",
    #     "topics": [
    #         "psychology", "discipline", "consistency",
    #         "mindset", "probabilistic_thinking"
    #     ],
    #     "path": "books/trading_in_the_zone.pdf"
    # },

    # "al_brooks_price_action": {
    #     "title": "Trading Price Action Trends",
    #     "author": "Al Brooks",
    #     "type": "scanned",
    #     "topics": [
    #         "price_action", "market_structure", "trend_bars",
    #         "reversals", "breakouts"
    #     ],
    #     "path": "books/al_brooks_price_action.pdf"
    # },

    # "japanese_candlesticks": {
    #     "title": "Japanese Candlestick Charting Techniques",
    #     "author": "Steve Nison",
    #     "type": "scanned",
    #     "topics": [
    #         "candlestick_patterns", "reversals",
    #         "continuations", "doji", "engulfing"
    #     ],
    #     "path": "books/nison_candlesticks.pdf"
    # },

    # "trade_your_way": {
    #     "title": "Trade Your Way to Financial Freedom",
    #     "author": "Van K. Tharp",
    #     "type": "text",
    #     "topics": [
    #         "position_sizing", "expectancy",
    #         "risk_management", "system_design", "r_multiples"
    #     ],
    #     "path": "books/van_tharp_trade_your_way.pdf"
    # },

    # "trading_for_a_living": {
    #     "title": "Trading for a Living",
    #     "author": "Alexander Elder",
    #     "type": "text",
    #     "topics": [
    #         "psychology", "technical_analysis",
    #         "risk_management", "triple_screen_system"
    #     ],
    #     "path": "books/elder_trading_for_a_living.pdf"
    # },

    # "reminiscences": {
    #     "title": "Reminiscences of a Stock Operator",
    #     "author": "Edwin Lefevre",
    #     "type": "text",
    #     "topics": [
    #         "psychology", "trend_following",
    #         "patience", "risk", "market_cycles"
    #     ],
    #     "path": "books/reminiscences_stock_operator.pdf"
    # },

    # "wyckoff_method": {
    #     "title": "The Wyckoff Method",
    #     "author": "Richard Wyckoff",
    #     "type": "scanned",
    #     "topics": [
    #         "market_structure", "accumulation",
    #         "distribution", "composite_man", "volume_analysis"
    #     ],
    #     "path": "books/wyckoff_method.pdf"
    # },
}

# =========================================================
# LEVEL 1 CACHE — RAW CONTENT
# Raw transcripts and book text saved on first fetch.
# YouTube and PDF APIs never called again for saved content.
# All cache functions now in db_handler.py
# =========================================================

# =========================================================
# GEMINI HELPERS
# =========================================================

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def _ask_gemini(prompt: str, json_mode: bool = False) -> str:
    """
    Sends a text prompt to Gemini.
    Retries up to 5 times with exponential backoff on failure.
    When json_mode=True forces Gemini to return pure JSON only.
    """
    model = genai.GenerativeModel(MODEL)

    if json_mode:
        gen_config = genai.GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json"
        )
    else:
        gen_config = genai.GenerationConfig(temperature=0.2)

    response = model.generate_content(prompt, generation_config=gen_config)
    return response.text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=30)
)
def _ask_gemini_with_pdf(pdf_path: str, prompt: str) -> str:
    """
    Uploads a PDF to Gemini and sends a prompt.
    Gemini reads the PDF natively — works for both text PDFs
    and scanned/chart-heavy PDFs.
    Called ONCE per book. Result cached so never repeated.

    Uses try/finally to guarantee the uploaded file is always
    deleted from Gemini servers even if generation fails.
    """
    model    = genai.GenerativeModel(MODEL)
    pdf_file = None

    print("  Uploading PDF to Gemini...")

    try:
        pdf_file = genai.upload_file(pdf_path)

        # Wait for Gemini to finish processing the file
        while pdf_file.state.name == "PROCESSING":
            time.sleep(2)
            pdf_file = genai.get_file(pdf_file.name)

        response = model.generate_content(
            [pdf_file, prompt],
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )

        return response.text

    finally:
        # Always clean up from Gemini servers
        # even if generate_content throws an exception
        if pdf_file is not None:
            try:
                genai.delete_file(pdf_file.name)
            except Exception:
                pass  # Don't let cleanup failure hide the real error


def _parse_json_response(text: str) -> dict:
    """
    Safely parses JSON from Gemini.
    When json_mode=True is used in _ask_gemini, the response
    is already pure JSON. This function handles the fallback
    case where stripping is still needed.
    """
    if not text:
        return {}

    cleaned = text.strip()

    # Strip markdown fences if present
    if cleaned.startswith("```"):
        lines   = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])

    # Strip any conversational preamble before the JSON
    brace = cleaned.find("{")
    if brace > 0:
        cleaned = cleaned[brace:]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

# =========================================================
# SCHEMA BUILDER
# =========================================================

def _build_schema(channel_key: str = None) -> dict:
    """
    Builds the JSON extraction schema.
    Adds channel-specific fields so unique concepts
    from each educator are captured properly.
    """
    schema = {
        "entry_conditions":  [],
        "exit_conditions":   [],
        "risk_management":   [],
        "market_structure":  [],
        "indicators":        [],
        "psychology":        [],
        "strategy_type":     [],
        "market_regime":     []
    }

    if channel_key and channel_key in CHANNEL_SCHEMAS:
        for field in CHANNEL_SCHEMAS[channel_key]["extra_fields"]:
            schema[field] = []

    return schema

# =========================================================
# YOUTUBE API
# =========================================================

def _get_youtube_client():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def get_channel_videos(
    channel_id: str,
    max_results: int = 20,
    order: str = "viewCount"
) -> List[Dict[str, Any]]:

    youtube = _get_youtube_client()

    response = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        type="video",
        order=order,
        maxResults=max_results
    ).execute()

    videos = []

    for item in response["items"]:
        snippet = item["snippet"]
        videos.append({
            "video_id":     item["id"]["videoId"],
            "title":        snippet["title"],
            "description":  snippet["description"],
            "published_at": snippet["publishedAt"]
        })

    return videos

# =========================================================
# TRANSCRIPT FETCHING — LEVEL 1 CACHED
# =========================================================

def get_video_transcript(video_id: str) -> str:
    """
    Returns transcript for a video.
    Checks Level 1 cache first — if saved, returns immediately
    without calling YouTube API at all.
    Only fetches from YouTube if not cached yet.
    """

    # Level 1 cache check
    cached = load_raw_transcript(video_id)
    if cached:
        print(f"  Transcript loaded from cache: {video_id}")
        return cached

    # Not cached — fetch from YouTube now
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        full_text = " ".join([line["text"] for line in transcript])

        # Quality check — too short means no real content
        if len(full_text.split()) < 200:
            print(f"  Transcript too short for {video_id} — skipping")
            return ""

        # Save to Level 1 cache immediately
        save_raw_transcript(video_id, full_text)

        return full_text

    except Exception as e:
        print(f"  Transcript error for {video_id}: {e}")
        return ""

# =========================================================
# TEXT CHUNKING
# =========================================================

def _chunk_text(text: str, chunk_size: int = 2500) -> List[str]:
    """
    Chunks text by sentences rather than blindly by word count.
    Prevents trading rules from being split across chunk boundaries.
    Groups sentences together until approaching chunk_size words.
    """
    # Split into sentences on period, newline, or exclamation
    import re
    sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())

    chunks       = []
    current      = []
    current_size = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        # If adding this sentence exceeds limit and we have content,
        # save current chunk and start a new one
        if current_size + word_count > chunk_size and current:
            chunks.append(" ".join(current))
            current      = []
            current_size = 0

        current.append(sentence)
        current_size += word_count

    # Don't forget the last chunk
    if current:
        chunks.append(" ".join(current))

    return chunks

# =========================================================
# RULE EXTRACTION
# =========================================================

def _extract_rules_from_text(
    text: str,
    channel_key: str = None
) -> dict:
    """
    Sends text to Gemini to extract trading rules.
    Chunks long content, extracts from each chunk,
    then consolidates into one rule set.
    """

    schema  = _build_schema(channel_key)
    chunks  = _chunk_text(text)

    print(f"  Extracting rules from {len(chunks)} chunks...")

    chunk_results = []

    for idx, chunk in enumerate(chunks):
        print(f"    Chunk {idx + 1}/{len(chunks)}")

        prompt = f"""
Extract trading concepts from this text.
Return JSON only matching this exact structure:
{json.dumps(schema, indent=2)}

If no trading concepts are found return the same structure
with empty lists. Do not invent content.

Text:
{chunk}
"""
        result = _parse_json_response(_ask_gemini(prompt, json_mode=True))
        if result:
            chunk_results.append(result)

        # Respect free tier rate limits (15 req/min)
        # tenacity handles retries if we hit the limit
        time.sleep(1)

    if not chunk_results:
        return {}

    return _consolidate_rules(chunk_results)


def _extract_rules_from_pdf(
    pdf_path: str,
    book_key: str
) -> dict:
    """
    Sends entire PDF to Gemini for native reading.
    Handles text, charts and diagrams in one pass.
    Replaces the old page-by-page image conversion approach.
    """

    book = BOOK_DATABASE[book_key]

    prompt = f"""
This is a trading book: "{book['title']}" by {book['author']}.
It covers: {', '.join(book['topics'])}.

Read the entire book carefully including all charts,
diagrams, and annotated examples.

Extract ALL trading rules, strategies and concepts.
Return JSON only:

{{
    "entry_conditions":  [{{"rule": "", "confidence": 0.0}}],
    "exit_conditions":   [{{"rule": "", "confidence": 0.0}}],
    "risk_management":   [{{"rule": "", "confidence": 0.0}}],
    "market_structure":  [{{"rule": "", "confidence": 0.0}}],
    "chart_patterns":    [{{"pattern": "", "description": ""}}],
    "indicators":        [{{"name": "", "how_to_use": ""}}],
    "psychology":        [],
    "strategy_type":     [],
    "key_concepts":      [],
    "conflicts":         []
}}

Return JSON only, no extra text.
"""

    result_text = _ask_gemini_with_pdf(pdf_path, prompt)
    return _parse_json_response(result_text)

# =========================================================
# CONSOLIDATION
# =========================================================

def _consolidate_rules(rule_sets: List[dict]) -> dict:
    """
    Merges multiple rule sets into one.
    Keeps high-confidence rules, removes duplicates,
    flags contradictions for Auditor AI LLM1.
    """

    valid_sets = [r for r in rule_sets if r]

    if not valid_sets:
        return {}

    if len(valid_sets) == 1:
        return valid_sets[0]

    prompt = f"""
Merge these trading rule sets into one unified set.

Instructions:
1. Keep rules that appear in multiple sets (high confidence)
2. Remove exact duplicates
3. Flag rules that directly contradict each other
4. Add confidence score 0.0 to 1.0 per rule

Return JSON only:
{{
    "entry_conditions":  [{{"rule": "", "confidence": 0.0}}],
    "exit_conditions":   [{{"rule": "", "confidence": 0.0}}],
    "risk_management":   [{{"rule": "", "confidence": 0.0}}],
    "market_structure":  [{{"rule": "", "confidence": 0.0}}],
    "indicators":        [],
    "psychology":        [],
    "strategy_type":     [],
    "market_regime":     [],
    "conflicts":         []
}}

Data:
{json.dumps(valid_sets, indent=2)}
"""

    parsed = _parse_json_response(_ask_gemini(prompt, json_mode=True))

    if not parsed:
        return valid_sets[0]

    # Log conflicts for Auditor AI LLM1 via db_handler
    if parsed.get("conflicts"):
        log_conflicts(parsed["conflicts"])

    return parsed


def _tournament_merge(rule_sets: List[dict]) -> dict:
    """
    Merges a large list of rule sets two at a time.
    Prevents sending too much data to Gemini at once.
    """

    active = [r for r in rule_sets if r]

    if not active:
        return {}

    while len(active) > 1:
        batch = []
        for i in range(0, len(active), 2):
            if i + 1 < len(active):
                merged = _consolidate_rules(
                    [active[i], active[i + 1]]
                )
                batch.append(merged)
            else:
                batch.append(active[i])
        active = batch

    return active[0]

# =========================================================
# YOUTUBE LEARNING PIPELINE
# =========================================================

def learn_from_channel(
    channel_key: str,
    topic: str,
    max_videos: int = 5,
    force_refresh: bool = False
) -> dict:
    """
    Full pipeline for learning from a YouTube channel.

    Level 1 cache: raw transcripts saved on first fetch.
    YouTube never called again for the same video.

    Level 2 cache: Gemini extraction saved after first run.
    Gemini never called again for the same content and topic.
    """

    if channel_key not in YOUTUBE_CHANNELS:
        raise ValueError(
            f"Unknown channel: {channel_key}. "
            f"Available: {list(YOUTUBE_CHANNELS.keys())}"
        )

    # Level 2 cache check
    if not force_refresh:
        existing = load_rules(channel_key, topic)
        if existing:
            print(
                f"Rules already extracted for "
                f"{channel_key} / {topic} — loaded from cache"
            )
            return existing

    channel = YOUTUBE_CHANNELS[channel_key]

    print(f"\nLearning from: {channel['name']}")
    print(f"Topic: {topic}")

    videos = get_channel_videos(
        channel["channel_id"],
        max_results=max_videos,
        order="viewCount"
    )

    if not videos:
        print("No videos found.")
        return {}

    # Gemini screens video titles before downloading anything
    video_text = "\n".join([
        f"{v['video_id']} :: {v['title']}"
        for v in videos
    ])

    selection_prompt = f"""
Select the 3 most relevant videos for learning about: {topic}

Videos:
{video_text}

Return ONLY the video IDs separated by commas.
No explanations, no extra text.
"""

    selected_raw = _ask_gemini(selection_prompt)
    selected_ids = [x.strip() for x in selected_raw.split(",")]

    print(f"Selected videos: {selected_ids}")

    all_rules = []

    for video in videos:

        if video["video_id"] not in selected_ids:
            continue

        print(f"\nProcessing: {video['title']}")

        # Level 1 cached — only fetches if not saved yet
        transcript = get_video_transcript(video["video_id"])

        if not transcript:
            print("  No transcript — skipping")
            continue

        rules = _extract_rules_from_text(transcript, channel_key)

        if rules:
            all_rules.append(rules)

    if not all_rules:
        print("No rules extracted.")
        return {}

    merged = _consolidate_rules(all_rules)

    # Level 2 cache — Gemini never asked this again
    save_rules(channel_key, topic, merged)

    return merged

# =========================================================
# BOOK LEARNING PIPELINE
# =========================================================

def learn_from_book(
    book_key: str,
    force_refresh: bool = False
) -> dict:
    """
    Full pipeline for learning from a trading book PDF.

    Level 1 cache: raw text extracted and saved once.
    Level 2 cache: Gemini extraction saved after first run.

    For scanned/visual PDFs Gemini reads the file natively —
    handles charts and diagrams automatically.
    """

    if book_key not in BOOK_DATABASE:
        raise ValueError(
            f"Unknown book: {book_key}. "
            f"Available: {list(BOOK_DATABASE.keys())}"
        )

    book = BOOK_DATABASE[book_key]

    # Level 2 cache check
    if not force_refresh:
        existing = load_rules(f"book_{book_key}", "full")
        if existing:
            print(f"Rules already extracted for: {book['title']}")
            return existing

    if not os.path.exists(book["path"]):
        raise FileNotFoundError(
            f"PDF not found at: {book['path']}\n"
            f"Please add the PDF to the books/ folder."
        )

    print(f"\nLearning from: {book['title']} by {book['author']}")

    if book["type"] == "text":
        return _learn_text_book(book_key, book)
    else:
        return _learn_scanned_book(book_key, book)


def _learn_text_book(book_key: str, book: dict) -> dict:
    """Pipeline for books with an extractable text layer."""

    # Level 1 cache check
    raw_text = load_raw_book_text(book_key)

    if not raw_text:
        print("  Extracting text from PDF...")

        text_parts = []

        with pdfplumber.open(book["path"]) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        raw_text = "\n".join(text_parts)

        if not raw_text:
            print("  No text found in PDF.")
            return {}

        # Level 1 cache — save raw text immediately
        save_raw_book_text(book_key, raw_text)

    else:
        print("  Raw book text loaded from cache.")

    rules = _extract_rules_from_text(raw_text)

    # Level 2 cache
    save_rules(f"book_{book_key}", "full", rules)

    return rules


def _learn_scanned_book(book_key: str, book: dict) -> dict:
    """
    Pipeline for scanned or chart-heavy PDFs.
    Sends the entire PDF to Gemini which reads it natively.
    One API call covers all pages, text and charts.
    """

    print("  Scanned PDF — uploading to Gemini...")

    rules = _extract_rules_from_pdf(book["path"], book_key)

    if not rules:
        print("  No rules extracted from PDF.")
        return {}

    # Level 2 cache — Gemini never reads this book again
    save_rules(f"book_{book_key}", "full", rules)

    return rules

# =========================================================
# ONE-OFF QUESTIONS — CACHED
# Ask Gemini a specific question about saved content.
# Same question is never sent twice.
# =========================================================

def ask_question(
    source_key: str,
    topic: str,
    question: str
) -> str:
    """
    Asks Gemini a specific question about already-extracted
    rules. Answer is cached by question hash.
    Same question is never sent to Gemini twice.

    Example:
        ask_question(
            "rayner_teo",
            "swing_trading",
            "What does Rayner say about stop losses?"
        )
    """

    # Unique hash from source + topic + question
    q_hash = hashlib.md5(
        f"{source_key}_{topic}_{question}".encode()
    ).hexdigest()[:12]

    # Check query cache via db_handler
    cached = load_query(q_hash)
    if cached:
        print(f"Answer loaded from cache")
        return cached.get("answer", "")

    # Load rules for context
    rules = load_rules(source_key, topic)

    if not rules:
        return (
            f"No extracted rules found for "
            f"{source_key} / {topic}. "
            f"Run learn_from_channel() or learn_from_book() first."
        )

    prompt = f"""
Based on these extracted trading rules:
{json.dumps(rules, indent=2)}

Answer this question clearly and concisely:
{question}
"""

    answer = _ask_gemini(prompt)

    # Cache the answer via db_handler
    save_query(q_hash, {
        "source_key": source_key,
        "topic":      topic,
        "question":   question,
        "answer":     answer,
        "timestamp":  str(datetime.now())
    })

    return answer

# =========================================================
# CROSS-CHANNEL MERGE
# =========================================================

def merge_all_channels(topic: str) -> dict:
    """
    Merges saved rules from all YouTube channels on a topic.
    Only uses Level 2 cache — no new API calls triggered.
    """

    all_rules = []

    for channel_key in YOUTUBE_CHANNELS:
        rules = load_rules(channel_key, topic)
        if rules:
            rules["_source"]      = channel_key
            rules["_source_type"] = "youtube"
            all_rules.append(rules)

    if not all_rules:
        print(f"No saved channel rules for topic: {topic}")
        return {}

    print(f"Merging {len(all_rules)} channel rule sets...")

    return _tournament_merge(all_rules)

# =========================================================
# MASTER KNOWLEDGE MERGE
# =========================================================

def merge_all_knowledge(topic: str) -> dict:
    """
    Merges everything — YouTube channels + books — into one
    master knowledge base. Books weighted 1.5x.
    Only uses cached data — no new API calls.
    """

    all_rules = []

    for channel_key in YOUTUBE_CHANNELS:
        rules = load_rules(channel_key, topic)
        if rules:
            rules["_source"]      = channel_key
            rules["_source_type"] = "youtube"
            rules["_weight"]      = 1.0
            all_rules.append(rules)

    for book_key in BOOK_DATABASE:
        rules = load_rules(f"book_{book_key}", "full")
        if rules:
            rules["_source"]      = book_key
            rules["_source_type"] = "book"
            rules["_weight"]      = 1.5
            all_rules.append(rules)

    if not all_rules:
        print(f"No knowledge found for topic: {topic}")
        return {}

    print(
        f"Merging {len(all_rules)} sources "
        f"(channels + books) on topic: {topic}..."
    )

    final = _tournament_merge(all_rules)

    save_rules("master_knowledge", topic, final)

    return final

# =========================================================
# NEW CONTENT CHECK
# =========================================================

def check_for_new_content(channel_key: str, topic: str) -> dict:
    """
    Checks if new videos posted since last learning run.
    Re-learns only if new content is found.
    Existing cached transcripts are still reused.
    """

    rules = load_rules(channel_key, topic)

    if not rules:
        print("No existing rules — running full pipeline.")
        return learn_from_channel(channel_key, topic)

    last_learned = rules.get("_last_updated", "2000-01-01")

    channel = YOUTUBE_CHANNELS[channel_key]

    latest = get_channel_videos(
        channel["channel_id"], max_results=5, order="date"
    )

    if not latest:
        return rules

    if latest[0]["published_at"] > last_learned:
        print(f"New content found for {channel_key} — re-learning.")
        return learn_from_channel(
            channel_key, topic, force_refresh=True
        )

    print(f"No new content for {channel_key} — cache is current.")
    return rules

# =========================================================
# CONFIDENCE UPDATING
# =========================================================

def update_rule_confidence(
    source_key: str,
    topic: str,
    rule_key: str,
    pnl: float
):
    """
    Called after each trade to update rule confidence scores.
    Low-confidence rules flagged for Auditor AI LLM1.
    Saves updated scores back to Level 2 cache.
    """

    rules = load_rules(source_key, topic)

    if not rules or rule_key not in rules:
        return

    if "score" not in rules[rule_key]:
        rules[rule_key]["score"] = 0.5

    rules[rule_key]["score"] += 0.02 if pnl > 0 else -0.02
    rules[rule_key]["score"]  = max(
        0.0, min(1.0, rules[rule_key]["score"])
    )

    if rules[rule_key]["score"] < 0.3:
        rules[rule_key]["status"] = "flagged"
        print(
            f"  Rule '{rule_key}' flagged — "
            f"confidence: {rules[rule_key]['score']:.2f}"
        )

    save_rules(source_key, topic, rules)

# =========================================================
# EXAMPLE
# =========================================================

if __name__ == "__main__":

    # Learn from a YouTube channel
    # Transcripts cached to: knowledge/raw/transcripts/
    # Rules cached to:       knowledge/extracted/
    results = learn_from_channel(
        channel_key="the_trading_channel",
        topic="risk_management",
        max_videos=10
    )
    print(json.dumps(results, indent=2))

    # Learn from a book (drop PDF in books/ folder first)
    # book_rules = learn_from_book("technical_analysis_murphy")

    # Ask a one-off question (answer cached, never repeated)
    # answer = ask_question(
    #     "rayner_teo",
    #     "risk_management",
    #     "What position sizing rules does Rayner recommend?"
    # )
    # print(answer)

    # Merge all channels on one topic
    # merged = merge_all_channels("risk_management")

    # Merge everything — channels + books
    # master = merge_all_knowledge("risk_management")
