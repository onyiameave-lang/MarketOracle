"""
knowledge_base.py

Learns trading knowledge from YouTube channels and PDF books.
Saves everything to a two-level cache so Gemini/YouTube
are never called twice for the same content.

Level 1 cache: raw transcripts/text  → knowledge/raw/
Level 2 cache: extracted JSON rules  → knowledge/extracted/

Connects to:
    - db_handler.py       → all file read/write operations
    - strategy_tester.py  → feeds learned rules into strategy configs
"""

import os
import io
import json
import time
import re
import base64
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from tenacity import retry, stop_after_attempt, wait_exponential
import pdfplumber

from experts.db_handler import (
    save_rules, load_rules,
    save_raw_transcript, load_raw_transcript,
    save_raw_book_text, load_raw_book_text,
    save_query, load_query,
    log_conflicts,
    DATA_DIR,
)

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "").strip()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. Add it to your .env file.")
if not YOUTUBE_API_KEY:
    print("WARNING: YOUTUBE_API_KEY not set. Add it to your .env file.")

genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-1.5-flash-latest"

# =========================================================
# CHANNEL DATABASE
# All channel IDs verified against YouTube directly.
# =========================================================

YOUTUBE_CHANNELS = {
    "the_trading_channel": {
        "name":       "The Trading Channel",
        "handle":     "@thetradingchannel",
        "channel_id": "UCGL9ubdGcvZh_dvSV2z1hoQ",
        "focus":      ["forex", "price_action", "risk_management"],
    },
    "rayner_teo": {
        "name":       "Rayner Teo",
        "handle":     "@tradingwithrayner.",
        "channel_id": "UCFSn-h8wTnhpKJMteN76Abg",
        "focus":      ["trend_following", "swing_trading", "price_action"],
    },
    "ict": {
        "name":       "ICT - Inner Circle Trader",
        "handle":     "@InnerCircleTrader",
        # Verified: youtube.com/channel/UCtjxa77NqamhVC8atV85Rog
        "channel_id": "UCtjxa77NqamhVC8atV85Rog",
        "focus":      ["smart_money", "liquidity", "market_structure", "order_blocks", "fair_value_gaps"],
    },
    "warrior_trading": {
        "name":       "Ross Cameron-Warrior Trading",
        "handle":     "@daytradewarrior",
        # Verified: youtube.com/channel/UCBayuhgYpKNbhJxfExYkPfA
        "channel_id": "UCBayuhgYpKNbhJxfExYkPfA",
        "focus":      ["momentum_trading", "day_trading", "stock_scalping"],
    },
    "adam_khoo": {
        "name":       "Adam Khoo",
        "handle":     "@AdamKhoo",
        "channel_id": "UCK-aOjEvZNJl3HINja0gAiQ",
        "focus":      ["stock_investing", "trading_psychology", "options", "macro_analysis"],
    },
    "quantreo": {
        "name":       "Quantreo",
        "handle":     "@Quantreo",
        "channel_id": "UCp7jckfiEglNf_Gj62VR0pw",
        "focus":      ["algorithmic_trading", "python", "backtesting"],
    },
    "trader_tom": {
        "name":       "Trader Tom",
        "handle":     "@TraderTom",
        "channel_id": "UC4C43bjs7bwwasNPecJm8bw",
        "focus":      ["discretionary_trading", "trader_psychology"],
    },
    "al_brooks_trading": {
        "name":       "Al Brooks Trading",
        "handle":     "@BrooksTradingCourse",
        "channel_id": "UCgkcoiJK7e33vMUbM5E-OQw",
        "focus":      ["price_action", "market_structure", "advanced_analysis"],
    },
}

# Extra schema fields per channel — captures concepts unique to each educator
CHANNEL_SCHEMAS = {
    "ict":              {"extra_fields": ["liquidity_levels", "order_blocks", "fair_value_gaps", "kill_zones"]},
    "warrior_trading":  {"extra_fields": ["momentum_triggers", "float_analysis", "halt_patterns"]},
    "rayner_teo":       {"extra_fields": ["trend_structure", "pullback_zones", "moving_average_rules"]},
    "al_brooks_trading":{"extra_fields": ["bar_patterns", "two_legged_pullbacks", "measured_moves"]},
}

# =========================================================
# BOOK DATABASE
# PDFs stored in knowledge/ folder (same level as the project)
# Filenames match exactly what's on disk.
# =========================================================

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOOK_DATABASE = {
    "technical_analysis_murphy": {
        "title":  "Technical Analysis of the Financial Markets",
        "author": "John Murphy",
        "type":   "scanned",
        "topics": ["chart_patterns", "indicators", "dow_theory", "volume", "intermarket_analysis"],
        "path":   os.path.join(_ROOT, "knowledge", "John_J._Murphy_-_Technical_Analysis_Of_The_Financial_Markets.PDF"),
    },
    "ict_trading_strategy": {
        "title":  "ICT Trading Strategy",
        "author": "ICT",
        "type":   "scanned",
        "topics": ["smart_money", "liquidity", "order_blocks", "fair_value_gaps", "market_structure"],
        "path":   os.path.join(_ROOT, "knowledge", "ICT-Trading-Strategy-1.PDF"),
    },
    "identifying_chart_patterns": {
        "title":  "Identifying Chart Patterns",
        "author": "Unknown",
        "type":   "scanned",
        "topics": ["chart_patterns", "technical_analysis", "reversals", "continuations"],
        "path":   os.path.join(_ROOT, "knowledge", "Idenitfying-Chart-Patterns.PDF"),
    },
    "liquidity_sweep_trading": {
        "title":  "Liquidity Sweep in Trading",
        "author": "Unknown",
        "type":   "scanned",
        "topics": ["liquidity", "smart_money", "liquidity_sweeps", "stop_hunts"],
        "path":   os.path.join(_ROOT, "knowledge", "Liquidity-Sweep-in-Trading.PDF"),
    },
    "smart_money_concept_strategy": {
        "title":  "Smart Money Concept Strategy",
        "author": "Unknown",
        "type":   "scanned",
        "topics": ["smart_money", "institutional_trading", "order_blocks", "market_structure"],
        "path":   os.path.join(_ROOT, "knowledge", "Smart-Money-Concept-trading-strategy-PDF.PDF"),
    },
    "technical_analysis_kanu_jain": {
        "title":  "Technical Analysis - Dr Kanu Jain",
        "author": "Dr Kanu Jain",
        "type":   "scanned",
        "topics": ["technical_analysis", "indicators", "chart_patterns"],
        "path":   os.path.join(_ROOT, "knowledge", "B.Com(Hons)_IIIyearVIsem_FundamentalsofInvestments_Week2_DrKanuJain.PDF"),
    },
}

# =========================================================
# GEMINI HELPERS
# =========================================================

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def _ask_gemini(prompt: str, json_mode: bool = False) -> str:
    """
    Sends a text prompt to Gemini.
    Retries up to 5x with exponential backoff.
    json_mode=True forces pure JSON response — no preamble text.
    """
    model = genai.GenerativeModel(MODEL)
    if json_mode:
        cfg = genai.GenerationConfig(temperature=0.2, response_mime_type="application/json")
    else:
        cfg = genai.GenerationConfig(temperature=0.2)
    response = model.generate_content(prompt, generation_config=cfg)
    return response.text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))
def _ask_gemini_with_images(images_b64: List[str], prompt: str) -> str:
    """
    Sends a batch of base64 JPEG images + prompt to Gemini inline.
    No file upload — avoids ResumableUploadError for large PDFs.
    Used for reading scanned trading books page by page.
    """
    model   = genai.GenerativeModel(MODEL)
    content = [prompt]
    for img in images_b64:
        content.append({"mime_type": "image/jpeg", "data": img})
    response = model.generate_content(
        content,
        generation_config=genai.GenerationConfig(temperature=0.2, response_mime_type="application/json")
    )
    return response.text


def _parse_json(text: str) -> dict:
    """Safely parses JSON from Gemini — handles markdown fences and preamble."""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])
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
    Builds the JSON extraction schema for a channel.
    Adds channel-specific fields so unique concepts
    (e.g. ICT order blocks, Warrior halt patterns) are captured.
    """
    schema = {
        "entry_conditions": [], "exit_conditions": [],
        "risk_management":  [], "market_structure": [],
        "indicators":       [], "psychology":       [],
        "strategy_type":    [], "market_regime":    [],
    }
    if channel_key and channel_key in CHANNEL_SCHEMAS:
        for field in CHANNEL_SCHEMAS[channel_key]["extra_fields"]:
            schema[field] = []
    return schema

# =========================================================
# YOUTUBE API
# Uses uploads playlist — more reliable than search.list
# =========================================================

def _get_youtube_client():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def get_channel_videos(
    channel_id:  str,
    max_results: int = 20,
    order:       str = "date"
) -> List[Dict[str, Any]]:
    """
    Gets videos from a channel using the uploads playlist.
    Uploads playlist ID = channel ID with UC replaced by UU.
    More reliable than search.list — never returns NotFound.
    """
    try:
        youtube             = _get_youtube_client()
        uploads_playlist_id = "UU" + channel_id[2:]

        response = youtube.playlistItems().list(
            part       = "snippet",
            playlistId = uploads_playlist_id,
            maxResults = max_results,
        ).execute()

        videos = []
        for item in response.get("items", []):
            snippet  = item["snippet"]
            video_id = snippet.get("resourceId", {}).get("videoId")
            if video_id:
                videos.append({
                    "video_id":    video_id,
                    "title":       snippet.get("title", ""),
                    "description": snippet.get("description", "")[:200],
                    "published_at": snippet.get("publishedAt", ""),
                })
        return videos

    except HttpError as e:
        print(f"  YouTube API error ({e.status_code}): {e.reason}")
        return []
    except Exception as e:
        print(f"  YouTube error: {e}")
        return []

# =========================================================
# TRANSCRIPT FETCHING — LEVEL 1 CACHED
# =========================================================

def get_video_transcript(video_id: str) -> str:
    """
    Returns transcript for a video.
    Level 1 cache: if saved already, returns immediately
    without calling YouTube API again.
    """
    cached = load_raw_transcript(video_id)
    if cached:
        print(f"  Transcript from cache: {video_id}")
        return cached
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text  = " ".join([line["text"] for line in transcript])
        if len(full_text.split()) < 200:
            print(f"  Transcript too short for {video_id} — skipping")
            return ""
        save_raw_transcript(video_id, full_text)
        return full_text
    except Exception as e:
        print(f"  Transcript error for {video_id}: {e}")
        return ""

# =========================================================
# TEXT CHUNKING — sentence-aware
# =========================================================

def _chunk_text(text: str, chunk_size: int = 2500) -> List[str]:
    """
    Chunks text by sentences — prevents trading rules from
    being split across chunk boundaries.
    """
    sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
    chunks, current, current_size = [], [], 0
    for sentence in sentences:
        wc = len(sentence.split())
        if current_size + wc > chunk_size and current:
            chunks.append(" ".join(current))
            current, current_size = [], 0
        current.append(sentence)
        current_size += wc
    if current:
        chunks.append(" ".join(current))
    return chunks

# =========================================================
# RULE EXTRACTION
# =========================================================

def _extract_rules_from_text(text: str, channel_key: str = None) -> dict:
    """Sends text to Gemini to extract structured trading rules."""
    schema  = _build_schema(channel_key)
    chunks  = _chunk_text(text)
    print(f"  Extracting from {len(chunks)} chunks...")
    results = []
    for idx, chunk in enumerate(chunks):
        print(f"    Chunk {idx+1}/{len(chunks)}")
        prompt = f"""Extract trading concepts from this text.
Return JSON only matching this exact structure:
{json.dumps(schema, indent=2)}

If no trading concepts found, return the same structure with empty lists.
Do not invent content.

Text:
{chunk}"""
        result = _parse_json(_ask_gemini(prompt, json_mode=True))
        if result:
            results.append(result)
        time.sleep(1)
    return _consolidate_rules(results) if results else {}


def _extract_rules_from_pdf_images(pdf_path: str, book_key: str) -> dict:
    """
    Reads a scanned PDF by converting pages to JPEG batches
    and sending them inline to Gemini.
    No file upload — avoids ResumableUploadError on large files.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("  pdf2image not installed — run: pip install pdf2image")
        return {}

    book   = BOOK_DATABASE[book_key]
    prompt = f"""Pages from trading book: "{book['title']}" by {book['author']}.
Topics: {', '.join(book['topics'])}.

Extract ALL trading rules and concepts visible on these pages.
Return JSON only:
{{
    "entry_conditions":  [{{"rule": "", "confidence": 0.0}}],
    "exit_conditions":   [{{"rule": "", "confidence": 0.0}}],
    "risk_management":   [{{"rule": "", "confidence": 0.0}}],
    "market_structure":  [{{"rule": "", "confidence": 0.0}}],
    "chart_patterns":    [{{"pattern": "", "description": ""}}],
    "indicators":        [{{"name": "", "how_to_use": ""}}],
    "psychology":        [],
    "key_concepts":      []
}}
Return JSON only."""

    print(f"  Converting PDF pages to images...")
    try:
        pages = convert_from_path(pdf_path, dpi=100)
    except Exception as e:
        print(f"  PDF conversion error: {e}")
        return {}

    total      = len(pages)
    batch_size = 8
    results    = []

    print(f"  Processing {total} pages in batches of {batch_size}...")
    for i in range(0, total, batch_size):
        batch     = pages[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_b   = (total + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_b} (pages {i+1}-{min(i+batch_size, total)})")

        imgs = []
        for page in batch:
            buf = io.BytesIO()
            page.save(buf, format="JPEG", quality=55)
            imgs.append(base64.b64encode(buf.getvalue()).decode())

        try:
            result = _parse_json(_ask_gemini_with_images(imgs, prompt))
            if result:
                results.append(result)
        except Exception as e:
            print(f"  Batch {batch_num} error: {e}")

        time.sleep(2)

    return _consolidate_rules(results) if results else {}

# =========================================================
# CONSOLIDATION
# =========================================================

def _consolidate_rules(rule_sets: List[dict]) -> dict:
    """Merges multiple rule sets — removes duplicates, flags conflicts."""
    valid = [r for r in rule_sets if r]
    if not valid:
        return {}
    if len(valid) == 1:
        return valid[0]

    prompt = f"""Merge these trading rule sets into one unified set.

Instructions:
1. Keep rules that appear in multiple sets (mark confidence high)
2. Remove exact duplicates
3. Flag rules that directly contradict each other
4. Add confidence score 0.0 to 1.0 per rule

Return JSON only:
{{
    "entry_conditions":  [{{"rule": "", "confidence": 0.0}}],
    "exit_conditions":   [{{"rule": "", "confidence": 0.0}}],
    "risk_management":   [{{"rule": "", "confidence": 0.0}}],
    "market_structure":  [{{"rule": "", "confidence": 0.0}}],
    "indicators": [], "psychology": [], "strategy_type": [],
    "market_regime": [], "conflicts": []
}}

Data:
{json.dumps(valid, indent=2)}"""

    parsed = _parse_json(_ask_gemini(prompt, json_mode=True))
    if not parsed:
        return valid[0]
    if parsed.get("conflicts"):
        log_conflicts(parsed["conflicts"])
    return parsed


def _tournament_merge(rule_sets: List[dict]) -> dict:
    """Merges a large list two at a time — prevents token limit issues."""
    active = [r for r in rule_sets if r]
    if not active:
        return {}
    while len(active) > 1:
        batch = []
        for i in range(0, len(active), 2):
            if i + 1 < len(active):
                batch.append(_consolidate_rules([active[i], active[i+1]]))
            else:
                batch.append(active[i])
        active = batch
    return active[0]

# =========================================================
# YOUTUBE LEARNING PIPELINE
# =========================================================

def learn_from_channel(
    channel_key:   str,
    topic:         str,
    max_videos:    int  = 5,
    force_refresh: bool = False
) -> dict:
    """
    Full YouTube learning pipeline.
    Level 1: raw transcripts cached on first fetch.
    Level 2: Gemini extraction cached after first run.
    """
    if channel_key not in YOUTUBE_CHANNELS:
        raise ValueError(f"Unknown channel: {channel_key}. Available: {list(YOUTUBE_CHANNELS.keys())}")

    if not force_refresh:
        existing = load_rules(channel_key, topic)
        if existing:
            print(f"  Cached rules loaded for {channel_key} / {topic}")
            return existing

    channel = YOUTUBE_CHANNELS[channel_key]
    print(f"\nLearning from: {channel['name']}")
    print(f"Topic: {topic}")

    videos = get_channel_videos(channel["channel_id"], max_results=max_videos)
    if not videos:
        print("  No videos found.")
        return {}

    # AI screens video titles before downloading transcripts
    video_list = "\n".join([f"{v['video_id']} :: {v['title']}" for v in videos])
    selection  = _ask_gemini(
        f"Select the 3 most relevant videos for learning about: {topic}\n\n"
        f"Videos:\n{video_list}\n\n"
        f"Return ONLY the video IDs separated by commas. No explanations."
    )
    selected_ids = [x.strip() for x in selection.split(",")]
    print(f"  Selected: {selected_ids}")

    all_rules = []
    for video in videos:
        if video["video_id"] not in selected_ids:
            continue
        print(f"\n  Processing: {video['title']}")
        transcript = get_video_transcript(video["video_id"])
        if not transcript:
            print("  No transcript — skipping")
            continue
        rules = _extract_rules_from_text(transcript, channel_key)
        if rules:
            all_rules.append(rules)

    if not all_rules:
        print("  No rules extracted.")
        return {}

    merged = _consolidate_rules(all_rules)
    save_rules(channel_key, topic, merged)
    return merged

# =========================================================
# BOOK LEARNING PIPELINE
# =========================================================

def learn_from_book(book_key: str, force_refresh: bool = False) -> dict:
    """
    Full book learning pipeline.
    Level 1: raw text cached after first extraction.
    Level 2: Gemini extraction cached after first run.
    Scanned PDFs processed as image batches — no upload errors.
    """
    if book_key not in BOOK_DATABASE:
        raise ValueError(f"Unknown book: {book_key}. Available: {list(BOOK_DATABASE.keys())}")

    book = BOOK_DATABASE[book_key]

    if not force_refresh:
        existing = load_rules(f"book_{book_key}", "full")
        if existing:
            print(f"  Cached rules loaded for: {book['title']}")
            return existing

    if not os.path.exists(book["path"]):
        raise FileNotFoundError(
            f"PDF not found at: {book['path']}\n"
            f"Expected in knowledge/ folder."
        )

    print(f"\nLearning from: {book['title']} by {book['author']}")

    if book["type"] == "text":
        return _learn_text_book(book_key, book)
    else:
        return _learn_scanned_book(book_key, book)


def _learn_text_book(book_key: str, book: dict) -> dict:
    raw_text = load_raw_book_text(book_key)
    if not raw_text:
        print("  Extracting text from PDF...")
        parts = []
        with pdfplumber.open(book["path"]) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    parts.append(txt)
        raw_text = "\n".join(parts)
        if not raw_text:
            print("  No text found in PDF.")
            return {}
        save_raw_book_text(book_key, raw_text)
    else:
        print("  Raw text loaded from cache.")
    rules = _extract_rules_from_text(raw_text)
    save_rules(f"book_{book_key}", "full", rules)
    return rules


def _learn_scanned_book(book_key: str, book: dict) -> dict:
    """
    Scanned PDFs processed as image batches.
    No Gemini file upload — eliminates ResumableUploadError.
    """
    print("  Scanned PDF — processing as image batches...")
    rules = _extract_rules_from_pdf_images(book["path"], book_key)
    if not rules:
        print("  No rules extracted.")
        return {}
    save_rules(f"book_{book_key}", "full", rules)
    return rules

# =========================================================
# MERGE FUNCTIONS
# =========================================================

def merge_all_channels(topic: str) -> dict:
    """Merges all cached YouTube channel rules for a topic."""
    all_rules = []
    for ck in YOUTUBE_CHANNELS:
        rules = load_rules(ck, topic)
        if rules:
            rules["_source"]      = ck
            rules["_source_type"] = "youtube"
            all_rules.append(rules)
    if not all_rules:
        print(f"No saved channel rules for topic: {topic}")
        return {}
    print(f"Merging {len(all_rules)} channel rule sets...")
    return _tournament_merge(all_rules)

def merge_all_knowledge(topic: str) -> dict:
    """
    Merges everything — YouTube channels + books — into one
    master knowledge base. Books weighted 1.5x (more structured).
    Only uses cached data — no new API calls.
    """
    all_rules = []
    for ck in YOUTUBE_CHANNELS:
        rules = load_rules(ck, topic)
        if rules:
            rules.update({"_source": ck, "_source_type": "youtube", "_weight": 1.0})
            all_rules.append(rules)
    for bk in BOOK_DATABASE:
        rules = load_rules(f"book_{bk}", "full")
        if rules:
            rules.update({"_source": bk, "_source_type": "book", "_weight": 1.5})
            all_rules.append(rules)
    if not all_rules:
        print(f"No knowledge found for topic: {topic}")
        return {}
    print(f"Merging {len(all_rules)} sources (channels + books) on topic: {topic}...")
    final = _tournament_merge(all_rules)
    save_rules("master_knowledge", topic, final)
    return final


def check_for_new_content(channel_key: str, topic: str) -> dict:
    """Re-learns from channel only if new videos have been posted."""
    rules = load_rules(channel_key, topic)
    if not rules:
        return learn_from_channel(channel_key, topic)
    last = rules.get("_last_updated", "2000-01-01")
    ch   = YOUTUBE_CHANNELS[channel_key]
    latest = get_channel_videos(ch["channel_id"], max_results=5, order="date")
    if not latest:
        return rules
    if latest[0]["published_at"] > last:
        print(f"New content found for {channel_key} — re-learning.")
        return learn_from_channel(channel_key, topic, force_refresh=True)
    print(f"No new content for {channel_key} — cache current.")
    return rules


def update_rule_confidence(source_key: str, topic: str, rule_key: str, pnl: float):
    """Called after each trade to update rule confidence scores."""
    rules = load_rules(source_key, topic)
    if not rules or rule_key not in rules:
        return
    if "score" not in rules[rule_key]:
        rules[rule_key]["score"] = 0.5
    rules[rule_key]["score"] += 0.02 if pnl > 0 else -0.02
    rules[rule_key]["score"]  = max(0.0, min(1.0, rules[rule_key]["score"]))
    if rules[rule_key]["score"] < 0.3:
        rules[rule_key]["status"] = "flagged"
        print(f"  Rule '{rule_key}' flagged — confidence: {rules[rule_key]['score']:.2f}")
    save_rules(source_key, topic, rules)


def ask_question(source_key: str, topic: str, question: str) -> str:
    """Ask Gemini a question about saved rules. Answer cached by hash."""
    q_hash = hashlib.md5(f"{source_key}_{topic}_{question}".encode()).hexdigest()[:12]
    cached = load_query(q_hash)
    if cached:
        return cached.get("answer", "")
    rules = load_rules(source_key, topic)
    if not rules:
        return f"No rules found for {source_key} / {topic}."
    answer = _ask_gemini(
        f"Based on these trading rules:\n{json.dumps(rules, indent=2)}\n\nAnswer concisely:\n{question}"
    )
    save_query(q_hash, {"source_key": source_key, "topic": topic, "question": question,
                        "answer": answer, "timestamp": str(datetime.now())})
    return answer
