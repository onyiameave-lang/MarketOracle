"""
db_handler.py

Lightweight JSON read/write handler for the knowledge base.
Both knowledge_base.py and strategy_tester.py import from here
instead of importing from each other — eliminates circular imports.
"""

import os
import json
from datetime import datetime
from typing import Optional

# =========================================================
# PATHS — single source of truth for all cache paths
# =========================================================

RAW_TRANSCRIPT_DIR = "knowledge/raw/transcripts"
RAW_BOOK_DIR       = "knowledge/raw/books"
EXTRACTED_DIR      = "knowledge/extracted"
QUERIES_DIR        = "knowledge/queries"
CONFLICTS_PATH     = "knowledge/conflicts_log.json"
OPTIMIZED_DIR      = "strategies/optimized"
MASTER_DIR         = "strategies/master"


def ensure_dirs():
    """Creates all required directories if they don't exist."""
    for path in [
        RAW_TRANSCRIPT_DIR,
        RAW_BOOK_DIR,
        EXTRACTED_DIR,
        QUERIES_DIR,
        OPTIMIZED_DIR,
        MASTER_DIR,
    ]:
        os.makedirs(path, exist_ok=True)


ensure_dirs()

# =========================================================
# KNOWLEDGE BASE — READ / WRITE
# =========================================================

def _extracted_path(source_key: str, topic: str) -> str:
    safe_topic = topic.replace(" ", "_").lower()
    return f"{EXTRACTED_DIR}/{source_key}_{safe_topic}.json"


def save_rules(source_key: str, topic: str, rules: dict):
    """
    Saves extracted rules to the knowledge cache.
    Stamps metadata (timestamp, source, topic) on every save.
    """
    path = _extracted_path(source_key, topic)

    rules["_last_updated"] = str(datetime.now())
    rules["_source"]       = source_key
    rules["_topic"]        = topic

    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)

    print(f"  Rules saved -> {path}")


def load_rules(source_key: str, topic: str) -> Optional[dict]:
    """
    Loads extracted rules from the knowledge cache.
    Returns None if no rules saved for this source/topic.
    """
    path = _extracted_path(source_key, topic)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


# =========================================================
# RAW CONTENT — READ / WRITE
# =========================================================

def transcript_cache_path(video_id: str) -> str:
    return f"{RAW_TRANSCRIPT_DIR}/{video_id}.txt"


def book_cache_path(book_key: str) -> str:
    return f"{RAW_BOOK_DIR}/{book_key}.txt"


def save_raw_transcript(video_id: str, text: str):
    path = transcript_cache_path(video_id)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Transcript cached -> {path}")


def load_raw_transcript(video_id: str) -> Optional[str]:
    path = transcript_cache_path(video_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def save_raw_book_text(book_key: str, text: str):
    path = book_cache_path(book_key)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Book text cached -> {path}")


def load_raw_book_text(book_key: str) -> Optional[str]:
    path = book_cache_path(book_key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


# =========================================================
# QUERY CACHE — READ / WRITE
# =========================================================

def query_path(question_hash: str) -> str:
    return f"{QUERIES_DIR}/{question_hash}.json"


def save_query(question_hash: str, data: dict):
    path = query_path(question_hash)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_query(question_hash: str) -> Optional[dict]:
    path = query_path(question_hash)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# =========================================================
# CONFLICTS LOG
# =========================================================

def log_conflicts(conflicts: list):
    """Appends rule conflicts to the conflicts log for Auditor AI."""
    existing = []

    if os.path.exists(CONFLICTS_PATH):
        try:
            with open(CONFLICTS_PATH) as f:
                existing = json.load(f)
        except Exception:
            existing = []

    existing.append({
        "timestamp": str(datetime.now()),
        "conflicts": conflicts
    })

    with open(CONFLICTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"  Conflicts logged -> {CONFLICTS_PATH}")


# =========================================================
# OPTIMIZED STRATEGIES — READ / WRITE
# =========================================================

def safe_symbol_name(symbol: str) -> str:
    """Converts symbol name to safe filename. EUR/USD -> EUR_USD"""
    return symbol.replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_optimized_strategy(symbol_name: str, strategy: dict):
    """Saves an optimized strategy for a specific symbol."""
    safe_name = safe_symbol_name(symbol_name)
    path      = f"{OPTIMIZED_DIR}/{safe_name}.json"

    strategy_copy = {k: v for k, v in strategy.items() if k != "data"}

    with open(path, "w") as f:
        json.dump(strategy_copy, f, indent=2)

    print(f"  Optimized strategy saved -> {path}")


def load_optimized_strategy(symbol_name: str) -> Optional[dict]:
    """Loads an optimized strategy for a symbol if it exists."""
    safe_name = safe_symbol_name(symbol_name)
    path      = f"{OPTIMIZED_DIR}/{safe_name}.json"

    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    return None


def save_master_strategy(symbol_name: str, strategy: dict):
    """Saves a master AI strategy for a specific symbol."""
    safe_name = safe_symbol_name(symbol_name)
    path      = f"{MASTER_DIR}/{safe_name}.json"

    with open(path, "w") as f:
        json.dump(strategy, f, indent=2)

    print(f"  Master strategy saved -> {path}")


def load_master_strategy(symbol_name: str) -> Optional[dict]:
    """Loads a master AI strategy for a symbol if it exists."""
    safe_name = safe_symbol_name(symbol_name)
    path      = f"{MASTER_DIR}/{safe_name}.json"

    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    return None
