"""
db_handler.py

Central read/write handler for all JSON and file operations.
Both knowledge_base.py and strategy_tester.py import from here
instead of importing from each other — eliminates circular imports.

No API calls made here — pure file IO only.
"""

import os
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# PATHS — single source of truth for the entire project
# =========================================================

ROOT_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DIR      = os.path.join(ROOT_DIR, "knowledge")
RAW_TRANSCRIPT_DIR = os.path.join(KNOWLEDGE_DIR, "raw", "transcripts")
RAW_BOOK_DIR       = os.path.join(KNOWLEDGE_DIR, "raw", "books")
EXTRACTED_DIR      = os.path.join(KNOWLEDGE_DIR, "extracted")
QUERIES_DIR        = os.path.join(KNOWLEDGE_DIR, "queries")
CONFLICTS_PATH     = os.path.join(KNOWLEDGE_DIR, "conflicts_log.json")
STRATEGIES_DIR     = os.path.join(ROOT_DIR, "strategies")
OPTIMIZED_DIR      = os.path.join(STRATEGIES_DIR, "optimized")
MASTER_DIR         = os.path.join(STRATEGIES_DIR, "master")
DATA_DIR           = os.path.join(ROOT_DIR, "data")
MODELS_DIR         = os.path.join(ROOT_DIR, "models")
RESULTS_DIR        = os.path.join(ROOT_DIR, "results")


def ensure_dirs():
    """Creates all required directories if they don't exist."""
    for path in [
        RAW_TRANSCRIPT_DIR, RAW_BOOK_DIR, EXTRACTED_DIR,
        QUERIES_DIR, OPTIMIZED_DIR, MASTER_DIR,
        DATA_DIR, MODELS_DIR, RESULTS_DIR,
    ]:
        os.makedirs(path, exist_ok=True)


ensure_dirs()

# =========================================================
# HELPERS
# =========================================================

def safe_symbol_name(symbol: str) -> str:
    """Converts symbol to a safe filename. EUR/USD -> EUR_USD"""
    return symbol.replace("/", "_").replace("\\", "_").replace(" ", "_")

# =========================================================
# KNOWLEDGE RULES — READ / WRITE
# =========================================================

def _extracted_path(source_key: str, topic: str) -> str:
    safe_topic = topic.replace(" ", "_").lower()
    return os.path.join(EXTRACTED_DIR, f"{source_key}_{safe_topic}.json")


def save_rules(source_key: str, topic: str, rules: dict):
    """Saves extracted rules to the knowledge cache."""
    path = _extracted_path(source_key, topic)
    rules["_last_updated"] = str(datetime.now())
    rules["_source"]       = source_key
    rules["_topic"]        = topic
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)
    print(f"  Rules saved -> {path}")


def load_rules(source_key: str, topic: str) -> Optional[dict]:
    """Loads extracted rules from cache. Returns None if not found."""
    path = _extracted_path(source_key, topic)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# =========================================================
# RAW CONTENT — READ / WRITE
# =========================================================

def save_raw_transcript(video_id: str, text: str):
    path = os.path.join(RAW_TRANSCRIPT_DIR, f"{video_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Transcript cached -> {path}")


def load_raw_transcript(video_id: str) -> Optional[str]:
    path = os.path.join(RAW_TRANSCRIPT_DIR, f"{video_id}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def save_raw_book_text(book_key: str, text: str):
    path = os.path.join(RAW_BOOK_DIR, f"{book_key}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Book text cached -> {path}")


def load_raw_book_text(book_key: str) -> Optional[str]:
    path = os.path.join(RAW_BOOK_DIR, f"{book_key}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None

# =========================================================
# QUERY CACHE — READ / WRITE
# =========================================================

def save_query(question_hash: str, data: dict):
    path = os.path.join(QUERIES_DIR, f"{question_hash}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_query(question_hash: str) -> Optional[dict]:
    path = os.path.join(QUERIES_DIR, f"{question_hash}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# =========================================================
# CONFLICTS LOG
# =========================================================

def log_conflicts(conflicts: list):
    """Appends rule conflicts for Auditor AI LLM1 to review."""
    existing = []
    if os.path.exists(CONFLICTS_PATH):
        try:
            with open(CONFLICTS_PATH) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append({"timestamp": str(datetime.now()), "conflicts": conflicts})
    with open(CONFLICTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Conflicts logged -> {CONFLICTS_PATH}")

# =========================================================
# OPTIMIZED STRATEGIES — READ / WRITE
# =========================================================

def save_optimized_strategy(symbol_name: str, strategy: dict):
    """Saves an optimized strategy config for a specific symbol."""
    path = os.path.join(OPTIMIZED_DIR, f"{safe_symbol_name(symbol_name)}.json")
    strategy_copy = {k: v for k, v in strategy.items() if k != "data"}
    with open(path, "w") as f:
        json.dump(strategy_copy, f, indent=2)
    print(f"  Optimized strategy saved -> {path}")


def load_optimized_strategy(symbol_name: str) -> Optional[dict]:
    """Loads an optimized strategy for a symbol if it exists."""
    path = os.path.join(OPTIMIZED_DIR, f"{safe_symbol_name(symbol_name)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_master_strategy(symbol_name: str, strategy: dict):
    """Saves an AI master strategy for a specific symbol."""
    path = os.path.join(MASTER_DIR, f"{safe_symbol_name(symbol_name)}.json")
    with open(path, "w") as f:
        json.dump(strategy, f, indent=2)
    print(f"  Master strategy saved -> {path}")


def load_master_strategy(symbol_name: str) -> Optional[dict]:
    """Loads an AI master strategy for a symbol if it exists."""
    path = os.path.join(MASTER_DIR, f"{safe_symbol_name(symbol_name)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None
