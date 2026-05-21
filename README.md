# MarketOracle — AI Trading System

An AI trading system that learns from YouTube channels and trading books,
develops strategies per symbol, and trains a Reinforcement Learning agent
to trade long and short across multiple markets.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add API keys
```bash
cp .env.example .env
# Edit .env and add your keys:
# GEMINI_API_KEY  → https://aistudio.google.com (free)
# YOUTUBE_API_KEY → https://console.cloud.google.com (free)
```

### 3. Download training data
```bash
python data_downloader.py --quick
```

### 4. Run system test
```bash
python test.py
```

### 5. Train
```bash
# First run (learns from YouTube/books, optimizes, trains)
python main.py

# Skip learning (use cached knowledge)
python main.py --skip-learning

# Quick test (1000 timesteps)
python main.py --skip-learning --skip-optimization --timesteps 1000
```

## Project Structure

```
MarketOracle/
├── .env                        ← API keys (never commit this)
├── .env.example                ← Template for .env
├── main.py                     ← Training pipeline entry point
├── data_downloader.py          ← Download market data
├── test.py                     ← System test (run before main.py)
├── requirements.txt
│
├── experts/
│   ├── db_handler.py           ← Centralized file IO (imported by all)
│   ├── knowledge_base.py       ← Learn from YouTube + books (Gemini)
│   ├── strategy_tester.py      ← Test + optimize strategies per symbol
│   ├── chart_expert.py         ← RL environment (long/short/switch)
│   ├── data_downloader.py      ← Data download module
│   ├── news_expert.py          ← News sentiment (future)
│   └── social_expert.py        ← Social sentiment (future)
│
├── core/
│   ├── boss_ai.py              ← Final trade decision (planned)
│   └── auditor_ai.py           ← Self-improving system (planned)
│
├── knowledge/                  ← PDF books + cached learning
│   ├── *.PDF                   ← Your trading books
│   ├── raw/transcripts/        ← Cached YouTube transcripts
│   ├── raw/books/              ← Cached book text
│   ├── extracted/              ← Cached Gemini rule extractions
│   └── queries/                ← Cached question answers
│
├── data/                       ← Downloaded price CSVs
├── models/                     ← Saved RL models
├── results/                    ← Evaluation results
└── strategies/
    ├── optimized/              ← Per-symbol optimized strategy configs
    └── master/                 ← AI-generated master strategies
```

## Module Connections

```
knowledge_base.py ──→ learns rules from YouTube + books
       ↓
strategy_tester.py ──→ tests + optimizes rules per symbol
       ↓                  (reads from db_handler)
chart_expert.py ──→ RL agent uses optimized strategy signals
       ↓                  (long/short/hold/close/switch)
main.py ──→ orchestrates the full pipeline
```

## Actions

| Action | Meaning |
|--------|---------|
| 0 | Hold |
| 1 | Buy (open long) |
| 2 | Sell (open short) |
| 3 | Close position |
| 4 | Switch symbol |

## Push to GitHub

```bash
git add -A
git commit -m "Update MarketOracle"
git push origin main
```
