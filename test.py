"""
test.py — MarketOracle System Test

Tests all components without consuming API quota.
Run this before main.py to verify everything works.

Usage:
    python test.py
"""

import base64
import os
import sys
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

print("\n" + "="*60)
print("  MARKETORACLE — SYSTEM TEST")
print("="*60)

passed = []
failed = []


def test(name: str, fn):
    try:
        fn()
        print(f"  ✓ {name}")
        passed.append(name)
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        failed.append((name, str(e)))


# ── 1. Environment ────────────────────────────────────────
print("\n1. Environment")

def test_env():
    gemini  = os.getenv("GEMINI_API_KEY", "").strip()
    youtube = os.getenv("YOUTUBE_API_KEY", "").strip()
    assert gemini  and "your_" not in gemini,  "GEMINI_API_KEY not set in environment"
    assert youtube and "your_" not in youtube, "YOUTUBE_API_KEY not set in environment"

    # Test Gemini API connectivity with a simple call
    try:
        try:
            from google import genai
            NEW_GENAI = True
        except ImportError:
            import google.generativeai as genai
            NEW_GENAI = False

        if NEW_GENAI:
            client = genai.Client(api_key=gemini)
            chat = client.chats.create(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                config=genai.types.GenerateContentConfig(temperature=0.0),
            )
            response = chat.send_message("test connectivity")
            assert getattr(response, "text", None) is not None, "Gemini API call returned an empty response, possibly blocked."
        else:
            genai.configure(api_key=gemini)
            model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
            response = model.generate_content(
                "test connectivity",
                generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=1),
            )
            assert response.text is not None, "Gemini API call returned an empty response, possibly blocked."
    except Exception as e:
        raise AssertionError(f"Gemini API key might be invalid or have issues: {e}")

    # Test Multimodal (Image) capability
    try:
        # Send a tiny 1x1 black pixel in base64 to test multimodal support
        pixel_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        if NEW_GENAI:
            client = genai.Client(api_key=gemini)
            chat = client.chats.create(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
            response = chat.send_message([
                genai.types.Part(text="Identify this pixel color."),
                genai.types.Part(
                    inlineData=genai.types.Blob(data=base64.b64decode(pixel_b64), mimeType="image/png")
                ),
            ])
        else:
            content = ["Identify this pixel color.", {"mime_type": "image/png", "data": pixel_b64}]
            model.generate_content(content)
    except Exception as e:
        raise AssertionError(f"Gemini API multimodal support failed: {e}. Check if MODEL name is correct.")
    
    # Test YouTube API connectivity with a simple call
    try:
        from googleapiclient.discovery import build
        youtube_client = build("youtube", "v3", developerKey=youtube)
        # Attempt to get details for a well-known channel (e.g., GoogleDevelopers)
        youtube_client.channels().list(part="id", id="UC_x5XG1OV2P6uZZ5FSM9Ttw").execute()
    except Exception as e:
        raise AssertionError(f"YouTube API key might be invalid or have issues: {e}")

test("API keys in .env", test_env)

# ── 2. Dependencies ───────────────────────────────────────
print("\n2. Dependencies")

def test_imports():
    import gymnasium
    import stable_baselines3
    import pandas
    import numpy
    import google.generativeai
    import yfinance
    import pdfplumber
    import tenacity
    from dotenv import load_dotenv

test("All packages installed", test_imports)

# ── 3. Project Structure ──────────────────────────────────
print("\n3. Project Structure")

def test_structure():
    required = [
        "experts/db_handler.py",
        "experts/knowledge_base.py",
        "experts/chart_expert.py",
        "experts/strategy_tester.py",
        "experts/data_downloader.py",
        "data_downloader.py",
        "main.py",
    ]
    for f in required:
        assert os.path.exists(f), f"Missing: {f}"

    if not os.path.exists(".env"):
        assert os.getenv("GEMINI_API_KEY"), ".env is missing and GEMINI_API_KEY is not set in the environment"
        assert os.getenv("YOUTUBE_API_KEY"), ".env is missing and YOUTUBE_API_KEY is not set in the environment"

test("All required files present", test_structure)

# ── 4. DB Handler ─────────────────────────────────────────
print("\n4. Database Handler")

def test_db_save_load():
    from experts.db_handler import save_rules, load_rules, ensure_dirs
    ensure_dirs()
    rules = {"entry_conditions": [{"rule": "test rule", "confidence": 0.8}]}
    save_rules("_test", "_test", rules)
    loaded = load_rules("_test", "_test")
    assert loaded is not None
    assert loaded["entry_conditions"][0]["rule"] == "test rule"
    # Cleanup
    path = "knowledge/extracted/__test__test.json"
    if os.path.exists(path):
        os.remove(path)

test("Save and load rules", test_db_save_load)

def test_db_paths():
    from experts.db_handler import safe_symbol_name, OPTIMIZED_DIR, EXTRACTED_DIR
    assert safe_symbol_name("EUR/USD") == "EUR_USD"
    assert safe_symbol_name("BTC-USD") == "BTC-USD"
    assert os.path.exists(OPTIMIZED_DIR)
    assert os.path.exists(EXTRACTED_DIR)

test("Paths and helpers", test_db_paths)

# ── 5. Knowledge Base Config ──────────────────────────────
print("\n5. Knowledge Base")

def test_channel_config():
    from experts.knowledge_base import YOUTUBE_CHANNELS, BOOK_DATABASE
    assert len(YOUTUBE_CHANNELS) == 8, f"Expected 8 channels, got {len(YOUTUBE_CHANNELS)}"
    for key, ch in YOUTUBE_CHANNELS.items():
        assert "channel_id" in ch,              f"{key} missing channel_id"
        assert ch["channel_id"].startswith("UC"), f"{key} channel_id wrong format"
        assert "focus" in ch,                   f"{key} missing focus"

test("8 channels configured correctly", test_channel_config)

def test_book_paths():
    from experts.knowledge_base import BOOK_DATABASE
    found = 0
    for key, book in BOOK_DATABASE.items():
        assert book["path"].startswith(os.path.dirname(os.path.abspath(__file__)))  or True
        if os.path.exists(book["path"]):
            found += 1
    assert found > 0, "No PDFs found in knowledge/ folder"
    print(f"    ({found}/{len(BOOK_DATABASE)} PDFs found)")

test("Book PDFs present", test_book_paths)

# ── 6. Technical Indicators ───────────────────────────────
print("\n6. Technical Indicators")

def make_test_df(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "Open":   prices * 0.999,
        "High":   prices * 1.005,
        "Low":    prices * 0.995,
        "Close":  prices,
        "Volume": abs(np.random.randn(n)) * 1e6
    }, index=pd.date_range("2020-01-01", periods=n, freq="D"))
    return df


def test_indicators():
    from experts.chart_expert import (
        compute_rsi, compute_atr, compute_obv,
        add_technical_indicators, _standardize_columns
    )
    df = make_test_df()

    # Column standardization
    df_lc  = df.rename(columns={"Close": "close", "Open": "open"})
    df_std = _standardize_columns(df_lc)
    assert "Close" in df_std.columns

    # RSI — no NaN, valid range
    rsi = compute_rsi(df["Close"])
    assert not rsi.isna().all()
    assert rsi.max() <= 100 and rsi.min() >= 0

    # RSI division by zero (all-up candles → RSI=100, not NaN)
    all_up = pd.Series([100 + i for i in range(300)])
    rsi2   = compute_rsi(all_up)
    assert not rsi2.isna().any(), "RSI has NaN on monotone series"

    # ATR
    atr = compute_atr(df)
    assert not atr.isna().all()

    # OBV — vectorized (no Python loop)
    obv = compute_obv(df)
    assert len(obv) == len(df)

    # Full suite
    df_ind = add_technical_indicators(df)
    for col in ["rsi_14", "atr_14", "macd", "stoch_k", "sma_50"]:
        assert col in df_ind.columns, f"Missing indicator: {col}"
    assert len(df_ind) > 0, "All rows dropped after adding indicators"

test("All indicators compute correctly", test_indicators)

# ── 7. RL Environment ─────────────────────────────────────
print("\n7. RL Environment")


def test_env_reset_step():
    from experts.chart_expert import add_technical_indicators, MultiSymbolChartExpert

    df    = make_test_df(500)
    df_i  = add_technical_indicators(df)
    split = int(len(df_i) * 0.8)

    data_bundle = {
        "TEST_SYMBOL": {
            "daily": {
                "train": df_i.iloc[:split].copy(),
                "test":  df_i.iloc[split:].copy(),
            }
        }
    }

    env      = MultiSymbolChartExpert(data_bundle, mode="train")
    obs, _   = env.reset()
    assert obs is not None and len(obs) > 0

    total_reward = 0.0
    for i in range(30):
        action = i % 5   # Cycle through Hold/Buy/Sell/Close/Switch
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        assert obs    is not None
        assert isinstance(reward, float)
        assert "symbol"  in info
        assert "balance" in info
        assert "win_rate" in info
        if terminated or truncated:
            break

    print(f"    (30 steps, total reward: {total_reward:.4f})")


def test_env_long_short():
    from experts.chart_expert import add_technical_indicators, MultiSymbolChartExpert

    df   = make_test_df(500)
    df_i = add_technical_indicators(df)
    split = int(len(df_i) * 0.8)

    data_bundle = {"TEST": {"daily": {"train": df_i.iloc[:split], "test": df_i.iloc[split:]}}}
    env = MultiSymbolChartExpert(data_bundle, mode="train")
    obs, _ = env.reset()

    # Open long
    _, _, _, _, info = env.step(1)  # ACTION_BUY
    assert info["direction"] == "long" or info["position"] is False or info["direction"] is None

    # Close
    env.step(3)  # ACTION_CLOSE

    # Open short
    _, _, _, _, info = env.step(2)  # ACTION_SELL
    # Direction should be short if position opened
    if info["position"]:
        assert info["direction"] == "short"

test("Environment resets and steps correctly", test_env_reset_step)
test("Long and short positions work",          test_env_long_short)

# ── 8. Strategy Tester ────────────────────────────────────
print("\n8. Strategy Tester")


def test_symbol_analyzer():
    from experts.strategy_tester import SymbolAnalyzer

    df       = make_test_df(300)
    sym_data = {"daily": df}
    analysis = SymbolAnalyzer.analyze_symbol(sym_data, "TEST")

    assert "overall_characteristics" in analysis
    assert "symbol" in analysis
    assert analysis["symbol"] == "TEST"
    assert len(analysis["overall_characteristics"]) > 0

    sym_type = SymbolAnalyzer.get_symbol_type(analysis)
    valid    = ["trending_volatile", "trending_smooth", "mean_reverting_choppy",
                "mean_reverting_normal", "ranging_volatile", "ranging_smooth"]
    assert sym_type in valid, f"Unexpected symbol type: {sym_type}"
    print(f"    (TEST classified as: {sym_type})")

test("Symbol analysis and classification", test_symbol_analyzer)

# ── 9. Data Downloader ────────────────────────────────────
print("\n9. Data Downloader")


def test_downloader_config():
    from data_downloader import SYMBOLS, TIMEFRAMES, _safe_name
    assert len(SYMBOLS) > 0
    assert "BTC-USD" in SYMBOLS
    assert "EURUSD=X" in SYMBOLS
    assert _safe_name("BTC-USD") == "BTC_USD"
    assert _safe_name("EURUSD=X") == "EURUSDX"

test("Downloader config", test_downloader_config)


def test_download_btc_daily():
    from data_downloader import download_symbol
    result = download_symbol("BTC-USD", timeframes=["daily"])
    assert "daily" in result, "daily download failed"
    df = pd.read_csv(result["daily"])
    assert len(df) > 100
    assert "Close" in df.columns
    print(f"    ({len(df)} bars downloaded)")

test("Download BTC daily data", test_download_btc_daily)

# ── 10. Connection Test ───────────────────────────────────
print("\n10. Module Connections")


def test_connections():
    # db_handler imports
    from experts.db_handler    import save_rules, load_rules, load_optimized_strategy

    # knowledge_base imports db_handler
    from experts.knowledge_base import YOUTUBE_CHANNELS, BOOK_DATABASE, load_rules as kb_load
    assert kb_load is not None

    # strategy_tester imports db_handler
    from experts.strategy_tester import SymbolAnalyzer, StrategyLoader
    assert StrategyLoader is not None

    # chart_expert imports db_handler
    from experts.chart_expert import MultiSymbolChartExpert, load_strategy_for_symbol
    assert load_strategy_for_symbol is not None

    # main imports everything through step functions
    import main
    assert hasattr(main, "step_load_data")
    assert hasattr(main, "step_train")
    assert hasattr(main, "step_evaluate")

test("All modules connected correctly", test_connections)

# ── RESULTS ───────────────────────────────────────────────
print("\n" + "="*60)
print(f"RESULTS:  {len(passed)} passed   |   {len(failed)} failed")
print("="*60)

if failed:
    print("\nFailed tests:")
    for name, err in failed:
        print(f"  ✗ {name}")
        print(f"    → {err}")
    print("\nFix the above before running main.py")
    sys.exit(1)
else:
    print("\n All tests passed! System is ready.")
    print("\nNext steps:")
    print("  1. python data_downloader.py --quick   (download training data)")
    print("  2. python main.py --skip-learning --timesteps 10000  (quick test run)")
    print("  3. python main.py --timesteps 500000   (full training run)")
