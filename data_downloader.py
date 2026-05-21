"""
data_downloader.py

Downloads OHLCV price data for all trading symbols
used in MarketOracle training.

Saves CSVs to data/ folder: SYMBOL_TIMEFRAME.csv
e.g. BTC_USD_daily.csv, AAPL_1h.csv

Usage:
    python data_downloader.py                        # Download all
    python data_downloader.py --symbols BTC-USD AAPL # Specific symbols
    python data_downloader.py --quick                # Fewer timeframes
    python data_downloader.py --verify               # Check existing data
"""

import os
import time
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# =========================================================
# SYMBOLS TO DOWNLOAD
# =========================================================

SYMBOLS = {
    # Crypto
    "BTC-USD":  "Bitcoin",
    "ETH-USD":  "Ethereum",
    "LTC-USD":  "Litecoin",
    # Stocks
    "AAPL":     "Apple",
    "MSFT":     "Microsoft",
    "GOOGL":    "Google",
    # Commodities
    "GC=F":     "Gold",
    "CL=F":     "Oil",
    # Forex
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
}

# =========================================================
# TIMEFRAME CONFIG
# =========================================================

TIMEFRAMES = {
    "weekly": {"interval": "1wk", "period": "10y",  "min_bars": 100},
    "daily":  {"interval": "1d",  "period": "5y",   "min_bars": 200},
    "4h":     {"interval": "1h",  "period": "730d", "min_bars": 100, "resample": True},
    "1h":     {"interval": "1h",  "period": "730d", "min_bars": 200},
    "30min":  {"interval": "30m", "period": "60d",  "min_bars": 100},
    "15min":  {"interval": "15m", "period": "60d",  "min_bars": 100},
    "5min":   {"interval": "5m",  "period": "60d",  "min_bars": 100},
}

QUICK_TIMEFRAMES = ["weekly", "daily", "1h", "15min"]


def _safe_name(symbol: str) -> str:
    return symbol.replace("/", "").replace("=", "").replace("-", "_")


def _resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    df = df_1h.copy()
    df.index = pd.to_datetime(df.index)
    return df.resample("4h").agg({
        "Open": "first", "High": "max",
        "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()


def download_symbol(symbol: str, timeframes: list = None) -> dict:
    """Downloads all requested timeframes for a symbol."""
    clean   = _safe_name(symbol)
    tfs     = timeframes or list(TIMEFRAMES.keys())
    results = {}
    ticker  = yf.Ticker(symbol)
    df_1h   = None

    print(f"\n{'='*40}")
    print(f"{symbol} ({SYMBOLS.get(symbol, 'Unknown')})")
    print(f"{'='*40}")

    for tf_name in tfs:
        if tf_name not in TIMEFRAMES:
            continue

        tf   = TIMEFRAMES[tf_name]
        path = f"{DATA_DIR}/{clean}_{tf_name}.csv"

        # Skip if downloaded in last 24h
        if os.path.exists(path):
            age = datetime.now().timestamp() - os.path.getmtime(path)
            if age < 86400:
                df_c = pd.read_csv(path)
                print(f"  {tf_name:<8} {len(df_c):>6} bars (cached)")
                results[tf_name] = path
                if tf_name == "1h":
                    df_1h = pd.read_csv(path, index_col=0, parse_dates=True)
                continue

        try:
            if tf.get("resample"):
                # Build 4h from 1h
                if df_1h is None:
                    p1h = f"{DATA_DIR}/{clean}_1h.csv"
                    if os.path.exists(p1h):
                        df_1h = pd.read_csv(p1h, index_col=0, parse_dates=True)
                    else:
                        raw = ticker.history(interval="1h", period="730d")
                        df_1h = raw[["Open","High","Low","Close","Volume"]].dropna()
                if df_1h is None or df_1h.empty:
                    print(f"  {tf_name:<8} Needs 1h data first")
                    continue
                df = _resample_4h(df_1h)
            else:
                raw = ticker.history(interval=tf["interval"], period=tf["period"])
                df  = raw[["Open","High","Low","Close","Volume"]].dropna()

            if df.empty or len(df) < tf["min_bars"]:
                print(f"  {tf_name:<8} Too few bars ({len(df)})")
                continue

            if tf_name == "1h":
                df_1h = df.copy()

            df.to_csv(path)
            print(f"  {tf_name:<8} {len(df):>6} bars → {path}")
            results[tf_name] = path
            time.sleep(0.5)

        except Exception as e:
            print(f"  {tf_name:<8} Error: {e}")

    return results


def download_all(symbols: list = None, timeframes: list = None, quick: bool = False) -> dict:
    """Downloads data for all symbols."""
    syms = symbols    or list(SYMBOLS.keys())
    tfs  = timeframes or (QUICK_TIMEFRAMES if quick else list(TIMEFRAMES.keys()))

    print(f"\nMarketOracle Data Downloader")
    print(f"Symbols: {len(syms)}  |  Timeframes: {tfs}")
    print(f"Output:  {DATA_DIR}/")

    results = {}
    failed  = []

    for sym in syms:
        try:
            r = download_symbol(sym, tfs)
            if r:
                results[sym] = r
            else:
                failed.append(sym)
        except Exception as e:
            print(f"  ERROR {sym}: {e}")
            failed.append(sym)
        time.sleep(1)

    csvs = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    size = sum(os.path.getsize(f"{DATA_DIR}/{f}") for f in csvs)

    print(f"\n{'='*40}")
    print(f"Done: {len(results)}/{len(syms)} symbols")
    if failed:
        print(f"Failed: {failed}")
    print(f"Files: {len(csvs)} CSVs ({size/(1024*1024):.1f} MB)")

    return results


def verify_data() -> bool:
    """Checks data is ready for training."""
    print("\nVerifying data...")
    csvs = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csvs:
        print("ERROR: No CSV files found in data/")
        return False

    syms = set()
    for f in csvs:
        parts = f.replace(".csv","").rsplit("_", 1)
        if len(parts) == 2:
            syms.add(parts[0])

    ready = []
    for s in syms:
        has_weekly = os.path.exists(f"{DATA_DIR}/{s}_weekly.csv")
        has_daily  = os.path.exists(f"{DATA_DIR}/{s}_daily.csv")
        has_entry  = any(os.path.exists(f"{DATA_DIR}/{s}_{tf}.csv") for tf in ["1h","15min","5min","30min"])
        if has_weekly and has_daily and has_entry:
            ready.append(s)

    print(f"Training-ready symbols: {len(ready)}/{len(syms)}")
    for s in ready:
        print(f"  ✓ {s}")

    return len(ready) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketOracle Data Downloader")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to download")
    parser.add_argument("--quick",   action="store_true", help="Quick mode (fewer timeframes)")
    parser.add_argument("--verify",  action="store_true", help="Verify existing data only")
    args = parser.parse_args()

    if args.verify:
        verify_data()
    else:
        download_all(symbols=args.symbols, quick=args.quick)
        verify_data()
