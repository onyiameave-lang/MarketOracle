"""
chart_expert.py

Multi-symbol RL trading environment.

The agent experiences the whole market across all symbols,
learns which strategies fit which charts best, and decides
when to go long, short, close, or switch to a better symbol.

Actions:
    0 = Hold
    1 = Buy  (open long)
    2 = Sell (open short)
    3 = Close position
    4 = Switch symbol

Connects to:
    - strategy_tester.py  → loads optimized strategy config per symbol
<<<<<<< HEAD
    - knowledger.py       → loads/saves cached strategies
=======
    - knowledge_base.py   → translates learned rules into numeric signals
    - db_handler.py       → loads/saves cached strategies
>>>>>>> b4a9b09bebd015dcd26a15f2a5f32024a3853a92
"""

import os
import re
import glob
import random
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

<<<<<<< HEAD
=======
from dotenv import load_dotenv
load_dotenv()
>>>>>>> b4a9b09bebd015dcd26a15f2a5f32024a3853a92
from experts.db_handler import load_optimized_strategy, load_rules

# =========================================================
# TIMEFRAME HIERARCHY
# These are fixed — minutes NEVER go in higher_timeframes
# =========================================================

HIGHER_TIMEFRAMES   = ["weekly", "daily", "4h", "1h"]
ANALYSIS_TIMEFRAMES = ["30min", "15min"]
ENTRY_TIMEFRAMES    = ["5min", "1min"]

ALL_TIMEFRAMES = HIGHER_TIMEFRAMES + ANALYSIS_TIMEFRAMES + ENTRY_TIMEFRAMES

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

TRAIN_RATIO = 0.8   # 80% train, 20% test

# =========================================================
# ACTIONS
# =========================================================

ACTION_HOLD   = 0
ACTION_BUY    = 1   # Open long
ACTION_SELL   = 2   # Open short
ACTION_CLOSE  = 3   # Close any open position
ACTION_SWITCH = 4   # Move to a different symbol

# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI — fixed min_periods to avoid invalid early values."""
    delta    = series.diff().fillna(0)
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    # fillna(100): all-up candles = max RSI = 100, not NaN
    return (100 - 100 / (1 + rs)).fillna(100)


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR — fixed min_periods."""
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=window).mean()
    return atr.bfill()


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume — vectorized.
    Fixed: replaces the old O(n) Python for-loop with
    vectorized pandas operations. ~100x faster on large datasets.
    """
    direction    = np.sign(df["Close"].diff().fillna(0))
    signed_vol   = direction * df["Volume"]
    return signed_vol.cumsum()


def compute_macd(series: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9):
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist   = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def compute_bollinger(series: pd.Series,
                      window: int = 20,
                      num_std: int = 2):
    """Bollinger Bands — fixed min_periods."""
    sma   = series.rolling(window, min_periods=window).mean()
    std   = series.rolling(window, min_periods=window).std()
    upper = sma + std * num_std
    lower = sma - std * num_std
    return upper.bfill(), sma.bfill(), lower.bfill()


def compute_stochastic(df: pd.DataFrame,
                       k_window: int = 14,
                       d_window: int = 3):
    """Stochastic Oscillator — fixed min_periods."""
    low_k  = df["Low"].rolling(k_window,  min_periods=k_window).min()
    high_k = df["High"].rolling(k_window, min_periods=k_window).max()
    range_ = (high_k - low_k).replace(0, 1)
    k      = 100 * (df["Close"] - low_k) / range_
    d      = k.rolling(d_window, min_periods=d_window).mean()
    return k.fillna(50), d.fillna(50)


def compute_williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Williams %R — fixed min_periods."""
    high = df["High"].rolling(window, min_periods=window).max()
    low  = df["Low"].rolling(window,  min_periods=window).min()
    r    = -100 * (high - df["Close"]) / (high - low).replace(0, 1)
    return r.fillna(-50)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to a DataFrame.
    All rolling calculations use proper min_periods.
    Drops early rows where indicators are not yet valid.
    """
    df = df.copy()

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    # ── Price features ────────────────────────────────
    df["return_1"]        = df["Close"].pct_change().fillna(0.0)
    df["return_5"]        = df["Close"].pct_change(5).fillna(0.0)
    df["high_low_ratio"]  = df["High"] / df["Low"]
    df["close_open_ratio"] = df["Close"] / df["Open"]

    # ── Moving averages — fixed min_periods ───────────
    df["sma_10"]  = df["Close"].rolling(10,  min_periods=10).mean()
    df["sma_20"]  = df["Close"].rolling(20,  min_periods=20).mean()
    df["sma_50"]  = df["Close"].rolling(50,  min_periods=50).mean()
    df["ema_12"]  = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"]  = df["Close"].ewm(span=26, adjust=False).mean()

    # ── Momentum ──────────────────────────────────────
    df["rsi_14"]              = compute_rsi(df["Close"], 14)
    df["rsi_7"]               = compute_rsi(df["Close"], 7)
    df["stoch_k"], df["stoch_d"] = compute_stochastic(df, 14, 3)
    df["williams_r"]          = compute_williams_r(df, 14)

    # ── Volatility ────────────────────────────────────
    df["atr_14"]                             = compute_atr(df, 14)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = compute_bollinger(df["Close"], 20, 2)

    # ── Volume ────────────────────────────────────────
    df["volume_sma"] = df["Volume"].rolling(10, min_periods=10).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma"].replace(0, 1)
    df["obv"]          = compute_obv(df)

    # ── MACD ──────────────────────────────────────────
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["Close"])

    # ── Candle shape ──────────────────────────────────
    df["body_size"]    = (df["Close"] - df["Open"]).abs() / df["Close"]
    df["upper_shadow"] = (df["High"] - df[["Open","Close"]].max(axis=1)) / df["Close"]
    df["lower_shadow"] = (df[["Open","Close"]].min(axis=1) - df["Low"]) / df["Close"]

    # Drop early rows where key indicators are invalid
    # (prevents NaN from polluting the observation space)
    df = df.dropna(subset=["sma_50", "rsi_14", "atr_14", "stoch_k"])

    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column casing — handles close/Close/CLOSE."""
    rename = {}
    for col in df.columns:
        cl = col.lower()
        if cl == "open":   rename[col] = "Open"
        elif cl == "high": rename[col] = "High"
        elif cl == "low":  rename[col] = "Low"
        elif cl == "close": rename[col] = "Close"
        elif cl == "volume": rename[col] = "Volume"
    return df.rename(columns=rename) if rename else df

# =========================================================
# DATA LOADER
# =========================================================

def load_data_bundle(data_dir: str = "data") -> dict:
    """
    Loads all CSV/XLSX price data from the data directory.
    Applies train/test split to each symbol/timeframe.
    Returns a nested dict: {symbol: {timeframe: {train: df, test: df}}}
    """
    timeframe_aliases = {
        "weekly":  ["weekly"],
        "daily":   ["daily"],
        "4h":      ["4h", "4hr"],
        "1h":      ["1h", "1hr", "hourly"],
        "30min":   ["30min", "30m"],
        "15min":   ["15min", "15m"],
        "5min":    ["5min", "5m"],
        "1min":    ["1min", "1m"],
    }

    raw_files = {}

    for timeframe, aliases in timeframe_aliases.items():
        for alias in aliases:
            for path in glob.glob(os.path.join(data_dir, f"*_{alias}.csv")):
                basename = os.path.basename(path)
                symbol   = "_".join(basename.split("_")[:-1])
                raw_files.setdefault(symbol, {})[timeframe] = path

    data_bundle = {}

    for symbol, paths in raw_files.items():

        # Require at least weekly + daily for trend context
        if "weekly" not in paths or "daily" not in paths:
            continue

        # Require at least one entry-level timeframe
        if not any(tf in paths for tf in ENTRY_TIMEFRAMES + ANALYSIS_TIMEFRAMES):
            continue

        symbol_data = {}

        for timeframe, path in paths.items():
            try:
                df = pd.read_csv(path)
            except UnicodeDecodeError:
                df = pd.read_excel(path, engine="openpyxl")

            if "Unnamed: 0" in df.columns:
                df = df.rename(columns={"Unnamed: 0": "Date"})
            if "Date" not in df.columns:
                print(f"WARNING: Missing Date column in {path} — skipping")
                continue

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            try:
                if df["Date"].dt.tz is not None:
                    df["Date"] = df["Date"].dt.tz_convert(None)
            except Exception:
                pass

            df = _standardize_columns(df)

            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
            if df.empty:
                continue

            df = df.sort_values("Date").set_index("Date")
            df = add_technical_indicators(df)

            if df.empty:
                continue

            # Train / test split — no data leakage
            split_idx = int(len(df) * TRAIN_RATIO)
            symbol_data[timeframe] = {
                "train": df.iloc[:split_idx].copy(),
                "test":  df.iloc[split_idx:].copy(),
            }

        if symbol_data:
            data_bundle[symbol] = symbol_data

    print(f"Loaded {len(data_bundle)} symbols")
    return data_bundle

# =========================================================
# KNOWLEDGE BRIDGE
# Translates text-based learned rules into numeric signals
# the RL agent can use as part of its observation space.
# =========================================================

# Keyword → observation index mapping
# These keywords come from learned rules in knowledge_base.py
<<<<<<< HEAD
RULE_KEYWORDS = {d
=======
RULE_KEYWORDS = {
    "trend":        0,
>>>>>>> b4a9b09bebd015dcd26a15f2a5f32024a3853a92
    "momentum":     1,
    "support":      2,
    "resistance":   2,
    "breakout":     3,
    "reversal":     4,
    "volume":       5,
    "liquidity":    6,
    "order_block":  7,
    "fair_value":   7,
    "risk":         8,
    "stop":         8,
    "psychology":   9,
}

KNOWLEDGE_FEATURE_SIZE = 10


def rules_to_signals(strategy_config: dict) -> np.ndarray:
    """
    Translates text-based learned rules into a numeric signal
    vector the RL agent can observe.

    Each position in the vector represents a trading concept.
    Value = average confidence of rules mentioning that concept.
    0.0 = concept not present, 1.0 = very high confidence concept.

    This is Gap 2 — the bridge between knowledge_base.py text rules
<<<<<<< HEAD

=======
    and the RL agent's numeric observation space.
    """
>>>>>>> b4a9b09bebd015dcd26a15f2a5f32024a3853a92
    signals = np.zeros(KNOWLEDGE_FEATURE_SIZE, dtype=np.float32)
    counts  = np.zeros(KNOWLEDGE_FEATURE_SIZE, dtype=np.float32)

    if not strategy_config:
        return signals

    # Collect all rule text from all fields
    rule_fields = [
        "entry_conditions",
        "exit_conditions",
        "risk_management",
        "market_structure",
        "indicators",
        "psychology",
    ]

    for field in rule_fields:
        rules = strategy_config.get(field, [])
        for rule in rules:
            if isinstance(rule, dict):
                text       = rule.get("rule", "") + " " + rule.get("description", "")
                confidence = float(rule.get("confidence", 0.5))
            elif isinstance(rule, str):
                text       = rule
                confidence = 0.5
            else:
                continue

            text_lower = text.lower()

            for keyword, idx in RULE_KEYWORDS.items():
                if keyword in text_lower:
                    signals[idx] += confidence
                    counts[idx]  += 1

    # Average confidence per concept
    nonzero = counts > 0
    signals[nonzero] = signals[nonzero] / counts[nonzero]

    return signals.clip(0.0, 1.0)


def load_strategy_for_symbol(symbol: str) -> dict:
    """
    Loads the optimized strategy config for a symbol from
    strategy_tester.py's cache. Falls back to empty dict
    if not yet optimized — agent still trains, just without
    strategy guidance signals.
    """
    config = load_optimized_strategy(symbol)

    if not config:
        # Try loading master knowledge as fallback
        master = load_rules("master_knowledge", "trading_strategy")
        if master:
            return master
        return {}

    return config

# =========================================================
# REWARD FUNCTION
# Rewards quality trades, punishes drawdown and bad habits.
# =========================================================

def compute_reward(
    pnl: float,
    entry_price: float,
    stop_loss_pct: float,
    equity: float,
    peak_equity: float,
    holding_steps: int,
    is_close: bool
) -> float:
    """
    Reward function designed to encourage:
    - High R:R trades (big wins relative to risk)
    - Quick decisive action
    - Avoiding large drawdowns

    Discourages:
    - Holding losers (no hold bonus)
    - Overtrading (no win bonus for small wins)
    - Large drawdowns (exponential penalty)
    """
    reward = 0.0

    if is_close and pnl != 0.0:
        # Scale reward by R:R ratio achieved
        risk_amount = entry_price * stop_loss_pct
        if risk_amount > 0:
            r_multiple = pnl / risk_amount
            # Reward grows with R:R, capped at 3R
            reward += pnl * (1.0 + min(r_multiple, 3.0) * 0.15)
        else:
            reward += pnl

        # Extra penalty for losses — discourages ignoring stop loss
        if pnl < 0:
            reward += pnl * 0.5  # Amplify loss penalty

    # Drawdown penalty — exponential above 5%
    if peak_equity > 0:
        drawdown = (peak_equity - equity) / peak_equity
        if drawdown > 0.05:
            reward -= (drawdown ** 2) * 10.0

    # Mild time decay — encourages decisive action
    reward -= 0.005

    # NO hold bonus — agent must earn reward through trades
    # NO win bonus — quality of trade matters, not just winning

    return float(reward)

# =========================================================
# MULTI-SYMBOL CHART EXPERT — MAIN RL ENVIRONMENT
# =========================================================

class MultiSymbolChartExpert(gym.Env):
    """
    RL environment that sees the whole market.

    The agent:
    - Trades any symbol in the bundle
    - Goes long OR short
    - Switches symbols when no opportunity exists
    - Uses optimized strategy signals per symbol
    - Learns which strategies work on which charts

    Observation space:
        - Window of entry-timeframe OHLCV candles
        - Higher timeframe snapshots (trend context)
        - Technical indicator signals
        - Knowledge signals (from learned strategy)

    Action space:
        0 = Hold
        1 = Buy  (open long)
        2 = Sell (open short)
        3 = Close position
        4 = Switch symbol
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_bundle: dict,
        mode: str = "train",
        window_size: int = 20
    ):
        super().__init__()

        self.data_bundle  = data_bundle
        self.mode         = mode        # "train" or "test"
        self.window_size  = window_size
        self.symbols      = sorted(data_bundle.keys())

        if not self.symbols:
            raise ValueError("data_bundle is empty — no symbols loaded")

        # ── Observation shape ─────────────────────────
        # Entry window: window_size candles × 5 OHLCV
        entry_features    = window_size * 5
        # Higher TF snapshots: each gives 5 OHLCV values
        context_features  = len(HIGHER_TIMEFRAMES) * 5
        # Technical indicator features from current bar
        indicator_features = 10
        # Knowledge signals from learned strategy
        knowledge_features = KNOWLEDGE_FEATURE_SIZE

        obs_size = (
            entry_features +
            context_features +
            indicator_features +
            knowledge_features
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        # ── Action space ──────────────────────────────
        # 0=Hold, 1=Buy, 2=Sell, 3=Close, 4=Switch
        self.action_space = spaces.Discrete(5)

        # ── State ─────────────────────────────────────
        self.current_symbol   = None
        self.current_data     = None
        self.entry_timeframe  = None
        self.strategy_signals = np.zeros(KNOWLEDGE_FEATURE_SIZE, dtype=np.float32)

        self.balance         = 10_000.0
        self.equity          = self.balance
        self.peak_equity     = self.balance
        self.previous_equity = self.balance
        self.position        = None
        self.current_step    = window_size
        self.holding_steps   = 0
        self.trade_history   = []
        self.current_trade   = None

    # ── Symbol selection ──────────────────────────────

    def _select_symbol(self, symbol: str = None):
        """
        Selects a symbol to trade.

        Priority:
        1. Use entry_tf from the optimized strategy config
           - Strategy can specify ANY timeframe as entry
           - A swing trader CAN use daily or 4H as entry
           - A scalper uses 1min/5min as entry
        2. If that tf has no data → skip to next symbol
        3. No strategy preference → find best available tf from data
        4. Never force a fallback just to fill — skip the symbol
        """
        candidates = (
            [symbol] if symbol
            else random.sample(self.symbols, len(self.symbols))
        )

        for candidate in candidates:
            self.current_symbol = candidate
            self.current_data   = self.data_bundle[candidate]

            # Load strategy first — it owns the entry_tf decision
            strategy_config       = load_strategy_for_symbol(candidate)
            self.strategy_signals = rules_to_signals(strategy_config)

            # Strategy specifies preferred entry timeframe
            # This can be ANY timeframe — daily for swing, 1min for scalp
            preferred_tf = strategy_config.get("entry_tf", None)

            if preferred_tf:
                tf_splits = self.current_data.get(preferred_tf, {})
                df        = tf_splits.get(self.mode)

                if df is not None and len(df) > self.window_size + 10:
                    # Strategy's preferred tf has data — use it
                    self.entry_timeframe = preferred_tf
                    self.current_step    = self.window_size
                    return
                else:
                    # Strategy's preferred tf has no data — skip symbol
                    # Don't force another tf, the strategy won't work right
                    continue

            # No strategy preference — find best available tf from data
            # Sort all available timeframes by granularity
            # Prefer lower timeframes (more data points) unless strategy says otherwise
            available_tfs = [
                tf for tf in self.current_data
                if self.current_data[tf].get(self.mode) is not None
                and len(self.current_data[tf].get(self.mode, pd.DataFrame())) > self.window_size + 10
            ]

            if not available_tfs:
                # No usable timeframe on this symbol — skip
                continue

            # Sort by position in ALL_TIMEFRAMES (lower = more granular)
            available_tfs.sort(
                key=lambda tf: ALL_TIMEFRAMES.index(tf)
                if tf in ALL_TIMEFRAMES else 999,
                reverse=True  # Higher index = lower timeframe = prefer
            )

            # Use most granular available timeframe
            # Higher TFs like daily/4H are available if they're the
            # only ones with enough data (e.g. swing trading symbols)
            self.entry_timeframe = available_tfs[0]
            self.current_step    = self.window_size
            return

        # All symbols exhausted with no valid entry tf found
        # True last resort — only hits if data is severely missing
        print("WARNING: All symbols exhausted — using first symbol as fallback")
        self.current_symbol   = self.symbols[0]
        self.current_data     = self.data_bundle[self.symbols[0]]
        self.entry_timeframe  = next(
            (tf for tf in ALL_TIMEFRAMES if tf in self.current_data),
            "daily"
        )
        self.current_step     = self.window_size
        strategy_config       = load_strategy_for_symbol(self.symbols[0])
        self.strategy_signals = rules_to_signals(strategy_config)

    # ── Observation builders ──────────────────────────

    def _get_entry_window(self) -> np.ndarray:
        """Returns the sliding window of entry-TF OHLCV candles."""
        df     = self.current_data[self.entry_timeframe][self.mode]
        window = df.iloc[
            self.current_step - self.window_size : self.current_step
        ]
        cols = [c for c in ["Close","Open","High","Low","Volume"] if c in window.columns]
        vals = window[cols].values.flatten()

        # Pad if some columns missing
        expected = self.window_size * 5
        if len(vals) < expected:
            vals = np.pad(vals, (0, expected - len(vals)))

        return vals.astype(np.float32)

    def _get_higher_tf_context(self) -> np.ndarray:
        """
        Returns the most recent snapshot from each higher timeframe.
        Gives the agent trend and bias context above entry level.
        """
        df_entry = self.current_data[self.entry_timeframe][self.mode]
        current_time = df_entry.index[self.current_step]

        context = []
        for tf in HIGHER_TIMEFRAMES:
            tf_splits = self.current_data.get(tf)
            df_tf     = tf_splits.get(self.mode) if tf_splits else None

            if df_tf is None or df_tf.empty:
                context.append(np.zeros(5, dtype=np.float32))
                continue

            frame = df_tf[df_tf.index <= current_time]
            row   = frame.iloc[-1] if not frame.empty else df_tf.iloc[-1]

            vals = np.array([
                float(row.get("Close",  0)),
                float(row.get("Open",   0)),
                float(row.get("High",   0)),
                float(row.get("Low",    0)),
                float(row.get("Volume", 0)),
            ], dtype=np.float32)
            context.append(vals)

        return np.concatenate(context)

    def _get_indicator_features(self) -> np.ndarray:
        """
        Returns key indicator values at the current bar.
        Normalized so they're on similar scales.
        """
        df  = self.current_data[self.entry_timeframe][self.mode]
        row = df.iloc[self.current_step]

        close = float(row.get("Close", 1.0))
        atr   = float(row.get("atr_14", close * 0.01))

        features = np.array([
            float(row.get("rsi_14",      50.0)) / 100.0,
            float(row.get("macd",         0.0)) / (atr + 1e-8),
            float(row.get("macd_hist",    0.0)) / (atr + 1e-8),
            float(row.get("stoch_k",     50.0)) / 100.0,
            float(row.get("williams_r", -50.0)) / -100.0,
            float(row.get("volume_ratio", 1.0)) / 5.0,
            float(row.get("body_size",    0.0)),
            float(row.get("upper_shadow", 0.0)),
            float(row.get("lower_shadow", 0.0)),
            (float(row.get("Close", 0)) - float(row.get("bb_lower", 0))) /
            (float(row.get("bb_upper", 1)) - float(row.get("bb_lower", 0)) + 1e-8),
        ], dtype=np.float32)

        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    def _get_observation(self) -> np.ndarray:
        entry     = self._get_entry_window()
        context   = self._get_higher_tf_context()
        indicators = self._get_indicator_features()
        knowledge  = self.strategy_signals

        obs = np.concatenate([entry, context, indicators, knowledge])
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    # ── Position management ───────────────────────────

    def _get_current_price(self) -> float:
        df = self.current_data[self.entry_timeframe][self.mode]
        return float(df.iloc[self.current_step]["Close"])

    def _open_position(self, direction: str, price: float):
        """
        Opens a long or short position.
        Risk sizing based on ATR — adapts to market volatility.
        """
        df  = self.current_data[self.entry_timeframe][self.mode]
        row = df.iloc[self.current_step]
        atr = float(row.get("atr_14", price * 0.01))

        # Dynamic SL/TP based on ATR
        stop_loss_pct   = max(0.005, (atr * 1.5) / price)
        take_profit_pct = stop_loss_pct * 2.0   # Minimum 1:2 R:R

        risk_per_trade = 0.01   # Risk 1% of balance per trade
        size = max(0.001, (self.balance * risk_per_trade) / (stop_loss_pct * price))

        if direction == "long":
            sl = price * (1.0 - stop_loss_pct)
            tp = price * (1.0 + take_profit_pct)
        else:  # short
            sl = price * (1.0 + stop_loss_pct)
            tp = price * (1.0 - take_profit_pct)

        self.position = {
            "direction":      direction,
            "entry_price":    price,
            "sl":             sl,
            "tp":             tp,
            "size":           size,
            "stop_loss_pct":  stop_loss_pct,
        }

        entry_time = df.index[self.current_step]

        self.current_trade = {
            "symbol":     self.current_symbol,
            "direction":  direction,
            "entry_time": str(entry_time),
            "entry_price": price,
            "size":        size,
            "sl":          sl,
            "tp":          tp,
        }

        self.holding_steps = 0

    def _close_position(self, price: float) -> float:
        """Closes the current position and returns PnL."""
        if self.position is None:
            return 0.0

        direction   = self.position["direction"]
        entry_price = self.position["entry_price"]
        size        = self.position["size"]

        if direction == "long":
            pnl = (price - entry_price) * size
        else:  # short
            pnl = (entry_price - price) * size

        self.balance += pnl

        if self.current_trade:
            self.current_trade["exit_price"] = price
            self.current_trade["pnl"]        = pnl
            self.current_trade["result"]     = (
                "win" if pnl > 0 else "loss" if pnl < 0 else "breakeven"
            )
            self.current_trade["hold_steps"] = self.holding_steps
            self.trade_history.append(self.current_trade)
            self.current_trade = None

        sl_pct            = self.position["stop_loss_pct"]
        self.position     = None
        self.holding_steps = 0

        return pnl, sl_pct

    def _check_sl_tp(self, price: float):
        """Check if SL or TP has been hit."""
        if self.position is None:
            return 0.0, 0.0

        pos       = self.position
        direction = pos["direction"]
        hit       = False

        if direction == "long":
            if price <= pos["sl"] or price >= pos["tp"]:
                hit = True
        else:  # short
            if price >= pos["sl"] or price <= pos["tp"]:
                hit = True

        if hit:
            pnl, sl_pct = self._close_position(price)
            return pnl, sl_pct

        return 0.0, getattr(self.position, "stop_loss_pct", 0.02) if self.position else 0.02

    def _get_equity(self, price: float) -> float:
        if self.position is None:
            return self.balance

        pos = self.position
        if pos["direction"] == "long":
            unrealized = (price - pos["entry_price"]) * pos["size"]
        else:
            unrealized = (pos["entry_price"] - price) * pos["size"]

        return self.balance + unrealized

    # ── Step ─────────────────────────────────────────

    def step(self, action: int):
        df       = self.current_data[self.entry_timeframe][self.mode]
        df_len   = len(df)
        price    = self._get_current_price()
        reward   = 0.0
        pnl      = 0.0
        sl_pct   = self.position["stop_loss_pct"] if self.position else 0.02

        # ── Process action ────────────────────────────
        if action == ACTION_BUY:
            if self.position is None:
                self._open_position("long", price)
            # If already in a position, treat as hold

        elif action == ACTION_SELL:
            if self.position is None:
                self._open_position("short", price)

        elif action == ACTION_CLOSE:
            if self.position is not None:
                pnl, sl_pct = self._close_position(price)
                reward += compute_reward(
                    pnl         = pnl,
                    entry_price = price,
                    stop_loss_pct = sl_pct,
                    equity      = self.balance,
                    peak_equity = self.peak_equity,
                    holding_steps = self.holding_steps,
                    is_close    = True
                )

        elif action == ACTION_SWITCH:
            # Close any open position before switching
            if self.position is not None:
                pnl, sl_pct = self._close_position(price)
                # Small penalty for switching mid-trade
                reward += pnl - 1.0
            self._select_symbol()

        # ── Check SL/TP ───────────────────────────────
        if self.position is not None:
            auto_pnl, sl_pct = self._check_sl_tp(price)
            if auto_pnl != 0.0:
                pnl = auto_pnl
                reward += compute_reward(
                    pnl           = auto_pnl,
                    entry_price   = price,
                    stop_loss_pct = sl_pct,
                    equity        = self.balance,
                    peak_equity   = self.peak_equity,
                    holding_steps = self.holding_steps,
                    is_close      = True
                )

        # ── Update equity and peak ────────────────────
        self.equity      = self._get_equity(price)
        self.peak_equity = max(self.peak_equity, self.equity)

        # ── Drawdown reward ───────────────────────────
        reward += compute_reward(
            pnl           = 0.0,
            entry_price   = price,
            stop_loss_pct = sl_pct,
            equity        = self.equity,
            peak_equity   = self.peak_equity,
            holding_steps = self.holding_steps,
            is_close      = False
        )

        self.previous_equity = self.equity

        # ── Advance step ─────────────────────────────
        self.current_step  += 1
        self.holding_steps += 1 if self.position else 0

        # ── Termination ───────────────────────────────
        terminated = self.current_step >= df_len - 1
        truncated  = False

        # Auto-close at episode end
        if terminated and self.position is not None:
            final_pnl, sl_pct = self._close_position(price)
            reward += compute_reward(
                pnl           = final_pnl,
                entry_price   = price,
                stop_loss_pct = sl_pct,
                equity        = self.balance,
                peak_equity   = self.peak_equity,
                holding_steps = self.holding_steps,
                is_close      = True
            )

        observation = self._get_observation()

        info = {
            "symbol":        self.current_symbol,
            "balance":       self.balance,
            "equity":        self.equity,
            "peak_equity":   self.peak_equity,
            "position":      self.position is not None,
            "direction":     self.position["direction"] if self.position else None,
            "pnl":           pnl,
            "trade_count":   len(self.trade_history),
            "win_rate": (
                sum(1 for t in self.trade_history if t.get("pnl", 0) > 0)
                / len(self.trade_history)
                if self.trade_history else 0.0
            ),
        }

        return observation, float(reward), terminated, truncated, info

    # ── Reset ─────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance         = 10_000.0
        self.equity          = self.balance
        self.peak_equity     = self.balance
        self.previous_equity = self.balance
        self.position        = None
        self.holding_steps   = 0
        self.trade_history   = []
        self.current_trade   = None

        self._select_symbol()

        return self._get_observation(), {}

    # ── Render ────────────────────────────────────────

    def render(self):
        trades = len(self.trade_history)
        wins   = sum(1 for t in self.trade_history if t.get("pnl", 0) > 0)
        wr     = wins / trades if trades > 0 else 0.0
        print(
            f"Symbol: {self.current_symbol:<12} "
            f"Balance: ${self.balance:>10.2f}  "
            f"Equity: ${self.equity:>10.2f}  "
            f"Trades: {trades:>4}  "
            f"Win rate: {wr:.1%}"
        )

# =========================================================
# STRATEGY TRAINER
# =========================================================

class StrategyTrainer:
    """
    Trains and evaluates the RL agent.
    Uses train split for training, test split for evaluation —
    win rate numbers are real, not memorized training data.
    """

    def __init__(self, data_bundle: dict):
        self.data_bundle = data_bundle

    def train(self, total_timesteps: int = 500_000) -> PPO:
        """Train PPO agent on the training split."""
        env = DummyVecEnv([
            lambda: MultiSymbolChartExpert(self.data_bundle, mode="train")
        ])

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,   # Encourages exploration
        )

        model.learn(total_timesteps=total_timesteps)
        model.save("models/market_oracle")
        print("Model saved -> models/market_oracle")
        return model

    def evaluate(self, model: PPO, episodes: int = 10) -> dict:
        """
        Evaluates agent on the TEST split only.
        This gives a real win rate — not seen during training.
        """
        env = MultiSymbolChartExpert(self.data_bundle, mode="test")

        all_trades  = []
        all_equity  = []
        total_reward = 0.0

        for ep in range(episodes):
            obs, _   = env.reset()
            ep_reward = 0.0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_reward  += reward
                all_equity.append(info["equity"])
                if terminated or truncated:
                    all_trades.extend(env.trade_history)
                    break

            total_reward += ep_reward
            print(
                f"Episode {ep+1}/{episodes}  "
                f"Reward: {ep_reward:>8.2f}  "
                f"Win rate: {info['win_rate']:.1%}"
            )

        wins   = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
        losses = sum(1 for t in all_trades if t.get("pnl", 0) < 0)
        total  = len(all_trades)

        # Max drawdown
        peak_eq = 10_000.0
        max_dd  = 0.0
        for eq in all_equity:
            peak_eq = max(peak_eq, eq)
            dd      = (peak_eq - eq) / peak_eq
            max_dd  = max(max_dd, dd)

        results = {
            "total_reward":   total_reward,
            "total_trades":   total,
            "wins":           wins,
            "losses":         losses,
            "win_rate":       wins / total if total else 0.0,
            "max_drawdown":   max_dd,
            "final_balance":  env.balance,
            "final_equity":   env.equity,
        }

        print("\n=== Evaluation Results (TEST SET) ===")
        print(f"Total trades : {total}")
        print(f"Win rate     : {results['win_rate']:.1%}")
        print(f"Max drawdown : {results['max_drawdown']:.1%}")
        print(f"Final balance: ${results['final_balance']:.2f}")

        return results

    def evaluate_symbol(self, model: PPO, symbol: str) -> dict:
        """Evaluate agent on a specific symbol (test split)."""
        if symbol not in self.data_bundle:
            raise ValueError(f"Symbol {symbol} not in data bundle")

        env      = MultiSymbolChartExpert(self.data_bundle, mode="test")
        obs, _   = env.reset()

        # Force the symbol
        env._select_symbol(symbol)
        obs = env._get_observation()

        trades = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                trades = env.trade_history
                break

        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        return {
            "symbol":    symbol,
            "trades":    len(trades),
            "win_rate":  wins / len(trades) if trades else 0.0,
            "balance":   env.balance,
            "equity":    env.equity,
<<<<<<< HEAD
        }
=======
        }
>>>>>>> b4a9b09bebd015dcd26a15f2a5f32024a3853a92
