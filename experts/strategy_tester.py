"""
Strategy Tester and Optimizer

Loads learned trading rules from knowledge_base.py
(YouTube channels + books + master merged knowledge).

Tests strategies on symbols, builds an AI-generated master
strategy from all sources, optimizes per symbol, and feeds
the best config to the RL agent in chart_expert.py.
"""

import os
import copy
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Import read/write from db_handler — no circular imports
from experts.db_handler import (
    load_rules,
    save_optimized_strategy,
    load_optimized_strategy,
    save_master_strategy,
    load_master_strategy as _load_master_from_db,
    safe_symbol_name as _safe_symbol_name,
)

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-1.5-flash"

# Topics to search when loading rules from knowledge base
# Tries each topic in order until rules are found
FALLBACK_TOPICS = [
    "trading_strategy",
    "risk_management",
    "price_action",
    "swing_trading",
    "full",
]

# =========================================================
# HELPERS
# =========================================================

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def _ask_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return response text."""
    model = genai.GenerativeModel(MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.2)
    )
    return response.text


def _parse_json(text: str) -> dict:
    """Safely parse JSON from Gemini response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])
    # Strip conversational preamble before JSON
    brace = cleaned.find("{")
    if brace > 0:
        cleaned = cleaned[brace:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes DataFrame column names to Title Case.
    Handles data sources that provide lowercase columns
    (close, high, low, open, volume) or mixed case.
    Prevents KeyError crashes from inconsistent column naming.
    """
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == "open":
            rename_map[col] = "Open"
        elif col_lower == "high":
            rename_map[col] = "High"
        elif col_lower == "low":
            rename_map[col] = "Low"
        elif col_lower == "close":
            rename_map[col] = "Close"
        elif col_lower == "volume":
            rename_map[col] = "Volume"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

# =========================================================
# SYMBOL ANALYZER — ANALYZES CHART CHARACTERISTICS
# =========================================================

class SymbolAnalyzer:
    """Analyze symbol characteristics to determine best-fit strategies."""

    @staticmethod
    def analyze_symbol(
        symbol_data: Dict[str, pd.DataFrame],
        symbol_name: str = "Unknown"
    ) -> Dict:
        """
        Analyze a symbol's characteristics across timeframes.
        Returns volatility, trend strength, mean reversion,
        and breakout frequency metrics.
        """
        analysis = {
            "symbol": symbol_name,  # Fixed: was always None
            "volatility_profile": {},
            "trend_strength": {},
            "mean_reversion": {},
            "breakout_frequency": {},
            "overall_characteristics": [],
        }

        # Higher timeframes inform direction
        # Lower timeframes inform entry
        # Minutes are NEVER in higher_timeframes
        higher_timeframes  = ["weekly", "daily", "4h", "1h"]
        analysis_timeframes = ["30min", "15min"]
        entry_timeframes   = ["5min", "1min"]

        all_timeframes = higher_timeframes + analysis_timeframes + entry_timeframes

        for tf in all_timeframes:
            if tf not in symbol_data:
                continue

            df = symbol_data[tf]
            if df.empty or len(df) < 50:
                continue

            # Standardize column names — handles close/Close/CLOSE
            df = _standardize_columns(df)
            high_low   = df["High"] - df["Low"]
            high_close = (df["High"] - df["Close"].shift()).abs()
            low_close  = (df["Low"]  - df["Close"].shift()).abs()
            tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=14).mean()
            volatility = (atr / df["Close"]) * 100
            vol_mean = float(volatility.mean())

            # ─────────────────────────────────────────────
            # Trend strength — MA slope
            # ─────────────────────────────────────────────
            sma_50  = df["Close"].rolling(50,  min_periods=50).mean()
            sma_200 = df["Close"].rolling(200, min_periods=200).mean()

            valid_idx = sma_50.notna() & sma_200.notna()
            if valid_idx.sum() > 20:
                ma_diff      = (sma_50 - sma_200)[valid_idx]
                trend_slope  = ma_diff.diff().rolling(20, min_periods=20).mean()
                close_std    = df["Close"].std()
                trend_strength = (
                    float(abs(trend_slope.mean()) / close_std)
                    if close_std > 0 else 0.0
                )
            else:
                trend_strength = 0.0

            # ─────────────────────────────────────────────
            # Mean reversion — RSI centering
            # ─────────────────────────────────────────────
            delta = df["Close"].diff().fillna(0)
            gain  = delta.where(delta > 0, 0).rolling(14, min_periods=14).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(14, min_periods=14).mean()
            rs    = gain / loss.replace(0, np.nan)
            # fillna(100): 14 consecutive up candles = max RSI = 100
            rsi   = (100 - 100 / (1 + rs)).fillna(100)
            rsi_mean = float(rsi.mean())
            mean_reversion_score = float(1 - (abs(rsi_mean - 50) / 50))

            # ─────────────────────────────────────────────
            # Breakout frequency
            # ─────────────────────────────────────────────
            high_20 = df["High"].rolling(20, min_periods=20).max()
            low_20  = df["Low"].rolling(20,  min_periods=20).min()
            breakouts = (
                ((df["High"] > high_20.shift(1)) | (df["Low"] < low_20.shift(1)))
                .astype(int)
                .rolling(20, min_periods=20)
                .sum()
            )
            breakout_freq = float(breakouts.mean() / 20) if len(breakouts) > 0 else 0.0

            analysis["volatility_profile"][tf] = {
                "atr_avg":         float(atr.mean()),
                "volatility_pct":  vol_mean,
                "volatility_level": (
                    "high"   if vol_mean > 2.5 else
                    "medium" if vol_mean > 1.5 else
                    "low"
                ),
            }

            analysis["trend_strength"][tf] = {
                "trend_score": trend_strength,
                "trend_type":  "strong" if trend_strength > 0.1 else "weak",
            }

            analysis["mean_reversion"][tf] = {
                "mean_reversion_score": mean_reversion_score,
                "rsi_mean":             rsi_mean,
                "tendency": (
                    "mean_reverting" if mean_reversion_score > 0.6 else "trending"
                ),
            }

            analysis["breakout_frequency"][tf] = {
                "breakout_freq":      breakout_freq,
                "breakout_tendency": (
                    "breakout_prone" if breakout_freq > 0.3 else "range_bound"
                ),
            }

        # Overall classification based on daily (primary timeframe)
        daily_vol    = analysis["volatility_profile"].get("daily", {}).get("volatility_level", "medium")
        daily_trend  = analysis["trend_strength"].get("daily", {}).get("trend_type", "weak")
        daily_revert = analysis["mean_reversion"].get("daily", {}).get("tendency", "trending")

        if daily_trend == "strong":
            analysis["overall_characteristics"].append("strong_trend")
        elif daily_revert == "mean_reverting":
            analysis["overall_characteristics"].append("mean_reverting")
        else:
            analysis["overall_characteristics"].append("mixed")

        if daily_vol == "high":
            analysis["overall_characteristics"].append("high_volatility")
        elif daily_vol == "low":
            analysis["overall_characteristics"].append("low_volatility")

        return analysis

    @staticmethod
    def get_symbol_type(analysis: Dict) -> str:
        """Classify symbol into a trading style category."""
        chars = analysis["overall_characteristics"]

        if "strong_trend" in chars and "high_volatility" in chars:
            return "trending_volatile"
        elif "strong_trend" in chars:
            return "trending_smooth"
        elif "mean_reverting" in chars and "low_volatility" in chars:
            return "mean_reverting_choppy"
        elif "mean_reverting" in chars:
            return "mean_reverting_normal"
        elif "high_volatility" in chars:
            return "ranging_volatile"
        else:
            return "ranging_smooth"

# =========================================================
# STRATEGY LOADER — LOADS FROM YOUTUBE + BOOKS + MASTER
# =========================================================

class StrategyLoader:
    """
    Loads learned trading strategies from knowledge_base.py.
    Sources: YouTube channels, trading books, and master merged knowledge.
    """

    @staticmethod
    def load_strategy_from_knowledge(
        source_key: str,
        topics: List[str] = None
    ) -> Optional[Dict]:
        """
        Load a strategy from the knowledge base.
        Tries multiple topics in fallback order until rules are found.
        """
        # load_rules imported from db_handler at top of file
        search_topics = topics or FALLBACK_TOPICS

        rules = None
        found_topic = None

        # Try each topic until we find saved rules
        for topic in search_topics:
            rules = load_rules(source_key, topic)
            if rules:
                found_topic = topic
                break

        if not rules:
            return None

        # Get entry conditions safely
        entry_conditions = rules.get("entry_conditions", [])

        # Fixed: use np.mean not min() for confidence
        if entry_conditions and isinstance(entry_conditions[0], dict):
            confidence = float(np.mean(
                [e.get("confidence", 0.5) for e in entry_conditions]
            ))
        else:
            confidence = 0.5

        strategy = {
            "_source":      source_key,
            "_topic":       found_topic,
            "_source_type": "youtube",
            "name":         f"{source_key.replace('_', ' ').title()} Strategy",
            "entry_conditions":  entry_conditions,
            "exit_conditions":   rules.get("exit_conditions",  []),
            "risk_management":   rules.get("risk_management",  []),
            "indicators":        rules.get("indicators",       []),
            "psychology":        rules.get("psychology",       []),
            "market_structure":  rules.get("market_structure", []),
            "strategy_type":     rules.get("strategy_type",    []),
            "market_regime":     rules.get("market_regime",    []),
            "confidence":        confidence,
        }

        return strategy

    @staticmethod
    def load_all_youtube_strategies() -> Dict[str, Dict]:
        """Load learned strategies from all YouTube channels."""
        from experts.knowledge_base import YOUTUBE_CHANNELS

        strategies = {}

        for channel_key in YOUTUBE_CHANNELS.keys():
            strategy = StrategyLoader.load_strategy_from_knowledge(channel_key)
            if strategy:
                strategies[channel_key] = strategy
                print(f"  Loaded YouTube strategy: {channel_key}")
            else:
                print(f"  No rules found for: {channel_key} — run learn_from_channel() first")

        return strategies

    @staticmethod
    def load_all_book_strategies() -> Dict[str, Dict]:
        """
        Load learned strategies from all trading books.
        Books are tagged with _source_type = 'book'.
        """
        from experts.knowledge_base import BOOK_DATABASE
        # load_rules imported from db_handler at top of file

        strategies = {}

        for book_key in BOOK_DATABASE.keys():
            # Books are saved under key "book_{book_key}" with topic "full"
            rules = load_rules(f"book_{book_key}", "full")

            if not rules:
                print(f"  No rules found for book: {book_key} — run learn_from_book() first")
                continue

            book_info = BOOK_DATABASE[book_key]

            entry_conditions = rules.get("entry_conditions", [])

            if entry_conditions and isinstance(entry_conditions[0], dict):
                confidence = float(np.mean(
                    [e.get("confidence", 0.5) for e in entry_conditions]
                ))
            else:
                confidence = 0.5

            strategy = {
                "_source":      f"book_{book_key}",
                "_topic":       "full",
                "_source_type": "book",
                "name":         f"{book_info['title']} Strategy",
                "author":       book_info["author"],
                "entry_conditions":  entry_conditions,
                "exit_conditions":   rules.get("exit_conditions",  []),
                "risk_management":   rules.get("risk_management",  []),
                "indicators":        rules.get("indicators",       []),
                "psychology":        rules.get("psychology",       []),
                "market_structure":  rules.get("market_structure", []),
                "chart_patterns":    rules.get("chart_patterns",   []),
                "key_concepts":      rules.get("key_concepts",     []),
                "confidence":        confidence,
            }

            strategies[f"book_{book_key}"] = strategy
            print(f"  Loaded book strategy: {book_info['title']}")

        return strategies

    @staticmethod
    def load_all_strategies() -> Dict[str, Dict]:
        """
        Loads ALL strategies from:
        - YouTube channels
        - Trading books
        - Master merged knowledge (if available)

        This is the main entry point for strategy loading.
        """
        print("\nLoading strategies from knowledge base...")

        all_strategies = {}

        # 1. YouTube channels
        youtube_strategies = StrategyLoader.load_all_youtube_strategies()
        all_strategies.update(youtube_strategies)

        # 2. Books
        book_strategies = StrategyLoader.load_all_book_strategies()
        all_strategies.update(book_strategies)

        # 3. Master merged knowledge (if it exists)
        master = StrategyLoader.load_master_strategy()
        if master:
            all_strategies["master_knowledge"] = master
            print("  Loaded master merged strategy")

        print(f"\nTotal strategies loaded: {len(all_strategies)}")
        return all_strategies

    @staticmethod
    def load_master_strategy() -> Optional[Dict]:
        """
        Load the master merged strategy if it exists.
        This is the strategy that combines all YouTube + book knowledge.
        """
        # load_rules imported from db_handler at top of file
        for topic in FALLBACK_TOPICS:
            rules = load_rules("master_knowledge", topic)
            if rules:
                rules["_source"]      = "master_knowledge"
                rules["_source_type"] = "master"
                rules["name"]         = "AI Master Strategy"
                return rules

        return None

# =========================================================
# AI MASTER STRATEGY BUILDER
# =========================================================

class MasterStrategyBuilder:
    """
    Uses Gemini to synthesize an AI-generated master strategy
    from all learned YouTube + book knowledge combined.
    The master strategy is symbol-aware — adapted to the
    specific characteristics of each chart.
    """

    @staticmethod
    def build_master_strategy(
        all_strategies: Dict[str, Dict],
        symbol_name: str,
        symbol_analysis: Dict,
        force_rebuild: bool = False
    ) -> Dict:
        """
        Builds an AI master strategy for a specific symbol
        by merging all available learned strategies.

        Saved to strategies/master/{symbol_name}.json
        Cached — Gemini only called if not already built.
        """
        # Load from cache if already built
        if not force_rebuild:
            cached = _load_master_from_db(symbol_name)
            if cached:
                print(f"  Master strategy loaded from cache")
                return cached

        if not all_strategies:
            print("  No strategies available to build master from")
            return {}

        symbol_type = SymbolAnalyzer.get_symbol_type(symbol_analysis)
        vol_level   = symbol_analysis["volatility_profile"].get(
            "daily", {}
        ).get("volatility_level", "medium")
        trend_type  = symbol_analysis["trend_strength"].get(
            "daily", {}
        ).get("trend_type", "weak")

        # Prepare condensed strategy summaries for Gemini
        # (avoid sending the full strategy dicts which can be huge)
        strategy_summaries = []

        for source_key, strategy in all_strategies.items():
            source_type = strategy.get("_source_type", "youtube")
            entries = strategy.get("entry_conditions", [])
            exits   = strategy.get("exit_conditions",  [])
            risk    = strategy.get("risk_management",  [])

            # Format entry rules
            entry_rules = []
            for e in entries[:5]:  # Limit to top 5 to save tokens
                if isinstance(e, dict):
                    entry_rules.append(e.get("rule", ""))
                elif isinstance(e, str):
                    entry_rules.append(e)

            # Format risk rules
            risk_rules = []
            for r in risk[:3]:
                if isinstance(r, dict):
                    risk_rules.append(r.get("rule", ""))
                elif isinstance(r, str):
                    risk_rules.append(r)

            strategy_summaries.append({
                "source":       source_key,
                "type":         source_type,
                "entries":      entry_rules,
                "exits":        [e.get("rule", e) if isinstance(e, dict) else e for e in exits[:3]],
                "risk":         risk_rules,
                "confidence":   strategy.get("confidence", 0.5),
            })

        prompt = f"""
You are an expert trading system designer.

I have a trading AI that needs to trade: {symbol_name}

Symbol characteristics:
- Symbol type: {symbol_type}
- Volatility: {vol_level}
- Trend: {trend_type}
- Overall: {symbol_analysis['overall_characteristics']}

I have learned trading rules from {len(all_strategies)} sources
(YouTube channels and trading books).

Here are the learned rules:
{json.dumps(strategy_summaries, indent=2)}

Your task:
1. Analyze which rules best fit this symbol's characteristics
2. Create ONE optimized master strategy specifically for {symbol_name}
3. Combine the best rules from all sources
4. Add any logical improvements you see
5. Make sure risk management is solid

Return JSON only:
{{
    "name": "Master Strategy for {symbol_name}",
    "symbol": "{symbol_name}",
    "symbol_type": "{symbol_type}",
    "entry_conditions": [
        {{"rule": "", "confidence": 0.0, "source": ""}}
    ],
    "exit_conditions": [
        {{"rule": "", "confidence": 0.0}}
    ],
    "risk_management": [
        {{"rule": "", "confidence": 0.0}}
    ],
    "indicators": [],
    "market_structure": [],
    "psychology": [],
    "adaptations": [],
    "overall_confidence": 0.0
}}

Return JSON only, no extra text.
"""

        print(f"  Building AI master strategy for {symbol_name}...")

        result_text = _ask_gemini(prompt)
        master = _parse_json(result_text)

        if not master:
            print("  Gemini returned invalid JSON — using merged fallback")
            master = MasterStrategyBuilder._fallback_merge(all_strategies)

        master["_source"]      = "master_knowledge"
        master["_source_type"] = "master"
        master["_built_for"]   = symbol_name
        master["_timestamp"]   = str(datetime.now())

        # Save to cache via db_handler
        save_master_strategy(symbol_name, master)

        return master

    @staticmethod
    def _fallback_merge(all_strategies: Dict[str, Dict]) -> Dict:
        """
        Simple fallback merge if Gemini fails.
        Collects all rules from all sources.
        """
        merged = {
            "name":              "Master Strategy (Merged)",
            "entry_conditions":  [],
            "exit_conditions":   [],
            "risk_management":   [],
            "indicators":        [],
            "market_structure":  [],
            "psychology":        [],
        }

        for strategy in all_strategies.values():
            for field in merged.keys():
                if field == "name":
                    continue
                existing = strategy.get(field, [])
                if isinstance(existing, list):
                    merged[field].extend(existing)

        return merged

# =========================================================
# STRATEGY TESTER — TESTS STRATEGIES AGAINST PRICE DATA
# =========================================================

class StrategyTester:
    """
    Tests strategies against actual price data.
    Scores based on how well entry conditions match
    the symbol's current market behaviour.
    """

    @staticmethod
    def test_strategy_on_symbol(
        strategy: Dict,
        symbol_data: Dict[str, pd.DataFrame],
        symbol_name: str,
        symbol_analysis: Dict
    ) -> Dict:
        """
        Tests a strategy against a symbol's price data.
        Scores how well the strategy fits the symbol's behaviour.
        """
        results = {
            "strategy_source": strategy.get("_source",      "Unknown"),
            "strategy_name":   strategy.get("name",         "Unknown"),
            "source_type":     strategy.get("_source_type", "unknown"),
            "symbol":          symbol_name,
            "fit_score":       0.0,
            "confidence":      strategy.get("confidence", 0.5),
            "component_scores": {},
            "fit_reasoning":   [],
        }

        # Use daily as primary test timeframe
        df = symbol_data.get("daily")
        if df is None or df.empty or len(df) < 50:
            results["fit_reasoning"].append("No daily data available")
            return results

        # Standardize column names — handles close/Close/CLOSE
        df = _standardize_columns(df)

        # Drop rows with missing price data
        required = [c for c in ["Close", "High", "Low"] if c in df.columns]
        if required:
            df = df.dropna(subset=required)

        fit_score = 0.0
        symbol_type = SymbolAnalyzer.get_symbol_type(symbol_analysis)

        # ─────────────────────────────────────────────
        # 1. Entry condition quality (40 points)
        # ─────────────────────────────────────────────
        entry_conditions = strategy.get("entry_conditions", [])
        if entry_conditions:
            if isinstance(entry_conditions[0], dict):
                confidences = [e.get("confidence", 0.5) for e in entry_conditions]
                avg_conf    = float(np.mean(confidences))
                count       = len(entry_conditions)
            else:
                avg_conf = 0.5
                count    = len(entry_conditions)

            entry_score = avg_conf * min(count / 3.0, 1.0) * 40
            fit_score  += entry_score
            results["component_scores"]["entry"] = round(entry_score, 2)
            results["fit_reasoning"].append(
                f"Entry: {count} conditions at {avg_conf:.0%} avg confidence"
            )
        else:
            results["fit_reasoning"].append("No entry conditions defined")

        # ─────────────────────────────────────────────
        # 2. Risk management completeness (20 points)
        # ─────────────────────────────────────────────
        risk_rules = strategy.get("risk_management", [])
        if risk_rules:
            risk_score = min(len(risk_rules) / 3.0, 1.0) * 20
            fit_score += risk_score
            results["component_scores"]["risk"] = round(risk_score, 2)
            results["fit_reasoning"].append(
                f"Risk: {len(risk_rules)} rules defined"
            )

        # ─────────────────────────────────────────────
        # 3. Market structure alignment (20 points)
        # Tests if the strategy suits this symbol type
        # ─────────────────────────────────────────────
        market_struct   = strategy.get("market_structure", [])
        strategy_type   = strategy.get("strategy_type",   [])
        source_type     = strategy.get("_source_type",    "youtube")

        structure_score = 0.0

        if market_struct:
            structure_score += 10

        # Books get a small bonus — more structured knowledge
        if source_type == "book":
            structure_score += 5

        # Master strategy gets the highest bonus
        if source_type == "master":
            structure_score += 10

        # Symbol type alignment
        trend_type = symbol_analysis["trend_strength"].get(
            "daily", {}
        ).get("trend_type", "weak")

        if "strong_trend" in symbol_analysis["overall_characteristics"]:
            # Trending symbol — favour trend-following strategies
            trend_keywords = ["trend", "momentum", "breakout", "continuation"]
            rules_text     = json.dumps(strategy).lower()
            if any(kw in rules_text for kw in trend_keywords):
                structure_score += 5

        elif "mean_reverting" in symbol_analysis["overall_characteristics"]:
            # Mean-reverting symbol — favour reversal strategies
            reversal_keywords = ["reversal", "support", "resistance", "oversold", "overbought"]
            rules_text        = json.dumps(strategy).lower()
            if any(kw in rules_text for kw in reversal_keywords):
                structure_score += 5

        structure_score = min(structure_score, 20)
        fit_score      += structure_score
        results["component_scores"]["market_structure"] = round(structure_score, 2)

        # ─────────────────────────────────────────────
        # 4. Indicator relevance (20 points)
        # ─────────────────────────────────────────────
        indicators = strategy.get("indicators", [])
        if indicators:
            indicator_score = min(len(indicators) / 4.0, 1.0) * 20
            fit_score      += indicator_score
            results["component_scores"]["indicators"] = round(indicator_score, 2)
            results["fit_reasoning"].append(
                f"Indicators: {len(indicators)} defined"
            )

        # Final clamp
        results["fit_score"] = round(min(100.0, max(0.0, fit_score)), 2)

        return results

# =========================================================
# STRATEGY OPTIMIZER — EVOLVES STRATEGIES PER SYMBOL
# =========================================================

class StrategyOptimizer:
    """
    Evolves a symbol-specific version of a strategy by
    generating meaningful variations and testing each one.
    """

    @staticmethod
    def optimize_for_symbol(
        base_strategy: Dict,
        symbol_data: Dict[str, pd.DataFrame],
        symbol_name: str,
        symbol_analysis: Dict
    ) -> Dict:
        """
        Takes a base strategy and evolves the best version
        for this specific symbol.
        """
        best_result   = StrategyTester.test_strategy_on_symbol(
            base_strategy, symbol_data, symbol_name, symbol_analysis
        )
        best_strategy = base_strategy.copy()
        best_score    = best_result["fit_score"]

        # Generate meaningful variations
        variations = StrategyOptimizer._generate_variations(
            base_strategy, symbol_analysis
        )

        for variation in variations:
            result = StrategyTester.test_strategy_on_symbol(
                variation, symbol_data, symbol_name, symbol_analysis
            )

            if result["fit_score"] > best_score:
                best_score    = result["fit_score"]
                best_strategy = variation

        best_strategy["optimized_for"]      = symbol_name
        best_strategy["optimization_score"] = best_score
        best_strategy["symbol_type"]        = SymbolAnalyzer.get_symbol_type(symbol_analysis)
        best_strategy["_optimized_at"]      = str(datetime.now())

        # Determine best entry_tf for this symbol based on its type
        # chart_expert.py reads this to pick the right timeframe
        symbol_type = SymbolAnalyzer.get_symbol_type(symbol_analysis)
        vol_level   = symbol_analysis["volatility_profile"].get(
            "daily", {}
        ).get("volatility_level", "medium")

        # Entry tf comes from symbol characteristics, not hardcoded
        if "trending" in symbol_type:
            if vol_level == "high":
                # Volatile trending — use 15min for precision
                best_strategy["entry_tf"] = "15min"
            else:
                # Smooth trending — 1h for cleaner signals
                best_strategy["entry_tf"] = "1h"
        elif "mean_reverting" in symbol_type:
            if vol_level == "low":
                # Calm mean-reverting — daily is fine
                best_strategy["entry_tf"] = "daily"
            else:
                # Active mean-reverting — 30min
                best_strategy["entry_tf"] = "30min"
        else:
            # Mixed/ranging — 15min default
            best_strategy["entry_tf"] = "15min"

        # Override with strategy-learned timeframe if it exists
        # (from YouTube/book knowledge that specified a timeframe)
        learned_tf = best_strategy.get("timeframe", None)
        if learned_tf and isinstance(learned_tf, str) and learned_tf != "":
            best_strategy["entry_tf"] = learned_tf

        return best_strategy

    @staticmethod
    def _generate_variations(
        strategy: Dict,
        symbol_analysis: Dict
    ) -> List[Dict]:
        """
        Generate meaningful variations of a strategy
        based on the symbol's characteristics.
        Each variation actually changes the strategy content.
        """
        variations     = []
        symbol_chars   = symbol_analysis["overall_characteristics"]
        vol_level      = symbol_analysis["volatility_profile"].get(
            "daily", {}
        ).get("volatility_level", "medium")

        entry_conditions = strategy.get("entry_conditions", [])
        risk_rules       = strategy.get("risk_management",  [])

        # ── Variation 1: Trend-following bias ─────────
        v1 = copy.deepcopy(strategy)
        v1["_adaptation"]           = "trend_following"
        v1["_confidence_threshold"] = 0.6
        # Filter entries to trend-relevant ones
        if entry_conditions and isinstance(entry_conditions[0], dict):
            trend_kws  = ["trend", "breakout", "momentum", "continuation"]
            trend_entries = [
                e for e in entry_conditions
                if any(kw in e.get("rule", "").lower() for kw in trend_kws)
            ]
            if trend_entries:
                v1["entry_conditions"] = trend_entries
        variations.append(v1)

        # ── Variation 2: Mean-reversion bias ──────────
        v2 = copy.deepcopy(strategy)
        v2["_adaptation"]           = "mean_reverting"
        v2["_confidence_threshold"] = 0.5
        if entry_conditions and isinstance(entry_conditions[0], dict):
            rev_kws = ["reversal", "support", "resistance", "bounce", "oversold", "overbought"]
            rev_entries = [
                e for e in entry_conditions
                if any(kw in e.get("rule", "").lower() for kw in rev_kws)
            ]
            if rev_entries:
                v2["entry_conditions"] = rev_entries
        variations.append(v2)

        # ── Variation 3: High-confidence entries only ─
        v3 = copy.deepcopy(strategy)
        v3["_adaptation"]           = "high_confidence_only"
        v3["_confidence_threshold"] = 0.7
        if entry_conditions and isinstance(entry_conditions[0], dict):
            high_conf_entries = [
                e for e in entry_conditions
                if e.get("confidence", 0.5) >= 0.7
            ]
            if high_conf_entries:
                v3["entry_conditions"] = high_conf_entries
        variations.append(v3)

        # ── Variation 4: Volatility-adapted ───────────
        v4 = copy.deepcopy(strategy)
        if vol_level == "high":
            v4["_adaptation"]         = "high_volatility_adapted"
            v4["_confidence_threshold"] = 0.75  # Higher bar in volatile markets
        elif vol_level == "low":
            v4["_adaptation"]         = "low_volatility_adapted"
            v4["_confidence_threshold"] = 0.4   # Lower bar in calm markets
        else:
            v4["_adaptation"]         = "normal_volatility"
            v4["_confidence_threshold"] = 0.55
        variations.append(v4)

        # ── Variation 5: Symbol-characteristic adapted ─
        v5 = copy.deepcopy(strategy)
        if "strong_trend" in symbol_chars:
            v5["_adaptation"] = "strong_trend_adapted"
            # Boost confidence of trend rules
            if entry_conditions and isinstance(entry_conditions[0], dict):
                adjusted = []
                for e in entry_conditions:
                    e_copy = e.copy()
                    trend_kws = ["trend", "breakout", "momentum"]
                    if any(kw in e_copy.get("rule", "").lower() for kw in trend_kws):
                        e_copy["confidence"] = min(1.0, e_copy.get("confidence", 0.5) + 0.1)
                    adjusted.append(e_copy)
                v5["entry_conditions"] = adjusted
        elif "mean_reverting" in symbol_chars:
            v5["_adaptation"] = "mean_reverting_adapted"
            if entry_conditions and isinstance(entry_conditions[0], dict):
                adjusted = []
                for e in entry_conditions:
                    e_copy = e.copy()
                    rev_kws = ["support", "resistance", "reversal", "bounce"]
                    if any(kw in e_copy.get("rule", "").lower() for kw in rev_kws):
                        e_copy["confidence"] = min(1.0, e_copy.get("confidence", 0.5) + 0.1)
                    adjusted.append(e_copy)
                v5["entry_conditions"] = adjusted
        else:
            v5["_adaptation"] = "balanced"
        variations.append(v5)

        return variations

    @staticmethod
    def save_optimized_strategy(symbol_name: str, strategy: Dict):
        """
        Save optimized strategy for a specific symbol.
        Delegates to db_handler which handles safe filenames.
        """
        save_optimized_strategy(symbol_name, strategy)

    @staticmethod
    def load_optimized_strategy(symbol_name: str) -> Optional[Dict]:
        """Load an optimized strategy for a symbol if it exists."""
        return load_optimized_strategy(symbol_name)

# =========================================================
# MASTER PIPELINE
# =========================================================

def test_all_strategies_on_symbol(
    symbol_name: str,
    symbol_data: Dict[str, pd.DataFrame],
    build_master: bool = True
) -> Dict:
    """
    Full pipeline:
    1. Analyze the symbol
    2. Load ALL strategies (YouTube + books + master)
    3. Build AI master strategy for this symbol
    4. Test all strategies against price data
    5. Optimize top 3
    6. Return everything ranked, ready for RL agent

    Args:
        symbol_name:  e.g. "BTC", "EUR/USD", "AAPL"
        symbol_data:  Dict of timeframe -> DataFrame
        build_master: Whether to build/use AI master strategy
    """

    print(f"\n{'='*50}")
    print(f"Strategy Pipeline: {symbol_name}")
    print(f"{'='*50}")

    # Step 1: Analyze symbol
    symbol_analysis = SymbolAnalyzer.analyze_symbol(symbol_data, symbol_name)
    symbol_type     = SymbolAnalyzer.get_symbol_type(symbol_analysis)

    print(f"Symbol type:       {symbol_type}")
    print(f"Characteristics:   {symbol_analysis['overall_characteristics']}")

    # Step 2: Load all strategies from knowledge base
    all_strategies = StrategyLoader.load_all_strategies()

    if not all_strategies:
        print("\nWARNING: No strategies found in knowledge base.")
        print("Run knowledge_base.py first to learn from YouTube/books.")
        return {
            "symbol":         symbol_name,
            "symbol_analysis": symbol_analysis,
            "symbol_type":    symbol_type,
            "strategies":     {},
            "top_strategy":   None,
            "error":          "No learned strategies available"
        }

    # Step 3: Build AI master strategy for this symbol
    if build_master:
        print("\nBuilding AI master strategy...")
        master = MasterStrategyBuilder.build_master_strategy(
            all_strategies, symbol_name, symbol_analysis
        )
        if master:
            all_strategies["ai_master"] = master

    # Step 4: Test all strategies
    print(f"\nTesting {len(all_strategies)} strategies...")

    test_results = {}

    for source_key, strategy in all_strategies.items():
        result = StrategyTester.test_strategy_on_symbol(
            strategy, symbol_data, symbol_name, symbol_analysis
        )
        test_results[source_key] = result
        print(
            f"  {source_key:<35} "
            f"score: {result['fit_score']:5.1f}  "
            f"type: {result['source_type']}"
        )

    # Step 5: Optimize top 3
    top_3 = sorted(
        test_results.items(),
        key=lambda x: x[1]["fit_score"],
        reverse=True
    )[:3]

    optimized_strategies = {}

    print(f"\nOptimizing top {len(top_3)} strategies for {symbol_name}...")

    for source_key, _ in top_3:
        base_strategy = all_strategies[source_key]

        print(f"  Optimizing: {source_key}")

        optimized = StrategyOptimizer.optimize_for_symbol(
            base_strategy, symbol_data, symbol_name, symbol_analysis
        )

        optimized_strategies[source_key] = optimized

        StrategyOptimizer.save_optimized_strategy(
            f"{symbol_name}_{source_key}",
            optimized
        )

    # Step 6: Rank results
    if not optimized_strategies:
        print("WARNING: No strategies were optimized")
        top_strategy        = None
        top_strategy_config = None
        final_ranking       = []
    else:
        final_ranking = sorted(
            optimized_strategies.items(),
            key=lambda x: x[1].get("optimization_score", 0),
            reverse=True
        )
        top_strategy        = final_ranking[0][0]
        top_strategy_config = final_ranking[0][1]

        print(f"\nBest strategy for {symbol_name}: {top_strategy}")
        print(f"Score: {top_strategy_config.get('optimization_score', 0):.1f}")

    return {
        "symbol":               symbol_name,
        "symbol_analysis":      symbol_analysis,
        "symbol_type":          symbol_type,
        "test_results":         test_results,
        "optimized_strategies": optimized_strategies,
        "ranked":               final_ranking,
        "top_strategy":         top_strategy,
        "top_strategy_config":  top_strategy_config,
    }

# =========================================================
# EXAMPLE
# =========================================================

if __name__ == "__main__":

    print("Strategy Tester — Ready")
    print("Loads from: YouTube channels + books + AI master strategy")
    print()
    print("Usage:")
    print("  from strategy_tester import test_all_strategies_on_symbol")
    print("  results = test_all_strategies_on_symbol('BTC', symbol_data)")
    print("  top_config = results['top_strategy_config']")