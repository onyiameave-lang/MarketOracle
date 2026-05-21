"""
main.py — MarketOracle Training Pipeline

Full pipeline:
1. Load price data
2. Learn from YouTube channels and books (knowledge base)
3. Test and optimize strategies per symbol
4. Train RL agent across all symbols
5. Evaluate on test set
6. Save results
"""

import os
from dotenv import load_dotenv
load_dotenv()
import json
import argparse
from datetime import datetime

# =========================================================
# PIPELINE STEPS
# =========================================================

def step_learn_knowledge(
    channels: list = None,
    books: list    = None,
    topic: str     = "trading_strategy",
    force_refresh: bool = False
):
    """
    Step 1 — Learn from YouTube channels and books.
    Skips anything already cached.
    Only calls Gemini for new content.
    """
    from experts.knowledge_base import (
        learn_from_channel,
        learn_from_book,
        merge_all_knowledge,
        YOUTUBE_CHANNELS,
        BOOK_DATABASE,
    )

    print("\n" + "="*60)
    print("STEP 1: KNOWLEDGE LEARNING")

    # Learn from channels
    learn_channels = channels or list(YOUTUBE_CHANNELS.keys())

    for channel_key in learn_channels:
        try:
            print(f"\nChannel: {channel_key}")
            learn_from_channel(
                channel_key   = channel_key,
                topic         = topic,
                max_videos    = 10,
                force_refresh = force_refresh
            )
        except Exception as e:
            print(f"  WARNING: Could not learn from {channel_key}: {e}")

    # Learn from books
    learn_books = books or list(BOOK_DATABASE.keys())

    for book_key in learn_books:
        book = BOOK_DATABASE[book_key]
        if not os.path.exists(book["path"]):
            print(f"\nBook not found: {book['title']} — skipping")
            print(f"  Add PDF to: {book['path']}")
            continue
        try:
            print(f"\nBook: {book['title']}")
            learn_from_book(
                book_key      = book_key,
                force_refresh = force_refresh
            )
        except Exception as e:
            print(f"  WARNING: Could not learn from {book_key}: {e}")

    # Merge all sources into master knowledge
    print("\nMerging all knowledge sources...")
    try:
        master = merge_all_knowledge(topic)
        if master:
            print(f"Master knowledge built from all sources")
        else:
            print("WARNING: Master knowledge is empty — check learning logs")
    except Exception as e:
        print(f"WARNING: Could not merge knowledge: {e}")


def step_load_data(data_dir: str = "data") -> dict:
    """
    Step 2 — Load and split all price data.
    Returns train/test split data bundle.
    """
    from experts.chart_expert import load_data_bundle

    print("\n" + "="*60)
    print("STEP 2: LOADING PRICE DATA")
    print("="*60)

    data_bundle = load_data_bundle(data_dir)

    if not data_bundle:
        raise RuntimeError(
            f"No data loaded from {data_dir}. "
            f"Add CSV files with naming: SYMBOL_TIMEFRAME.csv "
            f"e.g. BTC_daily.csv, BTC_1h.csv"
        )

    print(f"\nLoaded {len(data_bundle)} symbols:")
    for symbol, tfs in data_bundle.items():
        tf_list = list(tfs.keys())
        print(f"  {symbol:<15} timeframes: {tf_list}")

    return data_bundle


def step_optimize_strategies(data_bundle: dict) -> dict:
    """
    Step 3 — Test and optimize strategies per symbol.
    Loads learned rules from knowledge base.
    Finds best strategy fit for each symbol.
    Saves optimized configs for chart_expert to load.
    """
    from experts.strategy_tester import test_all_strategies_on_symbol

    print("\n" + "="*60)
    print("STEP 3: STRATEGY OPTIMIZATION PER SYMBOL")
    print("="*60)

    results = {}

    for symbol, symbol_data in data_bundle.items():
        print(f"\nOptimizing strategies for: {symbol}")

        # Build flat data dict for strategy_tester
        # (uses train split for strategy testing)
        flat_data = {
            tf: splits["train"]
            for tf, splits in symbol_data.items()
            if "train" in splits
        }

        try:
            result = test_all_strategies_on_symbol(
                symbol_name  = symbol,
                symbol_data  = flat_data,
                build_master = True
            )
            results[symbol] = result

            top = result.get("top_strategy")
            score = result.get("top_strategy_config", {}).get(
                "optimization_score", 0
            )

            if top:
                print(f"  Best strategy: {top} (score: {score:.1f})")
            else:
                print(f"  No strategy optimized — will use master knowledge")

        except Exception as e:
            print(f"  WARNING: Strategy optimization failed for {symbol}: {e}")

    print(f"\nStrategy optimization complete for {len(results)} symbols")
    return results


def step_train(
    data_bundle: dict,
    timesteps: int = 500_000
):
    """
    Step 4 — Train the RL agent.
    Agent sees all symbols, uses strategy signals,
    can go long or short, switches symbols when no setup.
    """
    from experts.chart_expert import StrategyTrainer

    print("\n" + "="*60)
    print("STEP 4: TRAINING RL AGENT")
    print("="*60)
    print(f"Symbols: {len(data_bundle)}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Actions: Hold / Buy(Long) / Sell(Short) / Close / Switch")

    os.makedirs("models", exist_ok=True)

    trainer = StrategyTrainer(data_bundle)
    model   = trainer.train(total_timesteps=timesteps)

    print("\nTraining complete")
    return trainer, model


def step_evaluate(
    trainer,
    model,
    data_bundle: dict,
    episodes: int = 10
) -> dict:
    """
    Step 5 — Evaluate on test set.
    Win rate here is real — agent has never seen this data.
    """
    print("\n" + "="*60)
    print("STEP 5: EVALUATION (TEST SET ONLY)")
    print("="*60)

    # Overall evaluation
    overall = trainer.evaluate(model, episodes=episodes)

    # Per-symbol evaluation
    per_symbol = {}
    for symbol in data_bundle.keys():
        try:
            result = trainer.evaluate_symbol(model, symbol)
            per_symbol[symbol] = result
            print(
                f"  {symbol:<15} "
                f"trades: {result['trades']:>4}  "
                f"win rate: {result['win_rate']:.1%}"
            )
        except Exception as e:
            print(f"  {symbol}: evaluation failed — {e}")

    overall["per_symbol"] = per_symbol

    return overall


def step_save_results(results: dict, output_dir: str = "results"):
    """Step 6 — Save evaluation results."""

    os.makedirs(output_dir, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/evaluation_{timestamp}.json"

    # Make results JSON serializable
    def _serialize(obj):
        if isinstance(obj, float):
            return round(obj, 4)
        if isinstance(obj, (list, tuple)):
            return [_serialize(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        return obj

    with open(output_path, "w") as f:
        json.dump(_serialize(results), f, indent=2)

    print(f"\nResults saved -> {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total trades:  {results.get('total_trades', 0)}")
    print(f"Win rate:      {results.get('win_rate', 0):.1%}")
    print(f"Max drawdown:  {results.get('max_drawdown', 0):.1%}")
    print(f"Final balance: ${results.get('final_balance', 10000):.2f}")

    win_rate = results.get("win_rate", 0)
    if win_rate >= 0.70:
        print("\n TARGET REACHED: Win rate >= 70%")
    elif win_rate >= 0.60:
        print("\n ACCEPTABLE: Win rate >= 60% — keep training")
    else:
        print("\n NEEDS WORK: Win rate below 60%")
        print("  → Check strategy optimization results")
        print("  → Review knowledge base learning logs")
        print("  → Consider more training timesteps")

# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="MarketOracle — AI Trading System"
    )

    parser.add_argument(
        "--skip-learning",
        action="store_true",
        help="Skip knowledge learning (use cached rules)"
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip strategy optimization (use cached configs)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Training timesteps (default: 500,000)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Evaluation episodes (default: 10)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing price CSV files (default: data/)"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="trading_strategy",
        help="Topic to learn from YouTube/books (default: trading_strategy)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-learning even if cached"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  MARKETORACLE — AI TRADING SYSTEM")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Step 1: Learn knowledge ───────────────────────────
    if not args.skip_learning:
        step_learn_knowledge(
            topic         = args.topic,
            force_refresh = args.force_refresh
        )
    else:
        print("\nSkipping knowledge learning (--skip-learning)")

    # ── Step 2: Load data ─────────────────────────────────
    data_bundle = step_load_data(args.data_dir)

    # ── Step 3: Optimize strategies ───────────────────────
    if not args.skip_optimization:
        step_optimize_strategies(data_bundle)
    else:
        print("\nSkipping strategy optimization (--skip-optimization)")

    # ── Step 4: Train ─────────────────────────────────────
    trainer, model = step_train(
        data_bundle = data_bundle,
        timesteps   = args.timesteps
    )

    # ── Step 5: Evaluate ──────────────────────────────────
    results = step_evaluate(
        trainer     = trainer,
        model       = model,
        data_bundle = data_bundle,
        episodes    = args.episodes
    )

    # ── Step 6: Save results ──────────────────────────────
    step_save_results(results)

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
