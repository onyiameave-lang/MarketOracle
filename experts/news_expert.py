"""
news_expert.py

News sentiment analysis for trading signals.

Fetches news headlines for each symbol and scores
their market impact using sentiment analysis.

Planned data sources:
    - NewsAPI (free tier: 100 requests/day)
    - CoinGecko news (free, no key needed for crypto)
    - Yahoo Finance news (via yfinance)

Status: PLANNED — to be connected to chart_expert.py
observation space once implemented.

When implemented, feeds 2 sentiment features into
the RL agent's observation space:
    - news_sentiment_score  (float: -1.0 to 1.0)
    - news_impact_magnitude (float: 0.0 to 1.0)
"""

# TODO: Implement NewsExpert class
