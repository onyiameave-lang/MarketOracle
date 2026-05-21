"""
social_expert.py

Social media sentiment analysis for trading signals.

Monitors social platforms for market sentiment
on each trading symbol.

Planned data sources:
    - Reddit (PRAW library — free, read-only)
    - StockTwits API (free tier available)

Status: PLANNED — to be connected to chart_expert.py
observation space once implemented.

When implemented, feeds 2 sentiment features into
the RL agent's observation space:
    - social_sentiment_score  (float: -1.0 to 1.0)
    - social_volume_spike     (float: 0.0 to 1.0)
"""

# TODO: Implement SocialExpert class
