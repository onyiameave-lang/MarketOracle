"""
boss_ai.py

Final trade decision layer.
Receives signals from all expert agents and makes the
ultimate go/no-go decision on whether to enter a trade.

Architecture:
    - Pure Python logic (no LLM — fast and deterministic)
    - Aggregates: chart_expert signal + strategy fit score
      + news sentiment + social sentiment
    - Weights each signal and computes a final confidence score
    - Only passes trades above confidence threshold

Status: PLANNED — to be built after expert agents are stable
"""

# TODO: Implement BossAI class
# See documentation in README.md for planned architecture
