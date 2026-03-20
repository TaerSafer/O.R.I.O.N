"""
Orion — Price Projections & Key Levels
"""

from __future__ import annotations

import numpy as np

from data.collector import get_history, ASSETS
from signals.indicators import atr, sma, rsi


# ═══════════════════════════════════════════════════════════════════════
#  SINGLE ASSET PROJECTION
# ═══════════════════════════════════════════════════════════════════════

def compute_projection(symbol: str) -> dict | None:
    """Compute price projections and key levels for a single asset.

    Uses RSI, SMA trend, and ATR to estimate:
    - 5-day probability of price increase (0-1)
    - 20-day probability of price increase (0-1)
    - Support level (close - 2*ATR)
    - Resistance level (close + 2*ATR)

    Returns None if insufficient data.
    """
    df = get_history(symbol, 90)

    if df.empty or len(df) < 50:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    current_price = float(close.iloc[-1])

    # ── 5-day probability (RSI-based) ─────────────────────────────────
    rsi_val = rsi(close, 14).iloc[-1]
    if np.isnan(rsi_val):
        prob_5d = 0.5
    elif rsi_val < 30:
        prob_5d = 0.70
    elif rsi_val > 70:
        prob_5d = 0.30
    else:
        prob_5d = 0.50 + (50 - rsi_val) / 100 * 0.50

    prob_5d = round(max(0.0, min(1.0, prob_5d)), 3)

    # ── 20-day probability (SMA trend-based) ──────────────────────────
    sma20 = sma(close, 20).iloc[-1]
    sma50 = sma(close, 50).iloc[-1] if len(close) >= 50 else np.nan

    if not np.isnan(sma20) and not np.isnan(sma50):
        if current_price > sma20 and sma20 > sma50:
            prob_20d = 0.65
        elif current_price < sma20 and sma20 < sma50:
            prob_20d = 0.35
        else:
            prob_20d = 0.50
    elif not np.isnan(sma20):
        prob_20d = 0.55 if current_price > sma20 else 0.45
    else:
        prob_20d = 0.50

    prob_20d = round(max(0.0, min(1.0, prob_20d)), 3)

    # ── Key levels (ATR-based) ────────────────────────────────────────
    atr_val = atr(high, low, close, 14).iloc[-1]
    if np.isnan(atr_val) or atr_val == 0:
        # Fallback: use simple price range
        atr_val = float((high.iloc[-14:].max() - low.iloc[-14:].min()) / 14)

    support = round(current_price - 2 * atr_val, 6)
    resistance = round(current_price + 2 * atr_val, 6)

    return {
        "symbol": symbol,
        "prob_5d": prob_5d,
        "prob_20d": prob_20d,
        "support": support,
        "resistance": resistance,
        "current_price": round(current_price, 6),
    }


# ═══════════════════════════════════════════════════════════════════════
#  ALL ASSETS PROJECTIONS
# ═══════════════════════════════════════════════════════════════════════

def compute_all_projections() -> list[dict]:
    """Compute projections for all assets in the Orion universe.

    Skips assets with insufficient data.
    Returns a list of projection dicts.
    """
    results = []
    for symbol in ASSETS:
        proj = compute_projection(symbol)
        if proj is not None:
            results.append(proj)
    return results
