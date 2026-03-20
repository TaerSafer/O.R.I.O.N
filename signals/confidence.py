"""
Orion — Global System Confidence Score (0-100%)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from risk.manager import detect_regime, get_portfolio, Regime
from signals.engine import scan_all
from data.collector import get_price


# ═══════════════════════════════════════════════════════════════════════
#  CONFIDENCE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_confidence() -> dict:
    """Compute a global system confidence score (0-100).

    Aggregates 4 sub-scores with equal 25% weight:
    - signal_quality:   quality of current trading signals
    - regime_stability: stability of the market regime
    - portfolio_health: health of the portfolio (drawdown, positions)
    - data_freshness:   freshness of market and macro data

    Returns a dict with score, level, sub-scores, and timestamp.
    """

    # ── 1. Signal Quality (25%) ────────────────────────────────────────
    try:
        signals = scan_all()
        if signals:
            strong = sum(1 for s in signals if abs(s.score) > 20)
            signal_quality = int((strong / len(signals)) * 100)
        else:
            signal_quality = 50
    except Exception:
        signal_quality = 50

    # ── 2. Regime Stability (25%) ──────────────────────────────────────
    try:
        regime_state = detect_regime()
        regime = regime_state.regime
        vix = regime_state.vix_simulated
        avg_corr = regime_state.avg_correlation

        if regime == Regime.EXPANSION and vix < 0.15:
            regime_stability = 90
        elif regime == Regime.CONTRACTION:
            regime_stability = 50
        elif regime == Regime.STRESS:
            regime_stability = 20
        else:
            # EXPANSION but VIX >= 0.15
            regime_stability = 70

        # Adjust by correlation: high correlation reduces stability
        # avg_corr typically 0.2-0.6; penalize above 0.4
        if avg_corr > 0.4:
            penalty = int((avg_corr - 0.4) * 100)
            regime_stability = max(0, regime_stability - penalty)
    except Exception:
        regime_stability = 50

    # ── 3. Portfolio Health (25%) ──────────────────────────────────────
    try:
        portfolio = get_portfolio()
        portfolio_health = 100

        # Subtract drawdown impact
        dd_pct = portfolio.drawdown_pct
        portfolio_health -= int(dd_pct * 200)

        # Subtract for excess positions (over 5)
        n_positions = len(portfolio.positions)
        if n_positions > 5:
            portfolio_health -= (n_positions - 5) * 5

        # Cooldown penalty
        if portfolio.is_in_cooldown:
            portfolio_health = 10

        # Clamp to 0-100
        portfolio_health = max(0, min(100, portfolio_health))
    except Exception:
        portfolio_health = 50

    # ── 4. Data Freshness (33%) ───────────────────────────────────────
    try:
        # Check price data recency for S&P 500
        price = get_price("^GSPC")
        if price and price.get("date"):
            price_date_str = price["date"][:10]
            try:
                price_date = datetime.strptime(price_date_str, "%Y-%m-%d")
                age_days = (datetime.now() - price_date).days
                if age_days <= 1:
                    data_freshness = 100
                elif age_days <= 3:
                    data_freshness = 70  # Weekend gap is ok
                else:
                    data_freshness = 30
            except ValueError:
                data_freshness = 50
        else:
            data_freshness = 30
    except Exception:
        data_freshness = 50

    # ── Global Score (3 pilliers : signaux, régime, portefeuille + fraîcheur) ─
    score = int(
        0.30 * signal_quality
        + 0.30 * regime_stability
        + 0.25 * portfolio_health
        + 0.15 * data_freshness
    )
    score = max(0, min(100, score))

    # Level classification
    if score >= 70:
        level = "high"
    elif score >= 40:
        level = "medium"
    else:
        level = "low"

    return {
        "score": score,
        "level": level,
        "signal_quality": signal_quality,
        "regime_stability": regime_stability,
        "portfolio_health": portfolio_health,
        "data_freshness": data_freshness,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
