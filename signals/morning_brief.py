"""
Orion — Morning Briefing Generator
Generates a daily French-language briefing at 8:00 AM.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

from risk.manager import detect_regime, get_portfolio, Regime
from signals.engine import scan_all
from journal.logger import log_event, get_events, stats

# ─── Module state for daily scheduling ───────────────────────────────
_last_briefing_date: str | None = None
_scheduler_thread: threading.Thread | None = None
_stop_event = threading.Event()


# ═══════════════════════════════════════════════════════════════════════
#  BRIEFING GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_briefing() -> dict:
    """Generate a complete French-language morning briefing.

    Returns a dict with keys: text, timestamp, regime, top_signals, capital, phase.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Current regime
    regime_state = detect_regime()
    regime_name = regime_state.regime.value
    vix = regime_state.vix_simulated

    # 2. Top 3 signals by |score|
    all_signals = scan_all()
    top_3 = sorted(all_signals, key=lambda s: abs(s.score), reverse=True)[:3]
    top_signals_list = [
        {"symbol": s.symbol, "action": s.action.value, "score": s.score}
        for s in top_3
    ]

    # 3. Portfolio status
    portfolio = get_portfolio()
    capital = portfolio.current_capital
    positions_count = len(portfolio.positions)
    drawdown_pct = portfolio.drawdown_pct

    # 4. Journal stats
    journal_stats = stats()

    # ── Build French text ──────────────────────────────────────────────

    # Regime description
    regime_descriptions = {
        "EXPANSION": "Le marche est en phase d'expansion. Conditions favorables pour le trading actif.",
        "CONTRACTION": "Le marche est en contraction. Prudence accrue, preference aux actifs defensifs.",
        "STRESS": "Le marche est en situation de stress. Mode protectif active, aucun nouveau trade.",
    }
    regime_text = regime_descriptions.get(regime_name, "Regime indetermine.")

    # Top signals text
    signals_lines = []
    for s in top_signals_list:
        signals_lines.append(f"  - {s['symbol']} : {s['action']} (score {s['score']:+.1f})")
    signals_text = "\n".join(signals_lines) if signals_lines else "  Aucun signal significatif."

    # Strategy conclusion
    if regime_name == "EXPANSION":
        strategie = "Rechercher des opportunites BUY sur les signaux forts, sizing normal."
    elif regime_name == "CONTRACTION":
        strategie = "Limiter les nouvelles positions aux actifs defensifs, sizing reduit de 30%."
    else:
        strategie = "Aucun nouveau trade autorise. Surveillance uniquement, protection du capital."

    text = (
        f"=== BRIEFING ORION — {timestamp[:10]} ===\n"
        f"\n"
        f"REGIME DE MARCHE\n"
        f"{regime_text} VIX simule : {vix:.1%}.\n"
        f"\n"
        f"TOP SIGNAUX\n"
        f"{signals_text}\n"
        f"\n"
        f"PORTEFEUILLE\n"
        f"Capital : {capital:,.2f} EUR. "
        f"Positions ouvertes : {positions_count}. "
        f"Drawdown : {drawdown_pct:.1%}.\n"
        f"\n"
        f"STRATEGIE DU JOUR\n"
        f"{strategie}\n"
    )

    return {
        "text": text,
        "timestamp": timestamp,
        "regime": regime_name,
        "top_signals": top_signals_list,
        "capital": capital,
    }


# ═══════════════════════════════════════════════════════════════════════
#  STORAGE & RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════

def store_briefing(briefing: dict):
    """Store a briefing in the journal as an event."""
    log_event("briefing", briefing["text"], "INFO", data=briefing)


def get_latest_briefing() -> dict | None:
    """Retrieve the most recent briefing from the journal.

    Returns the briefing dict, or None if no briefing has been stored.
    """
    df = get_events(category="briefing", limit=1)
    if df.empty:
        return None

    import json
    row = df.iloc[0]
    data_str = row.get("data")
    if data_str:
        try:
            return json.loads(data_str)
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: return basic info from the event row
    return {
        "text": row.get("message", ""),
        "timestamp": row.get("timestamp", ""),
    }


# ═══════════════════════════════════════════════════════════════════════
#  DAILY SCHEDULER
# ═══════════════════════════════════════════════════════════════════════

def _schedule_loop():
    """Background loop: generate and store briefing at 8:00 AM daily."""
    global _last_briefing_date

    while not _stop_event.is_set():
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        # Check if it's past 8:00 AM and we haven't generated today's briefing
        if now.hour >= 8 and _last_briefing_date != today:
            try:
                briefing = generate_briefing()
                store_briefing(briefing)
                _last_briefing_date = today
                print(f"  [Orion] Briefing du {today} genere et stocke.")
            except Exception as e:
                print(f"  [Orion] Erreur briefing : {e}")

        _stop_event.wait(timeout=60)


def schedule_daily_briefing():
    """Start the daily briefing scheduler in a daemon thread.

    Checks every 60 seconds. At 8:00 AM (first check after 8:00),
    generates and stores the briefing. Only runs once per day.
    """
    global _scheduler_thread
    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        return
    _stop_event.clear()
    _scheduler_thread = threading.Thread(
        target=_schedule_loop, daemon=True, name="orion-morning-brief"
    )
    _scheduler_thread.start()


def stop_daily_briefing():
    """Stop the daily briefing scheduler."""
    _stop_event.set()
    if _scheduler_thread is not None:
        _scheduler_thread.join(timeout=5)
