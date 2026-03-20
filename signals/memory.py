"""
Orion — System Memory (Daily Journal Entries)
Generates automatic daily entries stored in SQLite.
"""

from __future__ import annotations

import threading
from datetime import datetime

from risk.manager import detect_regime, get_portfolio
from signals.engine import scan_all
from journal.logger import log_event, get_events

# ─── Module state for daily scheduling ───────────────────────────────
_last_memory_date: str | None = None
_scheduler_thread: threading.Thread | None = None
_stop_event = threading.Event()


# ═══════════════════════════════════════════════════════════════════════
#  DAILY ENTRY GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_daily_entry() -> str:
    """Generate a French-language daily memory entry summarizing the day.

    Collects regime, signals, portfolio, and macro data to build a
    concise paragraph covering the system's state.

    Returns the text string.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    # Regime
    regime_state = detect_regime()
    regime_name = regime_state.regime.value
    vix = regime_state.vix_simulated

    # Top signal
    signals = scan_all()
    if signals:
        top = sorted(signals, key=lambda s: abs(s.score), reverse=True)[0]
        top_asset = top.symbol
        top_score = top.score
        top_text = f"{top_asset} en tete des signaux a {top_score:+.1f}"
    else:
        top_text = "Aucun signal genere"

    # Corrélation (Aladdin)
    avg_corr = regime_state.avg_correlation
    corr_text = f"Correlation moyenne {avg_corr:.2f}"

    # Positions ouvertes
    portfolio = get_portfolio()
    n_pos = len(portfolio.positions)
    if n_pos > 0:
        trades_text = f"{n_pos} position(s) ouverte(s)"
        reason = "gestion active"
    else:
        trades_text = "Aucun trade actif"
        reason = "pas de signal actionnable" if not signals else "signaux insuffisants"

    # Portfolio
    capital = portfolio.current_capital

    # System mode
    if regime_name == "STRESS":
        mode = "surveillance"
    elif regime_name == "CONTRACTION":
        mode = "trading defensif"
    else:
        mode = "trading actif"

    # Build paragraph
    entry = (
        f"{date_str} — Regime {regime_name}. "
        f"VIX simule {vix:.1%}. "
        f"{top_text}. "
        f"{corr_text}. "
        f"{trades_text} — {reason}. "
        f"Capital {capital:,.2f} EUR. "
        f"Systeme en {mode}."
    )

    return entry


# ═══════════════════════════════════════════════════════════════════════
#  STORAGE & RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════

def store_daily_entry(text: str):
    """Store a daily memory entry in the journal."""
    log_event("memory", text, "INFO")


def get_memory_entries(limit: int = 30) -> list[dict]:
    """Retrieve recent memory entries from the journal.

    Args:
        limit: Maximum number of entries to return (default 30).

    Returns:
        A list of dicts with keys 'date' and 'text', ordered by
        timestamp descending.
    """
    df = get_events(category="memory", limit=limit)
    if df.empty:
        return []

    entries = []
    for _, row in df.iterrows():
        ts = row.get("timestamp", "")
        date_str = ts[:10] if ts else ""
        entries.append({
            "date": date_str,
            "text": row.get("message", ""),
        })
    return entries


# ═══════════════════════════════════════════════════════════════════════
#  DAILY SCHEDULER
# ═══════════════════════════════════════════════════════════════════════

def _schedule_loop():
    """Background loop: generate and store daily memory at 22:00."""
    global _last_memory_date

    while not _stop_event.is_set():
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        # Check if it's past 22:00 and we haven't generated today's entry
        if now.hour >= 22 and _last_memory_date != today:
            try:
                text = generate_daily_entry()
                store_daily_entry(text)
                _last_memory_date = today
                print(f"  [Orion] Memoire du {today} generee et stockee.")
            except Exception as e:
                print(f"  [Orion] Erreur memoire : {e}")

        _stop_event.wait(timeout=60)


def schedule_daily_memory():
    """Start the daily memory scheduler in a daemon thread.

    Checks every 60 seconds. At 22:00 (first check after 22:00),
    generates and stores the daily entry. Only runs once per day.
    """
    global _scheduler_thread
    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        return
    _stop_event.clear()
    _scheduler_thread = threading.Thread(
        target=_schedule_loop, daemon=True, name="orion-daily-memory"
    )
    _scheduler_thread.start()


def stop_daily_memory():
    """Stop the daily memory scheduler."""
    _stop_event.set()
    if _scheduler_thread is not None:
        _scheduler_thread.join(timeout=5)
