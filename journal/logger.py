"""
Orion — Trading Journal
-------------------------
Journal de trading persistant SQLite pour l'enregistrement et l'analyse
de chaque trade, événement système, snapshot portefeuille et note manuelle.

Tables :
- trades       : historique complet de chaque trade (entrée, sortie, PnL)
- events       : événements système (régime, drawdown, cooldown, erreurs)
- snapshots    : état du portefeuille à intervalles réguliers
- notes        : notes manuelles du trader attachées à un trade ou un jour
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "orion.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "journal_conn"):
        _local.journal_conn = sqlite3.connect(str(DB_PATH))
        _local.journal_conn.execute("PRAGMA journal_mode=WAL")
        _local.journal_conn.execute("PRAGMA synchronous=NORMAL")
        _local.journal_conn.row_factory = sqlite3.Row
    return _local.journal_conn


def init_journal():
    """Crée les tables du journal si elles n'existent pas."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT NOT NULL,
            asset_class   TEXT,
            action        TEXT NOT NULL,
            entry_date    TEXT NOT NULL,
            entry_price   REAL NOT NULL,
            exit_date     TEXT,
            exit_price    REAL,
            size          REAL NOT NULL,
            stop_loss     REAL,
            take_profit   REAL,
            pnl           REAL,
            pnl_pct       REAL,
            commission    REAL DEFAULT 0,
            exit_reason   TEXT,
            regime        TEXT,
            sizing_method TEXT,
            signal_score  REAL,
            confidence    REAL,
            risk_amount   REAL,
            holding_days  INTEGER,
            tags          TEXT,
            created_at    TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(entry_date);
        CREATE INDEX IF NOT EXISTS idx_trades_exit ON trades(exit_date);

        CREATE TABLE IF NOT EXISTS events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT NOT NULL,
            category   TEXT NOT NULL,
            level      TEXT NOT NULL DEFAULT 'INFO',
            message    TEXT NOT NULL,
            data       TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_events_cat ON events(category);

        CREATE TABLE IF NOT EXISTS snapshots (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT NOT NULL,
            capital           REAL,
            equity            REAL,
            total_exposure    REAL,
            exposure_pct      REAL,
            total_risk        REAL,
            risk_pct          REAL,
            positions_count   INTEGER,
            drawdown_pct      REAL,
            regime            TEXT,
            positions_detail  TEXT,
            created_at        TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_snap_ts ON snapshots(timestamp);

        CREATE TABLE IF NOT EXISTS notes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            date       TEXT NOT NULL,
            trade_id   INTEGER,
            content    TEXT NOT NULL,
            tags       TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (trade_id) REFERENCES trades(id)
        );

        CREATE INDEX IF NOT EXISTS idx_notes_date ON notes(date);
    """)
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════
#  TRADES
# ═══════════════════════════════════════════════════════════════════════

def log_trade_open(symbol: str, asset_class: str, action: str,
                   entry_date: str, entry_price: float, size: float,
                   stop_loss: float = 0, take_profit: float = 0,
                   regime: str = "", sizing_method: str = "",
                   signal_score: float = 0, confidence: float = 0,
                   risk_amount: float = 0, tags: str = "") -> int:
    """Enregistre l'ouverture d'un trade. Retourne l'ID du trade."""
    conn = _get_conn()
    cur = conn.execute("""
        INSERT INTO trades
            (symbol, asset_class, action, entry_date, entry_price, size,
             stop_loss, take_profit, regime, sizing_method,
             signal_score, confidence, risk_amount, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (symbol, asset_class, action, entry_date, entry_price, size,
          stop_loss, take_profit, regime, sizing_method,
          signal_score, confidence, risk_amount, tags))
    conn.commit()
    return cur.lastrowid


def log_trade_close(trade_id: int, exit_date: str, exit_price: float,
                    pnl: float, pnl_pct: float, exit_reason: str = "",
                    commission: float = 0, holding_days: int = 0):
    """Enregistre la clôture d'un trade existant."""
    conn = _get_conn()
    conn.execute("""
        UPDATE trades SET
            exit_date = ?, exit_price = ?, pnl = ?, pnl_pct = ?,
            exit_reason = ?, commission = ?, holding_days = ?
        WHERE id = ?
    """, (exit_date, exit_price, pnl, pnl_pct, exit_reason,
          commission, holding_days, trade_id))
    conn.commit()


def log_trade_full(symbol: str, asset_class: str, action: str,
                   entry_date: str, entry_price: float,
                   exit_date: str, exit_price: float,
                   size: float, pnl: float, pnl_pct: float,
                   exit_reason: str = "", regime: str = "",
                   holding_days: int = 0, commission: float = 0,
                   signal_score: float = 0, confidence: float = 0,
                   risk_amount: float = 0, sizing_method: str = "",
                   tags: str = "") -> int:
    """Enregistre un trade complet (entrée + sortie) en une fois."""
    conn = _get_conn()
    cur = conn.execute("""
        INSERT INTO trades
            (symbol, asset_class, action, entry_date, entry_price,
             exit_date, exit_price, size, pnl, pnl_pct,
             exit_reason, regime, holding_days, commission,
             signal_score, confidence, risk_amount, sizing_method, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (symbol, asset_class, action, entry_date, entry_price,
          exit_date, exit_price, size, pnl, pnl_pct,
          exit_reason, regime, holding_days, commission,
          signal_score, confidence, risk_amount, sizing_method, tags))
    conn.commit()
    return cur.lastrowid


def get_trade(trade_id: int) -> dict | None:
    """Retourne un trade par ID."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
    return dict(row) if row else None


def get_open_trades() -> list[dict]:
    """Retourne les trades ouverts (sans exit_date)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM trades WHERE exit_date IS NULL ORDER BY entry_date DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_trades(symbol: str | None = None,
               start_date: str | None = None,
               end_date: str | None = None,
               action: str | None = None,
               regime: str | None = None,
               limit: int = 500) -> pd.DataFrame:
    """Recherche de trades avec filtres. Retourne un DataFrame."""
    conn = _get_conn()
    conditions = []
    params = []

    if symbol:
        conditions.append("symbol = ?")
        params.append(symbol)
    if start_date:
        conditions.append("entry_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("entry_date <= ?")
        params.append(end_date + " 23:59:59")
    if action:
        conditions.append("action = ?")
        params.append(action)
    if regime:
        conditions.append("regime = ?")
        params.append(regime)

    where = " AND ".join(conditions) if conditions else "1=1"
    params.append(limit)

    return pd.read_sql_query(
        f"SELECT * FROM trades WHERE {where} ORDER BY entry_date DESC LIMIT ?",
        conn, params=params)


# ═══════════════════════════════════════════════════════════════════════
#  ÉVÉNEMENTS
# ═══════════════════════════════════════════════════════════════════════

def log_event(category: str, message: str, level: str = "INFO",
              data: dict | None = None, timestamp: str | None = None):
    """Enregistre un événement système.

    Catégories : regime, drawdown, capital_protection, execution, error, system
    Niveaux : DEBUG, INFO, WARNING, CRITICAL
    """
    conn = _get_conn()
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_json = json.dumps(data, default=str) if data else None
    conn.execute(
        "INSERT INTO events (timestamp, category, level, message, data) VALUES (?, ?, ?, ?, ?)",
        (ts, category, level, message, data_json))
    conn.commit()


def get_events(category: str | None = None,
               level: str | None = None,
               start_date: str | None = None,
               end_date: str | None = None,
               limit: int = 200) -> pd.DataFrame:
    """Recherche d'événements avec filtres."""
    conn = _get_conn()
    conditions = []
    params = []

    if category:
        conditions.append("category = ?")
        params.append(category)
    if level:
        conditions.append("level = ?")
        params.append(level)
    if start_date:
        conditions.append("timestamp >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("timestamp <= ?")
        params.append(end_date + " 23:59:59")

    where = " AND ".join(conditions) if conditions else "1=1"
    params.append(limit)

    return pd.read_sql_query(
        f"SELECT * FROM events WHERE {where} ORDER BY timestamp DESC LIMIT ?",
        conn, params=params)


# ═══════════════════════════════════════════════════════════════════════
#  SNAPSHOTS PORTEFEUILLE
# ═══════════════════════════════════════════════════════════════════════

def log_snapshot(capital: float, equity: float,
                 total_exposure: float, exposure_pct: float,
                 total_risk: float, risk_pct: float,
                 positions_count: int, drawdown_pct: float,
                 regime: str = "",
                 positions_detail: dict | None = None,
                 timestamp: str | None = None):
    """Enregistre un snapshot de l'état du portefeuille."""
    conn = _get_conn()
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detail_json = json.dumps(positions_detail, default=str) if positions_detail else None
    conn.execute("""
        INSERT INTO snapshots
            (timestamp, capital, equity, total_exposure, exposure_pct,
             total_risk, risk_pct, positions_count, drawdown_pct,
             regime, positions_detail)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ts, capital, equity, total_exposure, exposure_pct,
          total_risk, risk_pct, positions_count, drawdown_pct,
          regime, detail_json))
    conn.commit()


def get_snapshots(start_date: str | None = None,
                  end_date: str | None = None,
                  limit: int = 500) -> pd.DataFrame:
    """Retourne les snapshots du portefeuille."""
    conn = _get_conn()
    conditions = []
    params = []
    if start_date:
        conditions.append("timestamp >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("timestamp <= ?")
        params.append(end_date + " 23:59:59")
    where = " AND ".join(conditions) if conditions else "1=1"
    params.append(limit)

    return pd.read_sql_query(
        f"SELECT * FROM snapshots WHERE {where} ORDER BY timestamp DESC LIMIT ?",
        conn, params=params)


def get_equity_curve(start_date: str | None = None,
                     end_date: str | None = None) -> pd.DataFrame:
    """Retourne la courbe d'equity à partir des snapshots."""
    df = get_snapshots(start_date, end_date, limit=10_000)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[["timestamp", "equity", "capital", "drawdown_pct", "regime"]].sort_values("timestamp")


# ═══════════════════════════════════════════════════════════════════════
#  NOTES DU TRADER
# ═══════════════════════════════════════════════════════════════════════

def add_note(content: str, date: str | None = None,
             trade_id: int | None = None, tags: str = "") -> int:
    """Ajoute une note manuelle au journal."""
    conn = _get_conn()
    d = date or datetime.now().strftime("%Y-%m-%d")
    cur = conn.execute(
        "INSERT INTO notes (date, trade_id, content, tags) VALUES (?, ?, ?, ?)",
        (d, trade_id, content, tags))
    conn.commit()
    return cur.lastrowid


def get_notes(date: str | None = None,
              trade_id: int | None = None,
              tags: str | None = None,
              limit: int = 100) -> pd.DataFrame:
    """Recherche de notes."""
    conn = _get_conn()
    conditions = []
    params = []
    if date:
        conditions.append("date = ?")
        params.append(date)
    if trade_id:
        conditions.append("trade_id = ?")
        params.append(trade_id)
    if tags:
        conditions.append("tags LIKE ?")
        params.append(f"%{tags}%")
    where = " AND ".join(conditions) if conditions else "1=1"
    params.append(limit)

    return pd.read_sql_query(
        f"SELECT * FROM notes WHERE {where} ORDER BY date DESC, created_at DESC LIMIT ?",
        conn, params=params)


# ═══════════════════════════════════════════════════════════════════════
#  ANALYTICS
# ═══════════════════════════════════════════════════════════════════════

def stats(start_date: str | None = None,
          end_date: str | None = None) -> dict:
    """Statistiques globales sur les trades clôturés."""
    conn = _get_conn()
    conditions = ["exit_date IS NOT NULL"]
    params = []
    if start_date:
        conditions.append("entry_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("entry_date <= ?")
        params.append(end_date + " 23:59:59")
    where = " AND ".join(conditions)

    rows = conn.execute(
        f"SELECT pnl, pnl_pct, holding_days, exit_reason, regime, asset_class "
        f"FROM trades WHERE {where}", params
    ).fetchall()

    if not rows:
        return {"total_trades": 0}

    pnls = [r["pnl"] for r in rows if r["pnl"] is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    # Séquences consécutives
    max_cw, max_cl, cw, cl = 0, 0, 0, 0
    for p in pnls:
        if p > 0:
            cw += 1; cl = 0; max_cw = max(max_cw, cw)
        else:
            cl += 1; cw = 0; max_cl = max(max_cl, cl)

    # Par régime
    by_regime: dict[str, dict] = {}
    for r in rows:
        reg = r["regime"] or "UNKNOWN"
        if reg not in by_regime:
            by_regime[reg] = {"trades": 0, "pnl": 0.0, "wins": 0}
        by_regime[reg]["trades"] += 1
        by_regime[reg]["pnl"] += r["pnl"] or 0
        by_regime[reg]["wins"] += int((r["pnl"] or 0) > 0)

    # Par classe
    by_class: dict[str, dict] = {}
    for r in rows:
        cls = r["asset_class"] or "unknown"
        if cls not in by_class:
            by_class[cls] = {"trades": 0, "pnl": 0.0, "wins": 0}
        by_class[cls]["trades"] += 1
        by_class[cls]["pnl"] += r["pnl"] or 0
        by_class[cls]["wins"] += int((r["pnl"] or 0) > 0)

    # Par raison de sortie
    by_exit: dict[str, dict] = {}
    for r in rows:
        reason = r["exit_reason"] or "unknown"
        if reason not in by_exit:
            by_exit[reason] = {"count": 0, "pnl": 0.0}
        by_exit[reason]["count"] += 1
        by_exit[reason]["pnl"] += r["pnl"] or 0

    holdings = [r["holding_days"] for r in rows if r["holding_days"] is not None]

    return {
        "total_trades": len(pnls),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(len(wins) / len(pnls), 4) if pnls else 0,
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl": round(float(np.mean(pnls)), 2) if pnls else 0,
        "avg_win": round(float(np.mean(wins)), 2) if wins else 0,
        "avg_loss": round(float(np.mean(losses)), 2) if losses else 0,
        "best_trade": round(max(pnls), 2) if pnls else 0,
        "worst_trade": round(min(pnls), 2) if pnls else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "avg_holding_days": round(float(np.mean(holdings)), 1) if holdings else 0,
        "max_consecutive_wins": max_cw,
        "max_consecutive_losses": max_cl,
        "by_regime": by_regime,
        "by_class": by_class,
        "by_exit": by_exit,
    }


def daily_pnl(start_date: str | None = None,
              end_date: str | None = None) -> pd.DataFrame:
    """P&L quotidien agrégé (trades clôturés par jour de sortie)."""
    conn = _get_conn()
    conditions = ["exit_date IS NOT NULL"]
    params = []
    if start_date:
        conditions.append("exit_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("exit_date <= ?")
        params.append(end_date + " 23:59:59")
    where = " AND ".join(conditions)

    return pd.read_sql_query(f"""
        SELECT substr(exit_date, 1, 10) as date,
               COUNT(*) as trades,
               SUM(pnl) as pnl,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins
        FROM trades WHERE {where}
        GROUP BY substr(exit_date, 1, 10)
        ORDER BY date ASC
    """, conn, params=params)


def monthly_pnl(start_date: str | None = None,
                end_date: str | None = None) -> pd.DataFrame:
    """P&L mensuel agrégé."""
    conn = _get_conn()
    conditions = ["exit_date IS NOT NULL"]
    params = []
    if start_date:
        conditions.append("exit_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("exit_date <= ?")
        params.append(end_date + " 23:59:59")
    where = " AND ".join(conditions)

    return pd.read_sql_query(f"""
        SELECT substr(exit_date, 1, 7) as month,
               COUNT(*) as trades,
               SUM(pnl) as pnl,
               ROUND(AVG(pnl), 2) as avg_pnl,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               ROUND(SUM(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) / COUNT(*), 4) as win_rate
        FROM trades WHERE {where}
        GROUP BY substr(exit_date, 1, 7)
        ORDER BY month ASC
    """, conn, params=params)


def streak_analysis() -> dict:
    """Analyse des séquences de gains/pertes actuelles."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT pnl FROM trades WHERE exit_date IS NOT NULL ORDER BY exit_date ASC"
    ).fetchall()

    if not rows:
        return {"current_streak": 0, "streak_type": "none"}

    current = 0
    streak_type = "none"
    for r in rows:
        p = r["pnl"] or 0
        if p > 0:
            if streak_type == "win":
                current += 1
            else:
                current = 1
                streak_type = "win"
        else:
            if streak_type == "loss":
                current += 1
            else:
                current = 1
                streak_type = "loss"

    return {"current_streak": current, "streak_type": streak_type}


def print_journal_report(start_date: str | None = None,
                         end_date: str | None = None):
    """Affiche un rapport complet du journal."""
    s = stats(start_date, end_date)

    if s["total_trades"] == 0:
        print("  Journal vide — aucun trade enregistré.")
        return

    print("=" * 60)
    print("  ORION JOURNAL — RAPPORT")
    print("=" * 60)
    print(f"  Trades:           {s['total_trades']}")
    print(f"  Win Rate:         {s['win_rate']:.1%} ({s['winning_trades']}W / {s['losing_trades']}L)")
    print(f"  P&L Total:        ${s['total_pnl']:+,.2f}")
    print(f"  Avg Trade:        ${s['avg_pnl']:+,.2f}")
    print(f"  Best / Worst:     ${s['best_trade']:+,.2f} / ${s['worst_trade']:+,.2f}")
    print(f"  Profit Factor:    {s['profit_factor']:.2f}")
    print(f"  Avg Holding:      {s['avg_holding_days']:.1f} jours")
    print(f"  Max Win Streak:   {s['max_consecutive_wins']}")
    print(f"  Max Loss Streak:  {s['max_consecutive_losses']}")

    streak = streak_analysis()
    print(f"  Current Streak:   {streak['current_streak']} {streak['streak_type']}")

    if s["by_regime"]:
        print("\n  --- Par régime ---")
        for reg, d in s["by_regime"].items():
            wr = d['wins'] / d['trades'] if d['trades'] else 0
            print(f"    {reg:<12} {d['trades']:>3} trades  "
                  f"WR={wr:.0%}  PnL=${d['pnl']:+,.2f}")

    if s["by_class"]:
        print("\n  --- Par classe ---")
        for cls, d in sorted(s["by_class"].items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = d['wins'] / d['trades'] if d['trades'] else 0
            print(f"    {cls:<12} {d['trades']:>3} trades  "
                  f"WR={wr:.0%}  PnL=${d['pnl']:+,.2f}")

    if s["by_exit"]:
        print("\n  --- Par sortie ---")
        for reason, d in sorted(s["by_exit"].items(), key=lambda x: x[1]["count"], reverse=True):
            print(f"    {reason:<18} {d['count']:>3}x  PnL=${d['pnl']:+,.2f}")

    mp = monthly_pnl(start_date, end_date)
    if not mp.empty:
        print("\n  --- P&L mensuel ---")
        for _, row in mp.iterrows():
            bar_len = min(30, max(1, int(abs(row['pnl']) / 100)))
            bar = "+" * bar_len if row['pnl'] >= 0 else "-" * bar_len
            print(f"    {row['month']}  {row['trades']:>3} trades  "
                  f"WR={row['win_rate']:.0%}  ${row['pnl']:>+10,.2f}  {bar}")

    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
#  EXÉCUTION DIRECTE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    init_journal()

    print("=" * 60)
    print("  Orion Trading Journal")
    print("=" * 60)

    print_journal_report()
