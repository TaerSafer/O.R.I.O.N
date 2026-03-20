"""
Orion Data Collector
--------------------
Collecte et stocke les données de marché pour 35 actifs via yfinance.
- Historique 10 ans au premier lancement
- Mise à jour automatique toutes les 15 minutes
- Stockage SQLite dans data/orion.db
"""

import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import pandas as pd

# ─── Univers de trading ─────────────────────────────────────────────

ASSETS = {
    # Forex
    "EURUSD=X":  "forex",
    "GBPUSD=X":  "forex",
    "USDJPY=X":  "forex",
    "USDCHF=X":  "forex",
    "AUDUSD=X":  "forex",
    "USDCAD=X":  "forex",
    "NZDUSD=X":  "forex",
    "EURGBP=X":  "forex",
    "EURJPY=X":  "forex",
    "GBPJPY=X":  "forex",
    # Matières premières
    "GC=F":      "commodity",   # Gold
    "SI=F":      "commodity",   # Silver
    "CL=F":      "commodity",   # Oil WTI
    "BZ=F":      "commodity",   # Oil Brent
    "HG=F":      "commodity",   # Cuivre
    "NG=F":      "commodity",   # Gaz naturel
    # Indices
    "^GSPC":     "index",       # S&P 500
    "^IXIC":     "index",       # NASDAQ
    "^GDAXI":    "index",       # DAX
    "^FCHI":     "index",       # CAC 40
    "^N225":     "index",       # Nikkei
    "^FTSE":     "index",       # FTSE 100
    # Actions
    "AAPL":      "stock",
    "MSFT":      "stock",
    "GOOGL":     "stock",
    "AMZN":      "stock",
    "TSLA":      "stock",
    "NVDA":      "stock",
    # Crypto
    "BTC-USD":   "crypto",
    "ETH-USD":   "crypto",
    "SOL-USD":   "crypto",
}

DB_PATH = Path(__file__).resolve().parent / "orion.db"
UPDATE_INTERVAL = 15 * 60  # 15 minutes en secondes
HISTORY_YEARS = 10

# ─── Base de données ─────────────────────────────────────────────────

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Une connexion SQLite par thread."""
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH))
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn


def _init_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            symbol    TEXT    NOT NULL,
            date      TEXT    NOT NULL,
            open      REAL,
            high      REAL,
            low       REAL,
            close     REAL,
            volume    REAL,
            asset_class TEXT,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_prices_symbol_date
        ON prices (symbol, date DESC)
    """)
    conn.commit()


# ─── Téléchargement ──────────────────────────────────────────────────

def _store_dataframe(symbol: str, df: pd.DataFrame, asset_class: str):
    """Insère un DataFrame yfinance dans SQLite (upsert)."""
    if df.empty:
        return 0

    conn = _get_conn()
    rows = []
    for date, row in df.iterrows():
        date_str = date.strftime("%Y-%m-%d %H:%M:%S") if hasattr(date, "strftime") else str(date)
        rows.append((
            symbol, date_str,
            float(row["Open"])   if pd.notna(row["Open"])   else None,
            float(row["High"])   if pd.notna(row["High"])   else None,
            float(row["Low"])    if pd.notna(row["Low"])    else None,
            float(row["Close"])  if pd.notna(row["Close"])  else None,
            float(row["Volume"]) if pd.notna(row.get("Volume", float("nan"))) else None,
            asset_class,
        ))

    conn.executemany("""
        INSERT INTO prices (symbol, date, open, high, low, close, volume, asset_class)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, date) DO UPDATE SET
            open=excluded.open, high=excluded.high,
            low=excluded.low,   close=excluded.close,
            volume=excluded.volume
    """, rows)
    conn.commit()
    return len(rows)


def fetch_history(symbol: str, asset_class: str, period: str = "10y") -> int:
    """Télécharge l'historique complet d'un symbole et le stocke."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, auto_adjust=True)
        if df.empty:
            print(f"  [!] {symbol}: aucune donnée retournée")
            return 0
        count = _store_dataframe(symbol, df, asset_class)
        print(f"  [+] {symbol}: {count} lignes stockées")
        return count
    except Exception as e:
        print(f"  [!] {symbol}: erreur — {e}")
        return 0


def fetch_latest(symbol: str, asset_class: str) -> int:
    """Télécharge les 5 derniers jours pour mise à jour incrémentale."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5d", auto_adjust=True)
        return _store_dataframe(symbol, df, asset_class)
    except Exception as e:
        print(f"  [!] {symbol}: mise à jour échouée — {e}")
        return 0


# ─── API publique ────────────────────────────────────────────────────

def get_price(symbol: str) -> dict | None:
    """Retourne le dernier prix connu pour un symbole.

    >>> get_price("AAPL")
    {'symbol': 'AAPL', 'date': '2026-03-18 00:00:00', 'open': ..., 'close': ..., ...}
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT symbol, date, open, high, low, close, volume, asset_class "
        "FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    if row is None:
        return None
    return dict(zip(
        ["symbol", "date", "open", "high", "low", "close", "volume", "asset_class"],
        row,
    ))


def get_history(symbol: str, days: int = 30) -> pd.DataFrame:
    """Retourne l'historique des N derniers jours sous forme de DataFrame.

    >>> df = get_history("AAPL", days=60)
    >>> df.columns.tolist()
    ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'asset_class']
    """
    conn = _get_conn()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        "SELECT symbol, date, open, high, low, close, volume, asset_class "
        "FROM prices WHERE symbol = ? AND date >= ? ORDER BY date ASC",
        conn,
        params=(symbol, cutoff),
    )
    return df


# ─── Boucle de mise à jour ───────────────────────────────────────────

_scheduler_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _update_loop():
    """Boucle de mise à jour toutes les 15 minutes."""
    _init_db()
    while not _stop_event.is_set():
        print(f"\n[Orion] Mise à jour des prix — {datetime.now():%Y-%m-%d %H:%M:%S}")
        for symbol, asset_class in ASSETS.items():
            if _stop_event.is_set():
                break
            fetch_latest(symbol, asset_class)
        _stop_event.wait(timeout=UPDATE_INTERVAL)


def start_scheduler():
    """Démarre la boucle de mise à jour en arrière-plan."""
    global _scheduler_thread
    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        print("[Orion] Le scheduler tourne déjà.")
        return
    _stop_event.clear()
    _scheduler_thread = threading.Thread(target=_update_loop, daemon=True, name="orion-collector")
    _scheduler_thread.start()
    print("[Orion] Scheduler démarré (intervalle : 15 min)")


def stop_scheduler():
    """Arrête proprement la boucle de mise à jour."""
    _stop_event.set()
    if _scheduler_thread is not None:
        _scheduler_thread.join(timeout=10)
    print("[Orion] Scheduler arrêté.")


# ─── Initialisation complète ──────────────────────────────────────────

def init(full_history: bool = True):
    """Initialise la base et charge l'historique complet si demandé.

    Args:
        full_history: Si True, télécharge 10 ans d'historique pour chaque actif.
                      Si False, ne télécharge que les 5 derniers jours.
    """
    _init_db()

    conn = _get_conn()
    existing = {row[0] for row in conn.execute("SELECT DISTINCT symbol FROM prices").fetchall()}
    missing = [s for s in ASSETS if s not in existing]

    if full_history and missing:
        print(f"[Orion] Chargement de l'historique {HISTORY_YEARS} ans pour {len(missing)} actifs...")
        for symbol in missing:
            fetch_history(symbol, ASSETS[symbol], period=f"{HISTORY_YEARS}y")
    elif not full_history:
        print(f"[Orion] Mise à jour rapide pour {len(ASSETS)} actifs...")
        for symbol, asset_class in ASSETS.items():
            fetch_latest(symbol, asset_class)

    already = len(existing)
    if already:
        print(f"[Orion] {already} actifs déjà en base — mise à jour incrémentale.")
        for symbol in existing:
            if symbol in ASSETS:
                fetch_latest(symbol, ASSETS[symbol])

    total = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    symbols_count = conn.execute("SELECT COUNT(DISTINCT symbol) FROM prices").fetchone()[0]
    print(f"[Orion] Base prête : {symbols_count} actifs, {total:,} lignes au total.")


# ─── Exécution directe ───────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Orion Data Collector")
    print("=" * 60)
    init(full_history=True)
    print()
    start_scheduler()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Orion] Arrêt demandé...")
        stop_scheduler()
