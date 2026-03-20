#!/usr/bin/env python3
"""
 ╔═══════════════════════════════════════════════════════════════╗
 ║                                                               ║
 ║    O.R.I.O.N.                                                 ║
 ║    Omniscient Risk-Integrated Orchestration Network            ║
 ║                                                               ║
 ║    Multi-Asset Trading Algorithm — Aladdin Philosophy          ║
 ║                                                               ║
 ╚═══════════════════════════════════════════════════════════════╝

 Point d'entrée unique du système.

 Usage:
     python orion.py                 Lancement complet (init + scheduler + dashboard)
     python orion.py --fast          Skip le téléchargement 10 ans, mise à jour rapide
     python orion.py --no-dashboard  Lance sans le dashboard web
     python orion.py --no-collect    Lance sans collecte de données (DB existante)
     python orion.py --port 9000     Dashboard sur un port différent
     python orion.py --status        Affiche l'état du système et quitte
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

BANNER = r"""
   ___  ____  ___ ___  _  __
  / _ \| __ \|_ _/ _ \| \| |
 | (_) |  _ \ | | (_) | .` |
  \___/|_| \_\___\___/|_|\_|

  Omniscient Risk-Integrated Orchestration Network
  Multi-Asset Trading Algorithm
"""

MODULES = {
    "data":      "Collecte de données de marché (35 actifs)",
    "signals":   "Moteur de signaux techniques (18 indicateurs)",
    "risk":      "Risk Manager Aladdin (régimes, drawdown, diversification)",
    "execution": "Broker IBKR paper trading",
    "dashboard": "Interface web FastAPI (temps réel)",
    "journal":   "Journal de trading SQLite",
    "backtest":  "Moteur de backtesting",
}


# ═══════════════════════════════════════════════════════════════════════
#  ÉTAPES DE DÉMARRAGE
# ═══════════════════════════════════════════════════════════════════════

def _step(label: str, fn, *args, **kwargs):
    """Exécute une étape avec affichage formaté."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}]  {label}", end="", flush=True)
    try:
        result = fn(*args, **kwargs)
        print(f"  \033[32mOK\033[0m")
        return result
    except Exception as e:
        print(f"  \033[31mERREUR\033[0m")
        print(f"           {e}")
        return None


def check_modules():
    """Vérifie que tous les modules sont présents."""
    print("\n  --- Vérification des modules ---")
    all_ok = True
    for name, desc in MODULES.items():
        path = BASE_DIR / name
        exists = path.exists() and (path / "__init__.py").exists()
        status = "\033[32mOK\033[0m" if exists else "\033[31mMANQUANT\033[0m"
        print(f"    {name:<12} {status}  {desc}")
        if not exists:
            all_ok = False
    return all_ok


def init_database():
    """Initialise la base SQLite (tables prices + journal)."""
    from data.collector import _init_db
    _init_db()

    from journal.logger import init_journal
    init_journal()


def load_data(full_history: bool = True):
    """Charge les données de marché."""
    from data.collector import init as collector_init
    collector_init(full_history=full_history)


def detect_regime_status():
    """Détecte et affiche le régime de marché courant."""
    try:
        from risk.manager import detect_regime
        reg = detect_regime()
        colors = {"EXPANSION": "\033[32m", "CONTRACTION": "\033[33m", "STRESS": "\033[31m"}
        c = colors.get(reg.regime.value, "")
        print(f"\n  Régime de marché : {c}{reg.regime.value}\033[0m")
        print(f"  VIX simulé :       {reg.vix_simulated:.1%}")
        print(f"  Corrélation avg :  {reg.avg_correlation:.2f}")
        return reg
    except Exception as e:
        print(f"\n  Régime : impossible à déterminer ({e})")
        return None


def start_data_scheduler():
    """Démarre le scheduler de collecte toutes les 15 minutes."""
    from data.collector import start_scheduler
    start_scheduler()


def _start_intelligence():
    """Démarre les modules d'intelligence : briefing matin + mémoire quotidienne."""
    try:
        from signals.morning_brief import schedule_daily_briefing
        schedule_daily_briefing()
    except Exception:
        pass
    try:
        from signals.memory import schedule_daily_memory
        schedule_daily_memory()
    except Exception:
        pass


def connect_ibkr() -> bool:
    """Teste la connexion IBKR et connecte le broker si TWS est disponible."""
    from execution.broker import (
        test_connection, OrionBroker,
        IBKR_HOST, IBKR_PORT, IBKR_ACCOUNT, IBKR_USERNAME,
    )

    print(f"\n  --- Connexion IBKR ---")
    print(f"    Compte :   {IBKR_ACCOUNT} ({IBKR_USERNAME})")
    print(f"    Adresse :  {IBKR_HOST}:{IBKR_PORT} (paper trading)")

    if not test_connection(IBKR_HOST, IBKR_PORT, timeout=3.0):
        print(f"    Status :   \033[33mTWS non détecté\033[0m")
        print(f"    → O.R.I.O.N. continue sans connexion IBKR.")
        print(f"    → Lancez TWS/Gateway puis redémarrez pour activer le trading.")
        return False

    try:
        broker = OrionBroker()
        if broker.connect():
            print(f"    Status :   \033[32mConnecté\033[0m")
            broker.start_monitor()
            print(f"    Monitor :  Aladdin actif")
            return True
        else:
            print(f"    Status :   \033[31mÉchec de connexion\033[0m")
            return False
    except Exception as e:
        print(f"    Status :   \033[31mErreur: {e}\033[0m")
        return False


def start_dashboard(host: str = "0.0.0.0", port: int = 8000):
    """Démarre le dashboard FastAPI dans un thread séparé."""
    import uvicorn
    from dashboard.app import app

    thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": host, "port": port, "log_level": "warning"},
        daemon=True,
        name="orion-dashboard",
    )
    thread.start()
    return thread


def show_status():
    """Affiche l'état complet du système."""
    print(BANNER)
    check_modules()

    print("\n  --- Base de données ---")
    db_path = BASE_DIR / "data" / "orion.db"
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"    Fichier :  {db_path}")
        print(f"    Taille :   {size_mb:.1f} MB")

        try:
            init_database()
            from data.collector import _get_conn, ASSETS
            conn = _get_conn()
            total = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
            symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM prices").fetchone()[0]
            latest = conn.execute("SELECT MAX(date) FROM prices").fetchone()[0]
            print(f"    Actifs :   {symbols}/{len(ASSETS)}")
            print(f"    Lignes :   {total:,}")
            print(f"    Dernière : {latest}")
        except Exception as e:
            print(f"    Erreur :   {e}")
    else:
        print("    Base non initialisée.")

    # Journal
    print("\n  --- Journal ---")
    try:
        from journal.logger import stats as journal_stats
        s = journal_stats()
        if s["total_trades"] > 0:
            print(f"    Trades :       {s['total_trades']}")
            print(f"    Win Rate :     {s['win_rate']:.1%}")
            print(f"    P&L Total :    ${s['total_pnl']:+,.2f}")
            print(f"    Profit Factor: {s['profit_factor']:.2f}")
        else:
            print("    Aucun trade enregistré.")
    except Exception:
        print("    Non disponible.")

    # IBKR
    print("\n  --- IBKR ---")
    try:
        from execution.broker import (
            test_connection, get_ibkr_status,
            IBKR_HOST, IBKR_PORT, IBKR_ACCOUNT, IBKR_USERNAME,
        )
        print(f"    Compte :   {IBKR_ACCOUNT} ({IBKR_USERNAME})")
        print(f"    Port :     {IBKR_PORT} (paper trading)")
        ibkr_st = get_ibkr_status()
        if ibkr_st["connected"]:
            print(f"    Status :   \033[32mConnecté\033[0m")
        else:
            tws_up = test_connection(IBKR_HOST, IBKR_PORT, timeout=2.0)
            if tws_up:
                print(f"    TWS :      Détecté (non connecté)")
            else:
                print(f"    TWS :      \033[33mNon détecté\033[0m")
            if ibkr_st["error"]:
                print(f"    Erreur :   {ibkr_st['error']}")
    except Exception:
        print("    Non disponible.")

    # Régime
    detect_regime_status()

    # Portfolio
    print("\n  --- Portefeuille ---")
    try:
        from risk.manager import get_portfolio, DEFAULT_CONFIG
        pf = get_portfolio()
        print(f"    Capital :    ${pf.current_capital:,.2f}")
        print(f"    Positions :  {len(pf.positions)}")
        print(f"    Drawdown :   {pf.drawdown_pct:.1%}")
        if pf.is_in_cooldown:
            print(f"    Cooldown :   ACTIF ({pf.cooldown_remaining})")
    except Exception:
        print("    Non disponible.")

    print()


# ═══════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="O.R.I.O.N. — Omniscient Risk-Integrated Orchestration Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fast", action="store_true",
                        help="Mise à jour rapide (5 jours) au lieu de 10 ans d'historique")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Ne pas lancer le dashboard web")
    parser.add_argument("--no-collect", action="store_true",
                        help="Ne pas collecter de données (utiliser la base existante)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port du dashboard (défaut: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host du dashboard (défaut: 0.0.0.0)")
    parser.add_argument("--status", action="store_true",
                        help="Afficher l'état du système et quitter")
    args = parser.parse_args()

    # ── Mode status ──
    if args.status:
        show_status()
        return

    # ── Démarrage complet ──
    print(BANNER)
    print(f"  Démarrage — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'=' * 50}")

    # 1. Modules
    if not check_modules():
        print("\n  ERREUR : modules manquants. Vérifiez l'installation.")
        sys.exit(1)

    # 2. Base de données
    _step("Initialisation de la base de données", init_database)

    # 3. Données de marché
    if not args.no_collect:
        full = not args.fast
        mode = "10 ans d'historique" if full else "mise à jour rapide (5j)"
        print(f"\n  --- Collecte de données ({mode}) ---")
        print(f"  Cela peut prendre plusieurs minutes au premier lancement...\n")
        load_data(full_history=full)
    else:
        print("\n  Collecte de données : SKIP (--no-collect)")

    # 4. Régime de marché
    detect_regime_status()

    # 5. Scheduler
    print()
    _step("Démarrage du scheduler de données (15 min)", start_data_scheduler)

    # 6. Intelligence (briefing matin 8h + mémoire 22h)
    _step("Démarrage modules d'intelligence", _start_intelligence)

    # 8. Connexion IBKR
    ibkr_ok = connect_ibkr()

    # 9. Dashboard
    if not args.no_dashboard:
        _step(f"\n  Démarrage du dashboard (http://{args.host}:{args.port})",
              start_dashboard, args.host, args.port)
    else:
        print("\n  Dashboard : SKIP (--no-dashboard)")

    # ── Résumé ──
    print(f"\n  {'=' * 50}")
    print(f"  O.R.I.O.N. est opérationnel.")
    print()
    if not args.no_dashboard:
        print(f"    Dashboard :  http://localhost:{args.port}")
    print(f"    Scheduler :  toutes les 15 minutes")
    ibkr_label = "\033[32mconnecté\033[0m" if ibkr_ok else "\033[33mnon connecté\033[0m"
    print(f"    IBKR :       {ibkr_label}")
    print(f"    Base :       {BASE_DIR / 'data' / 'orion.db'}")
    print(f"\n  Ctrl+C pour arrêter le système.")
    print(f"  {'=' * 50}\n")

    # ── Boucle principale ──
    shutdown = threading.Event()

    def _handle_signal(sig, frame):
        print(f"\n\n  Arrêt d'O.R.I.O.N. demandé...")
        shutdown.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not shutdown.is_set():
            shutdown.wait(timeout=1)
    except KeyboardInterrupt:
        pass

    # ── Arrêt propre ──
    print("\n  --- Arrêt du système ---")

    try:
        from data.collector import stop_scheduler
        stop_scheduler()
    except Exception:
        pass

    print("  O.R.I.O.N. arrêté.\n")


if __name__ == "__main__":
    main()
