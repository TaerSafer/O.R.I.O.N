"""
Orion — Execution Broker (IBKR Paper Trading)
------------------------------------------------
Connexion Interactive Brokers via ib_insync pour paper trading.
Toutes les protections Aladdin du risk manager sont intégrées :

- Vérification du régime avant chaque ordre
- Drawdown progressif avec réduction automatique
- Capital protection (pause après pertes consécutives)
- Trailing stop ATR-based
- Sync bidirectionnelle portefeuille IBKR ↔ Orion
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

# ib_insync requiert une event loop active à l'import
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import (
    IB, Stock, Forex, Future, Crypto, Contract,
    MarketOrder, LimitOrder, StopOrder,
    Trade, Order, Fill,
)

from data.collector import ASSETS
from signals.engine import Signal, Action
from risk.manager import (
    Regime, RegimeState, DrawdownAction,
    detect_regime, effective_config, evaluate_drawdown,
    positions_to_reduce, stress_reduce_positions,
    check_risk, update_trailing_stop, portfolio_report,
    get_portfolio, Portfolio, Position,
    RiskConfig, RiskCheck, PositionSize, DEFAULT_CONFIG,
)

logger = logging.getLogger("orion.execution")

# ─── Configuration IBKR ─────────────────────────────────────────────

IBKR_USERNAME = "bebitson13"
IBKR_ACCOUNT = "DUP485293"
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497           # Paper trading
IBKR_CLIENT_ID = 1

# État global de la connexion IBKR (consulté par le dashboard)
_ibkr_status = {
    "connected": False,
    "account": IBKR_ACCOUNT,
    "username": IBKR_USERNAME,
    "error": "",
}


def get_ibkr_status() -> dict:
    """Retourne l'état de la connexion IBKR."""
    return _ibkr_status.copy()


def test_connection(host: str = IBKR_HOST, port: int = IBKR_PORT,
                    timeout: float = 3.0) -> bool:
    """Teste si TWS/Gateway est accessible sur le port donné.

    Effectue une connexion TCP brute — ne nécessite pas ib_insync.
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        sock.close()
        logger.info(f"TWS détecté sur {host}:{port}")
        return True
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        logger.info(f"TWS non disponible sur {host}:{port} ({e})")
        return False
    finally:
        sock.close()


# ─── Mapping symboles Orion → contrats IBKR ─────────────────────────

# Forex : "EURUSD=X" → Forex("EUR", "USD")
# Futures : "GC=F" → Future("GC", exchange="COMEX")
# Indices : "^GSPC" → Future("ES", exchange="CME") (on trade le future)
# Stocks : "AAPL" → Stock("AAPL", "SMART", "USD")
# Crypto : "BTC-USD" → Crypto("BTC", "PAXOS", "USD")

_FOREX_MAP = {
    "EURUSD=X": ("EUR", "USD"), "GBPUSD=X": ("GBP", "USD"),
    "USDJPY=X": ("USD", "JPY"), "USDCHF=X": ("USD", "CHF"),
    "AUDUSD=X": ("AUD", "USD"), "USDCAD=X": ("USD", "CAD"),
    "NZDUSD=X": ("NZD", "USD"), "EURGBP=X": ("EUR", "GBP"),
    "EURJPY=X": ("EUR", "JPY"), "GBPJPY=X": ("GBP", "JPY"),
}

_FUTURE_MAP = {
    "GC=F": ("GC", "COMEX"),   "SI=F": ("SI", "COMEX"),
    "CL=F": ("CL", "NYMEX"),   "BZ=F": ("BZ", "NYMEX"),
    "HG=F": ("HG", "COMEX"),   "NG=F": ("NG", "NYMEX"),
}

_INDEX_FUTURE_MAP = {
    "^GSPC": ("ES", "CME"),    "^IXIC": ("NQ", "CME"),
    "^GDAXI": ("FDAX", "EUREX"), "^FCHI": ("CAC40", "MONEP"),
    "^N225": ("NIY", "CME"),   "^FTSE": ("Z", "ICEEU"),
}

_CRYPTO_MAP = {
    "BTC-USD": ("BTC", "PAXOS"), "ETH-USD": ("ETH", "PAXOS"),
    "SOL-USD": ("SOL", "PAXOS"),
}


def symbol_to_contract(symbol: str) -> Contract | None:
    """Convertit un symbole Orion en contrat IBKR."""
    if symbol in _FOREX_MAP:
        base, quote = _FOREX_MAP[symbol]
        pair = base + quote
        c = Forex(pair)
        c.exchange = "IDEALPRO"
        return c

    if symbol in _FUTURE_MAP:
        sym, exchange = _FUTURE_MAP[symbol]
        return Future(sym, exchange=exchange)

    if symbol in _INDEX_FUTURE_MAP:
        sym, exchange = _INDEX_FUTURE_MAP[symbol]
        return Future(sym, exchange=exchange)

    if symbol in _CRYPTO_MAP:
        sym, exchange = _CRYPTO_MAP[symbol]
        return Crypto(sym, exchange, "USD")

    asset_class = ASSETS.get(symbol)
    if asset_class == "stock":
        return Stock(symbol, "SMART", "USD")

    return None


# ═══════════════════════════════════════════════════════════════════════
#  TYPES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OrderResult:
    success: bool
    symbol: str
    action: str
    order_id: int | None = None
    fill_price: float | None = None
    fill_size: float | None = None
    status: str = ""
    error: str = ""
    risk_check: RiskCheck | None = None
    regime: str = ""
    timestamp: str = ""

    def __repr__(self):
        if self.success:
            return (f"[FILLED] {self.symbol} {self.action} "
                    f"size={self.fill_size} @ {self.fill_price} "
                    f"[{self.regime}] id={self.order_id}")
        return f"[FAILED] {self.symbol}: {self.error}"


# ═══════════════════════════════════════════════════════════════════════
#  BROKER
# ═══════════════════════════════════════════════════════════════════════

class OrionBroker:
    """Broker IBKR avec protections Aladdin intégrées."""

    def __init__(self, host: str = IBKR_HOST, port: int = IBKR_PORT,
                 client_id: int = IBKR_CLIENT_ID,
                 account: str = IBKR_ACCOUNT,
                 config: RiskConfig | None = None):
        """
        Args:
            host: Adresse TWS/Gateway.
            port: 7497 = paper trading, 7496 = live (bloqué par défaut).
            client_id: ID client unique pour cette connexion.
            account: Identifiant du compte IBKR.
            config: Configuration de risque.
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account = account
        self.config = config or DEFAULT_CONFIG
        self.ib = IB()
        self._connected = False
        self._regime: RegimeState | None = None
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # SÉCURITÉ : blocage du port live
        if port == 7496:
            raise ValueError(
                "Port 7496 (LIVE) bloqué par sécurité. "
                "Utilisez 7497 (paper trading) ou déverrouillez explicitement "
                "avec allow_live=True dans connect().")

    # ─── Connexion ───────────────────────────────────────────────────

    def connect(self, allow_live: bool = False) -> bool:
        """Connecte au TWS/Gateway IBKR."""
        global _ibkr_status

        if self.port == 7496 and not allow_live:
            logger.error("LIVE TRADING BLOQUÉ — utilisez allow_live=True")
            _ibkr_status["error"] = "Port live bloqué"
            return False

        # Vérifier d'abord que TWS est joignable
        if not test_connection(self.host, self.port):
            _ibkr_status["connected"] = False
            _ibkr_status["error"] = f"TWS non disponible sur {self.host}:{self.port}"
            return False

        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id,
                            account=self.account)
            self._connected = True
            acct = "Paper" if self.port == 7497 else "LIVE"
            logger.info(f"Connecté à IBKR ({acct}) @ {self.host}:{self.port} "
                        f"compte={self.account}")

            _ibkr_status["connected"] = True
            _ibkr_status["error"] = ""

            # Sync du capital
            account_values = self.ib.accountSummary(self.account)
            for av in account_values:
                if av.tag == "NetLiquidation" and av.currency == "USD":
                    capital = float(av.value)
                    self.config.capital = capital
                    pf = get_portfolio()
                    pf.current_capital = capital
                    pf.peak_capital = max(pf.peak_capital, capital)
                    logger.info(f"Capital synchronisé: ${capital:,.2f}")
                    break

            return True
        except Exception as e:
            logger.error(f"Connexion échouée: {e}")
            self._connected = False
            _ibkr_status["connected"] = False
            _ibkr_status["error"] = str(e)
            return False

    def disconnect(self):
        """Déconnecte proprement."""
        global _ibkr_status
        self.stop_monitor()
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            _ibkr_status["connected"] = False
            logger.info("Déconnecté d'IBKR")

    @property
    def is_connected(self) -> bool:
        return self._connected and self.ib.isConnected()

    # ─── Régime ──────────────────────────────────────────────────────

    def refresh_regime(self) -> RegimeState:
        """Recalcule et retourne le régime de marché."""
        self._regime = detect_regime()
        logger.info(f"Régime: {self._regime}")
        return self._regime

    @property
    def regime(self) -> RegimeState:
        if self._regime is None:
            self.refresh_regime()
        return self._regime

    # ─── Exécution d'ordres ──────────────────────────────────────────

    def execute_signal(self, signal: Signal,
                       method: str = "fixed",
                       order_type: str = "market") -> OrderResult:
        """Exécute un signal après validation Aladdin complète.

        Args:
            signal: Signal du moteur de signaux.
            method: "fixed", "kelly", "volatility".
            order_type: "market" ou "limit".

        Returns:
            OrderResult avec le résultat de l'exécution.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        reg = self.regime

        # ── Risk check complet ──
        rc = check_risk(signal, method=method, config=self.config, regime=reg)

        if not rc.approved:
            logger.warning(f"REJETÉ: {signal.symbol} — {rc.rejections}")
            return OrderResult(
                success=False, symbol=signal.symbol, action=signal.action.value,
                error="; ".join(rc.rejections), risk_check=rc,
                regime=reg.regime.value, timestamp=timestamp)

        if not self.is_connected:
            return OrderResult(
                success=False, symbol=signal.symbol, action=signal.action.value,
                error="Non connecté à IBKR", risk_check=rc,
                regime=reg.regime.value, timestamp=timestamp)

        pos = rc.position
        contract = symbol_to_contract(signal.symbol)
        if contract is None:
            return OrderResult(
                success=False, symbol=signal.symbol, action=signal.action.value,
                error=f"Impossible de mapper {signal.symbol} vers un contrat IBKR",
                risk_check=rc, regime=reg.regime.value, timestamp=timestamp)

        # ── Qualifier le contrat ──
        try:
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                return OrderResult(
                    success=False, symbol=signal.symbol, action=signal.action.value,
                    error=f"Contrat non qualifié: {contract}",
                    risk_check=rc, regime=reg.regime.value, timestamp=timestamp)
            contract = qualified[0]
        except Exception as e:
            return OrderResult(
                success=False, symbol=signal.symbol, action=signal.action.value,
                error=f"Qualification échouée: {e}",
                risk_check=rc, regime=reg.regime.value, timestamp=timestamp)

        # ── Créer l'ordre ──
        ib_action = "BUY" if pos.action == "BUY" else "SELL"
        size = abs(pos.size)

        if order_type == "limit":
            order = LimitOrder(ib_action, size, pos.entry_price)
        else:
            order = MarketOrder(ib_action, size)

        # ── Soumettre ──
        try:
            trade = self.ib.placeOrder(contract, order)
            # Attendre le fill (timeout 30s)
            timeout = 30
            while not trade.isDone() and timeout > 0:
                self.ib.sleep(1)
                timeout -= 1

            if trade.orderStatus.status == "Filled":
                fill_price = trade.orderStatus.avgFillPrice
                fill_size = trade.orderStatus.filled

                # Enregistrer la position dans le portefeuille Orion
                pf = get_portfolio()
                pf.add_position(Position(
                    symbol=signal.symbol,
                    asset_class=ASSETS.get(signal.symbol, "unknown"),
                    action=pos.action,
                    entry_price=fill_price,
                    size=fill_size,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                    timestamp=timestamp,
                    risk_amount=pos.risk_amount,
                ))

                # Placer le stop-loss chez IBKR
                sl_action = "SELL" if pos.action == "BUY" else "BUY"
                sl_order = StopOrder(sl_action, fill_size, pos.stop_loss)
                self.ib.placeOrder(contract, sl_order)

                logger.info(f"FILLED: {signal.symbol} {ib_action} "
                            f"{fill_size} @ {fill_price} SL={pos.stop_loss}")

                return OrderResult(
                    success=True, symbol=signal.symbol, action=pos.action,
                    order_id=trade.order.orderId,
                    fill_price=fill_price, fill_size=fill_size,
                    status="Filled", risk_check=rc,
                    regime=reg.regime.value, timestamp=timestamp)
            else:
                status = trade.orderStatus.status
                logger.warning(f"Ordre non rempli: {signal.symbol} status={status}")
                return OrderResult(
                    success=False, symbol=signal.symbol, action=pos.action,
                    order_id=trade.order.orderId,
                    status=status, error=f"Statut: {status}",
                    risk_check=rc, regime=reg.regime.value, timestamp=timestamp)

        except Exception as e:
            logger.error(f"Erreur d'exécution: {e}")
            return OrderResult(
                success=False, symbol=signal.symbol, action=signal.action.value,
                error=str(e), risk_check=rc,
                regime=reg.regime.value, timestamp=timestamp)

    # ─── Fermeture de position ───────────────────────────────────────

    def close_position(self, symbol: str, reason: str = "manual") -> OrderResult:
        """Ferme une position existante."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pf = get_portfolio()
        pos = pf.positions.get(symbol)

        if pos is None:
            return OrderResult(success=False, symbol=symbol, action="CLOSE",
                               error="Aucune position ouverte", timestamp=timestamp)

        contract = symbol_to_contract(symbol)
        if contract is None or not self.is_connected:
            return OrderResult(success=False, symbol=symbol, action="CLOSE",
                               error="Non connecté ou contrat inconnu", timestamp=timestamp)

        try:
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                return OrderResult(success=False, symbol=symbol, action="CLOSE",
                                   error="Contrat non qualifié", timestamp=timestamp)
            contract = qualified[0]
        except Exception as e:
            return OrderResult(success=False, symbol=symbol, action="CLOSE",
                               error=str(e), timestamp=timestamp)

        # Inverser la direction pour fermer
        close_action = "SELL" if pos.action == "BUY" else "BUY"
        order = MarketOrder(close_action, abs(pos.size))

        try:
            trade = self.ib.placeOrder(contract, order)
            timeout = 30
            while not trade.isDone() and timeout > 0:
                self.ib.sleep(1)
                timeout -= 1

            if trade.orderStatus.status == "Filled":
                fill_price = trade.orderStatus.avgFillPrice

                # Calculer P&L et enregistrer
                if pos.action == "BUY":
                    pnl = (fill_price - pos.entry_price) * pos.size
                else:
                    pnl = (pos.entry_price - fill_price) * abs(pos.size)

                is_win = pnl >= 0
                pf.record_trade_result(is_win)
                pf.remove_position(symbol)
                pf.update_capital(pf.current_capital + pnl)

                logger.info(f"CLOSED: {symbol} PnL=${pnl:+,.2f} ({reason})")
                return OrderResult(
                    success=True, symbol=symbol, action="CLOSE",
                    order_id=trade.order.orderId,
                    fill_price=fill_price, fill_size=abs(pos.size),
                    status=f"Closed ({reason}), PnL=${pnl:+,.2f}",
                    regime=self.regime.regime.value, timestamp=timestamp)
            else:
                return OrderResult(
                    success=False, symbol=symbol, action="CLOSE",
                    error=f"Statut: {trade.orderStatus.status}", timestamp=timestamp)

        except Exception as e:
            return OrderResult(success=False, symbol=symbol, action="CLOSE",
                               error=str(e), timestamp=timestamp)

    def reduce_position(self, symbol: str, factor: float,
                        reason: str = "drawdown") -> OrderResult:
        """Réduit une position d'un facteur (0.25 = réduire de 25%)."""
        pf = get_portfolio()
        pos = pf.positions.get(symbol)
        if pos is None:
            return OrderResult(success=False, symbol=symbol, action="REDUCE",
                               error="Aucune position", timestamp=datetime.now().isoformat())

        reduce_size = abs(pos.size) * factor
        contract = symbol_to_contract(symbol)
        if contract is None or not self.is_connected:
            return OrderResult(success=False, symbol=symbol, action="REDUCE",
                               error="Non connecté ou contrat inconnu",
                               timestamp=datetime.now().isoformat())

        try:
            qualified = self.ib.qualifyContracts(contract)
            if qualified:
                contract = qualified[0]
        except Exception:
            pass

        close_action = "SELL" if pos.action == "BUY" else "BUY"
        order = MarketOrder(close_action, reduce_size)

        try:
            trade = self.ib.placeOrder(contract, order)
            timeout = 30
            while not trade.isDone() and timeout > 0:
                self.ib.sleep(1)
                timeout -= 1

            if trade.orderStatus.status == "Filled":
                pos.size = pos.size * (1 - factor) if pos.action == "BUY" else pos.size * (1 - factor)
                pos.risk_amount *= (1 - factor)
                logger.info(f"REDUCED: {symbol} by {factor:.0%} ({reason})")
                return OrderResult(
                    success=True, symbol=symbol, action="REDUCE",
                    fill_price=trade.orderStatus.avgFillPrice,
                    fill_size=reduce_size,
                    status=f"Reduced {factor:.0%} ({reason})",
                    timestamp=datetime.now().isoformat())
            return OrderResult(success=False, symbol=symbol, action="REDUCE",
                               error=f"Statut: {trade.orderStatus.status}",
                               timestamp=datetime.now().isoformat())
        except Exception as e:
            return OrderResult(success=False, symbol=symbol, action="REDUCE",
                               error=str(e), timestamp=datetime.now().isoformat())

    # ─── Boucle de surveillance Aladdin ──────────────────────────────

    def _monitor_loop(self):
        """Boucle de surveillance continue avec protections Aladdin."""
        logger.info("Monitor Aladdin démarré")

        while not self._stop_event.is_set():
            if not self.is_connected:
                self._stop_event.wait(timeout=10)
                continue

            try:
                # 1. Rafraîchir le régime
                old_regime = self._regime.regime if self._regime else None
                self.refresh_regime()
                reg = self._regime

                # Transition vers STRESS → réduction automatique 50%
                if (reg.regime == Regime.STRESS
                        and old_regime != Regime.STRESS):
                    logger.warning("TRANSITION VERS STRESS — réduction 50%")
                    reductions = stress_reduce_positions()
                    for sym, factor in reductions:
                        self.reduce_position(sym, factor, reason="regime_stress")

                # 2. Évaluer le drawdown
                pf = get_portfolio()
                dd_action = evaluate_drawdown(self.config, pf)

                if dd_action == DrawdownAction.CLOSE_ALL:
                    logger.critical("DRAWDOWN CRITIQUE — FERMETURE TOTALE")
                    for sym in list(pf.positions.keys()):
                        self.close_position(sym, reason="drawdown_critical")
                    logger.critical("SYSTÈME ARRÊTÉ — drawdown max atteint")
                    self._stop_event.set()
                    break

                elif dd_action == DrawdownAction.REDUCE_50:
                    logger.warning("DRAWDOWN -8% — réduction 50%")
                    reductions = positions_to_reduce(dd_action, pf)
                    for sym, factor in reductions:
                        self.reduce_position(sym, factor, reason="drawdown_50")

                elif dd_action == DrawdownAction.REDUCE_25:
                    logger.warning("DRAWDOWN -5% — réduction 25%")
                    reductions = positions_to_reduce(dd_action, pf)
                    for sym, factor in reductions:
                        self.reduce_position(sym, factor, reason="drawdown_25")

                # 3. Trailing stops
                for sym, pos in list(pf.positions.items()):
                    price_data = None
                    contract = symbol_to_contract(sym)
                    if contract:
                        try:
                            qualified = self.ib.qualifyContracts(contract)
                            if qualified:
                                ticker = self.ib.reqMktData(qualified[0])
                                self.ib.sleep(2)
                                if ticker.last and ticker.last > 0:
                                    price_data = ticker.last
                                elif ticker.close and ticker.close > 0:
                                    price_data = ticker.close
                                self.ib.cancelMktData(qualified[0])
                        except Exception:
                            pass

                    if price_data:
                        result = update_trailing_stop(sym, price_data, self.config, pf)
                        if result and result["triggered"]:
                            logger.info(f"TRAILING STOP déclenché: {sym}")
                            self.close_position(sym, reason="trailing_stop")

                        # Vérifier take-profit
                        if pos.action == "BUY" and price_data >= pos.take_profit:
                            logger.info(f"TAKE PROFIT atteint: {sym}")
                            self.close_position(sym, reason="take_profit")
                        elif pos.action == "SELL" and price_data <= pos.take_profit:
                            logger.info(f"TAKE PROFIT atteint: {sym}")
                            self.close_position(sym, reason="take_profit")

                # 4. Sync capital
                try:
                    account_values = self.ib.accountSummary()
                    for av in account_values:
                        if av.tag == "NetLiquidation" and av.currency == "USD":
                            pf.update_capital(float(av.value))
                            break
                except Exception:
                    pass

                # 5. Log rapport
                report = portfolio_report(self.config, pf, reg)
                if report.warnings:
                    for w in report.warnings:
                        logger.warning(f"  {w}")

            except Exception as e:
                logger.error(f"Erreur monitor: {e}")

            # Intervalle de surveillance : 60 secondes
            self._stop_event.wait(timeout=60)

    def start_monitor(self):
        """Démarre la surveillance Aladdin en arrière-plan."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.info("Monitor déjà actif")
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="orion-aladdin-monitor")
        self._monitor_thread.start()

    def stop_monitor(self):
        """Arrête le monitor."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        logger.info("Monitor Aladdin arrêté")

    # ─── Sync portefeuille IBKR → Orion ──────────────────────────────

    def sync_positions(self):
        """Synchronise les positions IBKR vers le portefeuille Orion."""
        if not self.is_connected:
            return

        pf = get_portfolio()
        ib_positions = self.ib.positions()
        logger.info(f"Sync: {len(ib_positions)} positions IBKR")

        for ib_pos in ib_positions:
            contract = ib_pos.contract
            sym = self._contract_to_symbol(contract)
            if sym and sym in ASSETS:
                if sym not in pf.positions:
                    pf.add_position(Position(
                        symbol=sym,
                        asset_class=ASSETS[sym],
                        action="BUY" if ib_pos.position > 0 else "SELL",
                        entry_price=ib_pos.avgCost,
                        size=abs(ib_pos.position),
                        stop_loss=0,
                        take_profit=0,
                        timestamp=datetime.now().isoformat(),
                        risk_amount=0,
                    ))

    def _contract_to_symbol(self, contract: Contract) -> str | None:
        """Mapping inverse contrat IBKR → symbole Orion."""
        # Stocks
        if hasattr(contract, "symbol") and contract.secType == "STK":
            if contract.symbol in ASSETS:
                return contract.symbol

        # Forex
        if contract.secType == "CASH":
            for orion_sym, (base, quote) in _FOREX_MAP.items():
                if contract.symbol == base and contract.currency == quote:
                    return orion_sym

        # Futures
        if contract.secType == "FUT":
            for orion_sym, (fut_sym, _) in _FUTURE_MAP.items():
                if contract.symbol == fut_sym:
                    return orion_sym
            for orion_sym, (fut_sym, _) in _INDEX_FUTURE_MAP.items():
                if contract.symbol == fut_sym:
                    return orion_sym

        # Crypto
        if contract.secType == "CRYPTO":
            for orion_sym, (crypto_sym, _) in _CRYPTO_MAP.items():
                if contract.symbol == crypto_sym:
                    return orion_sym

        return None

    # ─── Status ──────────────────────────────────────────────────────

    def status(self) -> dict:
        """Retourne l'état complet du broker."""
        pf = get_portfolio()
        reg = self.regime
        dd = evaluate_drawdown(self.config, pf)
        report = portfolio_report(self.config, pf, reg)

        return {
            "connected": self.is_connected,
            "port": self.port,
            "mode": "PAPER" if self.port == 7497 else "LIVE",
            "regime": reg.regime.value,
            "vix": reg.vix_simulated,
            "avg_correlation": reg.avg_correlation,
            "capital": pf.current_capital,
            "exposure": report.total_exposure,
            "exposure_pct": report.exposure_pct,
            "risk_pct": report.risk_pct,
            "positions": report.positions_count,
            "drawdown_pct": report.drawdown_pct,
            "drawdown_action": dd.value,
            "cooldown": pf.is_in_cooldown,
            "consecutive_losses": pf._consecutive_losses,
            "warnings": report.warnings,
        }


# ═══════════════════════════════════════════════════════════════════════
#  EXÉCUTION DIRECTE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=" * 70)
    print("  Orion Execution Broker — IBKR Paper Trading")
    print("=" * 70)

    broker = OrionBroker(port=7497)

    print("\n  Connexion à IBKR Paper Trading (127.0.0.1:7497)...")
    if broker.connect():
        print(f"  Connecté! Mode: PAPER")

        # Status
        st = broker.status()
        print(f"\n  Régime:      {st['regime']}")
        print(f"  Capital:     ${st['capital']:,.2f}")
        print(f"  Positions:   {st['positions']}")
        print(f"  Drawdown:    {st['drawdown_pct']:.1%}")
        print(f"  VIX simulé:  {st['vix']:.1%}")

        # Sync positions existantes
        broker.sync_positions()

        # Démarrer le monitor Aladdin
        broker.start_monitor()
        print("\n  Monitor Aladdin actif (Ctrl+C pour arrêter)")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Arrêt...")
            broker.disconnect()
    else:
        print("  Échec de connexion.")
        print("  Vérifiez que TWS/Gateway est lancé en mode paper trading.")
        print("  Configuration → API → Enable ActiveX and Socket Clients")
        print("  Port: 7497 (paper)")
