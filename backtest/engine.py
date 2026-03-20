"""
Orion — Backtest Engine
------------------------
Moteur de backtesting complet avec intégration Aladdin :

- Simulation bar-by-bar sur données historiques
- Régime de marché simulé à chaque pas de temps
- Drawdown progressif, capital protection, diversification forcée
- Position sizing adaptatif (fixed / kelly / volatility)
- Trailing stops dynamiques
- Métriques complètes : Sharpe, Sortino, Calmar, max drawdown, win rate, etc.
- Comparaison avec benchmark (buy & hold)
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data.collector import get_history, ASSETS
from signals import indicators as ind
from signals.engine import (
    Action, Signal,
    _params_for, _score_trend, _score_momentum,
    _score_volatility, _score_volume, _score_pattern, WEIGHTS,
    BUY_THRESHOLD, SELL_THRESHOLD, _check_multi_timeframe,
)
from risk.manager import (
    Regime, RegimeState, DrawdownAction,
    RiskConfig, DEFAULT_CONFIG,
    VIX_CALM_THRESHOLD, VIX_ELEVATED_THRESHOLD,
    DEFENSIVE_ASSETS, _REGIME_MULTIPLIERS,
    CONTRACTION_ALLOWED, STRESS_ALLOWED, WATCH_ONLY,
)


# ═══════════════════════════════════════════════════════════════════════
#  TYPES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BacktestTrade:
    symbol: str
    asset_class: str
    action: str             # BUY / SELL
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str        # stop_loss / take_profit / trailing_stop / signal_reversal / end
    regime: str
    holding_days: int

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestPosition:
    symbol: str
    asset_class: str
    action: str
    entry_date: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    regime: str


@dataclass
class BacktestMetrics:
    # Rendement
    total_return: float
    total_return_pct: float
    annualized_return: float
    benchmark_return: float
    alpha: float
    # Risque
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_date: str
    avg_drawdown: float
    volatility: float
    downside_vol: float
    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_pnl: float
    best_trade: float
    worst_trade: float
    avg_holding_days: float
    # Séquences
    max_consecutive_wins: int
    max_consecutive_losses: int
    # Régimes
    trades_by_regime: dict = field(default_factory=dict)
    return_by_regime: dict = field(default_factory=dict)
    # Exposition
    avg_positions: float = 0.0
    max_positions: int = 0
    capital_protection_triggers: int = 0
    drawdown_reductions: int = 0

    def __repr__(self):
        lines = [
            "=" * 60,
            "  ORION BACKTEST — RÉSULTATS",
            "=" * 60,
            f"  Rendement total:     {self.total_return_pct:+.2%}  (${self.total_return:+,.2f})",
            f"  Rendement annualisé: {self.annualized_return:+.2%}",
            f"  Benchmark (B&H):     {self.benchmark_return:+.2%}",
            f"  Alpha:               {self.alpha:+.2%}",
            "",
            f"  Sharpe:              {self.sharpe_ratio:.2f}",
            f"  Sortino:             {self.sortino_ratio:.2f}",
            f"  Calmar:              {self.calmar_ratio:.2f}",
            f"  Volatilité:          {self.volatility:.2%}",
            f"  Max Drawdown:        {self.max_drawdown:.2%} ({self.max_drawdown_date})",
            "",
            f"  Trades:              {self.total_trades}",
            f"  Win Rate:            {self.win_rate:.1%} ({self.winning_trades}W / {self.losing_trades}L)",
            f"  Profit Factor:       {self.profit_factor:.2f}",
            f"  Avg Trade:           ${self.avg_trade_pnl:+,.2f}",
            f"  Best / Worst:        ${self.best_trade:+,.2f} / ${self.worst_trade:+,.2f}",
            f"  Avg Holding:         {self.avg_holding_days:.1f} jours",
            f"  Max Win Streak:      {self.max_consecutive_wins}",
            f"  Max Loss Streak:     {self.max_consecutive_losses}",
            "",
            f"  Avg Positions:       {self.avg_positions:.1f}",
            f"  Capital Protections: {self.capital_protection_triggers}",
            f"  Drawdown Reductions: {self.drawdown_reductions}",
        ]
        if self.trades_by_regime:
            lines.append("")
            lines.append("  --- Par régime ---")
            for reg, count in self.trades_by_regime.items():
                ret = self.return_by_regime.get(reg, 0)
                lines.append(f"    {reg:<12} {count:>3} trades  ret={ret:+.2%}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  FONCTIONS D'ANALYSE INLINE (sans appeler get_history)
# ═══════════════════════════════════════════════════════════════════════

def _analyze_bar(df_window: pd.DataFrame, symbol: str,
                 asset_class: str) -> Signal | None:
    """Génère un signal à partir d'une fenêtre de données (pas d'accès DB)."""
    if len(df_window) < 50:
        return None

    p = _params_for(asset_class)
    trend_score, _   = _score_trend(df_window, p)
    momentum_score, _ = _score_momentum(df_window, p)
    vol_score, _     = _score_volatility(df_window, p)
    volume_score, _  = _score_volume(df_window, p)
    pattern_score, _ = _score_pattern(df_window, p)

    raw_score = (
        WEIGHTS["trend"]      * trend_score
        + WEIGHTS["momentum"] * momentum_score
        + WEIGHTS["volatility"] * vol_score
        + WEIGHTS["volume"]   * volume_score
        + WEIGHTS["pattern"]  * pattern_score
    )
    score = float(np.clip(raw_score, -100, 100))

    if score >= BUY_THRESHOLD:
        action = Action.BUY
    elif score <= SELL_THRESHOLD:
        action = Action.SELL
    else:
        action = Action.HOLD

    # Confirmation multi-timeframe
    if action != Action.HOLD:
        if not _check_multi_timeframe(df_window["close"], action):
            action = Action.HOLD

    sub = [trend_score, momentum_score, vol_score, volume_score, pattern_score]
    signs = [1 if s > 0 else (-1 if s < 0 else 0) for s in sub]
    agreement = abs(sum(signs)) / len(signs)
    confidence = round(float(np.clip(agreement * (abs(score) / 100), 0, 1)), 2)

    return Signal(
        symbol=symbol, action=action, score=round(score, 1),
        confidence=confidence,
        timestamp=str(df_window.index[-1]) if hasattr(df_window.index[-1], 'strftime') else str(df_window.iloc[-1].get("date", "")),
        asset_class=asset_class, details={},
    )


def _compute_regime_at_bar(index_returns: pd.Series,
                           all_returns: pd.DataFrame) -> RegimeState:
    """Calcule le régime à un instant donné à partir des rendements."""
    # VIX simulé = vol réalisée 30j de l'indice
    if len(index_returns) >= 20:
        vix = float(index_returns.iloc[-30:].std() * np.sqrt(252)) if len(index_returns) >= 30 else float(index_returns.iloc[-20:].std() * np.sqrt(252))
    else:
        vix = 0.15

    # Corrélation moyenne
    if all_returns is not None and len(all_returns) >= 20:
        window = all_returns.iloc[-60:] if len(all_returns) >= 60 else all_returns.iloc[-20:]
        corr = window.corr()
        n = len(corr)
        if n >= 2:
            mask = np.ones(corr.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            avg_corr = float(np.abs(corr.values[mask]).mean())
        else:
            avg_corr = 0.3
    else:
        avg_corr = 0.3

    if vix >= VIX_ELEVATED_THRESHOLD and avg_corr >= 0.55:
        regime = Regime.STRESS
    elif vix >= VIX_ELEVATED_THRESHOLD or avg_corr >= 0.55:
        regime = Regime.CONTRACTION
    elif vix >= VIX_CALM_THRESHOLD:
        regime = Regime.CONTRACTION
    else:
        regime = Regime.EXPANSION

    return RegimeState(regime=regime, vix_simulated=round(vix, 4),
                       avg_correlation=round(avg_corr, 4), timestamp="")


def _compute_atr_at_bar(high: pd.Series, low: pd.Series,
                        close: pd.Series, period: int = 14) -> float:
    """ATR au dernier bar d'une fenêtre."""
    atr_series = ind.atr(high, low, close, period)
    val = atr_series.iloc[-1]
    return float(val) if not np.isnan(val) else 0.0


# ═══════════════════════════════════════════════════════════════════════
#  MOTEUR DE BACKTEST
# ═══════════════════════════════════════════════════════════════════════

class Backtest:
    """Moteur de backtesting Orion avec protections Aladdin."""

    def __init__(self,
                 symbols: list[str] | None = None,
                 start_date: str | None = None,
                 end_date: str | None = None,
                 capital: float = 100_000.0,
                 config: RiskConfig | None = None,
                 sizing_method: str = "fixed",
                 enable_aladdin: bool = True,
                 benchmark: str = "^GSPC",
                 signal_lookback: int = 200,
                 commission_pct: float = 0.001):
        """
        Args:
            symbols: Liste des actifs à backtester (défaut: tous les 35).
            start_date: Date de début "YYYY-MM-DD" (défaut: 5 ans).
            end_date: Date de fin (défaut: aujourd'hui).
            capital: Capital initial.
            config: Configuration de risque.
            sizing_method: "fixed", "kelly", "volatility".
            enable_aladdin: Active les protections (régime, drawdown, etc.).
            benchmark: Symbole de benchmark pour comparaison.
            signal_lookback: Nombre de barres pour calculer les signaux.
            commission_pct: Commission par trade (0.1% par défaut).
        """
        self.symbols = symbols or list(ASSETS.keys())
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = capital
        self.config = config or copy(DEFAULT_CONFIG)
        self.config.capital = capital
        self.sizing_method = sizing_method
        self.enable_aladdin = enable_aladdin
        self.benchmark = benchmark
        self.signal_lookback = signal_lookback
        self.commission_pct = commission_pct

        # État interne
        self._capital = capital
        self._peak_capital = capital
        self._positions: dict[str, BacktestPosition] = {}
        self._trades: list[BacktestTrade] = []
        self._equity_curve: list[tuple[str, float]] = []
        self._consecutive_losses = 0
        self._cooldown_until: str | None = None
        self._capital_protection_triggers = 0
        self._drawdown_reductions = 0

        # Change 1: Signal persistence — track consecutive signal days
        self._signal_streak: dict[str, tuple[str, int]] = {}
        # {symbol: (action, consecutive_days)} — needs 3 days to confirm
        self._signal_confirm_days = 3

        # Change 2: Bull market mode — track consecutive EXPANSION days
        self._expansion_streak = 0
        self._bull_mode_threshold = 60  # days of EXPANSION to activate
        self._bull_min_hold = 20        # minimum hold in bull mode

        # Change 3: Minimum holding period
        self._min_hold_days = 5

    # ─── Chargement des données ──────────────────────────────────────

    def _load_data(self) -> dict[str, pd.DataFrame]:
        """Charge les données historiques depuis la DB."""
        import sqlite3 as _sql
        from data.collector import DB_PATH

        if self.end_date:
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
        else:
            end = datetime.now()

        if self.start_date:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
        else:
            start = end - timedelta(days=5 * 365)

        # Load data from (start - lookback) to end
        load_from = (start - timedelta(days=self.signal_lookback + 60)).strftime("%Y-%m-%d")
        load_to = end.strftime("%Y-%m-%d 23:59:59")

        conn = _sql.connect(str(DB_PATH))
        data = {}
        for sym in self.symbols:
            df = pd.read_sql_query(
                "SELECT symbol, date, open, high, low, close, volume, asset_class "
                "FROM prices WHERE symbol = ? AND date >= ? AND date <= ? ORDER BY date ASC",
                conn, params=(sym, load_from, load_to),
            )
            if df.empty or len(df) < self.signal_lookback + 50:
                continue
            df = df.copy()
            df["date_dt"] = pd.to_datetime(df["date"])
            data[sym] = df
        conn.close()
        return data

    def _align_dates(self, data: dict[str, pd.DataFrame]) -> list[str]:
        """Retourne les dates communes, triées."""
        if not data:
            return []

        all_dates = set()
        for df in data.values():
            dates_in_range = df[df["date_dt"] >= (
                datetime.strptime(self.start_date, "%Y-%m-%d")
                if self.start_date
                else df["date_dt"].iloc[0] + timedelta(days=self.signal_lookback)
            )]["date"].tolist()
            all_dates.update(dates_in_range)

        if self.end_date:
            all_dates = {d for d in all_dates if d <= self.end_date + " 23:59:59"}

        return sorted(all_dates)

    # ─── Sizing ──────────────────────────────────────────────────────

    def _compute_size(self, symbol: str, action: str, entry: float,
                      atr_val: float, regime: RegimeState,
                      df_window: pd.DataFrame) -> tuple[float, float, float, float]:
        """Calcule size, SL, TP, risk_amount en tenant compte du régime."""
        cfg = self.config
        if atr_val == 0:
            return 0, 0, 0, 0

        sl_dist = cfg.atr_sl_multiplier * atr_val
        tp_dist = cfg.atr_tp_multiplier * atr_val

        if action == "BUY":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        risk_per_unit = sl_dist

        # Sizing de base
        if self.sizing_method == "volatility" and len(df_window) >= 30:
            returns = df_window["close"].pct_change().dropna()
            annual_vol = float(returns.std() * np.sqrt(252))
            if annual_vol > 0:
                size = (self._capital * 0.10) / (entry * annual_vol)
            else:
                size = (self._capital * cfg.risk_per_trade) / risk_per_unit
        elif self.sizing_method == "kelly":
            wins = [t for t in self._trades if t.is_win]
            losses = [t for t in self._trades if not t.is_win]
            if len(self._trades) >= 10 and losses:
                wr = len(wins) / len(self._trades)
                avg_wl = (np.mean([t.pnl for t in wins]) / abs(np.mean([t.pnl for t in losses]))
                          if wins else 1.0)
                kelly = max(0, wr - (1 - wr) / max(avg_wl, 0.01)) * cfg.kelly_fraction
            else:
                kelly = cfg.risk_per_trade
            risk_amount = self._capital * min(kelly, cfg.max_risk_per_trade)
            size = risk_amount / risk_per_unit
        else:
            risk_amount = self._capital * cfg.risk_per_trade
            size = risk_amount / risk_per_unit

        # Régime ajustement
        if self.enable_aladdin:
            mult = _REGIME_MULTIPLIERS[regime.regime]["sizing"]
            size *= mult

        # Plafonds
        max_value = self._capital * cfg.max_exposure_per_asset
        if size * entry > max_value:
            size = max_value / entry

        risk_amount = size * risk_per_unit
        if risk_amount > self._capital * cfg.max_risk_per_trade:
            size = (self._capital * cfg.max_risk_per_trade) / risk_per_unit
            risk_amount = size * risk_per_unit

        return size, sl, tp, risk_amount

    # ─── Vérifications Aladdin ───────────────────────────────────────

    def _check_aladdin_gates(self, symbol: str, regime: RegimeState,
                             current_date: str) -> tuple[bool, str]:
        """Vérifie toutes les protections Aladdin. Retourne (ok, reason)."""
        if not self.enable_aladdin:
            return True, ""

        # Watch-only stocks
        if symbol in WATCH_ONLY:
            return False, "watch_only"

        # Cooldown
        if self._cooldown_until and current_date < self._cooldown_until:
            return False, "capital_protection"

        # Filtrage par régime
        if regime.regime == Regime.STRESS:
            if symbol not in STRESS_ALLOWED:
                return False, "regime_stress"

        if regime.regime == Regime.CONTRACTION:
            if symbol not in CONTRACTION_ALLOWED:
                return False, "regime_contraction"

        # Position déjà ouverte
        if symbol in self._positions:
            return False, "already_open"

        # Max positions
        if len(self._positions) >= self.config.max_positions:
            return False, "max_positions"

        # Max par classe
        asset_class = ASSETS.get(symbol, "unknown")
        class_count = sum(1 for p in self._positions.values() if p.asset_class == asset_class)
        if class_count >= self.config.max_assets_per_class:
            return False, "max_per_class"

        # Exposition totale
        total_exposure = sum(p.size * p.entry_price for p in self._positions.values())
        if total_exposure > self._capital * 0.90:
            return False, "max_exposure"

        return True, ""

    def _apply_drawdown_rules(self, current_date: str,
                              regime: RegimeState | None = None) -> list[str]:
        """Applique les règles de drawdown progressif. Retourne les positions à fermer.

        Change 2: En bull market (EXPANSION > 60j), pas de réduction sauf si DD > 15%.
        """
        dd_pct = (self._peak_capital - self._capital) / self._peak_capital if self._peak_capital > 0 else 0
        to_close = []

        # Bull market mode: skip drawdown reductions unless critical
        is_bull = (self._expansion_streak >= self._bull_mode_threshold)
        if is_bull and regime and regime.regime == Regime.EXPANSION:
            if dd_pct < 0.15:
                return []  # Let positions breathe in bull market

        if dd_pct >= self.config.drawdown_level_3:
            to_close = list(self._positions.keys())
            self._drawdown_reductions += 1
        elif dd_pct >= self.config.drawdown_level_2:
            positions_sorted = sorted(
                self._positions.items(),
                key=lambda kv: self._unrealized_pnl(kv[0], kv[1], current_date)
            )
            n_close = max(1, len(positions_sorted) // 2)
            to_close = [sym for sym, _ in positions_sorted[:n_close]]
            self._drawdown_reductions += 1
        elif dd_pct >= self.config.drawdown_level_1:
            positions_sorted = sorted(
                self._positions.items(),
                key=lambda kv: self._unrealized_pnl(kv[0], kv[1], current_date)
            )
            n_close = max(1, len(positions_sorted) // 4)
            to_close = [sym for sym, _ in positions_sorted[:n_close]]
            self._drawdown_reductions += 1

        return to_close

    def _unrealized_pnl(self, symbol: str, pos: BacktestPosition,
                        current_date: str) -> float:
        """P&L non réalisé d'une position (approximation via dernier close)."""
        # On utilisera le prix courant passé par la boucle principale
        return 0.0  # Sera remplacé dans la boucle

    # ─── Boucle principale ───────────────────────────────────────────

    def run(self) -> BacktestResult:
        """Exécute le backtest complet."""
        print(f"[Backtest] Chargement des données pour {len(self.symbols)} actifs...")
        data = self._load_data()
        if not data:
            raise ValueError("Aucune donnée chargée. La base est-elle initialisée ?")

        dates = self._align_dates(data)
        if not dates:
            raise ValueError("Aucune date commune trouvée.")

        print(f"[Backtest] {len(data)} actifs chargés, {len(dates)} barres à simuler")
        print(f"[Backtest] Période: {dates[0][:10]} → {dates[-1][:10]}")
        print(f"[Backtest] Capital: ${self.initial_capital:,.0f}  Méthode: {self.sizing_method}")
        if self.enable_aladdin:
            print(f"[Backtest] Aladdin: ACTIVÉ")

        # Pré-calculer les rendements pour le régime
        index_data = data.get(self.benchmark)
        benchmark_closes = []

        # Préparer les rendements cross-asset pour corrélation
        all_close_series = {}
        sample_syms = [s for s in ["^GSPC", "AAPL", "GC=F", "CL=F", "EURUSD=X", "BTC-USD"] if s in data]
        for sym in sample_syms:
            all_close_series[sym] = data[sym].set_index("date")["close"]

        daily_positions_count = []

        # ─── Boucle bar-by-bar ───────────────────────────────────────
        for bar_idx, date in enumerate(dates):

            # Prix courants
            current_prices = {}
            for sym, df in data.items():
                row = df[df["date"] == date]
                if not row.empty:
                    current_prices[sym] = {
                        "open": float(row.iloc[0]["open"]),
                        "high": float(row.iloc[0]["high"]),
                        "low": float(row.iloc[0]["low"]),
                        "close": float(row.iloc[0]["close"]),
                        "volume": float(row.iloc[0]["volume"]) if pd.notna(row.iloc[0]["volume"]) else 0,
                    }

            if not current_prices:
                continue

            # Benchmark
            if self.benchmark in current_prices:
                benchmark_closes.append(current_prices[self.benchmark]["close"])

            # ── 1. Régime de marché + bull market tracking ──
            if index_data is not None and self.enable_aladdin:
                idx_mask = index_data["date"] <= date
                idx_closes = index_data[idx_mask]["close"]
                idx_returns = idx_closes.pct_change().dropna()

                cross_df = pd.DataFrame({
                    sym: series[series.index <= date].pct_change().dropna()
                    for sym, series in all_close_series.items()
                })
                regime = _compute_regime_at_bar(idx_returns, cross_df)
            else:
                regime = RegimeState(regime=Regime.EXPANSION, vix_simulated=0.1,
                                     avg_correlation=0.3, timestamp=date)

            # Track consecutive EXPANSION days for bull mode
            if regime.regime == Regime.EXPANSION:
                self._expansion_streak += 1
            else:
                self._expansion_streak = 0

            # ── 2. Drawdown progressif (regime-aware) ──
            if self.enable_aladdin and self._positions:
                dd_closes = self._apply_drawdown_rules(date, regime)
                for sym in dd_closes:
                    if sym in self._positions and sym in current_prices:
                        self._close_position(sym, current_prices[sym]["close"],
                                             date, "drawdown")

                # Arrêt total à -10%
                dd_pct = (self._peak_capital - self._capital) / self._peak_capital if self._peak_capital > 0 else 0
                if dd_pct >= self.config.drawdown_level_3:
                    # Fermer tout ce qui reste
                    for sym in list(self._positions.keys()):
                        if sym in current_prices:
                            self._close_position(sym, current_prices[sym]["close"],
                                                 date, "drawdown_critical")
                    continue  # Skip le reste de cette barre

            # ── 3. Stress → réduction 50% des positions existantes ──
            if (self.enable_aladdin and regime.regime == Regime.STRESS
                    and len(self._positions) > 0 and bar_idx % 5 == 0):
                # Fermer la moitié des positions (une fois tous les 5 jours en stress)
                sorted_pos = sorted(
                    [(s, p) for s, p in self._positions.items() if s in current_prices],
                    key=lambda kv: (current_prices[kv[0]]["close"] - kv[1].entry_price) * kv[1].size
                )
                n_close = max(1, len(sorted_pos) // 2)
                for sym, _ in sorted_pos[:n_close]:
                    self._close_position(sym, current_prices[sym]["close"],
                                         date, "regime_stress")

            # ── 4. Vérifier SL / TP / Trailing sur positions ouvertes ──
            for sym in list(self._positions.keys()):
                if sym not in current_prices:
                    continue
                pos = self._positions[sym]
                price = current_prices[sym]

                # Change 3: Compute holding days
                try:
                    entry_dt = datetime.strptime(pos.entry_date[:10], "%Y-%m-%d")
                    current_dt = datetime.strptime(date[:10], "%Y-%m-%d")
                    holding_days = (current_dt - entry_dt).days
                except ValueError:
                    holding_days = 999

                # Change 2: Bull mode minimum hold
                is_bull = (self._expansion_streak >= self._bull_mode_threshold)
                min_hold = self._bull_min_hold if is_bull else self._min_hold_days

                # Stop-loss (always enforced, but min hold for TP and trailing)
                if pos.action == "BUY" and price["low"] <= pos.stop_loss:
                    self._close_position(sym, pos.stop_loss, date, "stop_loss")
                    continue
                elif pos.action == "SELL" and price["high"] >= pos.stop_loss:
                    self._close_position(sym, pos.stop_loss, date, "stop_loss")
                    continue

                # Skip TP and trailing if below minimum holding period
                if holding_days < min_hold:
                    continue

                # Take-profit
                if pos.action == "BUY" and price["high"] >= pos.take_profit:
                    self._close_position(sym, pos.take_profit, date, "take_profit")
                    continue
                elif pos.action == "SELL" and price["low"] <= pos.take_profit:
                    self._close_position(sym, pos.take_profit, date, "take_profit")
                    continue

                # Trailing stop (mise à jour ATR)
                df_sym = data[sym]
                mask = df_sym["date"] <= date
                window = df_sym[mask].tail(30)
                if len(window) >= 15:
                    atr_val = _compute_atr_at_bar(
                        window["high"], window["low"], window["close"],
                        self.config.atr_period)
                    if atr_val > 0:
                        if pos.action == "BUY":
                            new_sl = price["close"] - self.config.atr_sl_multiplier * atr_val
                            pos.stop_loss = max(pos.stop_loss, new_sl)
                        else:
                            new_sl = price["close"] + self.config.atr_sl_multiplier * atr_val
                            pos.stop_loss = min(pos.stop_loss, new_sl)

            # ── 5. Générer des signaux et ouvrir des positions ──
            for sym in self.symbols:
                if sym not in data or sym not in current_prices:
                    continue

                # Fenêtre pour le signal
                df_sym = data[sym]
                mask = df_sym["date"] <= date
                window = df_sym[mask].tail(self.signal_lookback)
                if len(window) < 50:
                    continue

                signal = _analyze_bar(window, sym, ASSETS.get(sym, "unknown"))

                # Change 1: Signal persistence — track consecutive days
                if signal is None or signal.action == Action.HOLD:
                    self._signal_streak.pop(sym, None)
                    continue

                action_str = signal.action.value
                prev = self._signal_streak.get(sym)
                if prev and prev[0] == action_str:
                    self._signal_streak[sym] = (action_str, prev[1] + 1)
                else:
                    self._signal_streak[sym] = (action_str, 1)

                streak_days = self._signal_streak[sym][1]
                if streak_days < self._signal_confirm_days:
                    continue  # Not yet confirmed

                # Signal de fermeture (inversion) — respect min hold
                if sym in self._positions:
                    pos = self._positions[sym]
                    if (pos.action == "BUY" and signal.action == Action.SELL) or \
                       (pos.action == "SELL" and signal.action == Action.BUY):
                        # Check min hold before allowing reversal
                        try:
                            entry_dt = datetime.strptime(pos.entry_date[:10], "%Y-%m-%d")
                            current_dt = datetime.strptime(date[:10], "%Y-%m-%d")
                            held = (current_dt - entry_dt).days
                        except ValueError:
                            held = 999
                        if held >= self._min_hold_days:
                            self._close_position(sym, current_prices[sym]["close"],
                                                 date, "signal_reversal")
                    continue

                # Gates Aladdin
                ok, reason = self._check_aladdin_gates(sym, regime, date)
                if not ok:
                    continue

                # ATR pour sizing
                atr_val = _compute_atr_at_bar(
                    window["high"], window["low"], window["close"],
                    self.config.atr_period)
                if atr_val == 0:
                    continue

                entry_price = current_prices[sym]["close"]
                action = signal.action.value

                size, sl, tp, risk_amount = self._compute_size(
                    sym, action, entry_price, atr_val, regime, window)

                if size <= 0 or risk_amount <= 0:
                    continue

                # Commission
                commission = size * entry_price * self.commission_pct
                self._capital -= commission

                self._positions[sym] = BacktestPosition(
                    symbol=sym,
                    asset_class=ASSETS.get(sym, "unknown"),
                    action=action,
                    entry_date=date,
                    entry_price=entry_price,
                    size=size,
                    stop_loss=sl,
                    take_profit=tp,
                    risk_amount=risk_amount,
                    regime=regime.regime.value,
                )

            # ── 6. Mettre à jour l'equity ──
            unrealized = 0.0
            for sym, pos in self._positions.items():
                if sym in current_prices:
                    cp = current_prices[sym]["close"]
                    if pos.action == "BUY":
                        unrealized += (cp - pos.entry_price) * pos.size
                    else:
                        unrealized += (pos.entry_price - cp) * pos.size

            equity = self._capital + unrealized
            self._equity_curve.append((date, equity))
            daily_positions_count.append(len(self._positions))

            if equity > self._peak_capital:
                self._peak_capital = equity

            # Progress
            if bar_idx > 0 and bar_idx % 250 == 0:
                pct = bar_idx / len(dates) * 100
                print(f"  [{pct:5.1f}%] {date[:10]}  equity=${equity:,.0f}  "
                      f"positions={len(self._positions)}  trades={len(self._trades)}  "
                      f"regime={regime.regime.value}")

        # ── Fermer les positions restantes ──
        last_date = dates[-1] if dates else ""
        for sym in list(self._positions.keys()):
            if sym in data:
                df_sym = data[sym]
                last_row = df_sym[df_sym["date"] == last_date]
                if not last_row.empty:
                    self._close_position(sym, float(last_row.iloc[0]["close"]),
                                         last_date, "end")

        # ── Benchmark ──
        benchmark_return = 0.0
        if len(benchmark_closes) >= 2:
            benchmark_return = (benchmark_closes[-1] / benchmark_closes[0]) - 1

        # ── Métriques ──
        metrics = self._compute_metrics(benchmark_return, daily_positions_count)

        print(f"\n{metrics}")

        return BacktestResult(
            metrics=metrics,
            trades=self._trades,
            equity_curve=pd.DataFrame(self._equity_curve, columns=["date", "equity"]),
            config=self.config,
        )

    # ─── Fermeture de position ───────────────────────────────────────

    def _close_position(self, symbol: str, exit_price: float,
                        date: str, reason: str):
        """Ferme une position et enregistre le trade."""
        pos = self._positions.pop(symbol, None)
        if pos is None:
            return

        if pos.action == "BUY":
            pnl = (exit_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - exit_price) * pos.size

        # Commission de sortie
        commission = pos.size * exit_price * self.commission_pct
        pnl -= commission

        pnl_pct = pnl / (pos.entry_price * pos.size) if pos.entry_price * pos.size > 0 else 0

        entry_dt = datetime.strptime(pos.entry_date[:10], "%Y-%m-%d") if len(pos.entry_date) >= 10 else datetime.now()
        exit_dt = datetime.strptime(date[:10], "%Y-%m-%d") if len(date) >= 10 else datetime.now()
        holding = (exit_dt - entry_dt).days

        trade = BacktestTrade(
            symbol=symbol,
            asset_class=pos.asset_class,
            action=pos.action,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=date,
            exit_price=exit_price,
            size=pos.size,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            exit_reason=reason,
            regime=pos.regime,
            holding_days=max(holding, 1),
        )
        self._trades.append(trade)
        self._capital += pnl

        # Capital protection
        if pnl >= 0:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.config.consecutive_loss_limit:
                # Pause de 24h → dans le backtest = skip N jours
                try:
                    cooldown_dt = datetime.strptime(date[:10], "%Y-%m-%d") + timedelta(days=1)
                    self._cooldown_until = cooldown_dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass
                self._capital_protection_triggers += 1
                self._consecutive_losses = 0

    # ─── Calcul des métriques ────────────────────────────────────────

    def _compute_metrics(self, benchmark_return: float,
                         daily_positions: list[int]) -> BacktestMetrics:
        """Calcule toutes les métriques de performance."""
        total_return = self._capital - self.initial_capital
        total_return_pct = total_return / self.initial_capital

        # Equity curve → rendements quotidiens
        eq = pd.DataFrame(self._equity_curve, columns=["date", "equity"])
        if len(eq) < 2:
            return BacktestMetrics(
                total_return=total_return, total_return_pct=total_return_pct,
                annualized_return=0, benchmark_return=benchmark_return, alpha=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                max_drawdown=0, max_drawdown_date="", avg_drawdown=0,
                volatility=0, downside_vol=0,
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
                avg_trade_pnl=0, best_trade=0, worst_trade=0,
                avg_holding_days=0, max_consecutive_wins=0, max_consecutive_losses=0,
                avg_positions=0, max_positions=0,
                capital_protection_triggers=0, drawdown_reductions=0,
            )

        daily_returns = eq["equity"].pct_change().dropna()
        n_days = len(eq)
        n_years = n_days / 252

        # Rendement annualisé
        if n_years > 0 and total_return_pct > -1:
            annualized = (1 + total_return_pct) ** (1 / n_years) - 1
        else:
            annualized = 0

        # Volatilité
        vol = float(daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 1 else 0

        # Downside vol
        neg_returns = daily_returns[daily_returns < 0]
        downside_vol = float(neg_returns.std() * np.sqrt(252)) if len(neg_returns) > 1 else 0

        # Drawdown
        eq_values = eq["equity"].values
        peak = np.maximum.accumulate(eq_values)
        drawdowns = (peak - eq_values) / np.where(peak > 0, peak, 1)
        max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0
        max_dd_idx = int(drawdowns.argmax()) if len(drawdowns) > 0 else 0
        max_dd_date = eq.iloc[max_dd_idx]["date"][:10] if max_dd_idx < len(eq) else ""
        avg_dd = float(drawdowns[drawdowns > 0].mean()) if (drawdowns > 0).any() else 0

        # Ratios
        rf = 0.04  # Risk-free rate
        excess_daily = daily_returns - rf / 252
        sharpe = float(excess_daily.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        sortino = float(excess_daily.mean() / neg_returns.std() * np.sqrt(252)) if len(neg_returns) > 1 and neg_returns.std() > 0 else 0
        calmar = annualized / max_dd if max_dd > 0 else 0

        # Trades
        wins = [t for t in self._trades if t.is_win]
        losses = [t for t in self._trades if not t.is_win]
        n_trades = len(self._trades)
        win_rate = len(wins) / n_trades if n_trades > 0 else 0
        avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0
        avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0
        avg_pnl = float(np.mean([t.pnl for t in self._trades])) if self._trades else 0
        best = max((t.pnl for t in self._trades), default=0)
        worst = min((t.pnl for t in self._trades), default=0)
        avg_hold = float(np.mean([t.holding_days for t in self._trades])) if self._trades else 0

        # Séquences consécutives
        max_cw, max_cl, cw, cl = 0, 0, 0, 0
        for t in self._trades:
            if t.is_win:
                cw += 1; cl = 0
                max_cw = max(max_cw, cw)
            else:
                cl += 1; cw = 0
                max_cl = max(max_cl, cl)

        # Par régime
        trades_by_regime: dict[str, int] = {}
        return_by_regime: dict[str, float] = {}
        for t in self._trades:
            trades_by_regime[t.regime] = trades_by_regime.get(t.regime, 0) + 1
            return_by_regime[t.regime] = return_by_regime.get(t.regime, 0.0) + t.pnl

        for reg in return_by_regime:
            return_by_regime[reg] /= self.initial_capital

        # Alpha
        alpha = annualized - benchmark_return / max(n_years, 1)

        # Positions
        avg_pos = float(np.mean(daily_positions)) if daily_positions else 0
        max_pos = max(daily_positions) if daily_positions else 0

        return BacktestMetrics(
            total_return=round(total_return, 2),
            total_return_pct=round(total_return_pct, 4),
            annualized_return=round(annualized, 4),
            benchmark_return=round(benchmark_return, 4),
            alpha=round(alpha, 4),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            max_drawdown=round(max_dd, 4),
            max_drawdown_date=max_dd_date,
            avg_drawdown=round(avg_dd, 4),
            volatility=round(vol, 4),
            downside_vol=round(downside_vol, 4),
            total_trades=n_trades,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=round(win_rate, 4),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            avg_trade_pnl=round(avg_pnl, 2),
            best_trade=round(best, 2),
            worst_trade=round(worst, 2),
            avg_holding_days=round(avg_hold, 1),
            max_consecutive_wins=max_cw,
            max_consecutive_losses=max_cl,
            trades_by_regime=trades_by_regime,
            return_by_regime={k: round(v, 4) for k, v in return_by_regime.items()},
            avg_positions=round(avg_pos, 1),
            max_positions=max_pos,
            capital_protection_triggers=self._capital_protection_triggers,
            drawdown_reductions=self._drawdown_reductions,
        )


# ═══════════════════════════════════════════════════════════════════════
#  RÉSULTAT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    trades: list[BacktestTrade]
    equity_curve: pd.DataFrame
    config: RiskConfig

    def trades_df(self) -> pd.DataFrame:
        """Retourne les trades sous forme de DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "symbol": t.symbol, "class": t.asset_class,
                "action": t.action, "entry_date": t.entry_date[:10],
                "exit_date": t.exit_date[:10], "entry": round(t.entry_price, 4),
                "exit": round(t.exit_price, 4), "size": round(t.size, 4),
                "pnl": t.pnl, "pnl%": f"{t.pnl_pct:+.2%}",
                "reason": t.exit_reason, "regime": t.regime,
                "days": t.holding_days,
            }
            for t in self.trades
        ])

    def monthly_returns(self) -> pd.DataFrame:
        """Retourne les rendements mensuels."""
        eq = self.equity_curve.copy()
        if eq.empty:
            return pd.DataFrame()
        eq["date"] = pd.to_datetime(eq["date"])
        eq = eq.set_index("date")
        monthly = eq["equity"].resample("ME").last()
        returns = monthly.pct_change().dropna()
        df = returns.to_frame("return")
        df["year"] = df.index.year
        df["month"] = df.index.month
        return df.pivot_table(index="year", columns="month", values="return")

    def summary_by_class(self) -> pd.DataFrame:
        """Performance par classe d'actif."""
        if not self.trades:
            return pd.DataFrame()
        rows = {}
        for t in self.trades:
            cls = t.asset_class
            if cls not in rows:
                rows[cls] = {"trades": 0, "wins": 0, "pnl": 0.0}
            rows[cls]["trades"] += 1
            rows[cls]["wins"] += int(t.is_win)
            rows[cls]["pnl"] += t.pnl
        data = []
        for cls, r in rows.items():
            data.append({
                "class": cls,
                "trades": r["trades"],
                "win_rate": f"{r['wins']/r['trades']:.0%}" if r["trades"] else "0%",
                "pnl": round(r["pnl"], 2),
            })
        return pd.DataFrame(data).sort_values("pnl", ascending=False)

    def summary_by_exit(self) -> pd.DataFrame:
        """Répartition par raison de sortie."""
        if not self.trades:
            return pd.DataFrame()
        rows = {}
        for t in self.trades:
            reason = t.exit_reason
            if reason not in rows:
                rows[reason] = {"count": 0, "pnl": 0.0}
            rows[reason]["count"] += 1
            rows[reason]["pnl"] += t.pnl
        return pd.DataFrame([
            {"reason": r, "count": d["count"], "pnl": round(d["pnl"], 2)}
            for r, d in rows.items()
        ]).sort_values("count", ascending=False)


# ═══════════════════════════════════════════════════════════════════════
#  RACCOURCI
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(symbols: list[str] | None = None,
                 start_date: str | None = None,
                 end_date: str | None = None,
                 capital: float = 100_000.0,
                 sizing_method: str = "fixed",
                 enable_aladdin: bool = True,
                 **kwargs) -> BacktestResult:
    """Raccourci pour lancer un backtest."""
    bt = Backtest(
        symbols=symbols, start_date=start_date, end_date=end_date,
        capital=capital, sizing_method=sizing_method,
        enable_aladdin=enable_aladdin, **kwargs,
    )
    return bt.run()


# ═══════════════════════════════════════════════════════════════════════
#  EXÉCUTION DIRECTE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Orion Backtest Engine")
    print("=" * 60)

    result = run_backtest(
        capital=100_000,
        sizing_method="fixed",
        enable_aladdin=True,
    )

    print("\n  --- Trades par classe ---")
    print(result.summary_by_class().to_string(index=False))

    print("\n  --- Trades par raison de sortie ---")
    print(result.summary_by_exit().to_string(index=False))

    print(f"\n  {len(result.trades)} trades au total")
    print(f"  Equity finale: ${result.equity_curve['equity'].iloc[-1]:,.2f}")
