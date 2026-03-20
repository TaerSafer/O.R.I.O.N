"""
Orion — Risk Manager (Aladdin Philosophy)
-------------------------------------------
Gestion du risque inspirée d'Aladdin (BlackRock) :

1. Détection de régime de marché (EXPANSION / CONTRACTION / STRESS)
2. Paramètres de risque dynamiques par régime
3. Drawdown progressif avec réduction automatique des positions
4. Diversification forcée (max 3 par classe, corrélation < 0.7)
5. Capital protection mode (pause après 3 pertes consécutives)
6. Position sizing adaptatif (fixe, Kelly, volatilité)
7. Trailing stop ATR-based
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd

from data.collector import get_history, ASSETS
from signals.indicators import atr as calc_atr
from signals.engine import Signal, Action


# ═══════════════════════════════════════════════════════════════════════
#  RÉGIMES DE MARCHÉ
# ═══════════════════════════════════════════════════════════════════════

class Regime(Enum):
    EXPANSION = "EXPANSION"
    CONTRACTION = "CONTRACTION"
    STRESS = "STRESS"


# Actifs défensifs privilégiés en CONTRACTION
DEFENSIVE_ASSETS = {"GC=F", "SI=F", "USDCHF=X", "USDJPY=X", "EURJPY=X", "GBPJPY=X"}

# Filtrage par régime (Change 3)
CONTRACTION_ALLOWED = {"GC=F", "CL=F", "BZ=F", "USDCHF=X", "USDJPY=X", "EURUSD=X"}
STRESS_ALLOWED = {"GC=F", "USDCHF=X"}

# Actions en mode surveillance uniquement (Change 4)
WATCH_ONLY = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"}

# Proxies pour le VIX simulé (on utilise la vol réalisée du S&P 500)
VIX_PROXY_SYMBOL = "^GSPC"
VIX_CALM_THRESHOLD = 0.18      # Vol annualisée < 18% → calme
VIX_ELEVATED_THRESHOLD = 0.28  # Vol annualisée < 28% → élevée
# Au-dessus de 20% → stress


@dataclass
class RegimeState:
    regime: Regime
    vix_simulated: float          # Volatilité annualisée simulée
    avg_correlation: float        # Corrélation moyenne cross-asset
    timestamp: str
    details: dict = field(default_factory=dict)

    def __repr__(self):
        icons = {Regime.EXPANSION: "🟢", Regime.CONTRACTION: "🟡", Regime.STRESS: "🔴"}
        return (f"{icons[self.regime]} Régime: {self.regime.value}  "
                f"VIX={self.vix_simulated:.1%}  ρ_avg={self.avg_correlation:.2f}")


def _compute_simulated_vix(days: int = 30) -> float:
    """Volatilité réalisée annualisée du S&P 500 comme proxy VIX."""
    df = get_history(VIX_PROXY_SYMBOL, days=days + 10)
    if df.empty or len(df) < 10:
        # Fallback : moyenne de vol sur tous les indices
        vols = []
        for sym, cls in ASSETS.items():
            if cls == "index":
                d = get_history(sym, days=days + 10)
                if len(d) >= 10:
                    ret = d["close"].pct_change().dropna()
                    vols.append(float(ret.std() * np.sqrt(252)))
        return float(np.mean(vols)) if vols else 0.15

    returns = df["close"].pct_change().dropna().iloc[-days:]
    return float(returns.std() * np.sqrt(252))


def _compute_avg_correlation(days: int = 60) -> float:
    """Corrélation moyenne absolue entre toutes les paires d'actifs."""
    # Échantillon représentatif (un par classe) pour la performance
    sample = ["^GSPC", "AAPL", "GC=F", "CL=F", "EURUSD=X", "BTC-USD"]
    closes = {}
    for sym in sample:
        d = get_history(sym, days=days + 10)
        if not d.empty and len(d) > 10:
            closes[sym] = d.set_index("date")["close"]

    if len(closes) < 3:
        return 0.3  # Défaut neutre

    prices = pd.DataFrame(closes)
    returns = prices.pct_change().dropna()
    if returns.empty:
        return 0.3

    corr = returns.corr()
    n = len(corr)
    if n < 2:
        return 0.3

    # Moyenne des valeurs hors diagonale (en valeur absolue)
    mask = np.ones(corr.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    return float(np.abs(corr.values[mask]).mean())


def detect_regime() -> RegimeState:
    """Détecte le régime de marché actuel.

    - EXPANSION  : VIX bas + corrélations normales
    - CONTRACTION: VIX modéré OU corrélations en hausse
    - STRESS     : VIX élevé + corrélations élevées (flight to safety)
    """
    vix = _compute_simulated_vix(days=30)
    avg_corr = _compute_avg_correlation(days=60)
    details = {"vix_30d": round(vix, 4), "avg_corr_60d": round(avg_corr, 4)}

    # Logique de classification
    if vix >= VIX_ELEVATED_THRESHOLD and avg_corr >= 0.55:
        regime = Regime.STRESS
    elif vix >= VIX_ELEVATED_THRESHOLD or avg_corr >= 0.55:
        regime = Regime.CONTRACTION
    elif vix >= VIX_CALM_THRESHOLD:
        regime = Regime.CONTRACTION
    else:
        regime = Regime.EXPANSION

    return RegimeState(
        regime=regime,
        vix_simulated=round(vix, 4),
        avg_correlation=round(avg_corr, 4),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        details=details,
    )


# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION DE RISQUE PAR RÉGIME
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RiskConfig:
    """Paramètres de risque globaux (valeurs = régime EXPANSION)."""
    capital: float = 10_000.0
    risk_per_trade: float = 0.01         # 1% par trade
    max_risk_per_trade: float = 0.02     # Plafond 2%
    max_portfolio_risk: float = 0.06     # Risque total 6%
    max_positions: int = 10
    max_exposure_per_asset: float = 0.15 # 15% par actif
    max_exposure_per_class: float = 0.40 # 40% par classe
    max_correlated_exposure: float = 0.30
    max_drawdown: float = 0.10           # Arrêt total à 10%
    max_assets_per_class: int = 3        # Diversification forcée
    max_correlated_simultaneous: int = 1 # Max 1 paire corrélée (=2 actifs corrélés)
    correlation_threshold: float = 0.70  # Seuil de corrélation élevée
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 6.0
    atr_period: int = 14
    correlation_window: int = 60
    kelly_fraction: float = 0.25
    consecutive_loss_limit: int = 3      # Pause après N pertes
    cooldown_hours: int = 24             # Durée de pause

    # Drawdown progressif
    drawdown_level_1: float = 0.08       # -8% → réduction 25%
    drawdown_level_2: float = 0.12       # -12% → réduction 50%
    drawdown_level_3: float = 0.15       # -15% → fermeture totale


DEFAULT_CONFIG = RiskConfig()

# Multiplicateurs par régime appliqués sur les limites EXPANSION
_REGIME_MULTIPLIERS = {
    Regime.EXPANSION: {
        "sizing":            1.0,
        "max_exposure":      1.0,
        "new_trades":        True,
        "all_classes_active": True,
    },
    Regime.CONTRACTION: {
        "sizing":            0.70,   # -30% sizing
        "max_exposure":      0.70,
        "new_trades":        True,
        "all_classes_active": False,  # Préférence défensifs
    },
    Regime.STRESS: {
        "sizing":            0.50,   # -50% sizing
        "max_exposure":      0.333,  # Exposition max réduite à ~20% (0.40 * 0.333 ≈ 13%)
        "new_trades":        False,  # AUCUN nouveau trade
        "all_classes_active": False,
    },
}


def effective_config(config: RiskConfig, regime: RegimeState) -> RiskConfig:
    """Retourne une RiskConfig ajustée au régime courant."""
    m = _REGIME_MULTIPLIERS[regime.regime]
    from copy import copy
    cfg = copy(config)

    cfg.risk_per_trade *= m["sizing"]
    cfg.max_risk_per_trade *= m["sizing"]
    cfg.max_portfolio_risk *= m["sizing"]
    cfg.max_exposure_per_asset *= m["max_exposure"]
    cfg.max_exposure_per_class *= m["max_exposure"]
    cfg.max_correlated_exposure *= m["max_exposure"]

    # En STRESS : exposition max globale = 20% du capital
    if regime.regime == Regime.STRESS:
        cfg.max_exposure_per_class = min(cfg.max_exposure_per_class, 0.20)
        cfg.max_positions = min(cfg.max_positions, 5)

    return cfg


# ═══════════════════════════════════════════════════════════════════════
#  TYPES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PositionSize:
    symbol: str
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    value: float
    risk_amount: float
    risk_pct: float
    method: str
    regime: str = "EXPANSION"
    details: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"{self.symbol:<12} {self.action:>4}  "
                f"size={self.size:,.2f}  value=${self.value:,.0f}  "
                f"risk={self.risk_pct:.2%}  SL={self.stop_loss:.4f}  "
                f"TP={self.take_profit:.4f}  [{self.regime}]")


@dataclass
class PortfolioRisk:
    timestamp: str
    regime: str
    capital: float
    total_exposure: float
    total_risk: float
    exposure_pct: float
    risk_pct: float
    positions_count: int
    drawdown: float
    drawdown_pct: float
    drawdown_action: str
    peak_capital: float
    exposure_by_class: dict = field(default_factory=dict)
    risk_by_class: dict = field(default_factory=dict)
    positions_by_class: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
#  ÉTAT DU PORTEFEUILLE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    symbol: str
    asset_class: str
    action: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    timestamp: str
    risk_amount: float

    @property
    def value(self) -> float:
        return abs(self.size * self.entry_price)


class Portfolio:
    """Suivi de l'état du portefeuille avec protections Aladdin."""

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.positions: dict[str, Position] = {}
        self.peak_capital = self.config.capital
        self.current_capital = self.config.capital
        self._pnl_history: list[float] = [self.config.capital]

        # Capital protection : tracking des pertes consécutives
        self._consecutive_losses: int = 0
        self._last_trade_results: list[bool] = []  # True = win, False = loss
        self._cooldown_until: datetime | None = None

    def add_position(self, pos: Position):
        self.positions[pos.symbol] = pos

    def remove_position(self, symbol: str):
        self.positions.pop(symbol, None)

    def record_trade_result(self, is_win: bool):
        """Enregistre le résultat d'un trade pour le capital protection."""
        self._last_trade_results.append(is_win)
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.config.consecutive_loss_limit:
                self._cooldown_until = (
                    datetime.now() + timedelta(hours=self.config.cooldown_hours)
                )

    @property
    def is_in_cooldown(self) -> bool:
        if self._cooldown_until is None:
            return False
        if datetime.now() >= self._cooldown_until:
            # Cooldown expiré — reset
            self._cooldown_until = None
            self._consecutive_losses = 0
            return False
        return True

    @property
    def cooldown_remaining(self) -> timedelta | None:
        if self._cooldown_until is None or datetime.now() >= self._cooldown_until:
            return None
        return self._cooldown_until - datetime.now()

    def update_capital(self, new_capital: float):
        self.current_capital = new_capital
        self._pnl_history.append(new_capital)
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

    @property
    def total_exposure(self) -> float:
        return sum(p.value for p in self.positions.values())

    @property
    def total_risk(self) -> float:
        return sum(p.risk_amount for p in self.positions.values())

    @property
    def drawdown(self) -> float:
        if self.peak_capital == 0:
            return 0.0
        return max(0.0, self.peak_capital - self.current_capital)

    @property
    def drawdown_pct(self) -> float:
        if self.peak_capital == 0:
            return 0.0
        return self.drawdown / self.peak_capital

    def exposure_by_class(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for p in self.positions.values():
            result[p.asset_class] = result.get(p.asset_class, 0.0) + p.value
        return result

    def risk_by_class(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for p in self.positions.values():
            result[p.asset_class] = result.get(p.asset_class, 0.0) + p.risk_amount
        return result

    def positions_by_class(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for p in self.positions.values():
            result[p.asset_class] = result.get(p.asset_class, 0) + 1
        return result


# ─── Portefeuille global ─────────────────────────────────────────────

_portfolio = Portfolio()


def get_portfolio() -> Portfolio:
    return _portfolio


def set_portfolio(portfolio: Portfolio):
    global _portfolio
    _portfolio = portfolio


# ═══════════════════════════════════════════════════════════════════════
#  DRAWDOWN PROGRESSIF
# ═══════════════════════════════════════════════════════════════════════

class DrawdownAction(Enum):
    NONE = "NONE"
    REDUCE_25 = "REDUCE_25"     # -5% DD → couper 25%
    REDUCE_50 = "REDUCE_50"     # -8% DD → couper 50%
    CLOSE_ALL = "CLOSE_ALL"     # -10% DD → tout fermer, arrêt système


def evaluate_drawdown(config: RiskConfig | None = None,
                      portfolio: Portfolio | None = None) -> DrawdownAction:
    """Évalue le drawdown et retourne l'action requise."""
    cfg = config or DEFAULT_CONFIG
    pf = portfolio or _portfolio
    dd = pf.drawdown_pct

    if dd >= cfg.drawdown_level_3:
        return DrawdownAction.CLOSE_ALL
    elif dd >= cfg.drawdown_level_2:
        return DrawdownAction.REDUCE_50
    elif dd >= cfg.drawdown_level_1:
        return DrawdownAction.REDUCE_25
    return DrawdownAction.NONE


def positions_to_reduce(action: DrawdownAction,
                        portfolio: Portfolio | None = None) -> list[tuple[str, float]]:
    """Retourne les positions à réduire avec le facteur de réduction.

    Returns:
        [(symbol, reduction_factor), ...] — ex: ("AAPL", 0.25) = réduire de 25%
    """
    pf = portfolio or _portfolio
    if action == DrawdownAction.NONE:
        return []

    if action == DrawdownAction.CLOSE_ALL:
        return [(sym, 1.0) for sym in pf.positions]

    factor = 0.25 if action == DrawdownAction.REDUCE_25 else 0.50
    return [(sym, factor) for sym in pf.positions]


def stress_reduce_positions(portfolio: Portfolio | None = None) -> list[tuple[str, float]]:
    """En régime STRESS : réduction de 50% de toutes les positions."""
    pf = portfolio or _portfolio
    return [(sym, 0.50) for sym in pf.positions]


# ═══════════════════════════════════════════════════════════════════════
#  CORRÉLATION & DIVERSIFICATION
# ═══════════════════════════════════════════════════════════════════════

def correlation_matrix(symbols: list[str] | None = None,
                       days: int = 60) -> pd.DataFrame:
    """Matrice de corrélation des rendements quotidiens."""
    symbols = symbols or list(ASSETS.keys())
    closes = {}
    for sym in symbols:
        df = get_history(sym, days=days)
        if not df.empty and len(df) > 10:
            closes[sym] = df.set_index("date")["close"]

    if len(closes) < 2:
        return pd.DataFrame()

    prices = pd.DataFrame(closes)
    returns = prices.pct_change().dropna()
    return returns.corr()


def find_correlated_pairs(threshold: float = 0.7,
                          days: int = 60) -> list[tuple[str, str, float]]:
    """Paires d'actifs avec |corrélation| >= threshold."""
    corr = correlation_matrix(days=days)
    if corr.empty:
        return []

    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append((cols[i], cols[j], round(float(val), 3)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def _check_diversification(symbol: str, config: RiskConfig,
                           portfolio: Portfolio) -> list[str]:
    """Vérifie les règles de diversification forcée. Retourne les rejections."""
    rejections = []
    asset_class = ASSETS.get(symbol, "unknown")

    # Règle : max 3 actifs de la même classe
    class_count = portfolio.positions_by_class().get(asset_class, 0)
    if class_count >= config.max_assets_per_class:
        rejections.append(
            f"Diversification: déjà {class_count}/{config.max_assets_per_class} "
            f"positions en {asset_class}")

    # Règle : jamais 2 actifs corrélés à plus de 0.7 simultanément
    if portfolio.positions:
        existing_symbols = list(portfolio.positions.keys())
        check_symbols = existing_symbols + [symbol]
        corr = correlation_matrix(symbols=check_symbols, days=config.correlation_window)
        if not corr.empty and symbol in corr.columns:
            for existing_sym in existing_symbols:
                if existing_sym in corr.columns:
                    rho = abs(float(corr.loc[symbol, existing_sym]))
                    if rho >= config.correlation_threshold:
                        rejections.append(
                            f"Corrélation: {symbol} ↔ {existing_sym} "
                            f"(ρ={rho:.2f} ≥ {config.correlation_threshold})")

    return rejections


# ═══════════════════════════════════════════════════════════════════════
#  STOP-LOSS / TAKE-PROFIT
# ═══════════════════════════════════════════════════════════════════════

def compute_levels(symbol: str, action: str,
                   entry_price: float | None = None,
                   config: RiskConfig | None = None) -> dict:
    """Calcule SL/TP dynamiques basés sur l'ATR."""
    cfg = config or DEFAULT_CONFIG
    df = get_history(symbol, days=120)
    if df.empty or len(df) < cfg.atr_period + 1:
        return {}

    atr_val = float(calc_atr(df["high"], df["low"], df["close"], cfg.atr_period).iloc[-1])
    if np.isnan(atr_val) or atr_val == 0:
        return {}

    entry = entry_price or float(df["close"].iloc[-1])

    if action == "BUY":
        sl = entry - cfg.atr_sl_multiplier * atr_val
        tp = entry + cfg.atr_tp_multiplier * atr_val
    else:
        sl = entry + cfg.atr_sl_multiplier * atr_val
        tp = entry - cfg.atr_tp_multiplier * atr_val

    return {
        "entry": round(entry, 6),
        "stop_loss": round(sl, 6),
        "take_profit": round(tp, 6),
        "atr": round(atr_val, 6),
        "risk_per_unit": round(abs(entry - sl), 6),
    }


def update_trailing_stop(symbol: str, current_price: float,
                         config: RiskConfig | None = None,
                         portfolio: Portfolio | None = None) -> dict | None:
    """Trailing stop ATR-based (ne recule jamais)."""
    cfg = config or DEFAULT_CONFIG
    pf = portfolio or _portfolio
    pos = pf.positions.get(symbol)
    if pos is None:
        return None

    df = get_history(symbol, days=30)
    if df.empty or len(df) < cfg.atr_period + 1:
        return None

    atr_val = float(calc_atr(df["high"], df["low"], df["close"], cfg.atr_period).iloc[-1])
    if np.isnan(atr_val):
        return None

    old_sl = pos.stop_loss

    if pos.action == "BUY":
        new_sl = current_price - cfg.atr_sl_multiplier * atr_val
        new_sl = max(new_sl, old_sl)
        triggered = current_price <= new_sl
    else:
        new_sl = current_price + cfg.atr_sl_multiplier * atr_val
        new_sl = min(new_sl, old_sl)
        triggered = current_price >= new_sl

    pos.stop_loss = round(new_sl, 6)

    return {
        "old_sl": round(old_sl, 6),
        "new_sl": round(new_sl, 6),
        "triggered": triggered,
        "atr": round(atr_val, 6),
    }


# ═══════════════════════════════════════════════════════════════════════
#  POSITION SIZING (adaptatif au régime)
# ═══════════════════════════════════════════════════════════════════════

def _apply_regime_to_size(pos: PositionSize, regime: RegimeState,
                          config: RiskConfig) -> PositionSize:
    """Ajuste une PositionSize selon le régime."""
    m = _REGIME_MULTIPLIERS[regime.regime]
    factor = m["sizing"]

    pos.size = round(pos.size * factor, 6)
    pos.value = round(pos.value * factor, 2)
    pos.risk_amount = round(pos.risk_amount * factor, 2)
    pos.risk_pct = round(pos.risk_amount / config.capital, 4) if config.capital else 0
    pos.regime = regime.regime.value
    pos.details["regime_factor"] = factor
    return pos


def _resolve_capital(cfg: RiskConfig) -> float:
    """Capital dynamique composé : utilise le capital réel du portefeuille."""
    pf = _portfolio
    if pf.current_capital > 0 and pf.current_capital != cfg.capital:
        return pf.current_capital
    return cfg.capital


def size_fixed(signal: Signal, config: RiskConfig | None = None,
               regime: RegimeState | None = None) -> PositionSize | None:
    """Position sizing à risque fixe, ajusté au régime."""
    cfg = config or DEFAULT_CONFIG
    capital = _resolve_capital(cfg)
    action = signal.action.value
    if action == "HOLD":
        return None

    levels = compute_levels(signal.symbol, action, config=cfg)
    if not levels or levels["risk_per_unit"] == 0:
        return None

    risk_amount = capital * cfg.risk_per_trade
    max_risk = capital * cfg.max_risk_per_trade
    risk_amount = min(risk_amount, max_risk)

    size = risk_amount / levels["risk_per_unit"]
    value = size * levels["entry"]

    max_value = capital * cfg.max_exposure_per_asset
    if value > max_value:
        size = max_value / levels["entry"]
        value = max_value
        risk_amount = size * levels["risk_per_unit"]

    pos = PositionSize(
        symbol=signal.symbol, action=action,
        entry_price=levels["entry"], stop_loss=levels["stop_loss"],
        take_profit=levels["take_profit"],
        size=round(size, 6), value=round(value, 2),
        risk_amount=round(risk_amount, 2),
        risk_pct=round(risk_amount / capital, 4),
        method="fixed", details={"atr": levels["atr"]},
    )
    if regime:
        pos = _apply_regime_to_size(pos, regime, cfg)
    return pos


def size_kelly(signal: Signal, win_rate: float = 0.55, avg_win_loss: float = 1.5,
               config: RiskConfig | None = None,
               regime: RegimeState | None = None) -> PositionSize | None:
    """Kelly fractionnaire, ajusté au régime."""
    cfg = config or DEFAULT_CONFIG
    capital = _resolve_capital(cfg)
    action = signal.action.value
    if action == "HOLD":
        return None

    levels = compute_levels(signal.symbol, action, config=cfg)
    if not levels or levels["risk_per_unit"] == 0:
        return None

    kelly = win_rate - (1 - win_rate) / avg_win_loss
    kelly = max(0.0, kelly) * cfg.kelly_fraction
    kelly *= (0.5 + 0.5 * signal.confidence)

    risk_amount = capital * min(kelly, cfg.max_risk_per_trade)
    size = risk_amount / levels["risk_per_unit"]
    value = size * levels["entry"]

    max_value = capital * cfg.max_exposure_per_asset
    if value > max_value:
        size = max_value / levels["entry"]
        value = max_value
        risk_amount = size * levels["risk_per_unit"]

    pos = PositionSize(
        symbol=signal.symbol, action=action,
        entry_price=levels["entry"], stop_loss=levels["stop_loss"],
        take_profit=levels["take_profit"],
        size=round(size, 6), value=round(value, 2),
        risk_amount=round(risk_amount, 2),
        risk_pct=round(risk_amount / capital, 4),
        method="kelly",
        details={"kelly_raw": round(kelly, 4), "atr": levels["atr"],
                 "win_rate": win_rate, "avg_win_loss": avg_win_loss},
    )
    if regime:
        pos = _apply_regime_to_size(pos, regime, cfg)
    return pos


def size_volatility(signal: Signal, target_vol: float = 0.10,
                    config: RiskConfig | None = None,
                    regime: RegimeState | None = None) -> PositionSize | None:
    """Volatility targeting, ajusté au régime."""
    cfg = config or DEFAULT_CONFIG
    capital = _resolve_capital(cfg)
    action = signal.action.value
    if action == "HOLD":
        return None

    df = get_history(signal.symbol, days=120)
    if df.empty or len(df) < 30:
        return None

    levels = compute_levels(signal.symbol, action, config=cfg)
    if not levels or levels["risk_per_unit"] == 0:
        return None

    returns = df["close"].pct_change().dropna()
    annual_vol = float(returns.std() * np.sqrt(252))
    if annual_vol == 0:
        return None

    entry = levels["entry"]
    size = (capital * target_vol) / (entry * annual_vol)
    value = size * entry

    max_value = capital * cfg.max_exposure_per_asset
    if value > max_value:
        size = max_value / entry
        value = max_value

    risk_amount = size * levels["risk_per_unit"]
    if risk_amount > capital * cfg.max_risk_per_trade:
        risk_amount = capital * cfg.max_risk_per_trade
        size = risk_amount / levels["risk_per_unit"]
        value = size * entry

    pos = PositionSize(
        symbol=signal.symbol, action=action,
        entry_price=levels["entry"], stop_loss=levels["stop_loss"],
        take_profit=levels["take_profit"],
        size=round(size, 6), value=round(value, 2),
        risk_amount=round(risk_amount, 2),
        risk_pct=round(risk_amount / capital, 4),
        method="volatility",
        details={"annual_vol": round(annual_vol, 4), "target_vol": target_vol,
                 "atr": levels["atr"]},
    )
    if regime:
        pos = _apply_regime_to_size(pos, regime, cfg)
    return pos


# ═══════════════════════════════════════════════════════════════════════
#  CONTRÔLE DE RISQUE PRE-TRADE (Aladdin gate)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RiskCheck:
    approved: bool
    position: PositionSize | None
    regime: str = "EXPANSION"
    drawdown_action: str = "NONE"
    warnings: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)

    def __repr__(self):
        status = "APPROVED" if self.approved else "REJECTED"
        sym = self.position.symbol if self.position else "?"
        lines = [f"[{status}] {sym}  regime={self.regime}  dd_action={self.drawdown_action}"]
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        for r in self.rejections:
            lines.append(f"  ✗ {r}")
        return "\n".join(lines)


def check_risk(signal: Signal, method: str = "fixed",
               config: RiskConfig | None = None,
               portfolio: Portfolio | None = None,
               regime: RegimeState | None = None) -> RiskCheck:
    """Contrôle de risque complet Aladdin avant ouverture de position.

    Vérifie dans l'ordre :
    1. Signal non-HOLD
    2. Capital protection (cooldown après 3 pertes)
    3. Régime de marché (STRESS → aucun trade)
    4. CONTRACTION → préférence défensifs
    5. Drawdown progressif
    6. Position sizing
    7. Position déjà ouverte
    8. Nombre max de positions
    9. Risque total portefeuille
    10. Exposition par classe
    11. Diversification forcée (max 3/classe, corrélation < 0.7)
    12. Exposition corrélée
    """
    cfg = config or DEFAULT_CONFIG
    pf = portfolio or _portfolio
    reg = regime or detect_regime()
    warnings: list[str] = []
    rejections: list[str] = []

    # Appliquer le régime à la config
    eff_cfg = effective_config(cfg, reg)

    # ── 1. Signal HOLD ──
    if signal.action == Action.HOLD:
        return RiskCheck(approved=False, position=None,
                         regime=reg.regime.value, drawdown_action="NONE",
                         rejections=["Signal HOLD — pas d'action"])

    # ── 2. Capital protection : cooldown ──
    if pf.is_in_cooldown:
        remaining = pf.cooldown_remaining
        return RiskCheck(
            approved=False, position=None,
            regime=reg.regime.value, drawdown_action="NONE",
            rejections=[f"CAPITAL PROTECTION: pause après {cfg.consecutive_loss_limit} "
                        f"pertes consécutives (reste {remaining})"])

    # ── 3. Watch-only stocks ──
    if signal.symbol in WATCH_ONLY:
        return RiskCheck(
            approved=False, position=None,
            regime=reg.regime.value, drawdown_action="NONE",
            rejections=[f"WATCH-ONLY: {signal.symbol} en surveillance uniquement"])

    # ── 4. Filtrage par régime ──
    if reg.regime == Regime.STRESS:
        if signal.symbol not in STRESS_ALLOWED:
            return RiskCheck(
                approved=False, position=None,
                regime=reg.regime.value, drawdown_action="NONE",
                rejections=[f"STRESS: seuls {', '.join(sorted(STRESS_ALLOWED))} autorisés"])

    if reg.regime == Regime.CONTRACTION:
        if signal.symbol not in CONTRACTION_ALLOWED:
            return RiskCheck(
                approved=False, position=None,
                regime=reg.regime.value, drawdown_action="NONE",
                rejections=[f"CONTRACTION: {signal.symbol} non autorisé en contraction"])

    # ── 5. Drawdown progressif ──
    dd_action = evaluate_drawdown(cfg, pf)
    if dd_action == DrawdownAction.CLOSE_ALL:
        return RiskCheck(
            approved=False, position=None,
            regime=reg.regime.value, drawdown_action=dd_action.value,
            rejections=[f"DRAWDOWN CRITIQUE: {pf.drawdown_pct:.1%} ≥ {cfg.drawdown_level_3:.0%} "
                        f"— FERMETURE TOTALE requise"])
    elif dd_action == DrawdownAction.REDUCE_50:
        rejections.append(
            f"DRAWDOWN SÉVÈRE: {pf.drawdown_pct:.1%} ≥ {cfg.drawdown_level_2:.0%} "
            f"— réduction 50% en cours, pas de nouveau trade")
    elif dd_action == DrawdownAction.REDUCE_25:
        warnings.append(
            f"DRAWDOWN MODÉRÉ: {pf.drawdown_pct:.1%} ≥ {cfg.drawdown_level_1:.0%} "
            f"— réduction 25% active")

    # ── 6. Position sizing (avec régime) ──
    if method == "kelly":
        pos = size_kelly(signal, config=eff_cfg, regime=reg)
    elif method == "volatility":
        pos = size_volatility(signal, config=eff_cfg, regime=reg)
    else:
        pos = size_fixed(signal, config=eff_cfg, regime=reg)

    if pos is None:
        return RiskCheck(approved=False, position=None,
                         regime=reg.regime.value, drawdown_action=dd_action.value,
                         rejections=["Impossible de calculer la taille de position"])

    # ── 7. Position déjà ouverte ──
    if signal.symbol in pf.positions:
        return RiskCheck(approved=False, position=pos,
                         regime=reg.regime.value, drawdown_action=dd_action.value,
                         rejections=[f"Position déjà ouverte sur {signal.symbol}"])

    # ── 8. Nombre max de positions ──
    if len(pf.positions) >= eff_cfg.max_positions:
        rejections.append(f"Limite de {eff_cfg.max_positions} positions atteinte")

    # ── 9. Risque total portefeuille ──
    new_total_risk = pf.total_risk + pos.risk_amount
    if new_total_risk > eff_cfg.capital * eff_cfg.max_portfolio_risk:
        rejections.append(
            f"Risque total {new_total_risk/eff_cfg.capital:.1%} "
            f"dépasserait {eff_cfg.max_portfolio_risk:.1%}")

    # ── 10. Exposition par classe ──
    asset_class = ASSETS.get(signal.symbol, "unknown")
    class_exposure = pf.exposure_by_class().get(asset_class, 0.0) + pos.value
    if class_exposure > eff_cfg.capital * eff_cfg.max_exposure_per_class:
        rejections.append(
            f"Exposition {asset_class} ({class_exposure/eff_cfg.capital:.0%}) "
            f"> limite {eff_cfg.max_exposure_per_class:.0%}")

    # ── 11. Diversification forcée ──
    div_rejections = _check_diversification(signal.symbol, eff_cfg, pf)
    rejections.extend(div_rejections)

    # ── 12. Exposition corrélée ──
    if pf.positions:
        corr = correlation_matrix(days=eff_cfg.correlation_window)
        if not corr.empty and signal.symbol in corr.columns:
            correlated_exposure = 0.0
            for sym, existing_pos in pf.positions.items():
                if sym in corr.columns:
                    rho = abs(float(corr.loc[signal.symbol, sym]))
                    if rho >= eff_cfg.correlation_threshold:
                        correlated_exposure += existing_pos.value
                        warnings.append(f"Corrélation élevée avec {sym} (ρ={rho:.2f})")
            if correlated_exposure + pos.value > eff_cfg.capital * eff_cfg.max_correlated_exposure:
                rejections.append(
                    f"Exposition corrélée ({(correlated_exposure + pos.value)/eff_cfg.capital:.0%}) "
                    f"> limite {eff_cfg.max_correlated_exposure:.0%}")

    # ── Warnings ──
    if pos.risk_pct > eff_cfg.risk_per_trade:
        warnings.append(f"Risque ({pos.risk_pct:.2%}) > cible ({eff_cfg.risk_per_trade:.2%})")
    if signal.confidence < 0.3:
        warnings.append(f"Confiance faible ({signal.confidence:.0%})")
    if pf._consecutive_losses >= cfg.consecutive_loss_limit - 1:
        warnings.append(f"Attention: {pf._consecutive_losses} pertes consécutives "
                        f"(pause à {cfg.consecutive_loss_limit})")

    approved = len(rejections) == 0
    return RiskCheck(approved=approved, position=pos,
                     regime=reg.regime.value, drawdown_action=dd_action.value,
                     warnings=warnings, rejections=rejections)


# ═══════════════════════════════════════════════════════════════════════
#  RAPPORT DE RISQUE PORTEFEUILLE
# ═══════════════════════════════════════════════════════════════════════

def portfolio_report(config: RiskConfig | None = None,
                     portfolio: Portfolio | None = None,
                     regime: RegimeState | None = None) -> PortfolioRisk:
    """Rapport de risque complet du portefeuille."""
    cfg = config or DEFAULT_CONFIG
    pf = portfolio or _portfolio
    reg = regime or detect_regime()
    warnings = []

    exposure = pf.total_exposure
    risk = pf.total_risk
    exp_pct = exposure / cfg.capital if cfg.capital else 0
    risk_pct = risk / cfg.capital if cfg.capital else 0
    dd_action = evaluate_drawdown(cfg, pf)

    # Alertes drawdown
    if dd_action != DrawdownAction.NONE:
        warnings.append(f"DRAWDOWN: {pf.drawdown_pct:.1%} → action={dd_action.value}")

    # Alertes régime
    if reg.regime == Regime.STRESS:
        warnings.append(f"RÉGIME STRESS: réduction 50% de toutes les positions requise")
    elif reg.regime == Regime.CONTRACTION:
        warnings.append(f"RÉGIME CONTRACTION: sizing -30%, préférence actifs défensifs")

    # Alerte cooldown
    if pf.is_in_cooldown:
        warnings.append(f"CAPITAL PROTECTION: en pause ({pf.cooldown_remaining})")

    # Alertes classiques
    if risk_pct > cfg.max_portfolio_risk * 0.8:
        warnings.append(f"RISK: portefeuille à {risk_pct:.1%}")

    for cls, val in pf.exposure_by_class().items():
        pct = val / cfg.capital
        if pct > cfg.max_exposure_per_class * 0.8:
            warnings.append(f"EXPOSURE: {cls} à {pct:.0%}")

    for cls, count in pf.positions_by_class().items():
        if count >= cfg.max_assets_per_class:
            warnings.append(f"DIVERSIFICATION: {count}/{cfg.max_assets_per_class} en {cls}")

    if len(pf.positions) >= cfg.max_positions - 1:
        warnings.append(f"CAPACITY: {len(pf.positions)}/{cfg.max_positions} positions")

    return PortfolioRisk(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        regime=reg.regime.value,
        capital=pf.current_capital,
        total_exposure=round(exposure, 2),
        total_risk=round(risk, 2),
        exposure_pct=round(exp_pct, 4),
        risk_pct=round(risk_pct, 4),
        positions_count=len(pf.positions),
        drawdown=round(pf.drawdown, 2),
        drawdown_pct=round(pf.drawdown_pct, 4),
        drawdown_action=dd_action.value,
        peak_capital=pf.peak_capital,
        exposure_by_class={k: round(v, 2) for k, v in pf.exposure_by_class().items()},
        risk_by_class={k: round(v, 2) for k, v in pf.risk_by_class().items()},
        positions_by_class=pf.positions_by_class(),
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════════
#  EXÉCUTION DIRECTE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from signals.engine import analyze, scan_all

    print("=" * 70)
    print("  Orion Risk Manager — Aladdin Mode")
    print("=" * 70)

    # Détection du régime
    reg = detect_regime()
    print(f"\n  {reg}")

    eff = effective_config(DEFAULT_CONFIG, reg)
    print(f"  Config effective: risk/trade={eff.risk_per_trade:.2%}, "
          f"max_exposure/class={eff.max_exposure_per_class:.0%}")

    # Drawdown
    dd = evaluate_drawdown()
    print(f"  Drawdown action: {dd.value}")

    signals = scan_all()
    if not signals:
        print("\n  Aucun signal. Lance d'abord : python -m data.collector")
    else:
        print(f"\n  {len(signals)} actifs analysés\n")
        for sig in signals[:10]:
            rc = check_risk(sig, regime=reg)
            print(f"  {rc}")
            if rc.position:
                print(f"    → {rc.position}")
            print()

        # Corrélations
        pairs = find_correlated_pairs(threshold=0.7)
        if pairs:
            print("  === Paires corrélées (|ρ| ≥ 0.7) ===")
            for a, b, rho in pairs[:10]:
                print(f"    {a:<12} ↔ {b:<12}  ρ = {rho:+.3f}")
