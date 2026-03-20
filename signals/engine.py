"""
Orion — Moteur de Signaux
--------------------------
Analyse les données de marché et génère des signaux BUY / SELL / HOLD
en combinant plusieurs indicateurs techniques avec un système de score pondéré.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

from signals import indicators as ind
from data.collector import get_history, ASSETS


# ─── Types ───────────────────────────────────────────────────────────

class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    symbol: str
    action: Action
    score: float            # -100 (fort SELL) à +100 (fort BUY)
    confidence: float       # 0.0 à 1.0
    timestamp: str
    asset_class: str
    details: dict = field(default_factory=dict)

    def __repr__(self):
        arrow = {"BUY": "▲", "SELL": "▼", "HOLD": "●"}[self.action.value]
        return (f"{arrow} {self.symbol:<12} {self.action.value:>4}  "
                f"score={self.score:+6.1f}  conf={self.confidence:.0%}")


# ─── Seuils de signal ────────────────────────────────────────────────

BUY_THRESHOLD = 12
SELL_THRESHOLD = -12


def _check_multi_timeframe(close: pd.Series, action: Action) -> bool:
    """Confirmation multi-timeframe : vérifie l'alignement 1j / 5j / 20j."""
    if len(close) < 20:
        return False

    last = close.iloc[-1]
    sma5 = close.rolling(5).mean().iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]

    if np.isnan(sma5) or np.isnan(sma20):
        return False

    if action == Action.BUY:
        return last > sma5 and last > sma20
    elif action == Action.SELL:
        return last < sma5 and last < sma20
    return False

# ─── Paramètres par classe d'actif ──────────────────────────────────

DEFAULT_PARAMS = {
    "rsi_period":        14,
    "rsi_oversold":      30,
    "rsi_overbought":    70,
    "macd_fast":         12,
    "macd_slow":         26,
    "macd_signal":       9,
    "bb_period":         20,
    "bb_std":            2.0,
    "sma_fast":          20,
    "sma_slow":          50,
    "sma_trend":         200,
    "stoch_k":           14,
    "stoch_d":           3,
    "atr_period":        14,
    "cci_period":        20,
    "mfi_period":        14,
    "lookback_days":     365,
}

ASSET_CLASS_OVERRIDES = {
    "forex": {"rsi_oversold": 35, "rsi_overbought": 65, "lookback_days": 250},
    "crypto": {"rsi_oversold": 25, "rsi_overbought": 75, "lookback_days": 365},
}

# Poids de chaque sous-signal dans le score final
WEIGHTS = {
    "trend":     0.25,
    "momentum":  0.25,
    "volatility": 0.15,
    "volume":    0.15,
    "pattern":   0.20,
}


def _params_for(asset_class: str) -> dict:
    p = DEFAULT_PARAMS.copy()
    p.update(ASSET_CLASS_OVERRIDES.get(asset_class, {}))
    return p


# ─── Sous-analyseurs ────────────────────────────────────────────────
# Chaque fonction retourne un score entre -100 et +100.

def _score_trend(df: pd.DataFrame, p: dict) -> tuple[float, dict]:
    """Score de tendance (SMA, EMA, Ichimoku)."""
    close = df["close"]
    last = close.iloc[-1]
    details = {}

    scores = []

    # SMA crossover
    sma_f = ind.sma(close, p["sma_fast"]).iloc[-1]
    sma_s = ind.sma(close, p["sma_slow"]).iloc[-1]
    sma_t = ind.sma(close, p["sma_trend"]).iloc[-1]

    if not np.isnan(sma_f) and not np.isnan(sma_s):
        if sma_f > sma_s:
            scores.append(50)
            details["sma_cross"] = "bullish"
        else:
            scores.append(-50)
            details["sma_cross"] = "bearish"

    # Prix vs SMA 200
    if not np.isnan(sma_t):
        if last > sma_t:
            scores.append(40)
            details["sma200"] = "above"
        else:
            scores.append(-40)
            details["sma200"] = "below"

    # MACD
    m = ind.macd(close, p["macd_fast"], p["macd_slow"], p["macd_signal"])
    hist = m["histogram"].iloc[-1]
    hist_prev = m["histogram"].iloc[-2] if len(m) > 1 else 0
    if not np.isnan(hist):
        if hist > 0 and hist > hist_prev:
            scores.append(60)
        elif hist > 0:
            scores.append(20)
        elif hist < 0 and hist < hist_prev:
            scores.append(-60)
        else:
            scores.append(-20)
        details["macd_hist"] = round(float(hist), 4)

    return (np.mean(scores) if scores else 0.0, details)


def _score_momentum(df: pd.DataFrame, p: dict) -> tuple[float, dict]:
    """Score momentum (RSI, Stochastic, CCI, ROC)."""
    close, high, low = df["close"], df["high"], df["low"]
    details = {}
    scores = []

    # RSI
    r = ind.rsi(close, p["rsi_period"]).iloc[-1]
    if not np.isnan(r):
        if r < p["rsi_oversold"]:
            scores.append(70)     # oversold → signal achat
            details["rsi_zone"] = "oversold"
        elif r > p["rsi_overbought"]:
            scores.append(-70)    # overbought → signal vente
            details["rsi_zone"] = "overbought"
        else:
            scores.append((50 - r) * 1.4)  # linéaire centré
            details["rsi_zone"] = "neutral"
        details["rsi"] = round(float(r), 2)

    # Stochastic
    stoch = ind.stochastic(high, low, close, p["stoch_k"], p["stoch_d"])
    k_val = stoch["k"].iloc[-1]
    d_val = stoch["d"].iloc[-1]
    if not np.isnan(k_val):
        if k_val < 20 and k_val > d_val:
            scores.append(60)
        elif k_val > 80 and k_val < d_val:
            scores.append(-60)
        else:
            scores.append((50 - k_val) * 0.8)
        details["stoch_k"] = round(float(k_val), 2)

    # CCI
    c = ind.cci(high, low, close, p["cci_period"]).iloc[-1]
    if not np.isnan(c):
        if c < -100:
            scores.append(50)
        elif c > 100:
            scores.append(-50)
        else:
            scores.append(-c * 0.5)
        details["cci"] = round(float(c), 2)

    # ROC
    r12 = ind.roc(close, 12).iloc[-1]
    if not np.isnan(r12):
        scores.append(np.clip(r12 * 5, -60, 60))
        details["roc_12"] = round(float(r12), 2)

    return (np.mean(scores) if scores else 0.0, details)


def _score_volatility(df: pd.DataFrame, p: dict) -> tuple[float, dict]:
    """Score volatilité (Bollinger, ATR, Keltner)."""
    close, high, low = df["close"], df["high"], df["low"]
    details = {}
    scores = []

    # Bollinger %B
    bb = ind.bollinger(close, p["bb_period"], p["bb_std"])
    pct_b = bb["pct_b"].iloc[-1]
    bw = bb["bandwidth"].iloc[-1]
    if not np.isnan(pct_b):
        if pct_b < 0.0:
            scores.append(60)    # sous la bande basse
            details["bb_zone"] = "below_lower"
        elif pct_b > 1.0:
            scores.append(-60)   # au-dessus de la bande haute
            details["bb_zone"] = "above_upper"
        else:
            scores.append((0.5 - pct_b) * 80)
            details["bb_zone"] = "inside"
        details["bb_pct_b"] = round(float(pct_b), 4)

    # Squeeze detection (Bollinger inside Keltner)
    kelt = ind.keltner(high, low, close)
    bb_upper = bb["upper"].iloc[-1]
    bb_lower = bb["lower"].iloc[-1]
    k_upper = kelt["upper"].iloc[-1]
    k_lower = kelt["lower"].iloc[-1]
    if not any(np.isnan(x) for x in [bb_upper, bb_lower, k_upper, k_lower]):
        squeeze = bb_lower > k_lower and bb_upper < k_upper
        details["squeeze"] = squeeze
        if squeeze:
            scores.append(30)  # compression → breakout imminent

    # ATR relative (volatilité courante vs moyenne)
    a = ind.atr(high, low, close, p["atr_period"])
    if len(a.dropna()) > 50:
        atr_now = a.iloc[-1]
        atr_avg = a.iloc[-50:].mean()
        if not np.isnan(atr_now) and atr_avg > 0:
            ratio = atr_now / atr_avg
            details["atr_ratio"] = round(float(ratio), 2)

    return (np.mean(scores) if scores else 0.0, details)


def _score_volume(df: pd.DataFrame, p: dict) -> tuple[float, dict]:
    """Score volume (OBV, MFI). Retourne 0 si pas de volume."""
    close, high, low = df["close"], df["high"], df["low"]
    volume = df["volume"]
    details = {}
    scores = []

    if volume.sum() == 0 or volume.isna().all():
        details["volume"] = "unavailable"
        return (0.0, details)

    # OBV trend
    o = ind.obv(close, volume)
    if len(o.dropna()) > 20:
        obv_sma = o.rolling(20).mean()
        if o.iloc[-1] > obv_sma.iloc[-1]:
            scores.append(40)
            details["obv_trend"] = "bullish"
        else:
            scores.append(-40)
            details["obv_trend"] = "bearish"

    # MFI
    m = ind.mfi(high, low, close, volume, p["mfi_period"]).iloc[-1]
    if not np.isnan(m):
        if m < 20:
            scores.append(60)
            details["mfi_zone"] = "oversold"
        elif m > 80:
            scores.append(-60)
            details["mfi_zone"] = "overbought"
        else:
            scores.append((50 - m) * 1.0)
            details["mfi_zone"] = "neutral"
        details["mfi"] = round(float(m), 2)

    return (np.mean(scores) if scores else 0.0, details)


def _score_pattern(df: pd.DataFrame, p: dict) -> tuple[float, dict]:
    """Score patterns de prix (higher highs/lows, key levels)."""
    close, high, low = df["close"], df["high"], df["low"]
    details = {}
    scores = []

    # Higher highs / lower lows sur 10 barres
    recent_high = high.iloc[-10:]
    recent_low = low.iloc[-10:]

    hh = (recent_high.iloc[-1] > recent_high.iloc[-5]) and (recent_high.iloc[-5] > recent_high.iloc[0])
    ll = (recent_low.iloc[-1] < recent_low.iloc[-5]) and (recent_low.iloc[-5] < recent_low.iloc[0])
    hl = (recent_low.iloc[-1] > recent_low.iloc[-5]) and (recent_low.iloc[-5] > recent_low.iloc[0])
    lh = (recent_high.iloc[-1] < recent_high.iloc[-5]) and (recent_high.iloc[-5] < recent_high.iloc[0])

    if hh and hl:
        scores.append(60)
        details["structure"] = "uptrend"
    elif ll and lh:
        scores.append(-60)
        details["structure"] = "downtrend"
    else:
        scores.append(0)
        details["structure"] = "range"

    # Distance au plus haut / plus bas 52 semaines
    if len(close) >= 252:
        high_52w = high.iloc[-252:].max()
        low_52w = low.iloc[-252:].min()
        rng = high_52w - low_52w
        if rng > 0:
            pos = (close.iloc[-1] - low_52w) / rng
            details["52w_position"] = round(float(pos), 2)
            # Near 52w low = potential bounce, near high = potential exhaustion
            if pos < 0.1:
                scores.append(40)
            elif pos > 0.9:
                scores.append(-30)
            else:
                scores.append((0.5 - pos) * 40)

    return (np.mean(scores) if scores else 0.0, details)


# ─── Moteur principal ────────────────────────────────────────────────

def analyze(symbol: str, lookback_days: int | None = None) -> Signal | None:
    """Analyse un actif et retourne un Signal."""
    asset_class = ASSETS.get(symbol)
    if asset_class is None:
        return None

    p = _params_for(asset_class)
    days = lookback_days or p["lookback_days"]
    df = get_history(symbol, days=days)

    if df.empty or len(df) < 50:
        return None

    # Calcul des sous-scores
    trend_score, trend_det       = _score_trend(df, p)
    momentum_score, momentum_det = _score_momentum(df, p)
    vol_score, vol_det           = _score_volatility(df, p)
    volume_score, volume_det     = _score_volume(df, p)
    pattern_score, pattern_det   = _score_pattern(df, p)

    # Score composite pondéré
    raw_score = (
        WEIGHTS["trend"]      * trend_score
        + WEIGHTS["momentum"] * momentum_score
        + WEIGHTS["volatility"] * vol_score
        + WEIGHTS["volume"]   * volume_score
        + WEIGHTS["pattern"]  * pattern_score
    )
    score = float(np.clip(raw_score, -100, 100))

    # Action
    if score >= BUY_THRESHOLD:
        action = Action.BUY
    elif score <= SELL_THRESHOLD:
        action = Action.SELL
    else:
        action = Action.HOLD

    # Confirmation multi-timeframe
    if action != Action.HOLD:
        if not _check_multi_timeframe(df["close"], action):
            action = Action.HOLD

    # Confidence = accord entre les sous-signaux
    sub = [trend_score, momentum_score, vol_score, volume_score, pattern_score]
    signs = [1 if s > 0 else (-1 if s < 0 else 0) for s in sub]
    agreement = abs(sum(signs)) / len(signs)
    confidence = round(float(np.clip(agreement * (abs(score) / 100), 0, 1)), 2)

    details = {
        "scores": {
            "trend": round(trend_score, 1),
            "momentum": round(momentum_score, 1),
            "volatility": round(vol_score, 1),
            "volume": round(volume_score, 1),
            "pattern": round(pattern_score, 1),
        },
        **trend_det, **momentum_det, **vol_det, **volume_det, **pattern_det,
    }

    return Signal(
        symbol=symbol,
        action=action,
        score=round(score, 1),
        confidence=confidence,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        asset_class=asset_class,
        details=details,
    )


def scan_all(lookback_days: int | None = None) -> list[Signal]:
    """Scanne tous les 35 actifs et retourne la liste de signaux triée par score."""
    signals = []
    for symbol in ASSETS:
        sig = analyze(symbol, lookback_days)
        if sig is not None:
            signals.append(sig)
    signals.sort(key=lambda s: s.score, reverse=True)
    return signals


def scan_class(asset_class: str, lookback_days: int | None = None) -> list[Signal]:
    """Scanne les actifs d'une classe donnée (forex, commodity, index, stock, crypto)."""
    symbols = [s for s, c in ASSETS.items() if c == asset_class]
    signals = []
    for symbol in symbols:
        sig = analyze(symbol, lookback_days)
        if sig is not None:
            signals.append(sig)
    signals.sort(key=lambda s: s.score, reverse=True)
    return signals


def top_signals(n: int = 5, lookback_days: int | None = None) -> dict[str, list[Signal]]:
    """Retourne les N meilleurs signaux BUY et SELL."""
    all_sigs = scan_all(lookback_days)
    buys = [s for s in all_sigs if s.action == Action.BUY][:n]
    sells = [s for s in all_sigs if s.action == Action.SELL]
    sells.sort(key=lambda s: s.score)
    return {"buy": buys, "sell": sells[:n]}


# ─── Exécution directe ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Orion Signal Engine — Scan complet")
    print("=" * 60)

    signals = scan_all()

    if not signals:
        print("\n  Aucun signal — la base de données est-elle initialisée ?")
        print("  Lance d'abord : python -m data.collector")
    else:
        print(f"\n  {len(signals)} actifs analysés\n")
        for sig in signals:
            print(f"  {sig}")
        print()
        top = top_signals(3)
        print("  === TOP 3 BUY ===")
        for s in top["buy"]:
            print(f"    {s}")
        print("  === TOP 3 SELL ===")
        for s in top["sell"]:
            print(f"    {s}")
