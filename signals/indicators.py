"""
Orion — Indicateurs Techniques
-------------------------------
Bibliothèque d'indicateurs calculés sur des Series pandas (colonne close).
Chaque fonction retourne une Series ou un DataFrame aligné sur l'index d'entrée.
"""

import numpy as np
import pandas as pd


# ─── Tendance ────────────────────────────────────────────────────────

def sma(close: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return close.ewm(span=period, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD, Signal, Histogramme."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }, index=close.index)


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend indicator."""
    atr = _atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    st = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)  # 1 = up, -1 = down

    for i in range(period, len(close)):
        if close.iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1 and lower.iloc[i] < lower.iloc[i - 1]:
                lower.iloc[i] = lower.iloc[i - 1]
            if direction.iloc[i] == -1 and upper.iloc[i] > upper.iloc[i - 1]:
                upper.iloc[i] = upper.iloc[i - 1]

        st.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]

    return pd.DataFrame({"supertrend": st, "direction": direction}, index=close.index)


def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
             tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> pd.DataFrame:
    """Ichimoku Cloud (Tenkan, Kijun, Senkou A/B, Chikou)."""
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_line = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    chikou = close.shift(-kijun)
    return pd.DataFrame({
        "tenkan": tenkan_sen,
        "kijun": kijun_sen,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b_line,
        "chikou": chikou,
    }, index=close.index)


# ─── Momentum ───────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator (%K, %D)."""
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"k": k, "d": d}, index=close.index)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    """Williams %R."""
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def cci(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md).replace(0, np.nan)


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change (%)."""
    prev = close.shift(period)
    return ((close - prev) / prev.replace(0, np.nan)) * 100


# ─── Volatilité ─────────────────────────────────────────────────────

def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    """Average True Range (interne)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average True Range."""
    return _atr(high, low, close, period)


def bollinger(close: pd.Series, period: int = 20,
              std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands (upper, middle, lower, %B, bandwidth)."""
    middle = sma(close, period)
    rolling_std = close.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = ((upper - lower) / middle.replace(0, np.nan)) * 100
    return pd.DataFrame({
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "pct_b": pct_b,
        "bandwidth": bandwidth,
    }, index=close.index)


def keltner(high: pd.Series, low: pd.Series, close: pd.Series,
            ema_period: int = 20, atr_period: int = 10,
            multiplier: float = 1.5) -> pd.DataFrame:
    """Keltner Channels."""
    middle = ema(close, ema_period)
    atr_val = _atr(high, low, close, atr_period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower}, index=close.index)


# ─── Volume ──────────────────────────────────────────────────────────

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series) -> pd.Series:
    """VWAP (cumulatif intraday — approximation sur daily)."""
    tp = (high + low + close) / 3
    cum_tp_vol = (tp * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index."""
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    delta = tp.diff()
    pos_mf = pd.Series(np.where(delta > 0, raw_mf, 0), index=close.index)
    neg_mf = pd.Series(np.where(delta < 0, raw_mf, 0), index=close.index)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))
