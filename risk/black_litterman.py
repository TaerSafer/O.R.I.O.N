"""
Orion — Black-Litterman Portfolio Optimization
-----------------------------------------------
Implements the standard Black-Litterman model to combine market equilibrium
returns with views derived from Orion's signal engine.

Public API:
  - compute_bl_allocation() -> dict   (full pipeline)
  - get_equilibrium_returns() -> dict  (market prior only)
  - generate_views(signals) -> tuple   (P, Q, Omega matrices)
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from data.collector import get_history
from signals.engine import Signal, Action, scan_all


# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

BL_ASSETS = [
    "EURUSD=X", "GC=F", "CL=F", "^GSPC",
    "^GDAXI", "BTC-USD", "USDCHF=X", "^FCHI",
]

MARKET_CAP_WEIGHTS = {
    "^GSPC":     0.30,
    "^GDAXI":    0.10,
    "^FCHI":     0.08,
    "GC=F":      0.12,
    "CL=F":      0.08,
    "BTC-USD":   0.10,
    "EURUSD=X":  0.12,
    "USDCHF=X":  0.10,
}

DELTA = 2.5   # Risk aversion coefficient
TAU = 0.05    # Scaling factor for uncertainty in equilibrium returns


# ═══════════════════════════════════════════════════════════════════════
#  MARKET PRIOR
# ═══════════════════════════════════════════════════════════════════════

def _get_returns_matrix(days: int = 252) -> tuple[np.ndarray | None, list[str]]:
    """Fetch daily returns for BL_ASSETS.

    Returns (returns_matrix, valid_symbols) where returns_matrix has shape
    (n_days, n_assets) or None if insufficient data.
    """
    all_returns = {}
    for sym in BL_ASSETS:
        df = get_history(sym, days=days + 20)
        if df.empty or len(df) < 30:
            continue
        rets = df["close"].pct_change().dropna()
        if len(rets) < 20:
            continue
        all_returns[sym] = rets.values

    if not all_returns:
        return None, []

    # Align lengths to the shortest series
    min_len = min(len(v) for v in all_returns.values())
    valid_symbols = [s for s in BL_ASSETS if s in all_returns]
    matrix = np.column_stack([all_returns[s][-min_len:] for s in valid_symbols])
    return matrix, valid_symbols


def _compute_covariance(returns: np.ndarray) -> np.ndarray:
    """Annualised covariance matrix from daily returns."""
    return np.cov(returns, rowvar=False) * 252


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    """Inverse with fallback to pseudo-inverse for singular matrices."""
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def get_equilibrium_returns() -> dict:
    """Compute the market equilibrium (prior) returns.

    Returns dict with keys: pi (dict sym->float), sigma (2d list),
    symbols (list), weights (dict).
    """
    returns, symbols = _get_returns_matrix(days=252)

    if returns is None or len(symbols) == 0:
        # Fallback: equal weights, zero returns
        eq = {s: 0.0 for s in BL_ASSETS}
        return {"pi": eq, "sigma": [], "symbols": BL_ASSETS, "weights": MARKET_CAP_WEIGHTS}

    sigma = _compute_covariance(returns)

    # Market-cap weight vector aligned to valid symbols
    w_mkt = np.array([MARKET_CAP_WEIGHTS.get(s, 1.0 / len(symbols)) for s in symbols])
    w_mkt = w_mkt / w_mkt.sum()  # re-normalise in case some assets are missing

    pi = DELTA * sigma @ w_mkt  # equilibrium excess returns

    pi_dict = {symbols[i]: float(pi[i]) for i in range(len(symbols))}
    return {
        "pi": pi_dict,
        "sigma": sigma.tolist(),
        "symbols": symbols,
        "weights": {s: float(w_mkt[i]) for i, s in enumerate(symbols)},
    }


# ═══════════════════════════════════════════════════════════════════════
#  VIEWS FROM SIGNALS
# ═══════════════════════════════════════════════════════════════════════

def generate_views(
    signals: list[Signal],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create Black-Litterman view matrices from Orion signals.

    Parameters
    ----------
    signals : list[Signal]
        Output of scan_all() or a filtered subset.

    Returns
    -------
    P : np.ndarray  (K x N) — pick matrix
    Q : np.ndarray  (K,)    — expected returns per view
    Omega : np.ndarray (K x K) — diagonal uncertainty matrix

    Where K = number of views, N = number of BL_ASSETS with data.
    Returns empty arrays (shape (0, N), (0,), (0, 0)) when no views exist.
    """
    # We need the covariance for Omega computation
    returns, symbols = _get_returns_matrix(days=252)
    n = len(symbols) if symbols else len(BL_ASSETS)

    if returns is None or len(symbols) == 0:
        return np.zeros((0, n)), np.zeros(0), np.zeros((0, 0))

    sigma = _compute_covariance(returns)

    # Per-asset annualised volatility (diagonal of sigma)
    asset_vols = np.sqrt(np.diag(sigma))

    # Map symbol -> index in our matrix
    sym_idx = {s: i for i, s in enumerate(symbols)}

    rows_P = []
    rows_Q = []
    rows_omega_diag = []

    for sig in signals:
        if sig.action == Action.HOLD:
            continue
        if sig.symbol not in sym_idx:
            continue

        idx = sym_idx[sig.symbol]
        vol_i = asset_vols[idx]

        # P row: 1 for this asset, 0 elsewhere
        p_row = np.zeros(n)
        p_row[idx] = 1.0
        rows_P.append(p_row)

        # Q: expected return proportional to signal score and asset vol
        q_val = (sig.score / 100.0) * vol_i
        rows_Q.append(q_val)

        # Omega diagonal: uncertainty inversely proportional to confidence
        conf = max(sig.confidence, 0.01)  # avoid division issues
        omega_val = ((1.0 - conf) ** 2) * TAU * (vol_i ** 2)
        rows_omega_diag.append(omega_val)

    k = len(rows_P)
    if k == 0:
        return np.zeros((0, n)), np.zeros(0), np.zeros((0, 0))

    P = np.array(rows_P)           # (K, N)
    Q = np.array(rows_Q)           # (K,)
    Omega = np.diag(rows_omega_diag)  # (K, K)

    return P, Q, Omega


# ═══════════════════════════════════════════════════════════════════════
#  BLACK-LITTERMAN FORMULA
# ═══════════════════════════════════════════════════════════════════════

def _bl_posterior(
    sigma: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Black-Litterman posterior mean and covariance.

    mu_bl = inv(tau_sigma_inv + P.T @ omega_inv @ P) @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
    sigma_bl = inv(tau_sigma_inv + P.T @ omega_inv @ P)
    """
    tau_sigma = TAU * sigma
    tau_sigma_inv = _safe_inverse(tau_sigma)
    omega_inv = _safe_inverse(Omega)

    # Posterior covariance
    M = tau_sigma_inv + P.T @ omega_inv @ P
    sigma_bl = _safe_inverse(M)

    # Posterior mean
    mu_bl = sigma_bl @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

    return mu_bl, sigma_bl


# ═══════════════════════════════════════════════════════════════════════
#  PORTFOLIO OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════

def _neg_sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Negative Sharpe ratio (to minimise)."""
    port_return = w @ mu
    port_vol = np.sqrt(w @ sigma @ w)
    if port_vol < 1e-12:
        return 0.0
    return -(port_return / port_vol)


def optimize_portfolio(
    mu_bl: np.ndarray,
    sigma_bl: np.ndarray,
    symbols: list[str],
) -> dict[str, float]:
    """Optimise weights to maximise Sharpe ratio.

    Constraints: sum(w) = 1, each w in [-0.30, +0.30].
    Falls back to equal weights if scipy is unavailable.
    """
    n = len(symbols)
    if n == 0:
        return {}

    if not HAS_SCIPY:
        # Fallback: equal weights
        w_eq = 1.0 / n
        return {s: w_eq for s in symbols}

    bounds = [(-0.30, 0.30)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    w0 = np.array([1.0 / n] * n)

    result = minimize(
        _neg_sharpe,
        w0,
        args=(mu_bl, sigma_bl),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if result.success:
        weights = result.x
    else:
        # Fallback: use initial guess
        weights = w0

    return {symbols[i]: float(round(weights[i], 6)) for i in range(n)}


# ═══════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def compute_bl_allocation() -> dict:
    """Full Black-Litterman pipeline.

    Returns
    -------
    dict with keys:
        weights        : dict[str, float]  — optimal portfolio weights
        expected_return: float              — portfolio expected return
        risk           : float              — portfolio volatility
        mu_bl          : list[float]        — BL posterior expected returns
        views_count    : int                — number of views incorporated
    """
    # 1. Market prior
    eq = get_equilibrium_returns()
    symbols = eq["symbols"]

    if not symbols or not eq["sigma"]:
        # No data at all — fallback to equal weights
        n = len(BL_ASSETS)
        equal_w = {s: round(1.0 / n, 6) for s in BL_ASSETS}
        return {
            "weights": equal_w,
            "expected_return": 0.0,
            "risk": 0.0,
            "mu_bl": [0.0] * n,
            "views_count": 0,
        }

    sigma = np.array(eq["sigma"])
    pi = np.array([eq["pi"].get(s, 0.0) for s in symbols])

    # 2. Signal views
    signals = scan_all()
    P, Q, Omega = generate_views(signals)
    views_count = P.shape[0]

    # 3. BL posterior
    if views_count > 0:
        mu_bl, sigma_bl = _bl_posterior(sigma, pi, P, Q, Omega)
    else:
        # No views — return market equilibrium
        mu_bl = pi
        sigma_bl = TAU * sigma

    # 4. Optimize
    weights = optimize_portfolio(mu_bl, sigma_bl, symbols)

    # 5. Portfolio metrics
    w_arr = np.array([weights.get(s, 0.0) for s in symbols])
    expected_return = float(w_arr @ mu_bl)
    risk = float(np.sqrt(w_arr @ sigma_bl @ w_arr))

    return {
        "weights": weights,
        "expected_return": round(expected_return, 6),
        "risk": round(risk, 6),
        "mu_bl": [round(float(m), 6) for m in mu_bl],
        "views_count": views_count,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Orion — Black-Litterman Allocation")
    print("=" * 60)

    result = compute_bl_allocation()
    print(f"\n  Views incorporées : {result['views_count']}")
    print(f"  Rendement attendu : {result['expected_return']:.4%}")
    print(f"  Risque (vol)      : {result['risk']:.4%}")
    print("\n  Poids optimaux :")
    for sym, w in sorted(result["weights"].items(), key=lambda x: x[1], reverse=True):
        bar = "+" * int(abs(w) * 100) if w >= 0 else "-" * int(abs(w) * 100)
        print(f"    {sym:<12} {w:+.4f}  {bar}")
