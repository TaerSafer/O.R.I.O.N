"""
Orion — Dynamic Portfolio Rebalancer
--------------------------------------
Orchestrates Black-Litterman allocation, CVaR scenario validation,
and generates rebalance orders by comparing target vs current positions.

Public API:
  - compute_target_allocation(capital) -> dict
  - compute_rebalance_orders(target, current_positions, capital, threshold) -> list[dict]
  - rebalance_report() -> dict
"""

from __future__ import annotations

from datetime import datetime

from risk.black_litterman import (
    BL_ASSETS,
    MARKET_CAP_WEIGHTS,
    compute_bl_allocation,
)
from risk.scenario_engine import (
    compute_cvar,
    validate_allocation,
    get_scenario_report,
)
from risk.manager import get_portfolio, DEFAULT_CONFIG


# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

CVAR_SCALE_FACTOR = 0.80       # Scale risky weights by 20% per iteration
MAX_CVAR_ITERATIONS = 5        # Max CVaR reduction attempts
DEFAULT_REBALANCE_THRESHOLD = 0.03  # 3% weight delta to trigger an order
TRADE_COST_BPS = 0.001         # 0.1% estimated cost per trade


# ═══════════════════════════════════════════════════════════════════════
#  RISKY ASSET IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════

# Assets considered "risky" for CVaR scaling purposes
_RISKY_ASSETS = {"^GSPC", "^GDAXI", "^FCHI", "BTC-USD", "CL=F"}
_DEFENSIVE_ASSETS = {"GC=F", "USDCHF=X", "EURUSD=X"}


def _scale_risky_weights(weights: dict[str, float], factor: float) -> dict[str, float]:
    """Scale down risky asset weights and redistribute to defensive assets.

    Keeps sum(weights) == 1.
    """
    new_weights = dict(weights)
    released = 0.0

    for sym in list(new_weights.keys()):
        if sym in _RISKY_ASSETS:
            old = new_weights[sym]
            new_weights[sym] = old * factor
            released += old - new_weights[sym]

    # Redistribute released weight to defensive assets proportionally
    defensive_in_portfolio = [s for s in new_weights if s in _DEFENSIVE_ASSETS]
    if defensive_in_portfolio and released > 0:
        share = released / len(defensive_in_portfolio)
        for sym in defensive_in_portfolio:
            new_weights[sym] += share
    elif released > 0:
        # No defensive assets in portfolio — spread equally
        n = len(new_weights)
        if n > 0:
            share = released / n
            for sym in new_weights:
                new_weights[sym] += share

    # Re-normalise to exactly 1.0
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {s: w / total for s, w in new_weights.items()}

    return new_weights


# ═══════════════════════════════════════════════════════════════════════
#  TARGET ALLOCATION
# ═══════════════════════════════════════════════════════════════════════

def compute_target_allocation(capital: float | None = None) -> dict:
    """Full pipeline: BL optimisation + CVaR validation with auto-scaling.

    Steps:
        1. Get current signals via scan_all() (called inside compute_bl_allocation)
        2. Compute BL optimal weights
        3. Validate with CVaR
        4. If CVaR fails, scale down risky weights by 20% and retry (max 5 iterations)

    Parameters
    ----------
    capital : float or None
        Total portfolio capital. Defaults to current portfolio capital or
        DEFAULT_CONFIG.capital.

    Returns
    -------
    dict with keys:
        weights, expected_return, risk, cvar, cvar_ok, views_count, timestamp
    """
    if capital is None:
        pf = get_portfolio()
        capital = pf.current_capital if pf.current_capital > 0 else DEFAULT_CONFIG.capital

    # Step 1 & 2: BL allocation (internally calls scan_all for views)
    try:
        bl = compute_bl_allocation()
    except Exception:
        # Graceful fallback to equal weights
        n = len(BL_ASSETS)
        bl = {
            "weights": {s: round(1.0 / n, 6) for s in BL_ASSETS},
            "expected_return": 0.0,
            "risk": 0.0,
            "mu_bl": [0.0] * n,
            "views_count": 0,
        }

    weights = bl["weights"]

    # Step 3 & 4: CVaR validation with iterative scaling
    cvar_ok = False
    cvar_value = 0.0

    for iteration in range(MAX_CVAR_ITERATIONS + 1):
        ok, cvar_value, _ = validate_allocation(weights, capital)
        if ok:
            cvar_ok = True
            break
        if iteration < MAX_CVAR_ITERATIONS:
            weights = _scale_risky_weights(weights, CVAR_SCALE_FACTOR)

    # Round weights
    weights = {s: round(w, 6) for s, w in weights.items()}

    return {
        "weights": weights,
        "expected_return": bl["expected_return"],
        "risk": bl["risk"],
        "cvar": round(cvar_value, 2),
        "cvar_ok": cvar_ok,
        "views_count": bl["views_count"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ═══════════════════════════════════════════════════════════════════════
#  REBALANCE ORDERS
# ═══════════════════════════════════════════════════════════════════════

def compute_rebalance_orders(
    target: dict,
    current_positions: dict[str, float],
    capital: float,
    threshold: float = DEFAULT_REBALANCE_THRESHOLD,
) -> list[dict]:
    """Compare target allocation vs current and generate rebalance orders.

    Parameters
    ----------
    target : dict
        Output of compute_target_allocation() — must contain "weights" key.
    current_positions : dict[str, float]
        Current portfolio weights keyed by symbol (e.g. {"^GSPC": 0.28, ...}).
        Missing symbols are treated as 0.0.
    capital : float
        Total capital for notional calculations.
    threshold : float
        Minimum absolute weight delta to trigger an order (default 3%).

    Returns
    -------
    list[dict] with keys:
        symbol, action, target_weight, current_weight, delta_weight, notional, estimated_cost
    """
    target_weights = target.get("weights", {})
    orders = []

    # All symbols that appear in either target or current
    all_symbols = set(list(target_weights.keys()) + list(current_positions.keys()))

    for sym in sorted(all_symbols):
        tw = target_weights.get(sym, 0.0)
        cw = current_positions.get(sym, 0.0)
        delta = tw - cw

        if abs(delta) < threshold:
            continue

        action = "BUY" if delta > 0 else "SELL"
        notional = abs(delta) * capital
        estimated_cost = notional * TRADE_COST_BPS

        orders.append({
            "symbol": sym,
            "action": action,
            "target_weight": round(tw, 6),
            "current_weight": round(cw, 6),
            "delta_weight": round(delta, 6),
            "notional": round(notional, 2),
            "estimated_cost": round(estimated_cost, 2),
        })

    # Sort by notional descending (biggest trades first)
    orders.sort(key=lambda o: o["notional"], reverse=True)
    return orders


# ═══════════════════════════════════════════════════════════════════════
#  FULL REBALANCE REPORT
# ═══════════════════════════════════════════════════════════════════════

def rebalance_report() -> dict:
    """Generate a full rebalance report for the dashboard.

    Returns dict with keys:
        target_allocation, current_allocation, orders, cvar, expected_return,
        risk, timestamp, scenario_report
    """
    pf = get_portfolio()
    capital = pf.current_capital if pf.current_capital > 0 else DEFAULT_CONFIG.capital

    # Target allocation
    target = compute_target_allocation(capital)
    target_weights = target["weights"]

    # Current allocation from portfolio positions
    current_allocation: dict[str, float] = {}
    if pf.positions and capital > 0:
        for sym, pos in pf.positions.items():
            current_allocation[sym] = pos.value / capital
    # For assets not in current portfolio, weight is 0

    # Rebalance orders
    orders = compute_rebalance_orders(target, current_allocation, capital)

    # Scenario report on target allocation
    scenario_rep = get_scenario_report(target_weights, capital)

    return {
        "target_allocation": target_weights,
        "current_allocation": current_allocation,
        "orders": orders,
        "cvar": target["cvar"],
        "cvar_ok": target["cvar_ok"],
        "expected_return": target["expected_return"],
        "risk": target["risk"],
        "views_count": target["views_count"],
        "timestamp": target["timestamp"],
        "scenario_report": scenario_rep,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Orion — Dynamic Rebalancer")
    print("=" * 60)

    report = rebalance_report()

    print(f"\n  Timestamp     : {report['timestamp']}")
    print(f"  Views         : {report['views_count']}")
    print(f"  Expected Ret  : {report['expected_return']:.4%}")
    print(f"  Risk (vol)    : {report['risk']:.4%}")
    print(f"  CVaR (5%)     : ${report['cvar']:,.0f}  {'OK' if report['cvar_ok'] else 'BREACH'}")

    print("\n  --- Target Allocation ---")
    for sym, w in sorted(report["target_allocation"].items(), key=lambda x: x[1], reverse=True):
        print(f"    {sym:<12} {w:+.4f}")

    if report["orders"]:
        print("\n  --- Rebalance Orders ---")
        for o in report["orders"]:
            print(f"    {o['action']:>4} {o['symbol']:<12} "
                  f"delta={o['delta_weight']:+.4f}  "
                  f"notional=${o['notional']:,.0f}  "
                  f"cost=${o['estimated_cost']:.2f}")
    else:
        print("\n  No rebalance needed (all deltas below threshold).")

    print("\n  --- Scenarios ---")
    for s in report["scenario_report"]:
        sign = "+" if s["pnl"] >= 0 else ""
        print(f"    {s['name']:<20} P={s['probability']:.0%}  "
              f"P&L={sign}${s['pnl']:,.0f} ({s['pnl_pct']:+.2%})")
