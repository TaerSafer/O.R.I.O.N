"""
Orion — CVaR Scenario Simulation Engine
-----------------------------------------
Evaluates portfolio risk under predefined macro-economic scenarios.
Computes scenario P&L, weighted CVaR, and validates allocations against
risk budgets.

Public API:
  - compute_scenario_pnl(weights, scenario_name, capital) -> float
  - compute_cvar(weights, capital, alpha) -> float
  - validate_allocation(weights, capital, max_cvar_pct) -> tuple
  - get_scenario_report(weights, capital) -> list[dict]
"""

from __future__ import annotations

from risk.black_litterman import BL_ASSETS


# ═══════════════════════════════════════════════════════════════════════
#  PREDEFINED SCENARIOS
# ═══════════════════════════════════════════════════════════════════════

SCENARIOS: dict[str, dict] = {
    "RECESSION_US": {
        "impacts": {
            "^GSPC": -0.35, "^GDAXI": -0.30, "^FCHI": -0.25,
            "BTC-USD": -0.60, "GC=F": +0.15, "USDCHF=X": -0.10,
            "EURUSD=X": +0.05, "CL=F": -0.25,
        },
        "base_probability": 0.10,
    },
    "CRISE_LIQUIDITE": {
        "impacts": {
            "^GSPC": -0.25, "^GDAXI": -0.25, "^FCHI": -0.20,
            "BTC-USD": -0.50, "GC=F": +0.08, "USDCHF=X": +0.05,
            "EURUSD=X": -0.05, "CL=F": -0.15,
        },
        "base_probability": 0.08,
    },
    "DOLLAR_FORT": {
        "impacts": {
            "^GSPC": +0.02, "^GDAXI": -0.08, "^FCHI": -0.10,
            "BTC-USD": -0.15, "GC=F": -0.08, "USDCHF=X": +0.08,
            "EURUSD=X": -0.10, "CL=F": -0.12,
        },
        "base_probability": 0.15,
    },
    "STAGFLATION": {
        "impacts": {
            "^GSPC": -0.15, "^GDAXI": -0.12, "^FCHI": -0.12,
            "BTC-USD": -0.10, "GC=F": +0.25, "USDCHF=X": +0.05,
            "EURUSD=X": -0.03, "CL=F": +0.20,
        },
        "base_probability": 0.10,
    },
    "RISK_ON": {
        "impacts": {
            "^GSPC": +0.15, "^GDAXI": +0.12, "^FCHI": +0.10,
            "BTC-USD": +0.25, "GC=F": -0.05, "USDCHF=X": -0.03,
            "EURUSD=X": +0.02, "CL=F": +0.08,
        },
        "base_probability": 0.20,
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  SCENARIO P&L
# ═══════════════════════════════════════════════════════════════════════

def compute_scenario_pnl(
    weights: dict[str, float],
    scenario_name: str,
    capital: float,
) -> float:
    """Compute the P&L of a portfolio under a given scenario.

    Parameters
    ----------
    weights : dict
        Portfolio weights keyed by symbol (e.g. {"^GSPC": 0.30, ...}).
    scenario_name : str
        Key into SCENARIOS dict.
    capital : float
        Total capital deployed.

    Returns
    -------
    float — P&L in currency units (positive = profit, negative = loss).
    """
    scenario = SCENARIOS.get(scenario_name)
    if scenario is None:
        return 0.0

    impacts = scenario["impacts"]
    pnl = 0.0
    for sym, w in weights.items():
        impact = impacts.get(sym, 0.0)
        pnl += w * impact * capital

    return round(pnl, 2)


# ═══════════════════════════════════════════════════════════════════════
#  CVaR COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_cvar(
    weights: dict[str, float],
    capital: float,
    alpha: float = 0.05,
) -> float:
    """Probability-weighted CVaR across all predefined scenarios.

    The CVaR (Conditional Value at Risk) is computed as the expected loss
    in the worst scenarios whose cumulative probability does not exceed alpha.
    If total probability of loss scenarios < alpha, we use all loss scenarios.

    Parameters
    ----------
    weights : dict — portfolio weights.
    capital : float — total capital.
    alpha : float — confidence tail (default 5%).

    Returns
    -------
    float — CVaR as a positive number representing potential loss.
    """
    # Compute P&L and probability for each scenario
    scenario_data = []
    for name, spec in SCENARIOS.items():
        pnl = compute_scenario_pnl(weights, name, capital)
        prob = spec["base_probability"]
        scenario_data.append((pnl, prob, name))

    if not scenario_data:
        return 0.0

    # Sort by P&L ascending (worst first)
    scenario_data.sort(key=lambda x: x[0])

    # Accumulate probability from worst scenarios up to alpha
    cumulative_prob = 0.0
    weighted_loss = 0.0
    for pnl, prob, _ in scenario_data:
        if cumulative_prob >= alpha:
            break
        # How much of this scenario's probability falls within alpha
        remaining = alpha - cumulative_prob
        used_prob = min(prob, remaining)
        weighted_loss += pnl * used_prob
        cumulative_prob += used_prob

    if cumulative_prob <= 0:
        return 0.0

    # CVaR = expected loss in the tail (as positive number)
    cvar = -weighted_loss / cumulative_prob
    return round(max(cvar, 0.0), 2)


# ═══════════════════════════════════════════════════════════════════════
#  ALLOCATION VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def validate_allocation(
    weights: dict[str, float],
    capital: float,
    max_cvar_pct: float = 0.08,
) -> tuple[bool, float, list[dict]]:
    """Validate a portfolio allocation against CVaR budget.

    Parameters
    ----------
    weights : dict — portfolio weights.
    capital : float — total capital.
    max_cvar_pct : float — maximum acceptable CVaR as % of capital (default 8%).

    Returns
    -------
    (ok, cvar_value, scenario_details) where:
        ok : bool — True if CVaR <= max_cvar_pct * capital
        cvar_value : float — computed CVaR
        scenario_details : list[dict] — per-scenario breakdown
    """
    cvar = compute_cvar(weights, capital)
    report = get_scenario_report(weights, capital)
    ok = cvar <= max_cvar_pct * capital
    return ok, cvar, report


# ═══════════════════════════════════════════════════════════════════════
#  SCENARIO REPORT
# ═══════════════════════════════════════════════════════════════════════

def get_scenario_report(
    weights: dict[str, float],
    capital: float,
) -> list[dict]:
    """Full scenario report sorted by probability (descending).

    Returns list of dicts with keys:
        name, probability, pnl, pnl_pct, impacts
    """
    report = []
    for name, spec in SCENARIOS.items():
        pnl = compute_scenario_pnl(weights, name, capital)
        pnl_pct = pnl / capital if capital > 0 else 0.0
        report.append({
            "name": name,
            "probability": spec["base_probability"],
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "impacts": spec["impacts"],
        })

    report.sort(key=lambda x: x["probability"], reverse=True)
    return report


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from risk.black_litterman import MARKET_CAP_WEIGHTS

    print("=" * 60)
    print("  Orion — Scenario Engine")
    print("=" * 60)

    capital = 10_000.0
    weights = MARKET_CAP_WEIGHTS

    print(f"\n  Capital: ${capital:,.0f}")
    print(f"  Weights: market-cap equilibrium\n")

    report = get_scenario_report(weights, capital)
    for s in report:
        emoji = "+" if s["pnl"] >= 0 else ""
        print(f"  {s['name']:<20} P={s['probability']:.0%}  "
              f"P&L={emoji}${s['pnl']:,.0f} ({s['pnl_pct']:+.2%})")

    cvar = compute_cvar(weights, capital)
    print(f"\n  CVaR (5%) = ${cvar:,.0f} ({cvar/capital:.2%} du capital)")

    ok, _, _ = validate_allocation(weights, capital)
    status = "OK" if ok else "BREACH"
    print(f"  Validation CVaR: {status}")
