from risk.manager import (
    # Régime de marché
    Regime,
    RegimeState,
    detect_regime,
    effective_config,
    DEFENSIVE_ASSETS,
    # Drawdown progressif
    DrawdownAction,
    evaluate_drawdown,
    positions_to_reduce,
    stress_reduce_positions,
    # Position sizing
    check_risk,
    compute_levels,
    size_fixed,
    size_kelly,
    size_volatility,
    update_trailing_stop,
    # Corrélation
    correlation_matrix,
    find_correlated_pairs,
    # Portfolio
    portfolio_report,
    get_portfolio,
    set_portfolio,
    Portfolio,
    Position,
    # Config & types
    RiskConfig,
    RiskCheck,
    PositionSize,
    PortfolioRisk,
    DEFAULT_CONFIG,
)

from risk.black_litterman import (
    BL_ASSETS,
    MARKET_CAP_WEIGHTS,
    compute_bl_allocation,
    get_equilibrium_returns,
    generate_views,
    optimize_portfolio,
)

from risk.scenario_engine import (
    SCENARIOS,
    compute_scenario_pnl,
    compute_cvar,
    validate_allocation,
    get_scenario_report,
)

from risk.rebalancer import (
    compute_target_allocation,
    compute_rebalance_orders,
    rebalance_report,
)
