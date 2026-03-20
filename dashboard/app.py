"""
Orion Dashboard — FastAPI Application
---------------------------------------
REST API + WebSocket dashboard for the Orion trading system.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ─── Orion module imports ────────────────────────────────────────────

from data.collector import ASSETS, get_price, get_history
from signals.engine import analyze, scan_all, top_signals, Signal, Action
from risk.manager import (
    detect_regime, portfolio_report, get_portfolio, check_risk,
    correlation_matrix, find_correlated_pairs, evaluate_drawdown,
    Regime, RegimeState, DrawdownAction, Portfolio, Position,
    RiskConfig, DEFAULT_CONFIG, effective_config,
)
from journal.logger import (
    init_journal, get_trades, get_events, stats, daily_pnl, monthly_pnl,
    streak_analysis, log_event, get_snapshots, add_note, get_notes,
)
from backtest.engine import run_backtest, Backtest, BacktestResult, BacktestMetrics
from execution.broker import get_ibkr_status


# ─── App setup ───────────────────────────────────────────────────────

app = FastAPI(title="Orion Dashboard", version="1.0.0")

# ─── CORS (permet à Live Server :5500 d'appeler l'API :8000) ────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Servir les fichiers statiques (config.js, images, CSS custom)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Ensure journal tables exist
try:
    init_journal()
except Exception:
    pass

# ─── Helpers ─────────────────────────────────────────────────────────

def _sanitize(obj):
    """Recursively convert numpy/pandas types to native Python for JSON."""
    import numpy as _np
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj


def _signal_to_dict(sig: Signal) -> dict:
    return _sanitize({
        "symbol": sig.symbol,
        "action": sig.action.value,
        "score": sig.score,
        "confidence": sig.confidence,
        "timestamp": sig.timestamp,
        "asset_class": sig.asset_class,
        "details": sig.details,
    })


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))


def _safe(fn, default=None):
    """Call fn, return default on any exception."""
    try:
        return fn()
    except Exception:
        traceback.print_exc()
        return default


# ─── Pydantic models ────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    symbols: Optional[list[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    capital: float = 10_000.0
    sizing_method: str = "fixed"
    enable_aladdin: bool = True


class NoteRequest(BaseModel):
    content: str
    date: Optional[str] = None
    trade_id: Optional[int] = None
    tags: str = ""


class ConfigUpdate(BaseModel):
    capital: Optional[float] = None
    risk_per_trade: Optional[float] = None
    max_risk_per_trade: Optional[float] = None
    max_portfolio_risk: Optional[float] = None
    max_positions: Optional[int] = None
    max_exposure_per_asset: Optional[float] = None
    max_exposure_per_class: Optional[float] = None
    max_correlated_exposure: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_assets_per_class: Optional[int] = None
    max_correlated_simultaneous: Optional[int] = None
    correlation_threshold: Optional[float] = None
    atr_sl_multiplier: Optional[float] = None
    atr_tp_multiplier: Optional[float] = None
    atr_period: Optional[int] = None
    correlation_window: Optional[int] = None
    kelly_fraction: Optional[float] = None
    consecutive_loss_limit: Optional[int] = None
    cooldown_hours: Optional[int] = None
    drawdown_level_1: Optional[float] = None
    drawdown_level_2: Optional[float] = None
    drawdown_level_3: Optional[float] = None


# ─── Mutable config reference ───────────────────────────────────────

_config = RiskConfig()


def _get_config() -> RiskConfig:
    return _config


# ═══════════════════════════════════════════════════════════════════════
#  HTML PAGE
# ═══════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/public", response_class=HTMLResponse)
async def public_page(request: Request):
    return templates.TemplateResponse("public.html", {"request": request})


@app.get("/presentation", response_class=HTMLResponse)
async def presentation(request: Request):
    return templates.TemplateResponse("presentation.html", {"request": request})


# ═══════════════════════════════════════════════════════════════════════
#  REST API
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/overview")
async def api_overview():
    try:
        pf = get_portfolio()
        cfg = _get_config()
        regime_state = _safe(detect_regime)
        dd_action = _safe(lambda: evaluate_drawdown(cfg, pf), DrawdownAction.NONE)

        # Equity curve from snapshots
        snapshots = _safe(lambda: get_snapshots(limit=500), pd.DataFrame())
        equity_data = []
        if snapshots is not None and not snapshots.empty:
            for _, row in snapshots.iterrows():
                equity_data.append({
                    "timestamp": str(row.get("timestamp", "")),
                    "equity": float(row["equity"]) if pd.notna(row.get("equity")) else 0,
                })

        # Also add pnl_history from portfolio as fallback
        if not equity_data and hasattr(pf, "_pnl_history") and pf._pnl_history:
            for i, val in enumerate(pf._pnl_history):
                equity_data.append({"timestamp": str(i), "equity": val})

        # Journal stats for total P&L
        j_stats = _safe(lambda: stats(), {"total_pnl": 0})

        regime_info = None
        if regime_state:
            regime_info = {
                "regime": regime_state.regime.value,
                "vix_simulated": regime_state.vix_simulated,
                "avg_correlation": regime_state.avg_correlation,
                "timestamp": regime_state.timestamp,
                "details": regime_state.details,
            }

        return {
            "capital": pf.current_capital,
            "peak_capital": pf.peak_capital,
            "pnl_total": j_stats.get("total_pnl", 0) if j_stats else 0,
            "regime": regime_info,
            "drawdown": pf.drawdown,
            "drawdown_pct": pf.drawdown_pct,
            "drawdown_action": dd_action.value if dd_action else "NONE",
            "positions_count": len(pf.positions),
            "total_exposure": pf.total_exposure,
            "total_risk": pf.total_risk,
            "equity_curve": equity_data,
            "is_in_cooldown": pf.is_in_cooldown,
            "consecutive_losses": pf._consecutive_losses,
            "ibkr": _safe(get_ibkr_status, {"connected": False, "error": ""}),
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/api/positions")
async def api_positions():
    try:
        pf = get_portfolio()
        positions = []
        for sym, pos in pf.positions.items():
            price_data = _safe(lambda s=sym: get_price(s))
            current_price = price_data["close"] if price_data else pos.entry_price
            if pos.action == "BUY":
                unrealized = (current_price - pos.entry_price) * pos.size
            else:
                unrealized = (pos.entry_price - current_price) * pos.size
            positions.append({
                "symbol": pos.symbol,
                "asset_class": pos.asset_class,
                "action": pos.action,
                "entry_price": pos.entry_price,
                "current_price": current_price,
                "size": pos.size,
                "value": pos.value,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "timestamp": pos.timestamp,
                "risk_amount": pos.risk_amount,
                "unrealized_pnl": round(unrealized, 2),
            })
        return {"positions": positions}
    except Exception as e:
        traceback.print_exc()
        return {"positions": [], "error": str(e)}


@app.get("/api/allocation")
async def api_allocation():
    try:
        pf = get_portfolio()
        exposure_by_class = pf.exposure_by_class()
        positions_by_class = pf.positions_by_class()
        risk_by_class = pf.risk_by_class()
        return {
            "exposure_by_class": exposure_by_class,
            "positions_by_class": positions_by_class,
            "risk_by_class": risk_by_class,
            "total_exposure": pf.total_exposure,
            "capital": pf.current_capital,
        }
    except Exception as e:
        traceback.print_exc()
        return {"exposure_by_class": {}, "error": str(e)}


@app.get("/api/prices")
async def api_prices():
    """Tous les prix courants des 35 actifs pour le ticker et le tableau marchés."""
    try:
        prices = {}
        for symbol, asset_class in ASSETS.items():
            price_data = _safe(lambda s=symbol: get_price(s))
            if price_data:
                prices[symbol] = {
                    "symbol": symbol,
                    "asset_class": asset_class,
                    "close": price_data.get("close"),
                    "open": price_data.get("open"),
                    "high": price_data.get("high"),
                    "low": price_data.get("low"),
                    "date": price_data.get("date"),
                }
        return {"prices": prices, "count": len(prices)}
    except Exception as e:
        traceback.print_exc()
        return {"prices": {}, "error": str(e)}


@app.get("/api/signals")
async def api_signals():
    try:
        signals = scan_all()
        result = []
        for s in signals:
            d = _signal_to_dict(s)
            # Sparkline: 30-day close prices
            try:
                df = get_history(s.symbol, days=30)
                if not df.empty:
                    closes = df["close"].dropna().tolist()
                    d["sparkline"] = [round(float(c), 4) for c in closes]
                else:
                    d["sparkline"] = []
            except Exception:
                d["sparkline"] = []
            result.append(d)
        return {
            "signals": result,
            "count": len(result),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        traceback.print_exc()
        return {"signals": [], "error": str(e)}


@app.get("/api/correlation")
async def api_correlation():
    try:
        corr = correlation_matrix()
        if corr.empty:
            return {"matrix": {}, "symbols": [], "pairs": []}

        symbols = corr.columns.tolist()
        matrix = {}
        for sym in symbols:
            matrix[sym] = {s: round(float(corr.loc[sym, s]), 3) for s in symbols}

        pairs = find_correlated_pairs(threshold=0.7)
        pairs_data = [{"a": a, "b": b, "correlation": rho} for a, b, rho in pairs]

        return {"matrix": matrix, "symbols": symbols, "pairs": pairs_data}
    except Exception as e:
        traceback.print_exc()
        return {"matrix": {}, "symbols": [], "pairs": [], "error": str(e)}


# ─── Journal endpoints ──────────────────────────────────────────────

@app.get("/api/journal/trades")
async def api_journal_trades(
    symbol: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    limit: int = Query(500),
):
    try:
        df = get_trades(symbol=symbol, start_date=start, end_date=end, limit=limit)
        return {"trades": _df_to_records(df)}
    except Exception as e:
        traceback.print_exc()
        return {"trades": [], "error": str(e)}


@app.get("/api/journal/stats")
async def api_journal_stats():
    try:
        s = stats()
        streak = streak_analysis()
        s["streak"] = streak
        return s
    except Exception as e:
        traceback.print_exc()
        return {"total_trades": 0, "error": str(e)}


@app.get("/api/journal/daily-pnl")
async def api_journal_daily_pnl():
    try:
        df = daily_pnl()
        return {"data": _df_to_records(df)}
    except Exception as e:
        traceback.print_exc()
        return {"data": [], "error": str(e)}


@app.get("/api/journal/monthly-pnl")
async def api_journal_monthly_pnl():
    try:
        df = monthly_pnl()
        return {"data": _df_to_records(df)}
    except Exception as e:
        traceback.print_exc()
        return {"data": [], "error": str(e)}


@app.get("/api/journal/events")
async def api_journal_events(
    category: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    limit: int = Query(200),
):
    try:
        df = get_events(category=category, level=level, limit=limit)
        return {"events": _df_to_records(df)}
    except Exception as e:
        traceback.print_exc()
        return {"events": [], "error": str(e)}


@app.get("/api/journal/notes")
async def api_journal_notes():
    try:
        df = get_notes()
        return {"notes": _df_to_records(df)}
    except Exception as e:
        traceback.print_exc()
        return {"notes": [], "error": str(e)}


@app.post("/api/journal/notes")
async def api_journal_add_note(note: NoteRequest):
    try:
        note_id = add_note(
            content=note.content,
            date=note.date,
            trade_id=note.trade_id,
            tags=note.tags,
        )
        return {"id": note_id, "status": "ok"}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ─── Backtest endpoint ──────────────────────────────────────────────

@app.post("/api/backtest/run")
async def api_backtest_run(req: BacktestRequest):
    try:
        result = run_backtest(
            symbols=req.symbols,
            start_date=req.start_date,
            end_date=req.end_date,
            capital=req.capital,
            sizing_method=req.sizing_method,
            enable_aladdin=req.enable_aladdin,
        )

        # Convert metrics
        m = result.metrics
        metrics_dict = {
            "total_return": m.total_return,
            "total_return_pct": m.total_return_pct,
            "annualized_return": m.annualized_return,
            "benchmark_return": m.benchmark_return,
            "alpha": m.alpha,
            "sharpe_ratio": m.sharpe_ratio,
            "sortino_ratio": m.sortino_ratio,
            "calmar_ratio": m.calmar_ratio,
            "max_drawdown": m.max_drawdown,
            "max_drawdown_date": m.max_drawdown_date,
            "avg_drawdown": m.avg_drawdown,
            "volatility": m.volatility,
            "downside_vol": m.downside_vol,
            "total_trades": m.total_trades,
            "winning_trades": m.winning_trades,
            "losing_trades": m.losing_trades,
            "win_rate": m.win_rate,
            "avg_win": m.avg_win,
            "avg_loss": m.avg_loss,
            "profit_factor": m.profit_factor,
            "avg_trade_pnl": m.avg_trade_pnl,
            "best_trade": m.best_trade,
            "worst_trade": m.worst_trade,
            "avg_holding_days": m.avg_holding_days,
            "max_consecutive_wins": m.max_consecutive_wins,
            "max_consecutive_losses": m.max_consecutive_losses,
            "trades_by_regime": m.trades_by_regime,
            "return_by_regime": m.return_by_regime,
            "avg_positions": m.avg_positions,
            "max_positions": m.max_positions,
            "capital_protection_triggers": m.capital_protection_triggers,
            "drawdown_reductions": m.drawdown_reductions,
        }

        # Equity curve
        eq = result.equity_curve
        equity_data = []
        if not eq.empty:
            for _, row in eq.iterrows():
                equity_data.append({
                    "date": str(row["date"])[:10] if row["date"] else "",
                    "equity": float(row["equity"]),
                })

        # Trades
        trades_data = []
        for t in result.trades:
            trades_data.append({
                "symbol": t.symbol,
                "asset_class": t.asset_class,
                "action": t.action,
                "entry_date": t.entry_date[:10] if t.entry_date else "",
                "exit_date": t.exit_date[:10] if t.exit_date else "",
                "entry_price": round(t.entry_price, 4),
                "exit_price": round(t.exit_price, 4),
                "size": round(t.size, 4),
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
                "regime": t.regime,
                "holding_days": t.holding_days,
            })

        # Summaries
        by_class = _df_to_records(result.summary_by_class())
        by_exit = _df_to_records(result.summary_by_exit())

        return {
            "metrics": metrics_dict,
            "equity_curve": equity_data,
            "trades": trades_data,
            "summary_by_class": by_class,
            "summary_by_exit": by_exit,
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ─── Config endpoints ───────────────────────────────────────────────

@app.get("/api/config")
async def api_config():
    try:
        cfg = _get_config()
        regime_state = _safe(detect_regime)
        eff = None
        if regime_state:
            eff_cfg = effective_config(cfg, regime_state)
            eff = {
                "risk_per_trade": eff_cfg.risk_per_trade,
                "max_risk_per_trade": eff_cfg.max_risk_per_trade,
                "max_portfolio_risk": eff_cfg.max_portfolio_risk,
                "max_exposure_per_asset": eff_cfg.max_exposure_per_asset,
                "max_exposure_per_class": eff_cfg.max_exposure_per_class,
                "max_positions": eff_cfg.max_positions,
            }

        return {
            "config": {
                "capital": cfg.capital,
                "risk_per_trade": cfg.risk_per_trade,
                "max_risk_per_trade": cfg.max_risk_per_trade,
                "max_portfolio_risk": cfg.max_portfolio_risk,
                "max_positions": cfg.max_positions,
                "max_exposure_per_asset": cfg.max_exposure_per_asset,
                "max_exposure_per_class": cfg.max_exposure_per_class,
                "max_correlated_exposure": cfg.max_correlated_exposure,
                "max_drawdown": cfg.max_drawdown,
                "max_assets_per_class": cfg.max_assets_per_class,
                "max_correlated_simultaneous": cfg.max_correlated_simultaneous,
                "correlation_threshold": cfg.correlation_threshold,
                "atr_sl_multiplier": cfg.atr_sl_multiplier,
                "atr_tp_multiplier": cfg.atr_tp_multiplier,
                "atr_period": cfg.atr_period,
                "correlation_window": cfg.correlation_window,
                "kelly_fraction": cfg.kelly_fraction,
                "consecutive_loss_limit": cfg.consecutive_loss_limit,
                "cooldown_hours": cfg.cooldown_hours,
                "drawdown_level_1": cfg.drawdown_level_1,
                "drawdown_level_2": cfg.drawdown_level_2,
                "drawdown_level_3": cfg.drawdown_level_3,
            },
            "effective": eff,
            "regime_multipliers": {
                "EXPANSION": {"sizing": 1.0, "max_exposure": 1.0, "new_trades": True},
                "CONTRACTION": {"sizing": 0.70, "max_exposure": 0.70, "new_trades": True},
                "STRESS": {"sizing": 0.50, "max_exposure": 0.333, "new_trades": False},
            },
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/api/config")
async def api_config_update(update: ConfigUpdate):
    global _config
    try:
        cfg = _config
        for field_name, value in update.model_dump(exclude_none=True).items():
            if hasattr(cfg, field_name):
                setattr(cfg, field_name, value)
        return {"status": "ok", "config": {
            k: getattr(cfg, k) for k in update.model_fields.keys()
            if hasattr(cfg, k)
        }}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ─── New feature endpoints ────────────────────────────────────────────

@app.get("/api/briefing")
async def api_briefing():
    try:
        from signals.morning_brief import get_latest_briefing, generate_briefing
        brief = _safe(get_latest_briefing)
        if not brief:
            brief = _safe(generate_briefing, {})
        return _sanitize(brief or {"text": "Briefing non disponible"})
    except Exception:
        return {"text": "Briefing non disponible"}


@app.get("/api/confidence")
async def api_confidence():
    try:
        from signals.confidence import compute_confidence
        return _sanitize(_safe(compute_confidence, {"score": 50, "level": "medium"}))
    except Exception:
        return {"score": 50, "level": "medium"}


@app.get("/api/projections")
async def api_projections():
    try:
        from signals.projections import compute_all_projections
        projs = _safe(compute_all_projections, [])
        return _sanitize({"projections": projs})
    except Exception:
        return {"projections": []}


@app.get("/api/memory")
async def api_memory():
    try:
        from signals.memory import get_memory_entries
        entries = _safe(lambda: get_memory_entries(30), [])
        return _sanitize({"entries": entries})
    except Exception:
        return {"entries": []}


@app.post("/api/protection")
async def api_protection():
    try:
        pf = get_portfolio()
        closed = []
        for sym in list(pf.positions.keys()):
            pf.remove_position(sym)
            closed.append(sym)
        from journal.logger import log_event
        log_event("protection", f"Protection totale activée — {len(closed)} positions fermées",
                  "CRITICAL", data={"closed": closed})
        return {"status": "ok", "closed": closed}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  WEBSOCKET
# ═══════════════════════════════════════════════════════════════════════

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        payload = json.dumps(data, default=str)
        for ws in list(self.active):
            try:
                await ws.send_text(payload)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()


async def _build_ws_payload() -> dict:
    """Build the overview + positions payload for WS broadcast."""
    try:
        pf = get_portfolio()
        cfg = _get_config()
        regime_state = _safe(detect_regime)
        dd_action = _safe(lambda: evaluate_drawdown(cfg, pf), DrawdownAction.NONE)

        regime_info = None
        if regime_state:
            regime_info = {
                "regime": regime_state.regime.value,
                "vix_simulated": regime_state.vix_simulated,
                "avg_correlation": regime_state.avg_correlation,
            }

        positions = []
        for sym, pos in pf.positions.items():
            price_data = _safe(lambda s=sym: get_price(s))
            current_price = price_data["close"] if price_data else pos.entry_price
            if pos.action == "BUY":
                unrealized = (current_price - pos.entry_price) * pos.size
            else:
                unrealized = (pos.entry_price - current_price) * pos.size
            positions.append({
                "symbol": pos.symbol,
                "asset_class": pos.asset_class,
                "action": pos.action,
                "entry_price": pos.entry_price,
                "current_price": current_price,
                "size": pos.size,
                "value": pos.value,
                "unrealized_pnl": round(unrealized, 2),
            })

        return {
            "type": "update",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overview": {
                "capital": pf.current_capital,
                "peak_capital": pf.peak_capital,
                "drawdown": pf.drawdown,
                "drawdown_pct": pf.drawdown_pct,
                "drawdown_action": dd_action.value if dd_action else "NONE",
                "positions_count": len(pf.positions),
                "total_exposure": pf.total_exposure,
                "total_risk": pf.total_risk,
                "regime": regime_info,
                "ibkr": _safe(get_ibkr_status, {"connected": False, "error": ""}),
            },
            "positions": positions,
        }
    except Exception as e:
        return {"type": "error", "message": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send initial data
        payload = await _build_ws_payload()
        await ws.send_text(json.dumps(payload, default=str))

        while True:
            # Wait 15 seconds, or receive a message
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=15.0)
            except asyncio.TimeoutError:
                pass

            # Send update
            payload = await _build_ws_payload()
            await ws.send_text(json.dumps(payload, default=str))
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
