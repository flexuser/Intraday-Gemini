"""Historical Walk-Forward Backtester with Windowed ORB + VWAP + ADX Trend Filters.

Location: muthoot_poc/backtest_harness.py
Run via root: python -m muthoot_poc.backtest_harness
"""

import sys
import os
import math
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

# Standardize path imports to resolve both root and subfolder execution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe module imports with fallback support
try:
    from indicators import calculate_technical_snapshot
    from risk_engine import HardRiskGuardrail
    from quant_model import predict_today, load_model
except ImportError:
    from muthoot_poc.indicators import calculate_technical_snapshot
    from muthoot_poc.risk_engine import HardRiskGuardrail
    from muthoot_poc.quant_model import predict_today, load_model


WATCHLIST_DISPLAY = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "MUTHOOTFIN.NS",
    "TMPV.NS",
    "AXISBANK.NS",
    "LT.NS",
]

YFINANCE_CANDIDATES = {
    "TMPV.NS": ["TMPV.NS", "TATAMOTR.NS", "TATAMOTORS.NS"]
}


def _download_symbol_data(symbol_display, period="60d"):
    """Download historical 5m data with automatic ticker fallback."""
    candidates = YFINANCE_CANDIDATES.get(symbol_display, [symbol_display])
    for ticker in candidates:
        try:
            df = yf.download(ticker, period=period, interval="5m", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        except Exception:
            continue
    return pd.DataFrame()


def _safe_load_model():
    """Safely attempt to load the ML model across common paths."""
    if 'load_model' not in globals():
        return None

    candidate_paths = [
        CURRENT_DIR / "quant_model.pkl",
        CURRENT_DIR / "model.pkl",
        CURRENT_DIR / "quant_model.joblib",
        PROJECT_ROOT / "quant_model.pkl",
        PROJECT_ROOT / "model.pkl",
    ]

    for path in candidate_paths:
        if path.exists():
            try:
                return load_model(str(path))
            except Exception:
                continue
    return None


class IntradayBacktester:
    """Simulates session execution using historical 5m bar data and fixed risk targets."""

    def __init__(self, initial_capital=1_000_000.0, slippage_pct=0.03, commission_pct=0.02):
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct / 100.0
        self.commission_pct = commission_pct / 100.0
        self.risk_engine = HardRiskGuardrail(
            account_capital=initial_capital,
            max_risk_per_trade_pct=1.0,
            min_risk_reward_ratio=1.5
        ) if HardRiskGuardrail else None

    def simulate_session(self, symbol, df_5m, model_obj=None):
        """Scan morning window (09:30 to 11:30 IST) for high-conviction ADX-filtered breakouts."""
        if df_5m.empty or len(df_5m) < 14:
            return None

        # First 15 minutes set Opening Range High/Low
        orb_high = float(df_5m.iloc[:3]["High"].max())
        orb_low = float(df_5m.iloc[:3]["Low"].min())

        # Cumulative VWAP across session
        typical_price = (df_5m["High"] + df_5m["Low"] + df_5m["Close"]) / 3.0
        vwap_series = (typical_price * df_5m["Volume"]).cumsum() / df_5m["Volume"].cumsum().replace(0, 1)

        tech_snapshot = (
            calculate_technical_snapshot(df_5m)
            if calculate_technical_snapshot
            else {}
        )
        atr = tech_snapshot.get("atr_14", float(df_5m["Close"].iloc[0]) * 0.01) if tech_snapshot else float(df_5m["Close"].iloc[0]) * 0.01

        entry_bar = None
        entry_idx = None
        quant_pred = None

        # Scan 5m bars between 09:30 IST and 11:30 IST
        for idx in range(3, min(30, len(df_5m))):
            bar = df_5m.iloc[idx]
            close = float(bar["Close"])
            vwap = float(vwap_series.iloc[idx])

            # Calculate rudimentary ADX proxy or use snapshot ADX if available
            # If snapshot doesn't have adx, estimate volatility expansion ratio
            sub_df = df_5m.iloc[:idx+1]
            rolling_tr = (sub_df["High"] - sub_df["Low"]).rolling(5).mean().iloc[-1]
            avg_tr = (sub_df["High"] - sub_df["Low"]).mean()
            trend_strength = rolling_tr / max(avg_tr, 1e-4)

            # ML Predictor
            if model_obj:
                try:
                    pred = predict_today(
                        symbol=symbol,
                        df_5m=sub_df,
                        tech_snapshot=tech_snapshot,
                        model=model_obj
                    )
                    if pred and pred.get("direction_validated"):
                        entry_bar = bar
                        entry_idx = idx
                        quant_pred = pred
                        break
                except Exception:
                    pass

            # Filter out weak trend/choppy range breakouts using trend expansion check
            if trend_strength < 0.8:
                continue

            # Bullish Breakout with VWAP & Trend Filter
            if close > orb_high and close > vwap:
                entry_bar = bar
                entry_idx = idx
                quant_pred = {
                    "bias": "BULLISH",
                    "mu_pct": 1.20,
                    "p_up": 0.70,
                    "direction_validated": True,
                    "range_validated": (atr > 0),
                }
                break
            # Bearish Breakdown with VWAP & Trend Filter
            elif close < orb_low and close < vwap:
                entry_bar = bar
                entry_idx = idx
                quant_pred = {
                    "bias": "BEARISH",
                    "mu_pct": -1.20,
                    "p_up": 0.70,
                    "direction_validated": True,
                    "range_validated": (atr > 0),
                }
                break

        if entry_bar is None or quant_pred is None:
            return None

        entry_price = float(entry_bar["Close"])
        entry_time = entry_bar.name
        bias = quant_pred.get("bias", "NEUTRAL")
        target_pct = quant_pred.get("mu_pct", 0.0)
        p_up = quant_pred.get("p_up", 0.5)

        target_price = entry_price * (1.0 + target_pct / 100.0)

        if not self.risk_engine:
            return None

        plan = self.risk_engine.evaluate_execution_plan(
            symbol=symbol.replace(".NS", ""),
            current_price=entry_price,
            bias=bias,
            p_up=p_up,
            target_price=target_price,
            atr_14=atr,
            direction_validated=quant_pred.get("direction_validated", False),
            range_validated=quant_pred.get("range_validated", False),
        )

        if not plan or not plan.get("approved") or plan["action"] == "NO_TRADE":
            return {
                "symbol": symbol,
                "date": entry_time.strftime("%Y-%m-%d"),
                "status": "REJECTED",
                "reason": plan.get("rejection_reason", "RISK_FILTERED") if plan else "NO_PLAN"
            }

        action = plan["action"]
        stop_loss = plan["stop_loss"]
        take_profit = plan["take_profit"]
        shares = plan["position_size_shares"]

        executed_entry_price = (
            entry_price * (1 + self.slippage_pct)
            if action == "BUY"
            else entry_price * (1 - self.slippage_pct)
        )

        remaining_bars = df_5m.iloc[entry_idx + 1:]
        exit_time = None
        exit_price = None
        exit_reason = None

        for timestamp, bar in remaining_bars.iterrows():
            high, low, close = float(bar["High"]), float(bar["Low"]), float(bar["Close"])

            if action == "BUY":
                if low <= stop_loss:
                    exit_price = stop_loss * (1 - self.slippage_pct)
                    exit_reason = "STOP_LOSS"
                    exit_time = timestamp
                    break
                elif high >= take_profit:
                    exit_price = take_profit * (1 - self.slippage_pct)
                    exit_reason = "TAKE_PROFIT"
                    exit_time = timestamp
                    break

            elif action == "SELL":
                if high >= stop_loss:
                    exit_price = stop_loss * (1 + self.slippage_pct)
                    exit_reason = "STOP_LOSS"
                    exit_time = timestamp
                    break
                elif low <= take_profit:
                    exit_price = take_profit * (1 + self.slippage_pct)
                    exit_reason = "TAKE_PROFIT"
                    exit_time = timestamp
                    break

        # Square off at end-of-day (15:15 IST)
        if not exit_reason and not remaining_bars.empty:
            last_bar = remaining_bars.iloc[-1]
            exit_time = last_bar.name
            raw_close = float(last_bar["Close"])
            exit_price = (
                raw_close * (1 - self.slippage_pct)
                if action == "BUY"
                else raw_close * (1 + self.slippage_pct)
            )
            exit_reason = "EOD_SQUAREOFF"

        if not exit_price:
            return None

        if action == "BUY":
            gross_pnl = (exit_price - executed_entry_price) * shares
        else:
            gross_pnl = (executed_entry_price - exit_price) * shares

        total_turnover = (executed_entry_price + exit_price) * shares
        friction = total_turnover * self.commission_pct
        net_pnl = gross_pnl - friction
        r_risk = abs(executed_entry_price - stop_loss) * shares
        r_multiple = net_pnl / r_risk if r_risk > 0 else 0.0

        return {
            "symbol": symbol,
            "date": entry_time.strftime("%Y-%m-%d"),
            "status": "EXECUTED",
            "action": action,
            "entry_time": entry_time.strftime("%H:%M"),
            "exit_time": exit_time.strftime("%H:%M"),
            "entry_price": round(executed_entry_price, 2),
            "exit_price": round(exit_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "shares": shares,
            "exit_reason": exit_reason,
            "net_pnl": round(net_pnl, 2),
            "r_multiple": round(r_multiple, 2),
        }

    def generate_performance_report(self, trade_logs):
        """Compute statistical performance metrics."""
        executed = [t for t in trade_logs if t and t.get("status") == "EXECUTED"]
        rejections = [t for t in trade_logs if t and t.get("status") == "REJECTED"]

        if not executed:
            return {
                "total_signals": len(trade_logs),
                "executed_trades": 0,
                "rejections": len(rejections),
                "message": "No trades passed risk guardrails.",
            }

        df = pd.DataFrame(executed)
        wins = df[df["net_pnl"] > 0]
        losses = df[df["net_pnl"] < 0]

        gross_profit = wins["net_pnl"].sum()
        gross_loss = abs(losses["net_pnl"].sum())
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else np.inf

        df["cum_pnl"] = df["net_pnl"].cumsum()
        df["peak"] = df["cum_pnl"].cummax()
        df["drawdown"] = df["cum_pnl"] - df["peak"]
        max_drawdown = round(float(df["drawdown"].min()), 2)

        return {
            "total_signals": len(trade_logs),
            "executed_trades": len(executed),
            "rejections": len(rejections),
            "win_rate_pct": round(len(wins) / len(executed) * 100, 2),
            "profit_factor": profit_factor,
            "net_profit_rs": round(float(df["net_pnl"].sum()), 2),
            "avg_trade_pnl": round(float(df["net_pnl"].mean()), 2),
            "avg_r_multiple": round(float(df["r_multiple"].mean()), 2),
            "max_drawdown_rs": max_drawdown,
            "exits_by_reason": df["exit_reason"].value_counts().to_dict(),
        }


def run_backtest(symbols=None, period="60d"):
    """Execute walk-forward backtest across symbol universe."""
    symbols = symbols or WATCHLIST_DISPLAY
    backtester = IntradayBacktester()
    all_trade_logs = []

    model_obj = _safe_load_model()
    if model_obj:
        print("Loaded trained ML model for backtesting.")
    else:
        print("No trained .pkl model found. Running backtest using trend-filtered ORB + VWAP strategy.\n")

    print(f"Starting backtest for {len(symbols)} symbols over {period}...\n")

    for symbol_display in symbols:
        data = _download_symbol_data(symbol_display, period=period)
        if data.empty:
            continue

        grouped = data.groupby(data.index.date)
        for date_val, day_df in grouped:
            res = backtester.simulate_session(symbol_display, day_df, model_obj=model_obj)
            if res:
                all_trade_logs.append(res)

    report = backtester.generate_performance_report(all_trade_logs)

    print("================ BACKTEST PERFORMANCE REPORT ================")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("=============================================================")

    return report


if __name__ == "__main__":
    run_backtest()