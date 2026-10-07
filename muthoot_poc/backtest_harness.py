"""
Muthoot POC - Hybrid Quant Backtest Harness (v4 - hardened engine; same strategy rules as v3.8)
File: muthoot_poc/backtest_harness.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
SYMBOLS = [
    "MUTHOOTFIN.NS", "MANAPPURAM.NS", "SBIN.NS", "ICICIBANK.NS",
    "AXISBANK.NS", "HDFCBANK.NS", "BAJFINANCE.NS", "CHOLAFIN.NS",
    "RELIANCE.NS", "INFY.NS", "TCS.NS"
]
NIFTY_SYMBOL = "^NSEI"
ALLOW_SYNTHETIC = os.getenv("ALLOW_SYNTHETIC", "") == "1"   # offline dry runs only; never silently
MIN_TRADES = 30                 # below this the numbers are noise and no verdict is given
RUN_INFO = {"source": "unknown"}

TOTAL_DAYS = 60
TRAIN_DAYS = 34
TEST_DAYS = 24

RVOL_THRESHOLD = 1.35          # Increased volume confirmation threshold
ML_PROB_THRESHOLD = 0.50       # Classifier / Ensemble cutoff
ATR_SL_MULTIPLIER = 1.2        # Dynamic stop loss: 1.2x ATR
ATR_TP_MULTIPLIER = 2.0        # Target profit: 2.0x ATR (1:1.67 R/R)
MAX_EMA_DISPLACEMENT = 2.0     # Max ATR distance from EMA20
SLIPPAGE_PCT = 0.0005          # 0.05% slippage on fills

FEATURE_COLS = ["rvol", "dist_vwap", "dist_ema20", "dist_sma50", "squeeze_duration", "atr_pct"]
REQUIRED_COLS = FEATURE_COLS + ["atr", "bb_upper", "vwap", "sma50", "ema20"]


# =====================================================================
# TECHNICAL INDICATORS & FEATURE ENGINEERING
# =====================================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes technical indicators, breakout conditions, and ML features."""
    df = df.copy()

    # 1. Average True Range (ATR 14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()          # no bfill: it copied FUTURE values into the first rows
    df["atr_pct"] = df["atr"] / df["close"]

    # 2. INTRADAY SESSION-RESET VWAP
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = df["volume"] * tp
    dates = df.index.date

    cum_pv = pv.groupby(dates).cumsum()
    cum_vol = df["volume"].groupby(dates).cumsum()
    df["vwap"] = cum_pv / cum_vol

    # 3. Moving Averages & Bands
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["std20"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["sma20"] + (2 * df["std20"])
    df["bb_lower"] = df["sma20"] - (2 * df["std20"])

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["kc_upper"] = df["ema20"] + (1.5 * df["atr"])
    df["kc_lower"] = df["ema20"] - (1.5 * df["atr"])

    # Base Squeeze Condition
    df["is_squeeze"] = (df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])

    # Squeeze Memory: Look back up to 5 bars for recent consolidation
    df["had_recent_squeeze"] = (
        df["is_squeeze"].shift(1).rolling(5).max().fillna(0).astype(bool)
    )

    # 4. Relative Volume (RVOL 20-MA)
    vol_ma = df["volume"].rolling(20).mean().replace(0, np.nan)
    df["rvol"] = (df["volume"] / vol_ma).fillna(1.0)

    # 5. Trend & Overextension Features
    df["dist_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]
    df["dist_ema20"] = (df["close"] - df["ema20"]) / df["atr"]
    df["dist_sma50"] = (df["close"] - df["sma50"]) / df["sma50"]
    
    is_sq = df["is_squeeze"].astype(int)
    df["squeeze_duration"] = is_sq.groupby((~df["is_squeeze"]).cumsum()).cumsum()

    return df


# =====================================================================
# DATA FETCHING / SYNTHESIS FALLBACK
# =====================================================================
def fetch_data():
    """Attempts to fetch real data via yfinance, falling back to synthetic dataset if offline."""
    data = {}
    try:
        import yfinance as yf
        print(f"[*] Fetching {TOTAL_DAYS}d data for {len(SYMBOLS)} stocks + Nifty Index...")

        tickers_to_fetch = SYMBOLS + [NIFTY_SYMBOL]
        raw_data = yf.download(
            tickers_to_fetch,
            period=f"{TOTAL_DAYS}d",
            interval="15m",
            progress=False,
            group_by="ticker",
        )

        for sym in SYMBOLS:
            if isinstance(raw_data.columns, pd.MultiIndex):
                if sym in raw_data.columns.levels[0]:
                    df = raw_data[sym].dropna().copy()
                    df.columns = [c.lower() for c in df.columns]
                    data[sym] = compute_indicators(df)
            elif sym in raw_data:
                df = raw_data[sym].dropna().copy()
                df.columns = [c.lower() for c in df.columns]
                data[sym] = compute_indicators(df)

        if NIFTY_SYMBOL in raw_data:
            ndf = (
                raw_data[NIFTY_SYMBOL].dropna().copy()
                if isinstance(raw_data.columns, pd.MultiIndex)
                else raw_data.dropna().copy()
            )
            ndf.columns = [c.lower() for c in ndf.columns]
            ndf["ema20"] = ndf["close"].ewm(span=20, adjust=False).mean()
            ndf["regime_bullish"] = ndf["close"] > ndf["ema20"]
            data["NIFTY"] = ndf

        missing = [s for s in SYMBOLS if s not in data]
        if "NIFTY" not in data or len(missing) > 2:
            raise ValueError(f"Insufficient data from yfinance (missing: {missing}, index present: {'NIFTY' in data})")
        if missing:
            print(f"[!] Continuing without: {missing}")
        RUN_INFO["source"] = "yfinance"

    except Exception as e:
        if not ALLOW_SYNTHETIC:
            print(f"[X] Live data unavailable ({e}). Refusing to report on synthetic data (set ALLOW_SYNTHETIC=1 for offline dry runs).")
            sys.exit(2)
        print(f"[!] Live fetch unavailable ({e}). USING SYNTHETIC DATA - nothing below is a real backtest.")
        RUN_INFO["source"] = "synthetic"
        np.random.seed(42)
        timestamps = pd.date_range(end=datetime.now(), periods=TOTAL_DAYS * 25, freq="15min")

        # Synthesize Nifty
        nifty_closes = 24000 + np.cumsum(np.random.randn(len(timestamps)) * 15)
        n_df = pd.DataFrame({
            "open": nifty_closes, "high": nifty_closes + 10,
            "low": nifty_closes - 10, "close": nifty_closes, "volume": 100000
        }, index=timestamps)
        n_df["ema20"] = n_df["close"].ewm(span=20, adjust=False).mean()
        n_df["regime_bullish"] = n_df["close"] > n_df["ema20"]
        data["NIFTY"] = n_df

        # Synthesize Stocks
        for sym in SYMBOLS:
            base_price = np.random.uniform(200, 2000)
            returns = np.random.normal(0.0001, 0.008, len(timestamps))
            prices = base_price * np.exp(np.cumsum(returns))

            s_df = pd.DataFrame({
                "open": prices,
                "high": prices * (1 + np.abs(np.random.normal(0, 0.003, len(timestamps)))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.003, len(timestamps)))),
                "close": prices,
                "volume": np.random.randint(10000, 500000, len(timestamps))
            }, index=timestamps)

            data[sym] = compute_indicators(s_df)

    return data


# =====================================================================
# ML MODEL & HEURISTIC EVALUATOR
# =====================================================================
def train_classifier(data_dict, split_ts):
    """Extracts historical squeeze setups and trains ML classifier or heuristic score."""
    X_train, y_train = [], []

    for sym, df in data_dict.items():
        if sym == "NIFTY":
            continue

        train_sub = df[df.index < split_ts]          # split by TIMESTAMP so every symbol shares one cut-off
        for i in range(20, len(train_sub) - 10):
            row = train_sub.iloc[i]
            if row[REQUIRED_COLS].isna().any():
                continue

            is_breakout = (row["close"] > row["bb_upper"]) and (row["close"] > row["vwap"]) and (row["close"] > row["sma50"])
            is_not_overextended = row["dist_ema20"] <= MAX_EMA_DISPLACEMENT
            
            if row["had_recent_squeeze"] and is_breakout and is_not_overextended and (row["rvol"] >= RVOL_THRESHOLD):
                future_bars = train_sub.iloc[i + 1 : i + 11]
                future_bars = future_bars[future_bars.index.date == row.name.date()]   # same-day outcomes, like the live rules
                if future_bars.empty:
                    continue
                entry = row["close"]
                tp = entry + (ATR_TP_MULTIPLIER * row["atr"])
                sl = entry - (ATR_SL_MULTIPLIER * row["atr"])

                hit_tp = (future_bars["high"] >= tp).any()
                hit_sl = (future_bars["low"] <= sl).any()

                label = 1 if (hit_tp and not hit_sl) else 0
                X_train.append(row[FEATURE_COLS].values)
                y_train.append(label)

    if len(X_train) >= 25:
        X_train, y_train = np.array(X_train), np.array(y_train)
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_train, y_train)
        return clf, len(X_train)
    else:
        # Heuristic evaluator if sample size is too small
        return None, len(X_train)


def evaluate_signal(clf, bar):
    """Returns probability score from ML model or fallback heuristic matrix."""
    if clf is not None:
        features = bar[FEATURE_COLS].values.reshape(1, -1)
        return clf.predict_proba(features)[0][1]
    
    # Fallback Quantitative Score
    score = 0.50
    if bar["rvol"] >= 1.5: score += 0.10
    if bar["dist_sma50"] > 0.005: score += 0.08
    if bar["squeeze_duration"] >= 3: score += 0.07
    if bar["dist_ema20"] > 1.8: score -= 0.15
    return score


# =====================================================================
# BACKTEST ENGINE
# =====================================================================
def fill_entry(bar, next_bar):
    """Limit order just above the upper band. If price never trades down to it the order is NOT filled (no trade).
    v3.8 bought at the open instead, which kept every runaway winner a real limit order would have missed."""
    limit = bar["bb_upper"] + 0.15 * bar["atr"]
    if next_bar["open"] <= limit:
        return next_bar["open"] * (1 + SLIPPAGE_PCT)
    if next_bar["low"] <= limit:
        return limit * (1 + SLIPPAGE_PCT)
    return None


def simulate_trade(test_df, i, entry_price, atr_val, ema20):
    """Manage one long from the bar after signal bar i. Same-day only: never held overnight."""
    day = test_df.index[i].date()
    stop = min(entry_price - ATR_SL_MULTIPLIER * atr_val, ema20)
    take_profit = entry_price + ATR_TP_MULTIPLIER * atr_val
    risk = entry_price - stop
    exit_price, reason, last_j = None, None, i + 1
    for j in range(i + 1, min(i + 16, len(test_df))):
        f = test_df.iloc[j]
        if f.name.date() != day:
            break
        last_j = j
        if f["low"] <= stop:                                     # stop first: conservative when both levels trade in one bar
            exit_price, reason = stop * (1 - SLIPPAGE_PCT), "STOP_LOSS"
            break
        if f["high"] >= take_profit:
            exit_price, reason = take_profit, "TAKE_PROFIT"
            break
        if f["close"] < f["vwap"]:
            exit_price, reason = f["close"] * (1 - SLIPPAGE_PCT), "VWAP_BREAKDOWN"
            break
    if exit_price is None:
        exit_price = test_df.iloc[last_j]["close"] * (1 - SLIPPAGE_PCT)
        reason = "TIME_STOP" if (last_j - i) >= 15 else "END_OF_DAY"
    return {"ret_pct": (exit_price / entry_price - 1) * 100.0, "pnl": exit_price - entry_price,
            "r_multiple": (exit_price - entry_price) / risk if risk > 0 else 0.0, "exit_reason": reason, "exit_idx": last_j}


def bootstrap_mean_ci(trades_df, n_boot=2000, seed=0):
    """95% interval for the mean return, resampling whole trading DAYS (same-day trades are correlated)."""
    g = trades_df.groupby(trades_df["entry_ts"].str[:10])["ret_pct"].agg(["sum", "count"]).values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    s = g[idx].sum(axis=1)
    means = s[:, 0] / s[:, 1]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_backtest():
    print("================ HYBRID QUANT BACKTEST (v4 - hardened) ================")
    data = fetch_data()
    nifty_df = data["NIFTY"]
    split_ts = nifty_df.index[int(len(nifty_df) * (TRAIN_DAYS / TOTAL_DAYS))]
    nifty_regime = nifty_df["regime_bullish"][~nifty_df.index.duplicated(keep="last")]
    print(f"[*] Data source: {RUN_INFO['source']} | train < {split_ts} <= test (split by timestamp, identical for every symbol)")

    clf, train_samples = train_classifier(data, split_ts)
    model_type = "RandomForest" if clf is not None else "Quantitative Heuristic Ensemble"
    print(f"[+] Signal Evaluator ({model_type}) initialized with {train_samples} training samples.\n")

    diagnostics = {"Total Scanned Bars": 0, "Passed Squeeze Memory": 0, "Passed Trend & Breakout": 0,
                   "Passed Overextension Check": 0, "Passed RVOL Gate": 0, "Passed Signal Gate": 0,
                   "Unfilled limit orders": 0, "Skipped (already in position)": 0, "Executed Trades": 0}
    trades = []
    for sym, df in data.items():
        if sym == "NIFTY":
            continue
        test_df = df[df.index >= split_ts].copy()
        busy_until = -1                                          # one position per symbol at a time
        for i in range(20, len(test_df) - 15):
            if i <= busy_until:
                diagnostics["Skipped (already in position)"] += 1
                continue
            bar, next_bar = test_df.iloc[i], test_df.iloc[i + 1]
            diagnostics["Total Scanned Bars"] += 1
            if bar[REQUIRED_COLS].isna().any():
                continue
            if not bar["had_recent_squeeze"]:
                continue
            diagnostics["Passed Squeeze Memory"] += 1
            if not (bar["close"] > bar["bb_upper"] and bar["close"] > bar["vwap"] and bar["close"] > bar["sma50"]):
                continue
            diagnostics["Passed Trend & Breakout"] += 1
            if bar["dist_ema20"] > MAX_EMA_DISPLACEMENT:
                continue
            diagnostics["Passed Overextension Check"] += 1
            if bar["rvol"] < RVOL_THRESHOLD:
                continue
            diagnostics["Passed RVOL Gate"] += 1
            regime = nifty_regime.get(bar.name)
            if regime is not None and not bool(regime):
                continue
            if evaluate_signal(clf, bar) < ML_PROB_THRESHOLD:
                continue
            diagnostics["Passed Signal Gate"] += 1
            if next_bar.name.date() != bar.name.date():
                continue                                         # never open a position into an overnight gap
            entry = fill_entry(bar, next_bar)
            if entry is None:
                diagnostics["Unfilled limit orders"] += 1
                continue
            trade = simulate_trade(test_df, i, entry, bar["atr"], bar["ema20"])
            busy_until = trade.pop("exit_idx")
            trades.append({"symbol": sym, "entry_ts": str(bar.name), **trade})
            diagnostics["Executed Trades"] += 1

    print("--- PIPELINE FILTER DIAGNOSTICS ---")
    for k, v in diagnostics.items():
        print(f"  {k:<30}: {v}")
    print("-----------------------------------\n")

    n = len(trades)
    print("================ OUT-OF-SAMPLE PERFORMANCE REPORT ================")
    print(f"  data_source: {RUN_INFO['source']}" + ("   <-- SYNTHETIC: NOT A BACKTEST" if RUN_INFO["source"] == "synthetic" else ""))
    print(f"  out_of_sample_trades: {n}   (at least {MIN_TRADES} needed before any verdict is given)")
    verdict = "INSUFFICIENT_SAMPLE"
    if n:
        td = pd.DataFrame(trades)
        r = td["ret_pct"]
        wins, losses = r[r > 0], r[r < 0]
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
        cum = r.cumsum()
        lo, hi = bootstrap_mean_ci(td)
        print(f"  win_rate_pct: {len(wins) / n * 100:.1f}")
        print(f"  avg_return_pct_per_trade: {r.mean():+.3f}   (95% range {lo:+.3f} to {hi:+.3f}; after {SLIPPAGE_PCT * 200:.2f}% round-trip slippage)")
        print(f"  profit_factor: {pf:.2f}")
        print(f"  avg_r_multiple: {td['r_multiple'].mean():+.2f}")
        print(f"  max_drawdown_pct_points: -{(cum.cummax() - cum).max():.2f}")
        print(f"  exits_by_reason: {td['exit_reason'].value_counts().to_dict()}")
        if n >= MIN_TRADES and RUN_INFO["source"] != "synthetic":
            verdict = "EDGE_SUPPORTED" if lo > 0 else "NO_EDGE_DETECTED"
    if RUN_INFO["source"] == "synthetic":
        verdict = "SYNTHETIC_DATA"
    print(f"  verdict: {verdict}")
    print("  note: thresholds in this file were tuned over several versions; treat the test window as partly in-sample.")
    print("==================================================================")
    return verdict

if __name__ == "__main__":
    run_backtest()