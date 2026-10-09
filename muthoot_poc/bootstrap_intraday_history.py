"""Build an initial intraday training set from Yahoo's recent five-minute history.

Yahoo supplies only a short intraday window, so this is a bootstrap and shadow-model
source, not a substitute for a durable market-data vendor.  Features are created
from each bar's past only; outcomes are attached after the session slice is complete.
"""
import argparse
import json
import os
import time

import pandas as pd
import pytz
import yfinance as yf

from muthoot_poc import intraday_model as model
from muthoot_poc import intraday_observations as observations
from muthoot_poc.indicators import calculate_technical_snapshot


IST = pytz.timezone("Asia/Kolkata")
SYMBOLS = ("MUTHOOTFIN", "RELIANCE", "TMPV", "INFY", "HDFCBANK", "ICICIBANK",
           "TCS", "SBIN", "BHARTIARTL", "LT", "SUNPHARMA")


def _write_json(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def _normalise(frame):
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    index = data.index
    data.index = index.tz_localize("UTC").tz_convert(IST) if index.tz is None else index.tz_convert(IST)
    data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return data.between_time("09:15", "15:25")


def _session_metrics(frame):
    volume = frame["Volume"].astype(float)
    baseline = float(volume.mean()) if len(volume) else 0.0
    return {"vol_ratio": round(float(volume.iloc[-1]) / baseline, 2) if baseline > 0 else 1.0}


def build_observations(symbol, frame):
    rows = []
    if frame.empty:
        return rows
    for _, session in frame.groupby(frame.index.date):
        session = session.copy()
        session_rows = []
        for index in range(6, len(session)):
            known = session.iloc[:index + 1]
            technical = calculate_technical_snapshot(known)
            row = observations.capture(symbol, known, technical, _session_metrics(known))
            if row:
                session_rows.append(row)
        rows.extend(observations.update_outcomes(session_rows, session))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bootstrap causal intraday observations from recent Yahoo 5-minute bars.")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--history", default="public/data_store/intraday_training_history.json")
    parser.add_argument("--model", default="public/data_store/intraday_model.json")
    parser.add_argument("--report", default="public/data_store/intraday_backtest_report.json")
    args = parser.parse_args(argv)

    try:
        with open(args.history, encoding="utf-8") as handle:
            history = json.load(handle)
    except FileNotFoundError:
        history = []

    fresh = []
    for symbol in SYMBOLS:
        ticker = f"{symbol}.NS"
        try:
            frame = _normalise(yf.Ticker(ticker).history(period=args.period, interval="5m", auto_adjust=False))
            symbol_rows = build_observations(symbol, frame)
            fresh.extend(symbol_rows)
            print(f"{symbol}: {len(symbol_rows)} causal observations")
        except Exception as error:
            print(f"{symbol}: skipped ({type(error).__name__}: {error})")
        time.sleep(0.5)

    merged = observations.merge_history(history if isinstance(history, list) else [], fresh)
    _write_json(args.history, merged)
    trained, report = model.train(merged)
    report["bootstrap"] = {"source": "Yahoo Finance 5-minute history", "fresh_rows": len(fresh)}
    _write_json(args.report, report)
    if trained:
        _write_json(args.model, trained)
    print(f"Saved {len(merged)} observations; intraday model is {report['status']}.")


if __name__ == "__main__":
    main()
