"""Train and serve causal 15/30/60-minute intraday return forecasts.

The trainer uses only feature rows recorded before their outcomes were known.  It
validates with expanding, day-by-day walk-forward predictions and leaves a horizon
in shadow mode unless both the pooled and stock-specific results pass their gates.
"""
import argparse
import datetime
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from muthoot_poc.intraday_observations import FEATURE_VERSION, HORIZONS_MINUTES


FEATURE_NAMES = (
    "return_5m_pct", "return_15m_pct", "return_30m_pct", "move_since_open_pct",
    "vwap_distance_pct", "rsi_14", "macd_hist", "atr_pct", "last_bar_range_pct",
    "last_bar_volume_ratio", "session_volume_ratio", "minutes_since_open",
)
MIN_HISTORY_SESSIONS = 40
MIN_TRAIN_SESSIONS = 20
MIN_OOS_SESSIONS = 15
RIDGE_LAMBDA = 25.0


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


def _safe_corr(predicted: np.ndarray, actual: np.ndarray) -> float:
    if len(predicted) < 2 or np.std(predicted) == 0 or np.std(actual) == 0:
        return 0.0
    return float(np.corrcoef(predicted, actual)[0, 1])


def records_to_frame(records: List[Dict[str, Any]], horizon: int) -> pd.DataFrame:
    """Convert scored observations into model rows for one horizon."""
    rows = []
    key = str(horizon)
    for observation in records or []:
        if observation.get("feature_version") != FEATURE_VERSION:
            continue
        features = observation.get("features") or {}
        outcome = (observation.get("outcomes") or {}).get(key) or {}
        if outcome.get("status") != "scored" or not _finite(outcome.get("actual_return_pct")):
            continue
        values = [features.get(name) for name in FEATURE_NAMES]
        if not all(_finite(value) for value in values):
            continue
        rows.append({
            "date": str(pd.Timestamp(observation["bar_time"]).date()),
            "symbol": observation["symbol"],
            "target": float(outcome["actual_return_pct"]),
            **{name: float(features[name]) for name in FEATURE_NAMES},
        })
    return pd.DataFrame(rows)


def fit_ridge(rows: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if rows.empty or len(rows) < 100:
        return None
    x = rows.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    y = rows["target"].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * RIDGE_LAMBDA
    penalty[0, 0] = 0.0
    system = design.T @ design + penalty
    target = design.T @ y
    try:
        weights = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        # Retain a usable shadow forecast if a thin or collinear sample makes
        # the normal equation numerically singular.
        weights = np.linalg.lstsq(system, target, rcond=None)[0]
    residuals = y - design @ weights
    return {
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "intercept": float(weights[0]),
        "weights": weights[1:].tolist(),
        "residual_sd": float(max(np.std(residuals), 0.01)),
    }


def predict_rows(fit: Dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    x = rows.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    mean = np.asarray(fit["mean"], dtype=float)
    scale = np.asarray(fit["scale"], dtype=float)
    weights = np.asarray(fit["weights"], dtype=float)
    return float(fit["intercept"]) + ((x - mean) / scale) @ weights


def walk_forward(rows: pd.DataFrame) -> pd.DataFrame:
    """Expanding day-by-day validation. A day's bars never train its forecast."""
    dates = sorted(rows["date"].unique())
    predictions = []
    for index in range(MIN_TRAIN_SESSIONS, len(dates)):
        train = rows[rows["date"].isin(dates[:index])]
        test = rows[rows["date"] == dates[index]].copy()
        fit = fit_ridge(train)
        if fit is None or test.empty:
            continue
        test["prediction"] = predict_rows(fit, test)
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def _metrics(rows: pd.DataFrame, n_boot: int = 300, seed: int = 0) -> Dict[str, Any]:
    if rows.empty:
        return {"n": 0, "n_days": 0, "passes_gate": False}
    daily = []
    for _, group in rows.groupby("date"):
        predicted = group["prediction"].to_numpy(dtype=float)
        actual = group["target"].to_numpy(dtype=float)
        daily.append([
            len(group), predicted.sum(), actual.sum(), (predicted * actual).sum(),
            (predicted * predicted).sum(), (actual * actual).sum(),
            float((np.sign(predicted) == np.sign(actual)).sum()),
            float(np.abs(predicted - actual).sum()), float(np.abs(actual).sum()),
        ])
    blocks = np.asarray(daily, dtype=float)
    total = blocks.sum(axis=0)

    def values(block):
        n, sx, sy, sxy, sxx, syy, hit, mae_model, mae_flat = block
        corr_den = math.sqrt(max(n * sxx - sx * sx, 0.0) * max(n * syy - sy * sy, 0.0))
        return (
            (n * sxy - sx * sy) / corr_den if corr_den > 0 else 0.0,
            hit / n if n else 0.0,
            mae_model / n if n else 0.0,
            mae_flat / n if n else 0.0,
        )

    rng = np.random.default_rng(seed)
    draws = blocks[rng.integers(0, len(blocks), size=(n_boot, len(blocks)))].sum(axis=1)
    boot = np.asarray([values(draw) for draw in draws])
    ic, hit_rate, mae_model, mae_flat = values(total)
    ic_ci = np.quantile(boot[:, 0], [0.025, 0.975]).tolist()
    hit_ci = np.quantile(boot[:, 1], [0.025, 0.975]).tolist()
    passes = bool(
        len(blocks) >= MIN_OOS_SESSIONS and ic_ci[0] > 0 and hit_ci[0] > 0.5 and mae_model <= mae_flat
    )
    return {
        "n": int(total[0]), "n_days": int(len(blocks)),
        "ic": round(float(ic), 4), "ic_ci95": [round(float(value), 4) for value in ic_ci],
        "hit_rate": round(float(hit_rate), 4), "hit_ci95": [round(float(value), 4) for value in hit_ci],
        "mae_model_pct": round(float(mae_model), 4), "mae_flat_pct": round(float(mae_flat), 4),
        "passes_gate": passes,
    }


def train(records: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    sessions = sorted({str(pd.Timestamp(row["bar_time"]).date()) for row in records or [] if row.get("bar_time")})
    report = {
        "feature_version": FEATURE_VERSION,
        "sessions_available": len(sessions),
        "minimum_sessions": MIN_HISTORY_SESSIONS,
        "horizons": {},
    }
    if len(sessions) < MIN_HISTORY_SESSIONS:
        report["status"] = "collecting"
        report["reason"] = "More completed sessions are required before training."
        return None, report

    trained_horizons, symbol_gates = {}, {}
    for horizon in HORIZONS_MINUTES:
        rows = records_to_frame(records, horizon)
        oos = walk_forward(rows)
        overall = _metrics(oos, seed=horizon)
        by_symbol, symbol_fits = {}, {}
        for index, (symbol, symbol_rows) in enumerate(rows.groupby("symbol")):
            symbol_oos = walk_forward(symbol_rows)
            by_symbol[symbol] = _metrics(symbol_oos, seed=horizon + index + 1)
            symbol_fit = fit_ridge(symbol_rows)
            if symbol_fit is not None:
                symbol_fits[symbol] = symbol_fit
        fit = fit_ridge(rows)
        report["horizons"][str(horizon)] = {"overall": overall, "by_symbol": by_symbol}
        if fit is not None:
            trained_horizons[str(horizon)] = {
                "fit": fit,
                "gate": overall["passes_gate"],
                "symbol_fits": symbol_fits,
            }
            symbol_gates[str(horizon)] = {symbol: metrics["passes_gate"] for symbol, metrics in by_symbol.items()}

    if not trained_horizons:
        report["status"] = "insufficient_rows"
        report["reason"] = "Recorded sessions did not contain enough complete feature rows."
        return None, report

    now = datetime.datetime.now(datetime.timezone.utc)
    model = {
        "version": "intraday-v1-" + now.strftime("%Y%m%d"),
        "trained_at": now.isoformat(timespec="seconds"),
        "feature_version": FEATURE_VERSION,
        "sessions": len(sessions),
        "horizons": trained_horizons,
        "symbol_gates": symbol_gates,
    }
    report["status"] = "trained"
    return model, report


def predict(model: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
    """Return raw estimates and validation state for each trained horizon."""
    if not model or observation.get("feature_version") != model.get("feature_version"):
        return {"status": "unavailable", "forecasts": {}}
    features = observation.get("features") or {}
    if not all(_finite(features.get(name)) for name in FEATURE_NAMES):
        return {"status": "incomplete_features", "forecasts": {}}

    row = pd.DataFrame([{name: float(features[name]) for name in FEATURE_NAMES}])
    entry = float(observation["entry_price"])
    forecasts = {}
    for horizon, payload in (model.get("horizons") or {}).items():
        # A stock's published forecast is fitted to that stock's own history.
        # The pooled fit remains research fallback only and can never validate a
        # stock that lacks its own out-of-sample evidence.
        fit = (payload.get("symbol_fits") or {}).get(observation["symbol"]) or payload.get("fit")
        if not fit:
            continue
        expected_return = float(predict_rows(fit, row)[0])
        residual_sd = float(fit.get("residual_sd", 1.0))
        symbol_valid = bool((model.get("symbol_gates", {}).get(horizon) or {}).get(observation["symbol"]))
        validated = symbol_valid
        forecasts[horizon] = {
            "expected_return_pct": round(expected_return, 4),
            "target_price": round(entry * (1.0 + expected_return / 100.0), 2),
            "band_low_pct": round(expected_return - 1.28 * residual_sd, 4),
            "band_high_pct": round(expected_return + 1.28 * residual_sd, 4),
            "status": "validated" if validated else "shadow",
        }
    return {"status": "ready" if forecasts else "unavailable", "forecasts": forecasts,
            "model_version": model.get("version")}


def _write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the causal intraday forecast model from recorded observations.")
    parser.add_argument("--input", default="public/data_store/intraday_training_history.json")
    parser.add_argument("--model", default="public/data_store/intraday_model.json")
    parser.add_argument("--report", default="public/data_store/intraday_backtest_report.json")
    args = parser.parse_args(argv)
    try:
        with open(args.input, encoding="utf-8") as handle:
            records = json.load(handle)
    except FileNotFoundError:
        records = []
    model, report = train(records if isinstance(records, list) else [])
    _write_json(args.report, report)
    if model:
        _write_json(args.model, model)
    print(f"Intraday model: {report['status']} ({report['sessions_available']}/{report['minimum_sessions']} sessions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
