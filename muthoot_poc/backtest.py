"""Walk-forward validation + training of the quantitative model.

    python -m muthoot_poc.backtest                  # download ~10y of daily bars, validate, write model + report
    python -m muthoot_poc.backtest --synthetic      # offline dry run on simulated data (checks the machinery only)

Writes (to DATA_DIR, default public/data_store):
    backtest_report.json   out-of-sample evidence, with day-block bootstrap confidence intervals
    quant_model.json       the model trained on all data, plus which gates it passed

The live pipeline only publishes a directional forecast if the direction gate passed, and a range band if the
range gate passed. Nothing here is ever tuned on the out-of-sample predictions it reports.
"""
import argparse
import datetime
import json
import math
import os
import sys

import numpy as np
import pandas as pd

from muthoot_poc import quant_model as qm

DATA_DIR = os.getenv("DATA_DIR", "public/data_store")
INDEX_TICKER = "^NSEI"
WATCHLIST = ["MUTHOOTFIN", "RELIANCE", "TMPV", "INFY", "HDFCBANK", "ICICIBANK", "TCS", "SBIN", "BHARTIARTL", "LT", "SUNPHARMA"]
# More liquid large caps = more training data. The model is volatility-scaled, so it transfers to the watchlist.
UNIVERSE_EXTRA = ["ITC", "KOTAKBANK", "AXISBANK", "HINDUNILVR", "BAJFINANCE", "MARUTI", "ASIANPAINT", "TITAN",
                  "ULTRACEMCO", "WIPRO", "NTPC", "POWERGRID", "ONGC", "TATASTEEL", "JSWSTEEL", "HCLTECH", "TECHM",
                  "NESTLEIND", "COALINDIA", "BAJAJFINSV", "DRREDDY", "CIPLA", "HINDALCO", "GRASIM", "EICHERMOT"]
MIN_TRAIN_ROWS = 500


# --------------------------------------------------------------------------- serialization helper
def sanitize_for_json(obj):
    """Recursively converts NumPy scalars and replaces float NaN/Inf values with None for standard JSON encoding."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return [sanitize_for_json(v) for v in obj.tolist()]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


# --------------------------------------------------------------------------- data
def download_daily(tickers, period="10y"):
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period, interval="1d")
            if df is not None and len(df) > 300:
                out[t] = qm.clean_daily(df)
            else:
                print(f"[skip] {t}: only {0 if df is None else len(df)} rows")
        except Exception as e:
            print(f"[skip] {t}: {type(e).__name__}: {e}")
    return out


def make_synthetic(n_days=1500, n_sym=8, beta=0.0, seed=1):
    """Simulated market: clustering volatility, a shared market factor, optional planted gap-reversal edge (beta)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_days)
    f_gap, f_oc = rng.standard_normal(n_days), rng.standard_normal(n_days)

    def series(base_sig, own_beta, is_index):
        ls, logsig = np.log(base_sig), np.empty(n_days)
        for t in range(n_days):
            ls = 0.97 * ls + 0.03 * np.log(base_sig) + 0.12 * rng.standard_normal()
            logsig[t] = ls
        sig, close, rows = np.exp(logsig), 100.0, []
        for t in range(n_days):
            gz = f_gap[t] if is_index else 0.5 * f_gap[t] + 0.87 * rng.standard_normal()
            noise = f_oc[t] if is_index else 0.5 * f_oc[t] + 0.87 * rng.standard_normal()
            o = close * math.exp(0.6 * sig[t] * gz)
            c = o * math.exp(sig[t] * (-own_beta * gz + noise))
            hi = max(o, c) * math.exp(0.5 * sig[t] * abs(rng.standard_normal()))
            lo = min(o, c) * math.exp(-0.5 * sig[t] * abs(rng.standard_normal()))
            rows.append((o, hi, lo, c))
            close = c
        return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=dates)

    idx = series(0.008, 0.0, True)
    return {f"SYN{i}": series(rng.uniform(0.01, 0.02), beta, False) for i in range(n_sym)}, idx


def build_panel(daily, idx):
    parts = []
    for sym, d in daily.items():
        f = qm.build_features(d, idx)
        f["symbol"], f["date"] = sym, f.index
        parts.append(f)
    panel = pd.concat(parts).dropna(subset=["sig", "r_oc"])
    return panel.reset_index(drop=True)


# --------------------------------------------------------------------------- walk-forward
def walk_forward(panel, init_days=504, step=21):
    """Expanding window: fit on everything before a block, predict the block, move on. No row is ever predicted by
    a model that saw it (or anything after it)."""
    dates = np.array(sorted(panel["date"].unique()))
    if len(dates) < init_days + 2 * step:
        init_days = max(120, int(len(dates) * 0.5))
    keep = ["date", "symbol", "r_oc", "sig", "z_oc", "y_range", "lr5", "ret1_z", "gap_z"]
    chunks = []
    for s in range(init_days, len(dates), step):
        train = panel[panel["date"] < dates[s]]
        test = panel[panel["date"] >= dates[s]] if s + step >= len(dates) else \
            panel[(panel["date"] >= dates[s]) & (panel["date"] < dates[s + step])]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        pred = qm.predict_rows(qm.fit_models(train), test)
        chunks.append(pd.concat([test[keep], pred], axis=1))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks).dropna(subset=["mu_z", "range_f"]).reset_index(drop=True)


# --------------------------------------------------------------------------- statistics
def _corr(s, c):
    n, sx, sy, sxy, sxx, syy = (s[c[k]] for k in ("one", "x", "y", "xy", "xx", "yy"))
    den = math.sqrt(max(n * sxx - sx * sx, 0) * max(n * syy - sy * sy, 0))
    return (n * sxy - sx * sy) / den if den > 0 else 0.0


def evaluate(oos, cost_bps=6.0, n_boot=500, seed=0):
    r, sig = oos["r_oc"].values, oos["sig"].values
    mu_ret = oos["mu_z"].values * sig
    d = np.sign(mu_ret)
    x, y = oos["mu_z"].values, oos["z_oc"].clip(-qm.CLIP, qm.CLIP).values
    hit = (d == np.sign(r)).astype(float)
    net = d * r * 1e4 - cost_bps                              # bps per trade if you traded the sign every time
    inband = ((r >= oos["lo"].values) & (r <= oos["hi"].values)).astype(float)

    T = pd.DataFrame({"one": 1.0, "x": x, "y": y, "xy": x * y, "xx": x * x, "yy": y * y,
                      "hit": hit, "net": net, "inband": inband})
    per = T.groupby(oos["date"].values).sum()                 # one row per trading day -> block bootstrap
    S, c = per.values, {k: i for i, k in enumerate(per.columns)}
    brng = np.random.default_rng(seed)
    draws = S[brng.integers(0, len(S), size=(n_boot, len(S)))].sum(axis=1)

    def ci(func):
        v = np.array([func(row) for row in draws])
        return [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]

    tot = S.sum(axis=0)
    ic, hit_rate, net_mean, cov = _corr(tot, c), tot[c["hit"]] / tot[c["one"]], tot[c["net"]] / tot[c["one"]], tot[c["inband"]] / tot[c["one"]]
    ic_ci = ci(lambda s: _corr(s, c))
    hit_ci = ci(lambda s: s[c["hit"]] / s[c["one"]])
    net_ci = ci(lambda s: s[c["net"]] / s[c["one"]])
    cov_ci = ci(lambda s: s[c["inband"]] / s[c["one"]])

    mae_model, mae_flat = float(np.mean(np.abs(r - mu_ret)) * 1e4), float(np.mean(np.abs(r)) * 1e4)
    thr = np.quantile(np.abs(oos["mu_z"].values), 0.8)
    top = np.abs(oos["mu_z"].values) >= thr
    log_model = float(np.mean(np.abs(oos["y_range"].values - oos["range_log"].values)))
    log_naive = float(np.mean(np.abs(oos["y_range"].values - oos["lr5"].values)))

    by_symbol = {}
    for sym_index, (sym, g) in enumerate(oos.groupby("symbol")):
        gr, gz = g["r_oc"].values, g["mu_z"].values
        gm, gy = gz * g["sig"].values, g["z_oc"].clip(-qm.CLIP, qm.CLIP).values
        sym_inband = ((gr >= g["lo"].values) & (gr <= g["hi"].values)).astype(float)
        sym_net = np.sign(gm) * gr * 1e4 - cost_bps
        sym_stats = pd.DataFrame({
            "one": 1.0, "x": gz, "y": gy, "xy": gz * gy, "xx": gz * gz, "yy": gy * gy,
            "hit": (np.sign(gm) == np.sign(gr)).astype(float), "net": sym_net, "inband": sym_inband,
        }, index=g.index).groupby(g["date"].values).sum()
        sym_columns = {k: i for i, k in enumerate(sym_stats.columns)}
        sym_tot = sym_stats.values.sum(axis=0)
        sym_ic = _corr(sym_tot, sym_columns)
        sym_hit = sym_tot[sym_columns["hit"]] / sym_tot[sym_columns["one"]]
        sym_rng = np.random.default_rng(seed + sym_index + 1)
        sym_draws = sym_stats.values[sym_rng.integers(0, len(sym_stats), size=(n_boot, len(sym_stats)))].sum(axis=1)

        def sym_ci(metric):
            return [round(float(np.quantile(metric, 0.025)), 4), round(float(np.quantile(metric, 0.975)), 4)]

        sym_ic_ci = sym_ci(np.array([_corr(sample, sym_columns) for sample in sym_draws]))
        sym_hit_ci = sym_ci(sym_draws[:, sym_columns["hit"]] / sym_draws[:, sym_columns["one"]])
        sym_net_ci = sym_ci(sym_draws[:, sym_columns["net"]] / sym_draws[:, sym_columns["one"]])
        sym_mae = float(np.mean(np.abs(gr - gm)) * 1e4)
        sym_flat_mae = float(np.mean(np.abs(gr)) * 1e4)
        sym_range_mae = float(np.mean(np.abs(g["y_range"].values - g["range_log"].values)))
        sym_range_naive = float(np.mean(np.abs(g["y_range"].values - g["lr5"].values)))
        sym_coverage = float(sym_tot[sym_columns["inband"]] / sym_tot[sym_columns["one"]])
        enough_days = len(sym_stats) >= 252
        sym_direction = bool(enough_days and sym_ic_ci[0] > 0 and sym_hit_ci[0] > 0.5 and sym_mae <= sym_flat_mae)
        sym_tradeable = bool(sym_direction and sym_net_ci[0] > 0)
        sym_range_gate = bool(enough_days and abs(sym_coverage - 0.8) <= 0.04 and sym_range_mae < sym_range_naive)

        by_symbol[sym] = {
            "n": int(len(g)), "n_days": int(len(sym_stats)),
            "hit_rate": round(float(sym_hit), 3), "hit_ci95": sym_hit_ci,
            "ic": round(float(sym_ic), 3), "ic_ci95": sym_ic_ci,
            "net_ci95": sym_net_ci,
            "mae_model_bps": round(sym_mae, 2), "mae_flat_bps": round(sym_flat_mae, 2),
            "coverage_80": round(sym_coverage, 3),
            "gates": {"direction": sym_direction, "tradeable": sym_tradeable, "range": sym_range_gate},
        }

    direction = {"ic": round(ic, 4), "ic_ci95": ic_ci, "hit_rate": round(float(hit_rate), 4), "hit_ci95": hit_ci,
                 "mae_model_bps": round(mae_model, 2), "mae_flat_bps": round(mae_flat, 2),
                 "mean_net_bps_per_trade": round(float(net_mean), 2), "net_ci95": net_ci,
                 "top_quintile": {"hit_rate": round(float(hit[top].mean()), 4), "mean_net_bps": round(float(net[top].mean()), 2)},
                 "baseline_hit_rates": {"momentum": round(float(np.mean(np.sign(oos["ret1_z"]) == np.sign(r))), 4),
                                        "gap_fade": round(float(np.mean(-np.sign(oos["gap_z"]) == np.sign(r))), 4)}}
    direction["passes_gate"] = bool(ic_ci[0] > 0 and hit_ci[0] > 0.5 and mae_model <= mae_flat)
    direction["tradeable_after_costs"] = bool(net_ci[0] > 0)
    rng_ = {"coverage_80": round(float(cov), 4), "coverage_ci95": cov_ci,
            "mae_logrange_model": round(log_model, 4), "mae_logrange_naive_5d_mean": round(log_naive, 4)}
    rng_["passes_gate"] = bool(abs(cov - 0.8) <= 0.04 and log_model < log_naive)
    return {"n_obs": int(len(oos)), "n_days": int(len(per)), "n_symbols": int(oos["symbol"].nunique()),
            "oos_start": str(pd.Timestamp(oos["date"].min()).date()), "oos_end": str(pd.Timestamp(oos["date"].max()).date()),
            "cost_bps_assumed": cost_bps, "direction": direction, "range": rng_, "by_symbol": by_symbol}


# --------------------------------------------------------------------------- orchestration
def run(daily, idx, cost_bps=6.0, n_boot=500, init_days=504, step=21):
    panel = build_panel(daily, idx)
    oos = walk_forward(panel, init_days=init_days, step=step)
    if oos.empty:
        raise RuntimeError("Not enough history for a walk-forward test.")
    report = evaluate(oos, cost_bps=cost_bps, n_boot=n_boot)
    models = qm.fit_models(panel)
    gates = {"direction": report["direction"]["passes_gate"], "range": report["range"]["passes_gate"],
             "tradeable": report["direction"]["tradeable_after_costs"]}
    now = datetime.datetime.now(datetime.timezone.utc)
    symbol_gates = {sym: details["gates"] for sym, details in report["by_symbol"].items()}
    model = {"version": "q2-" + now.strftime("%Y%m%d"), "trained_at": now.isoformat(timespec="seconds"),
             "n_obs": int(len(panel)), "n_symbols": int(panel["symbol"].nunique()),
             "data_span": [str(pd.Timestamp(panel["date"].min()).date()), str(pd.Timestamp(panel["date"].max()).date())],
             "gates": gates, "symbol_gates": symbol_gates, "dir": models["dir"], "rng": models["rng"],
             "oos_summary": {"direction_hit_rate": report["direction"]["hit_rate"], "ic": report["direction"]["ic"],
                             "coverage_80": report["range"]["coverage_80"]}}
    report.update({"generated_at": now.isoformat(timespec="seconds"), "universe": sorted(daily.keys()), "gates": gates,
                   "caveats": ["Out-of-sample means each day was predicted by a model trained only on earlier days.",
                               "Per-symbol gates use day-block bootstrap confidence intervals and are screening evidence, not guarantees.",
                               "Per-symbol intervals are not adjusted for screening across the full universe.",
                               "Results are for open->close moves on liquid large caps; they say nothing about other horizons.",
                               "Costs are a flat per-trade assumption; real slippage can be higher.",
                               "A passed gate is evidence, not a guarantee: re-check every month."]})
    return report, model


def _fmt(val, fmt_spec):
    """Safely formats numerical values, defaulting to N/A for None or NaN."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "N/A"
    return f"{val:{fmt_spec}}"


def summary_text(report):
    d, r = report["direction"], report["range"]
    hit_ci0 = _fmt(d['hit_ci95'][0], '.1%') if isinstance(d['hit_ci95'], list) and len(d['hit_ci95']) > 0 else 'N/A'
    hit_ci1 = _fmt(d['hit_ci95'][1], '.1%') if isinstance(d['hit_ci95'], list) and len(d['hit_ci95']) > 1 else 'N/A'
    ic_ci0 = _fmt(d['ic_ci95'][0], '+.3f') if isinstance(d['ic_ci95'], list) and len(d['ic_ci95']) > 0 else 'N/A'
    ic_ci1 = _fmt(d['ic_ci95'][1], '+.3f') if isinstance(d['ic_ci95'], list) and len(d['ic_ci95']) > 1 else 'N/A'
    net_ci0 = _fmt(d['net_ci95'][0], '+.1f') if isinstance(d['net_ci95'], list) and len(d['net_ci95']) > 0 else 'N/A'
    net_ci1 = _fmt(d['net_ci95'][1], '+.1f') if isinstance(d['net_ci95'], list) and len(d['net_ci95']) > 1 else 'N/A'

    lines = [f"Out-of-sample: {report['n_obs']:,} stock-days over {report['n_days']} trading days "
             f"({report['oos_start']} to {report['oos_end']}), {report['n_symbols']} symbols",
             "", "DIRECTION (open -> close)",
             f"  hit-rate {_fmt(d['hit_rate'], '.1%')}  (95% range {hit_ci0} - {hit_ci1})   coin flip = 50%",
             f"  IC       {_fmt(d['ic'], '+.3f')}  (95% range {ic_ci0} to {ic_ci1})",
             f"  baselines: momentum {_fmt(d['baseline_hit_rates']['momentum'], '.1%')}, gap-fade {_fmt(d['baseline_hit_rates']['gap_fade'], '.1%')}",
             f"  net per trade after {_fmt(report['cost_bps_assumed'], '.0f')} bps cost: {_fmt(d['mean_net_bps_per_trade'], '+.1f')} bps "
             f"(95% range {net_ci0} to {net_ci1})",
             f"  GATE: {'PASSED - directional forecasts will be published' if d['passes_gate'] else 'NOT PASSED - no directional forecast will be shown'}"
             f"{'' if not d['passes_gate'] else ('; tradeable after costs: ' + ('yes' if d['tradeable_after_costs'] else 'NO'))}",
             "", "RANGE (how far will it move today)",
             f"  80% band coverage {_fmt(r['coverage_80'], '.1%')} (target 80%)   log-range error {_fmt(r['mae_logrange_model'], '.3f')} vs naive {_fmt(r['mae_logrange_naive_5d_mean'], '.3f')}",
             f"  GATE: {'PASSED - calibrated band will be shown' if r['passes_gate'] else 'NOT PASSED - no band will be shown'}"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="offline dry run on simulated data")
    ap.add_argument("--beta", type=float, default=0.0, help="(synthetic) planted gap-reversal strength")
    ap.add_argument("--period", default="10y")
    ap.add_argument("--cost-bps", type=float, default=6.0)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--out", default=DATA_DIR)
    a = ap.parse_args(argv)

    if a.synthetic:
        daily, idx = make_synthetic(beta=a.beta)
    else:
        idx_all = download_daily([INDEX_TICKER], a.period)
        if INDEX_TICKER not in idx_all:
            sys.exit("Could not download the index history (^NSEI).")
        idx = idx_all[INDEX_TICKER]
        daily = download_daily([f"{s}.NS" for s in dict.fromkeys(WATCHLIST + UNIVERSE_EXTRA)], a.period)
        if len(daily) < 8:
            sys.exit(f"Only {len(daily)} symbols downloaded - refusing to train on so little data.")
    report, model = run(daily, idx, cost_bps=a.cost_bps, n_boot=a.boot)
    os.makedirs(a.out, exist_ok=True)

    summary = summary_text(report)

    report = sanitize_for_json(report)
    model = sanitize_for_json(model)

    with open(os.path.join(a.out, "backtest_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    with open(os.path.join(a.out, "quant_model.json"), "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, allow_nan=False)

    print(summary)
    print(f"\nWrote backtest_report.json and quant_model.json to {a.out}")


if __name__ == "__main__":
    main()
