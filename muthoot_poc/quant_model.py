"""Quantitative day-ahead model (open -> 15:30 close). Pure numpy/pandas, no ML dependency.

Two models, because they have very different predictability:
  * RANGE model     - how far will the stock move today? Volatility clusters, so this is genuinely predictable.
                      Output: expected high-low range + an 80% band for the close, calibrated on out-of-sample data.
  * DIRECTION model - which way? Barely predictable for liquid large caps. It is only published when the
                      walk-forward backtest (muthoot_poc.backtest) proves an edge; otherwise the forecast is "no view".

Every feature for day t uses data from days < t plus day t's OPEN only (known at 09:15), so training and
live serving share one code path and cannot leak the future.
"""
import json
import math

import numpy as np
import pandas as pd

# V1 = original live model (existing quant_model.json keeps working).
DIR_FEATURES_V1 = ["gap_z", "rel_gap_z", "ret1_z", "ret5_z", "ret20_z", "idx_gap_z", "idx_ret1_z", "close_pos"]
# V2 = sector relative + overnight US (only after retrain with --feature-set v2).
DIR_FEATURES_V2 = DIR_FEATURES_V1 + ["sec_gap_z", "sec_ret1_z", "us_ret1_z"]
DIR_FEATURES = list(DIR_FEATURES_V1)  # import compatibility
RNG_FEATURES = ["lr1", "lr5", "lr20", "abs_gap", "idx_lr1", "idx_lr5"]
BAND_LO_Q, BAND_HI_Q = 0.10, 0.90
RIDGE_LAMBDA = 10.0
CLIP = 4.0


def dir_feature_list(model=None):
    """Feature names the loaded model was trained on. Default V1 — never breaks old JSON."""
    if isinstance(model, dict):
        names = model.get("dir_features")
        if isinstance(names, (list, tuple)) and len(names) >= 4:
            return list(names)
        w = (model.get("dir") or {}).get("w")
        if isinstance(w, (list, tuple)):
            if len(w) == len(DIR_FEATURES_V2):
                return list(DIR_FEATURES_V2)
            if len(w) == len(DIR_FEATURES_V1):
                return list(DIR_FEATURES_V1)
    return list(DIR_FEATURES_V1)


# --------------------------------------------------------------------------- features
def clean_daily(df):
    """Date-only index, positive OHLC only."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    cols = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in cols):
        return pd.DataFrame(columns=cols)

    df = df[cols].astype(float).copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[(df > 0).all(axis=1)]


def _vol20(d):
    return np.log(d["Close"]).diff().shift(1).rolling(20, min_periods=15).std()


def _build_features(d, idx, sector=None, overseas=None):
    """Row t uses days < t and day t's Open only. Optional sector / US frames add V2 columns."""
    d = clean_daily(d)
    idx = clean_daily(idx)
    if d.empty or idx.empty:
        return pd.DataFrame()

    idx = idx.reindex(d.index).ffill()
    sig, isig = _vol20(d), _vol20(idx)
    gap = np.log(d["Open"] / d["Close"].shift(1))
    igap = np.log(idx["Open"] / idx["Close"].shift(1))
    lc, ilc = np.log(d["Close"]), np.log(idx["Close"])
    rng, irng = np.log(d["High"] / d["Low"]), np.log(idx["High"] / idx["Low"])
    hl_prev = (d["High"] - d["Low"]).shift(1).replace(0, np.nan)

    f = pd.DataFrame(index=d.index)
    f["gap_z"] = gap / sig
    f["rel_gap_z"] = (gap - igap) / sig
    f["ret1_z"] = (lc.shift(1) - lc.shift(2)) / sig
    f["ret5_z"] = (lc.shift(1) - lc.shift(6)) / (sig * math.sqrt(5))
    f["ret20_z"] = (lc.shift(1) - lc.shift(21)) / (sig * math.sqrt(20))
    f["idx_gap_z"] = igap / isig
    f["idx_ret1_z"] = (ilc.shift(1) - ilc.shift(2)) / isig
    f["close_pos"] = (d["Close"].shift(1) - d["Low"].shift(1)) / hl_prev - 0.5
    f["lr1"] = np.log(rng.shift(1))
    f["lr5"] = np.log(rng.shift(1).rolling(5).mean())
    f["lr20"] = np.log(rng.shift(1).rolling(20).mean())
    f["abs_gap"] = gap.abs()
    f["idx_lr1"] = np.log(irng.shift(1))
    f["idx_lr5"] = np.log(irng.shift(1).rolling(5).mean())
    f["sig"] = sig
    f["open"] = d["Open"]
    f["r_oc"] = np.log(d["Close"] / d["Open"])
    f["y_range"] = np.log(rng)
    f["z_oc"] = f["r_oc"] / sig

    sec = clean_daily(sector) if sector is not None else pd.DataFrame()
    if not sec.empty:
        sec = sec.reindex(d.index).ffill()
        ssig = _vol20(sec).replace(0, np.nan)
        sgap = np.log(sec["Open"] / sec["Close"].shift(1))
        slc = np.log(sec["Close"])
        f["sec_gap_z"] = sgap / ssig
        f["sec_ret1_z"] = (slc.shift(1) - slc.shift(2)) / ssig
    else:
        f["sec_gap_z"] = f["idx_gap_z"]
        f["sec_ret1_z"] = f["idx_ret1_z"]

    us = clean_daily(overseas) if overseas is not None else pd.DataFrame()
    if not us.empty:
        us = us.reindex(d.index).ffill()
        usig = _vol20(us).replace(0, np.nan)
        ulc = np.log(us["Close"])
        f["us_ret1_z"] = (ulc.shift(1) - ulc.shift(2)) / usig
    else:
        f["us_ret1_z"] = 0.0

    return f.replace([np.inf, -np.inf], np.nan)


def build_features(d, idx, sector=None, overseas=None):
    with np.errstate(divide="ignore", invalid="ignore"):
        return _build_features(d, idx, sector=sector, overseas=overseas)


# --------------------------------------------------------------------------- ridge
def ridge_fit(X, y, lam=RIDGE_LAMBDA):
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.clip((X - mu) / sd, -CLIP, CLIP)
    b = float(y.mean())
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - b))
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "b": b}


def ridge_predict(m, X):
    Z = np.clip((np.asarray(X, float) - np.array(m["mu"])) / np.array(m["sd"]), -CLIP, CLIP)
    return Z @ np.array(m["w"]) + m["b"]


def fit_models(panel, lam=RIDGE_LAMBDA, dir_features=None):
    """Fit both models on a training panel (rows = symbol-days with features and realised targets).

    dir_features: optional override (default V1 for compatibility). Pass DIR_FEATURES_V2 after
    sector/US columns exist on the panel to train the extended model.
    """
    dcols = list(dir_features) if dir_features is not None else list(DIR_FEATURES_V1)
    td = panel.dropna(subset=dcols + ["z_oc"])
    dm = ridge_fit(td[dcols], td["z_oc"].clip(-CLIP, CLIP), lam)
    dm["resid_sd"] = float(np.std(td["z_oc"].clip(-CLIP, CLIP).values - ridge_predict(dm, td[dcols])))
    dm["feature_names"] = dcols

    tr = panel.dropna(subset=RNG_FEATURES + ["y_range", "r_oc"])
    rm = ridge_fit(tr[RNG_FEATURES], tr["y_range"], lam)
    resid = tr["y_range"].values - ridge_predict(rm, tr[RNG_FEATURES])
    rm["smear"] = float(np.mean(np.exp(resid)))
    forecast = np.exp(ridge_predict(rm, tr[RNG_FEATURES])) * rm["smear"]
    u = tr["r_oc"].values / forecast
    rm["q_lo"], rm["q_hi"] = float(np.quantile(u, BAND_LO_Q)), float(np.quantile(u, BAND_HI_Q))
    return {"dir": dm, "rng": rm, "dir_features": dcols}


def predict_rows(models, rows, dir_features=None):
    """Vectorised prediction for a frame of feature rows (used by the backtest and live)."""
    dcols = list(dir_features) if dir_features is not None else list(
        models.get("dir_features") or (models.get("dir") or {}).get("feature_names") or DIR_FEATURES_V1
    )
    out = pd.DataFrame(index=rows.index)
    ok_d = rows[dcols].notna().all(axis=1) if all(c in rows.columns for c in dcols) else pd.Series(False, index=rows.index)
    ok_r = rows[RNG_FEATURES].notna().all(axis=1)
    out["mu_z"] = np.nan
    out["range_f"] = np.nan
    out["range_log"] = np.nan
    if ok_d.any():
        out.loc[ok_d, "mu_z"] = ridge_predict(models["dir"], rows.loc[ok_d, dcols].values)
    if ok_r.any():
        raw = ridge_predict(models["rng"], rows.loc[ok_r, RNG_FEATURES].values)
        out.loc[ok_r, "range_log"] = raw
        out.loc[ok_r, "range_f"] = np.exp(raw) * models["rng"]["smear"]
    out["lo"] = models["rng"]["q_lo"] * out["range_f"]                # log-return band for the close
    out["hi"] = models["rng"]["q_hi"] * out["range_f"]
    return out


# --------------------------------------------------------------------------- live
def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def save_model(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


def load_model(path):
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        return m if m.get("dir") and m.get("rng") else None
    except Exception:
        return None


def predict_today(model, feature_row, min_prob_edge=0.08, symbol=None):
    """Live forecast from one feature row. Returns None if any feature is missing (never guess)."""
    if not model or not isinstance(model, dict) or not model.get("dir") or not model.get("rng"):
        return None
    if feature_row is None or feature_row.empty:
        return None

    row = feature_row
    dcols = dir_feature_list(model)
    req_cols = dcols + RNG_FEATURES + ["sig"]
    if not all(col in row.index for col in req_cols) or row[req_cols].isna().any():
        return None

    models = {"dir": model["dir"], "rng": model["rng"], "dir_features": dcols}
    p = predict_rows(models, row.to_frame().T.astype(float), dir_features=dcols)
    if p.empty or p["mu_z"].isna().any() or p["range_f"].isna().any():
        return None

    mu_z = float(p["mu_z"].iloc[0])
    F = float(p["range_f"].iloc[0])
    lo = float(p["lo"].iloc[0])
    hi = float(p["hi"].iloc[0])
    sig = float(row["sig"])

    resid_sd = float(model["dir"].get("resid_sd", 1.0))
    resid_sd = resid_sd if (resid_sd and not math.isnan(resid_sd) and resid_sd > 1e-9) else 1e-9

    p_up = _phi(mu_z / resid_sd)
    gates = model.get("gates", {})
    symbol_gates = model.get("symbol_gates")
    has_symbol_gates = isinstance(symbol_gates, dict) and bool(symbol_gates)
    symbol_gate = symbol_gates.get(symbol) if (symbol and has_symbol_gates) else None
    symbol_gate_available = bool(symbol_gate) if symbol and has_symbol_gates else (False if symbol else None)
    dir_ok = bool(gates.get("direction")) and (bool(symbol_gate.get("direction")) if symbol_gate else True)
    tradeable_ok = bool(gates.get("tradeable")) and (bool(symbol_gate and symbol_gate.get("tradeable")) if symbol and has_symbol_gates else not symbol)
    rng_ok = bool(gates.get("range")) and (bool(symbol_gate.get("range")) if symbol_gate else True)
    bias = "NEUTRAL"
    if dir_ok and abs(p_up - 0.5) >= min_prob_edge:
        bias = "BULLISH" if p_up > 0.5 else "BEARISH"

    return {
        "mu_pct": round((math.exp(mu_z * sig) - 1) * 100, 3),
        "p_up": round(p_up, 3),
        "p_up_calibrated": False,  # model score in [0,1], not a calibrated probability
        "bias": bias,
        "expected_range_pct": round((math.exp(F) - 1) * 100, 2),
        "band_lo_pct": round((math.exp(lo) - 1) * 100, 2),
        "band_hi_pct": round((math.exp(hi) - 1) * 100, 2),
        "direction_validated": dir_ok,
        "tradeable_validated": tradeable_ok,
        "range_validated": rng_ok,
        "symbol_gate_available": symbol_gate_available,
        "symbol_gate_passed": (bool(symbol_gate and symbol_gate.get("direction")) if has_symbol_gates else None) if symbol else None,
        "model_version": model.get("version"),
        "feature_set": "v2" if len(dcols) > len(DIR_FEATURES_V1) else "v1",
        "dir_features_used": dcols,
    }
