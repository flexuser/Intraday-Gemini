"""Quant model tests (offline, simulated data).  pytest -q tests"""
import pandas as pd, pytest
from muthoot_poc import quant_model as qm, backtest as bt

COLS = qm.DIR_FEATURES + qm.RNG_FEATURES + ["sig"]

@pytest.fixture(scope="module")
def planted():
    daily, idx = bt.make_synthetic(n_days=1300, n_sym=6, beta=0.10, seed=3)
    return bt.run(daily, idx, n_boot=200, init_days=400), daily, idx

@pytest.fixture(scope="module")
def null():
    daily, idx = bt.make_synthetic(n_days=1300, n_sym=6, beta=0.0, seed=4)
    return bt.run(daily, idx, n_boot=200, init_days=400)

def test_features_never_read_the_future():
    daily, idx = bt.make_synthetic(n_days=200, n_sym=1, seed=5)
    d, last = daily["SYN0"], daily["SYN0"].index[-1]
    f1 = qm.build_features(d, idx)
    d2, i2 = d.copy(), idx.copy()
    d2.loc[last, ["High", "Low", "Close"]] = d.loc[last, ["High", "Low", "Close"]].values * [1.5, 0.5, 1.3]
    i2.loc[last, ["High", "Low", "Close"]] = idx.loc[last, ["High", "Low", "Close"]].values * 1.2
    f2 = qm.build_features(d2, i2)
    pd.testing.assert_series_equal(f1.loc[last, COLS], f2.loc[last, COLS])      # only day t's OPEN may matter

def test_live_features_match_training_features():
    daily, idx = bt.make_synthetic(n_days=200, n_sym=1, seed=6)
    d, k = daily["SYN0"], 150
    full = qm.build_features(d, idx)
    live = d.iloc[:k + 1].copy()
    live.iloc[-1, live.columns.get_indexer(["High", "Low", "Close"])] = live.iloc[-1]["Open"]   # what the live path does
    part = qm.build_features(live, idx)
    pd.testing.assert_series_equal(full.iloc[k][COLS], part.iloc[-1][COLS])

def test_finds_a_planted_edge_and_calibrates_the_band(planted):
    (report, model), _, _ = planted
    d, r = report["direction"], report["range"]
    assert model["gates"]["direction"] is True and d["ic"] > 0.04 and d["hit_ci95"][0] > 0.5
    assert model["gates"]["range"] is True and 0.75 <= r["coverage_80"] <= 0.85

def test_refuses_to_publish_when_there_is_no_edge(null):
    report, model = null
    assert model["gates"]["direction"] is False and model["gates"]["range"] is True

def test_predict_today_and_roundtrip(planted, tmp_path):
    (report, model), daily, idx = planted
    qm.save_model(str(tmp_path / "m.json"), model)
    loaded = qm.load_model(str(tmp_path / "m.json"))
    feats = qm.build_features(daily["SYN0"], idx).dropna(subset=COLS)
    p = qm.predict_today(loaded, feats.iloc[-1])
    assert p["direction_validated"] and p["range_validated"] and p["band_lo_pct"] < 0 < p["band_hi_pct"]
    assert 0.0 < p["p_up"] < 1.0 and p["expected_range_pct"] > 0
    broken = feats.iloc[-1].copy(); broken["gap_z"] = float("nan")
    assert qm.predict_today(loaded, broken) is None                              # never guess on missing data
    assert qm.load_model(str(tmp_path / "missing.json")) is None


@pytest.mark.parametrize("drift", [-0.08, 0.08])
def test_drift_without_skill_cannot_pass_any_direction_gate(drift):
    """Real open->close drift is negative (~46.6% up days). A model that only leans with the drift must not pass."""
    daily, idx = bt.make_synthetic(n_days=1300, n_sym=6, beta=0.0, drift=drift, seed=12)
    report, model = bt.run(daily, idx, n_boot=200, init_days=400)
    d = report["direction"]
    assert max(d["baseline_hit_rates"]["always_long"], d["baseline_hit_rates"]["always_short"]) > 0.52   # the drift is visible...
    assert model["gates"]["direction"] is False                                                           # ...but it is not skill
    assert sum(1 for g in model["symbol_gates"].values() if g["direction"]) <= 1                          # chance false positives only

def test_planted_edge_passes_the_drift_immune_gate_globally_and_per_symbol(planted):
    (report, model), _, _ = planted
    d = report["direction"]
    assert model["gates"]["direction"] is True and d["spread_up_prob_ci95"][0] > 0 and d["spread_up_prob"] > 0.03
    assert sum(1 for g in model["symbol_gates"].values() if g["direction"]) >= 3

def test_v2_features_never_read_the_future():
    """Sector row of day t (H/L/C) and the SAME-DATE US session both happen after the Indian open and must not matter."""
    daily, idx = bt.make_synthetic(n_days=250, n_sym=3, seed=21)
    d, sec, us = daily["SYN0"], daily["SYN1"], daily["SYN2"]
    last = d.index[-1]
    cols = qm.DIR_FEATURES_V2 + qm.RNG_FEATURES + ["sig"]
    f1 = qm.build_features(d, idx, sector=sec, overseas=us)
    sec2, us2 = sec.copy(), us.copy()
    sec2.loc[last, ["High", "Low", "Close"]] = sec.loc[last, ["High", "Low", "Close"]].values * 1.4
    us2.loc[last, ["Open", "High", "Low", "Close"]] = us.loc[last, ["Open", "High", "Low", "Close"]].values * 1.4
    f2 = qm.build_features(d, idx, sector=sec2, overseas=us2)
    pd.testing.assert_series_equal(f1.loc[last, cols], f2.loc[last, cols])
