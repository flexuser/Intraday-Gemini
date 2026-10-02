"""Offline tests (no network, no API key needed):  pip install pytest && pytest -q"""
import datetime, json, os
import numpy as np, pandas as pd, pytest
import muthoot_poc.archive_once as ao

IST = ao.IST
def at(h, m, day=(2026, 10, 5)):
    return IST.localize(datetime.datetime(*day, h, m))

def bars(n, day=(2026, 10, 5), base=100.0, drift=0.0, gaps=()):
    s = IST.localize(datetime.datetime(*day, 9, 15))
    idx = [s + datetime.timedelta(minutes=5 * i) for i in range(n) if i not in gaps]
    px = base + np.linspace(0, drift, len(idx))
    return pd.DataFrame({"Open": px, "High": px + .1, "Low": px - .1, "Close": px,
                         "Volume": np.full(len(idx), 1000)}, index=pd.DatetimeIndex(idx))

GOOD = json.dumps({"bias": "BULLISH", "target_pct": 1.0, "archetype": "MOMENTUM_BREAKOUT", "reasoning": "ok."})

class LLM:
    def __init__(self): self.calls, self.out = 0, GOOD
    def __call__(self, prompt):
        self.calls += 1
        if isinstance(self.out, Exception): raise self.out
        return self.out

class HttpErr(Exception):
    def __init__(self, code): super().__init__(f"HTTP {code}"); self.code = code

@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(ao, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ao.time, "sleep", lambda s: None)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.delenv("FORCE_RUN", raising=False)
    monkeypatch.setattr(ao, "fetch_news", lambda t: [{"title": "t", "link": "#", "publisher": "p"}])
    llm = LLM(); monkeypatch.setattr(ao, "llm_generate", llm)
    monkeypatch.setattr(ao, "fetch_daily_context",
                        lambda t, today: {"prev_close": 99.0, "prev_return_pct": 0.5, "avg_daily_range_pct": 1.4})
    del ao.RUN_ERRORS[:]
    state = {"df": bars(4)}
    monkeypatch.setattr(ao, "fetch_5m", lambda t, today: state["df"])
    return llm, state, tmp_path

def saved(tmp_path, sym="INFY"):
    return json.load(open(tmp_path / f"{sym}_prediction.json", encoding="utf-8"))

# ---- data shape ----
def test_multiindex_columns_are_flattened():
    raw = bars(5); raw.index = raw.index.tz_convert("UTC")
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["RELIANCE.NS"]], names=["Price", "Ticker"])
    df = ao.normalize_frame(raw)
    assert list(df.columns)[:2] == ["Open", "High"] and str(df.index.tz) == "Asia/Kolkata"
    assert ao.compute_session_metrics(df, at(9, 45))["last_close"] == 100.0

def test_gaps_anchor_at_real_slot():
    df = bars(14, gaps=(2, 3), drift=1.0)
    tf, actual, pred, _ = ao.build_curves(df, 101.5)
    last = max(i for i, v in enumerate(actual) if v is not None)
    assert last == 13 and pred[last] == actual[last]

def test_news_link_always_string():
    assert ao._news_url({"content": {"canonicalUrl": {"url": "https://a.b/x"}}}) == "https://a.b/x"
    assert ao._news_url({"link": {"url": "https://c.d"}}) == "https://c.d"
    assert ao._news_url({"content": {}}) == "#" and ao._news_url({"link": "javascript:alert(1)"}) == "#"

def test_nan_never_reaches_json(tmp_path):
    ao.write_json(str(tmp_path / "a.json"), {"x": float("nan"), "y": [float("inf"), 1.0]})
    assert json.load(open(tmp_path / "a.json")) == {"x": None, "y": [None, 1.0]}

# ---- session guard ----
@pytest.mark.parametrize("now,df,expect", [
    (at(9, 20), bars(2), True),
    (at(12, 0), bars(30), True),
    (at(10, 0), pd.DataFrame(), False),            # holiday / Yahoo has no bar for today
    (at(8, 30), bars(2), False),                   # pre-open
    (at(16, 5), bars(75), False),                  # after window
    (at(10, 0, (2026, 10, 3)), bars(30), False),   # Saturday
])
def test_guard(monkeypatch, now, df, expect):
    monkeypatch.delenv("FORCE_RUN", raising=False)
    monkeypatch.setattr(ao, "fetch_5m", lambda t, today: df)
    assert ao.is_live_session(now) is expect

def test_force_run(monkeypatch):
    monkeypatch.setenv("FORCE_RUN", "1")
    assert ao.is_live_session(at(2, 0)) is True

# ---- LLM validation / retry ----
@pytest.mark.parametrize("bad", [
    {"bias": "STRONG BUY", "target_pct": 1, "archetype": "RANGE_BOUND", "reasoning": "x"},
    {"bias": "BULLISH", "target_pct": 1, "archetype": "TREND_REVERSAL", "reasoning": "x"},
    {"bias": "BEARISH", "target_pct": 2.0, "archetype": "RANGE_BOUND", "reasoning": "x"},
    {"bias": "BULLISH", "target_pct": 9.0, "archetype": "RANGE_BOUND", "reasoning": "x"},
    {"bias": "BULLISH", "target_pct": float("nan"), "archetype": "RANGE_BOUND", "reasoning": "x"},
    {"bias": "BULLISH", "target_pct": "1.25%", "archetype": "RANGE_BOUND", "reasoning": "x"},
])
def test_invalid_output_retries_then_neutral_fallback(env, bad):
    llm, state, _ = env
    llm.out = json.dumps(bad) if "nan" not in str(bad["target_pct"]) else '{"bias":"BULLISH","target_pct":NaN,"archetype":"RANGE_BOUND","reasoning":"x"}'
    r = ao.query_gemini_regime("INFY", ao.compute_session_metrics(state["df"], at(9, 30)), [], at(9, 30))
    assert llm.calls == ao.RETRIES and r["source"] == "fallback"
    assert (r["bias"], r["target_pct"], r["archetype"]) == ("NEUTRAL", 0.0, "RANGE_BOUND") and r["llm_error"]

def test_permanent_error_is_not_retried(env):
    llm, state, _ = env
    llm.out = HttpErr(404)
    r = ao.query_gemini_regime("INFY", ao.compute_session_metrics(state["df"], at(9, 30)), [], at(9, 30))
    assert llm.calls == 1 and "404" in r["llm_error"]

def test_sdk_accepts_config(monkeypatch):
    from google.genai import types
    types.GenerateContentConfig(response_mime_type="application/json", response_schema=ao.RESPONSE_SCHEMA,
                                thinking_config=types.ThinkingConfig(thinking_level="low"))

# ---- forecast lifecycle + honest scoring ----
def test_refresh_window_and_opening_forecast_is_immutable(env):
    llm, state, tmp = env
    ao.process_symbol("INFY", at(9, 30)); assert llm.calls == 1
    first = saved(tmp)["opening_forecast"]
    state["df"] = bars(7)
    ao.process_symbol("INFY", at(9, 45)); assert llm.calls == 1            # < 30 min: no API call
    state["df"] = bars(11)
    llm.out = json.dumps({"bias": "BEARISH", "target_pct": -2.0, "archetype": "BREAKDOWN", "reasoning": "x"})
    ao.process_symbol("INFY", at(10, 5)); assert llm.calls == 2           # refreshed
    d = saved(tmp)
    assert d["forecast"]["target_pct"] == -2.0 and d["opening_forecast"] == first

def test_failed_refresh_keeps_last_real_forecast(env):
    llm, state, tmp = env
    ao.process_symbol("INFY", at(9, 30))
    llm.out = HttpErr(503); state["df"] = bars(11)
    ao.process_symbol("INFY", at(10, 5))
    d = saved(tmp)
    assert d["prediction"]["source"] == "llm" and d["prediction"]["target_pct"] == 1.0 and d["prediction"]["llm_error"]

def test_error_is_not_always_zero_and_beats_baseline(env):
    llm, state, tmp = env
    ao.process_symbol("INFY", at(9, 30))                                    # LLM: +1.0% from open 100
    state["df"] = bars(75, drift=0.8)                                       # closes at 100.8
    row = ao.process_symbol("INFY", at(15, 35))
    d = saved(tmp)
    assert d["session"]["complete"] and d["curves"]["error_pct"] == 0.2
    assert row["baseline_error_pct"] == 0.8 and row["direction_hit"] is True and row["verdict"] == "SUCCESS"

def test_mid_session_is_not_scored(env):
    llm, state, tmp = env
    state["df"] = bars(22)
    ao.process_symbol("INFY", at(11, 0))
    d = saved(tmp)
    assert d["session"]["verdict"] == "IN_PROGRESS" and d["curves"]["error_pct"] is None

def test_no_llm_forecast_means_no_forecast_verdict_not_success(env):
    llm, state, tmp = env
    llm.out = HttpErr(404)
    ao.process_symbol("INFY", at(9, 30))
    state["df"] = bars(75, drift=0.3)
    row = ao.process_symbol("INFY", at(15, 35))
    assert saved(tmp)["opening_forecast"] is None and row["verdict"] == "NO_FORECAST" and row["error_pct"] is None

def test_finalize_idempotent_and_midsession_shows_last_completed(env):
    llm, state, tmp = env
    ao.process_symbol("INFY", at(9, 30)); state["df"] = bars(75, drift=0.8)
    row = ao.process_symbol("INFY", at(15, 35))
    for _ in range(2): ao.finalize([row], at(15, 35))                       # re-run inside 15:30-15:50 window
    hist = json.load(open(tmp / "history.json"))
    assert len(hist) == 1 and hist[0]["verdict"] == "SUCCESS"
    summ = json.load(open(tmp / "performance_summary.json"))["overall"]
    assert summ["sessions"] == 1 and summ["direction_hit_rate"] == 1.0 and summ["beat_flat_baseline_rate"] == 1.0
    # next morning: in-progress result -> audit table keeps showing the last completed session
    state["df"] = bars(4, day=(2026, 10, 6))
    inprog = ao.process_symbol("INFY", at(9, 30, (2026, 10, 6)))
    ao.finalize([inprog], at(9, 30, (2026, 10, 6)))
    eod = {r["symbol"]: r for r in json.load(open(tmp / "eod_evaluation.json"))}
    assert eod["INFY"]["verdict"] == "SUCCESS" and eod["INFY"]["session_date"] == "2026-10-05"


# ---- v3: context, baselines, statistics, health ----
def test_prompt_includes_context_and_no_none():
    m = ao.compute_session_metrics(bars(4), at(9, 30))
    p = ao.build_prompt("INFY", m, [], at(9, 30), {"prev_close": 99.0, "prev_return_pct": 0.5, "avg_daily_range_pct": 1.4})
    assert "opening gap: +1.01%" in p and "Previous session's move: +0.50%" in p and "1.40%" in p and "None" not in p
    assert "None" not in ao.build_prompt("INFY", m, [], at(9, 30), {})

def test_daily_context_excludes_today(monkeypatch):
    days = pd.date_range("2026-09-28", periods=7, freq="B", tz="Asia/Kolkata")          # last row = Oct 6 (today)
    d = pd.DataFrame({"Open": 100.0, "High": 102.0, "Low": 98.0, "Close": np.linspace(100, 106, 7), "Volume": 1}, index=days)
    class T:
        def __init__(self, t): pass
        def history(self, period, interval): return d
    monkeypatch.setattr(ao.yf, "Ticker", T)
    c = ao.fetch_daily_context("INFY.NS", datetime.date(2026, 10, 6))
    assert c["prev_close"] == 105.0 and c["prev_return_pct"] == round((105 / 104 - 1) * 100, 2) and c["avg_daily_range_pct"] > 0

def test_baselines_are_frozen_and_scored(env):
    llm, state, tmp = env
    ao.process_symbol("INFY", at(9, 30))                                    # open 100, prev close 99 -> gap +1.01%
    assert saved(tmp)["opening_forecast"]["baselines"] == {"flat": 0.0, "momentum": 0.5, "gap_fade": -1.01}
    state["df"] = bars(75, drift=0.8)
    row = ao.process_symbol("INFY", at(15, 35))
    b = row["baselines"]
    assert b["momentum"]["error_pct"] == 0.3 and b["momentum"]["direction_hit"] is True
    assert b["gap_fade"]["direction_hit"] is False and b["flat"]["error_pct"] == 0.8

def test_summary_wilson_interval_and_enough_data():
    rows = [{"symbol": "INFY", "date": f"2026-10-{d:02d}", "source": "llm", "error_pct": 0.5, "baseline_error_pct": 0.8,
             "direction_hit": d <= 6, "baselines": {"flat": {"error_pct": 0.8, "direction_hit": None}}} for d in range(1, 11)]
    o = ao.summarize(rows)["overall"]
    assert o["direction_hit_rate"] == 0.6 and o["directional_calls"] == 10
    lo, hi = o["direction_hit_ci95"]; assert 0.25 < lo < 0.35 and 0.8 < hi < 0.9      # 6/10 is NOT distinguishable from luck
    assert o["days"] == 10 and o["enough_data"] is False and o["baselines"]["flat"]["mean_error_pct"] == 0.8

def test_stale_data_is_skipped_and_reported(env):
    llm, state, tmp = env
    state["df"] = bars(4)                                                   # newest bar 09:30 but "now" is 11:00
    assert ao.process_symbol("INFY", at(11, 0)) is None
    assert not (tmp / "INFY_prediction.json").exists() and ao.RUN_ERRORS[0][0] == "INFY" and llm.calls == 0

def test_health_status_and_alert_cooldown(env, monkeypatch):
    llm, state, tmp = env
    sent = []
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "http://example.invalid/hook")
    monkeypatch.setattr(ao.requests, "post", lambda url, json, timeout: sent.append(json["text"]))
    bad = [{"symbol": s, "forecast_source": "fallback", "llm_error": "404", "complete": False, "date": "2026-10-05",
            "verdict": "IN_PROGRESS", "bias": "NEUTRAL", "final_pred": 1, "final_actual": 1, "error_pct": None,
            "baseline_error_pct": None, "direction_hit": None, "baselines": {}, "source": "fallback", "target_pct": None}
           for s in ao.SYMBOLS]
    good = [dict(r, forecast_source="llm", llm_error=None) for r in bad]
    assert ao.finalize(bad, at(10, 0))["status"] == "DEGRADED" and len(sent) == 1
    ao.finalize(bad, at(10, 5)); assert len(sent) == 1                      # inside cooldown: no spam
    assert ao.finalize(good, at(10, 10))["status"] == "OK" and "recovered" in sent[-1]
    assert ao.finalize([], at(10, 15))["status"] == "FAILED"
