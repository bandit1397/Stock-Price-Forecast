"""핵심 안전장치 테스트. 네트워크 없이 실행된다."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from stockai import alerts, backtest, features, filings, forecast, fundamentals, news, prices, risk, signals


def ohlcv(n=600, seed=1, drift=0.0005, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": rng.integers(1_000_000, 2_000_000, n).astype(float)}, index=idx)


def test_features_do_not_use_future_data():
    df = ohlcv()
    full = features.stock_features(df)
    cut = features.stock_features(df.iloc[:400])
    pd.testing.assert_frame_equal(full.iloc[:400], cut)


def test_market_features_lag_one_day():
    idx = pd.bdate_range("2021-01-01", periods=60)
    vix = pd.DataFrame({"close": np.arange(60.0) + 10}, index=idx)
    m = features._lagged_market({"^VIX": vix}, idx)
    assert m["vix"].iloc[10] == vix["close"].iloc[9]


def test_negative_futures_price_is_ignored():
    idx = pd.bdate_range("2020-01-01", periods=60)
    close = np.full(60, 50.0)
    close[40] = -37.0
    with np.errstate(invalid="raise"):
        m = features._lagged_market({"CL=F": pd.DataFrame({"close": close}, index=idx)}, idx)
    assert np.isnan(m["oil_ret21"].iloc[41])


def _table(n_days=3000, h=21, signal=False, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2009-01-01", periods=n_days)
    rows = []
    for t in range(6):
        x = rng.normal(size=n_days)
        noise = rng.normal(size=n_days)
        y = (x if signal else 0) * 1.0 + noise
        rows.append(pd.DataFrame({"date": dates, "ticker": str(t), "f1": x, "timing": 50.0, f"y_{h}": y, f"x_{h}": y,
                                  f"end_{h}": pd.Series(dates).shift(-h).values}))
    return pd.concat(rows, ignore_index=True)


def test_walk_forward_labels_finish_before_test_period(monkeypatch):
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append(X.index)
            return self

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

    monkeypatch.setattr(forecast, "_model", Spy)
    t = _table()
    oos = forecast.walk_forward(t, 21, "up", ["f1"])
    assert seen and not oos.empty
    for year, idx in zip(sorted(oos["fold"].unique()), seen):
        assert t.loc[idx, "end_21"].max() < pd.Timestamp(year, 1, 1)


def test_gate_rejects_pure_noise():
    t = _table(signal=False)
    assert not forecast.gate(forecast.walk_forward(t, 21, "up", ["f1"]), 21)["passed"]


def test_gate_accepts_real_signal():
    t = _table(signal=True)
    res = forecast.gate(forecast.walk_forward(t, 21, "up", ["f1"]), 21)
    assert res["passed"] and res["auc"] > 0.6


def test_predict_hides_probability_without_validated_model():
    row = pd.Series({"vol_63": 0.02, "vol_long": 0.02, "f1": 0.0})
    out = forecast.predict(row, 100.0, {"cols": ["f1"], "models": {}})
    assert len(out) == 8
    assert all(r["p_up"] is None and r["p_beat"] is None for r in out)
    assert {r["status"] for r in out} == {"예측력 미입증", "시나리오"}
    assert all(r["low"] < r["mid"] < r["high"] for r in out)
    widths = [r["high"] / r["low"] for r in out]
    assert widths == sorted(widths)


def test_implied_growth_round_trip():
    g = fundamentals.implied_growth(1000, 50)
    assert -0.5 < g < 1
    assert fundamentals.implied_growth(2000, 50) > g
    assert fundamentals.implied_growth(1000, -5) is None


def test_cycle_peak_earnings_are_normalised():
    f = {"mcap": 1000, "fcf": 300, "ni": 300, "revenue": 1000, "om": 0.6, "om_avg": 0.1, "rev_cagr": 0.05, "fwd_pe": 2.5,
         "currency": "KRW", "fin_currency": "KRW"}
    cyc, non = fundamentals.valuation(f, True), fundamentals.valuation(f, False)
    assert cyc["value"] <= 60 and cyc["implied_g"] > non["implied_g"]
    assert any("선행 PER" in w for w in cyc["warn"]) and any("고점" in w for w in cyc["warn"])


def test_currency_mismatch_skips_valuation():
    assert fundamentals.valuation({"currency": "USD", "fin_currency": "KRW", "mcap": 1}, False)["value"] is None


def test_low_confidence_moat_cannot_pass_gate():
    res = fundamentals.tech_score("232680.KQ", {"om": 0.5, "gm": 0.9, "rev_cagr": 0.5})
    assert res["tech"] <= 59


FEAT = pd.Series({"ma200_gap": 0.1, "ma20_gap": 0.01, "atr_pct": 0.02, "vol_63": 0.02, "ret_5": 0.01})
GOOD = {"tech": 80, "demand": 12, "rev_g": 0.1, "fin": 70, "value": 60, "timing": 75, "abnormal": 0}


def test_buy_requires_all_gates():
    assert signals.decide(GOOD, FEAT, "중립")["action"] == signals.BUY
    assert signals.decide({**GOOD, "value": 20}, FEAT, "중립")["action"] == signals.WAIT
    assert signals.decide({**GOOD, "value": None}, FEAT, "중립")["action"] == signals.WAIT
    assert signals.decide({**GOOD, "tech": 40}, FEAT, "중립")["action"] == signals.HOLD


@pytest.mark.parametrize("extra,feat,stage", [
    ({"earnings_days": 2}, {}, "중립"),
    ({"red_flags": [{"word": "유상증자"}]}, {}, "중립"),
    ({}, {"ret_5": -0.3}, "중립"),
    ({}, {"ma20_gap": 0.2}, "중립"),
    ({}, {}, "위험 회피"),
])
def test_conditions_that_block_a_buy(extra, feat, stage):
    res = signals.decide({**GOOD, **extra}, pd.Series({**FEAT.to_dict(), **feat}), stage)
    assert res["action"] == signals.WAIT and res["reasons"]


def test_news_keyword_alone_never_forces_avoid():
    res = signals.decide({**GOOD, "red_flags": [{"word": "유상증자"}]}, FEAT, "중립")
    assert res["action"] != signals.AVOID


def test_filing_and_abnormal_force_avoid():
    assert signals.decide({**GOOD, "filing_flags": ["x"]}, FEAT, "중립")["action"] == signals.AVOID
    assert signals.decide({**GOOD, "abnormal": 80}, FEAT, "중립")["action"] == signals.AVOID


def test_broken_chart_sells_only_weak_companies():
    feat = pd.Series({**FEAT.to_dict(), "ma200_gap": -0.1})
    assert signals.decide({**GOOD, "timing": 20}, feat, "중립")["action"] == signals.WAIT
    assert signals.decide({**GOOD, "timing": 20, "fin": 20}, feat, "중립")["action"] == signals.SELL


def test_view_scores_report_missing_evidence():
    v = signals.view_scores({"tech": 80, "fin": None, "value": None, "macro": None})
    assert v["장기"]["score"] == 80 and v["장기"]["coverage"] == 0.4


def test_position_hysteresis():
    s = pd.Series([50, 61, 50, 44, 50, 70, np.nan, 50], dtype=float)
    assert backtest.positions(s).tolist() == [0, 1, 1, 0, 0, 1, 0, 0]


def test_backtest_holiday_gap_keeps_position_and_charges_once(monkeypatch):
    a, b = ohlcv(800, 1), ohlcv(800, 2)
    b = b.drop(b.index[300::40])                           # 휴장일이 다른 시장
    monkeypatch.setattr(backtest, "stocks", lambda: [{"ticker": "A", "market": "US"}, {"ticker": "B", "market": "KR"}])
    table = pd.concat([pd.DataFrame({"date": d.index, "ticker": t, "timing": 80.0}) for t, d in (("A", a), ("B", b))])
    res = backtest.run({"A": a, "B": b}, table, start="2020-01-01")
    assert res["turnover_year"] < 1                        # 진입 1회 외 재매매가 없어야 함


def test_drop_incomplete_daily_bar():
    df = ohlcv(5, start="2026-11-02")
    during = datetime(2026, 11, 6, 2, 0, tzinfo=timezone.utc)    # 서울 11:00
    after = datetime(2026, 11, 6, 7, 0, tzinfo=timezone.utc)     # 서울 16:00
    assert len(prices.drop_incomplete(df, "005930.KS", during)) == 4
    assert len(prices.drop_incomplete(df, "005930.KS", after)) == 5


def test_abnormal_detects_pump_and_dump():
    df = ohlcv(120)
    c = df["close"].to_numpy().copy()
    c[-15:-5] = c[-16] * np.linspace(1.1, 2.0, 10)
    c[-5:] = c[-6] * np.linspace(0.9, 0.65, 5)
    df["close"] = c
    df["high"], df["low"], df["open"] = c * 1.01, c * 0.99, c
    feat = features.stock_features(df).iloc[-1]
    res = risk.abnormal(df, feat, "KR", 0)
    assert res["abnormal"] >= 30 and any("급등" in r for r in res["reasons"])


def test_news_scoring_and_failure_is_not_silence():
    now = datetime(2026, 11, 5, tzinfo=timezone.utc)
    items = [{"title": f"Acme wins contract {i} - Wire", "when": now, "link": ""} for i in range(4)]
    res = news.score_company(items, ["Acme"], now)
    assert res["news"] > 50 and res["count"] == 4
    assert news.score_company(items[:2], ["Acme"], now)["news"] is None
    assert news.intensity(None) is None


def test_keyword_boundary():
    assert news._hits("Acme stock upgraded by analysts", news.POS) == ["upgraded"]
    assert news._hits("recalled memories", news.NEG) == []


def test_relative_intensity():
    rows = news.relative_intensity([{"raw": 1.0}, {"raw": 2.0}, {"raw": 8.0}, {"raw": None}])
    assert [r["intensity"] for r in rows] == [0.5, 1.0, 4.0, None]


def test_sec_flags_parsing():
    recent = {"form": ["8-K", "8-K", "NT 10-Q", "8-K"], "filingDate": ["2026-11-01", "2026-11-02", "2026-11-03", "2020-01-01"],
              "items": ["4.02,9.01", "2.02", "", "1.03"]}
    out = filings.sec_flags(recent, datetime(2026, 10, 1).date())
    assert len(out) == 2 and any("재무제표" in x for x in out) and any("지연" in x for x in out)


def test_telegram_without_token_prints_only(monkeypatch, capsys):
    monkeypatch.setattr(alerts, "env", lambda k: "")
    assert alerts.send("hello") is False
    assert "hello" in capsys.readouterr().out


def test_chunks_respect_limit():
    parts = alerts.chunks("\n".join(["가" * 100] * 100), 1000)
    assert len(parts) > 1 and all(len(p) <= 1000 for p in parts)


def test_config_is_consistent():
    from stockai.config import moat, stocks, universe
    tickers = [s["ticker"] for s in stocks()]
    assert len(tickers) == len(set(tickers))
    assert set(tickers) == set(moat()["scores"])
    limits = [15, 15, 15, 15, 10, 10]
    for t, m in moat()["scores"].items():
        assert len(m["s"]) == 6 and all(v <= c for v, c in zip(m["s"], limits)), t
    assert all(s["theme"] in universe()["themes"] or s["theme"] == "components" for s in stocks())
