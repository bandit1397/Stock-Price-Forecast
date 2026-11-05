"""전체 분석 파이프라인."""
from __future__ import annotations

import json
import logging
from datetime import date

import numpy as np
import pandas as pd

from . import filings, flows, forecast, fundamentals, macro, news, prices, risk, signals, store, themes, watchlist
from .config import REPORTS, env, macro_themes, portfolio, stocks
from .features import build_panel, last_moves

log = logging.getLogger(__name__)
KEEP = ["rev_g", "rev_cagr", "gm", "om", "om_avg", "roe", "de", "fwd_pe", "ev_ebitda", "ps", "rd_intensity", "mcap", "earnings_date"]


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) or np.isinf(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def upcoming_events(fund: dict, days: int = 21) -> list[dict]:
    today = pd.Timestamp.now().normalize()
    out = [{"date": str(e["date"]), "name": e["name"]} for e in macro_themes()["events"]
           if 0 <= (pd.Timestamp(e["date"]) - today).days <= days]
    for s in stocks():
        d = fund.get(s["ticker"], {}).get("earnings_date")
        if d and 0 <= (pd.Timestamp(d) - today).days <= days:
            out.append({"date": d, "name": f'{s["name"]} 실적 발표'})
    return sorted(out, key=lambda x: x["date"])


def portfolio_check(results: list[dict]) -> list[dict]:
    """보유 종목별 손익과 손절 상태, 업종 쏠림 점검."""
    by = {r["ticker"]: r for r in results}
    rows = []
    for h in portfolio():
        r = by.get(h["ticker"])
        if not r:
            rows.append({"ticker": h["ticker"], "note": "감시 목록에 없음"})
            continue
        pnl = r["close"] / h["avg_price"] - 1
        stop = h.get("stop") or h["avg_price"] * 0.85
        rows.append({"ticker": h["ticker"], "name": r["name"], "category": r["category"], "pnl": pnl, "action": r["action"],
                     "value": r["close"] * h.get("qty", 0), "stop": stop, "stop_hit": bool(r["close"] <= stop)})
    return rows


def run(use_network_extras: bool = True) -> dict:
    panel = prices.load()
    table = build_panel(panel)
    latest = table.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    bundle = forecast.load_bundle()
    tickers = [s["ticker"] for s in stocks()]
    fund = fundamentals.load_all(tickers)
    snapshot = macro.fred_snapshot()
    reg = macro.regime(panel, snapshot)
    issues = news.issue_scan(macro_themes()["issues"]) if use_network_extras else []
    tilt = news.sector_tilt(issues)
    cnews = news.company_news(stocks()) if use_network_extras else {}
    kr_flow = flows.korean_flows(tickers) if use_network_extras else {}
    us = [s["ticker"] for s in stocks() if s["market"] == "US"]
    kr = [s["ticker"] for s in stocks() if s["market"] == "KR"]
    filing = {**filings.sec_recent(us), **filings.dart_recent(kr)} if use_network_extras else {}
    issues_q = prices.quality_issues(panel, tickers)
    if use_network_extras:                                 # 점검을 못 한 것이 '문제없음'으로 읽히면 안 된다
        if not env("SEC_CONTACT"):
            issues_q.append("미국 공시 점검 꺼짐: .env 의 SEC_CONTACT 미설정")
        if not env("DART_API_KEY"):
            issues_q.append("한국 공시 점검 꺼짐: .env 의 DART_API_KEY 미설정")
        missing = [s["name"] for s in stocks() if not cnews.get(s["ticker"])]
        if missing:
            issues_q.append("뉴스 수집 실패: " + ", ".join(missing))

    results = []
    for s in stocks():
        t = s["ticker"]
        if t not in panel or t not in latest.index:
            continue
        df, feat = panel[t], latest.loc[t]
        f = fund.get(t, {})
        nw = cnews.get(t) or {}
        ab = risk.abnormal(df, feat, s["market"], nw.get("recent7"))
        val = fundamentals.valuation(f, bool(s.get("cyclical")))
        tech = fundamentals.tech_score(t, f)
        close = float(df["close"].iloc[-1])
        timing = None if pd.isna(feat["timing"]) else float(feat["timing"])
        c = {"ticker": t, "name": s["name"], "market": s["market"], "category": s["category"], "theme": s["theme"],
             "date": str(df.index[-1].date()), "close": close, "timing": timing, "fin": fundamentals.fin_score(f),
             "value": val["value"], "implied_g": val.get("implied_g"), "capable_g": val.get("capable_g"),
             "value_warn": val["warn"], "flow": flows.flow_score(df, kr_flow.get(t), f.get("short_pct")),
             "kr_flow": kr_flow.get(t), "news": nw.get("news"), "news_count": nw.get("count"),
             "headlines": nw.get("headlines", []), "red_flags": nw.get("red_flags", []),
             "filing_flags": filing.get(t, []), "abnormal": ab["abnormal"], "abnormal_level": ab["level"],
             "abnormal_reasons": ab["reasons"], "safety": 100 - ab["abnormal"],
             "macro": macro.stock_macro(reg, s, panel, tilt), "rev_g": f.get("rev_g"), **tech, **last_moves(df["close"]),
             "earnings_days": (pd.Timestamp(f["earnings_date"]) - pd.Timestamp.now().normalize()).days if f.get("earnings_date") else None,
             "fundamentals": {k: f.get(k) for k in KEEP}}
        c.update(signals.decide(c, feat, reg[s["market"]]["stage"]))
        c["views"] = signals.view_scores(c)
        c["plan"] = signals.trade_plan(close, float(feat["atr_pct"]), float(feat["ma20_gap"])) if pd.notna(feat["atr_pct"]) else None
        c["forecast"] = forecast.predict(feat, close, bundle)
        results.append(c)

    theme_rank = themes.rank(panel, fund, use_network_extras)
    day = str(date.today())
    report = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "day": day, "regime": reg, "fred": snapshot,
              "issues": sorted([{k: i.get(k) for k in ("name", "intensity", "titles")} for i in issues],
                               key=lambda x: -(x["intensity"] or 0)),
              "themes": theme_rank, "stocks": results, "data_issues": issues_q, "events": upcoming_events(fund),
              "portfolio": portfolio_check(results), "scorecard": store.live_scorecard(panel),
              "watch": watchlist.score_all(watchlist.build(results, reg, tilt, panel), panel),
              "watch_scorecard": store.watch_scorecard(panel),
              "model": {"trained": bundle["trained"], "gate": bundle["gate"], "coverage": bundle["coverage"]} if bundle else None}
    bt = REPORTS / "backtest.json"
    report["backtest"] = json.loads(bt.read_text(encoding="utf-8")) if bt.exists() else None
    report = _clean(report)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report

