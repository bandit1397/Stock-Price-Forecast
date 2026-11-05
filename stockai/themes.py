"""급부상 산업 조기 탐지. 고정 종목과 별개로 매일 15개 산업을 비교해 상위 5개를 뽑는다."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import stocks, universe
from .features import rsi
from .fundamentals import lin
from .news import fetch_rss, intensity

WEIGHTS = {"rs": 25, "volume": 20, "news": 20, "earnings": 15, "trend": 10, "safety": 10}


def _raw(tid: str, th: dict, panel: dict, fund: dict) -> dict | None:
    df = panel.get(th["etf"])
    if df is None or len(df) < 260:
        return None
    c = df["close"]
    spy = panel["^GSPC"]["close"].reindex(c.index, method="ffill")
    members = [s["ticker"] for s in stocks() if s["theme"] == tid]
    breadth = [panel[m]["close"].iloc[-1] > panel[m]["close"].tail(50).mean() for m in members if m in panel]
    vol = (c * df["volume"]).astype(float)
    growth = [fund[m]["rev_g"] for m in members if fund.get(m, {}).get("rev_g") is not None]
    r = np.log(c).diff()
    r21 = float(c.iloc[-1] / c.iloc[-22] - 1)
    return {"id": tid, "name": th["name"], "etf": th["etf"], "members": members,
            "rs21": r21 - float(spy.iloc[-1] / spy.iloc[-22] - 1),
            "rs63": float(c.iloc[-1] / c.iloc[-64] - 1) - float(spy.iloc[-1] / spy.iloc[-64] - 1),
            "ret21": r21, "volume_ratio": float(vol.tail(5).mean() / vol.tail(60).mean()),
            "breadth": float(np.mean(breadth)) if breadth else None,
            "growth": float(np.median(growth)) if growth else None,
            "rsi": float(rsi(c).iloc[-1]), "z21": r21 / float(r.tail(252).std() * np.sqrt(21)),
            "above50": bool(c.iloc[-1] > c.tail(50).mean()), "above200": bool(c.iloc[-1] > c.tail(200).mean())}


def stage(t: dict) -> str:
    if t["rsi"] > 75 or t["z21"] > 2.5:
        return "과열 경고"
    if t["above50"] and t["above200"] and (t["breadth"] or 0) >= 0.6 and t["rs63"] > 0:
        return "상승 확인"
    if t["rs21"] > 0 and t["rs63"] <= 0.05 and t["volume_ratio"] > 1:
        return "조기 발견"
    return "관찰"


def rank(panel: dict, fund: dict, use_news: bool = True) -> list[dict]:
    themes = universe()["themes"]
    rows = [x for x in (_raw(k, v, panel, fund) for k, v in themes.items()) if x]
    if not rows:
        return []
    for t in rows:
        t["news_ratio"] = intensity(fetch_rss(themes[t["id"]]["news"])) if use_news else None
    df = pd.DataFrame(rows)
    pct = lambda s: s.rank(pct=True) * 100
    rs = pct(0.5 * df["rs21"] + 0.5 * df["rs63"])
    vol = pct(df["volume_ratio"])
    news = pct(df["news_ratio"].astype(float)) if df["news_ratio"].notna().any() else pd.Series(np.nan, index=df.index)
    for i, t in enumerate(rows):
        parts = {"rs": rs[i], "volume": vol[i], "news": news[i],
                 "earnings": lin(t["growth"], -0.05, 0.30),
                 "trend": 50 * t["above50"] + 50 * t["above200"],
                 "safety": lin(t["z21"], 3, 1)}
        ok = {k: v for k, v in parts.items() if v is not None and not pd.isna(v)}
        t["score"] = round(sum(ok[k] * WEIGHTS[k] for k in ok) / sum(WEIGHTS[k] for k in ok), 1)
        t["parts"] = {k: round(float(v), 0) for k, v in ok.items()}
        t["stage"] = stage(t)
    return sorted(rows, key=lambda x: -x["score"])
