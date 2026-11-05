"""재무, 가치평가, 기술 경쟁력 점수.

핵심 질문을 분리한다.
- 좋은 회사인가: 기술 점수, 재무 점수
- 지금 가격이 적정한가: 역DCF로 주가에 반영된 성장률을 실제 성장 능력과 비교
"""
from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import yfinance as yf

from .config import DATA, moat

log = logging.getLogger(__name__)
PATH = DATA / "fundamentals.json"
MAX_AGE_DAYS = 3


def lin(x, lo, hi) -> float | None:
    """x를 lo→0점, hi→100점으로 변환 (lo>hi면 낮을수록 고득점)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return max(0.0, min(100.0, (x - lo) / (hi - lo) * 100))


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _row(stmt, name: str) -> list[float]:
    if stmt is None or stmt.empty or name not in stmt.index:
        return []
    return [float(v) for v in stmt.loc[name].dropna().tolist()]   # 최신 연도부터


def fetch_one(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    i = t.info or {}
    out = {k: i.get(v) for k, v in {
        "currency": "currency", "fin_currency": "financialCurrency", "mcap": "marketCap",
        "fwd_pe": "forwardPE", "ttm_pe": "trailingPE", "pb": "priceToBook", "ev_ebitda": "enterpriseToEbitda",
        "ps": "priceToSalesTrailing12Months", "rev_g": "revenueGrowth", "gm": "grossMargins",
        "om": "operatingMargins", "fcf": "freeCashflow", "ni": "netIncomeToCommon", "revenue": "totalRevenue",
        "roe": "returnOnEquity", "de": "debtToEquity", "short_pct": "shortPercentOfFloat"}.items()}
    try:
        s = t.income_stmt
        rev, op, rd = _row(s, "Total Revenue"), _row(s, "Operating Income"), _row(s, "Research And Development")
        if len(rev) >= 3 and rev[-1] > 0 and rev[0] > 0:
            out["rev_cagr"] = (rev[0] / rev[-1]) ** (1 / (len(rev) - 1)) - 1
        if rev and op:
            n = min(len(rev), len(op))
            margins = [op[k] / rev[k] for k in range(n) if rev[k] > 0]
            out["om_avg"] = sum(margins) / len(margins) if margins else None
        if rev and rd and rev[0] > 0:
            out["rd_intensity"] = rd[0] / rev[0]
    except Exception as e:
        log.debug("연간 재무 실패 %s: %s", ticker, e)
    try:
        cal = t.calendar or {}
        d = cal.get("Earnings Date") or []
        out["earnings_date"] = str(d[0]) if d else None
    except Exception:
        out["earnings_date"] = None
    return out


def load_all(tickers: list[str], force: bool = False) -> dict[str, dict]:
    cache = json.loads(PATH.read_text(encoding="utf-8")) if PATH.exists() else {}
    today = date.today()

    def stale(t: str) -> bool:
        c = cache.get(t)
        return force or not c or (today - date.fromisoformat(c["_at"])).days >= MAX_AGE_DAYS

    def job(t: str):
        try:
            return t, {**fetch_one(t), "_at": str(today)}
        except Exception as e:
            log.warning("재무 수집 실패 %s: %s", t, e)
            return t, None

    with ThreadPoolExecutor(4) as ex:
        for t, res in ex.map(job, [t for t in tickers if stale(t)]):
            if res:
                cache[t] = res
    DATA.mkdir(exist_ok=True)
    PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return cache


def fin_score(f: dict) -> float | None:
    fcf_m = f["fcf"] / f["revenue"] if f.get("fcf") is not None and f.get("revenue") else None
    parts = [lin(f.get("rev_g"), -0.10, 0.25), lin(f.get("om"), 0.0, 0.30), lin(fcf_m, -0.05, 0.20),
             lin(f.get("roe"), 0.0, 0.25), lin(f.get("de"), 200, 30)]
    return _mean(parts) if sum(p is not None for p in parts) >= 3 else None


def implied_growth(mcap: float, earnings: float, r: float = 0.10, tg: float = 0.03, years: int = 10) -> float | None:
    """현재 시가총액을 정당화하려면 향후 10년간 이익이 연 몇 % 늘어야 하는가 (역DCF)."""
    if not mcap or not earnings or earnings <= 0:
        return None

    def value(g: float) -> float:
        e, pv = earnings, 0.0
        for t in range(1, years + 1):
            e *= 1 + g
            pv += e / (1 + r) ** t
        return pv + e * (1 + tg) / (r - tg) / (1 + r) ** years

    lo, hi = -0.5, 1.0
    if value(hi) < mcap:
        return hi
    if value(lo) > mcap:
        return lo
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if value(mid) < mcap else (lo, mid)
    return (lo + hi) / 2


def valuation(f: dict, cyclical: bool) -> dict:
    """가치 점수와 경고. 이익이 없거나 데이터가 의심되면 점수를 주지 않는다."""
    warn: list[str] = []
    if f.get("currency") and f.get("fin_currency") and f["currency"] != f["fin_currency"]:
        return {"value": None, "implied_g": None, "warn": ["주가와 재무제표 통화가 달라 가치평가 생략"]}
    om, om_avg, rev = f.get("om"), f.get("om_avg"), f.get("revenue")
    peak = bool(cyclical and om and om_avg and om_avg > 0 and om > 1.5 * om_avg)
    positives = [x for x in (f.get("fcf"), f.get("ni")) if x and x > 0]
    earnings = sum(positives) / len(positives) if positives else None
    if peak and rev:
        earnings = rev * om_avg * 0.8                      # 경기순환 고점 이익 대신 평균 마진 사용
        warn.append(f"이익 고점 가능성: 현재 영업이익률 {om:.0%}, 평균 {om_avg:.0%}")
    g = implied_growth(f.get("mcap"), earnings)
    if g is None:
        return {"value": None, "implied_g": None, "warn": warn + ["이익 부재로 가치 산정 불가"]}
    cagr = f.get("rev_cagr")
    capable = 0.6 * min(cagr, 0.30) + 0.4 * 0.04 if cagr is not None else 0.05   # 장기 성장 능력 (평균 회귀 가정)
    dcf = lin(capable - g, -0.15, 0.10)
    pe = f.get("fwd_pe")
    if pe is not None and 0 < pe < 4:
        warn.append(f"선행 PER {pe:.1f}배: 데이터 오류 또는 이익 고점 의심, 가치 계산에서 제외")
        pe = None
    pe_score = lin(1 / pe, 0.02, 0.08) if pe and pe > 0 else None
    value = dcf if pe_score is None else 0.7 * dcf + 0.3 * pe_score
    if peak:
        value = min(value, 60.0)
    return {"value": value, "implied_g": g, "capable_g": capable, "warn": warn}


def tech_score(ticker: str, f: dict) -> dict:
    """기술 점수 = 평가표 80점 + 실제 재무로 계산한 매출·이익 기여 20점."""
    m = moat()["scores"].get(ticker)
    if not m:
        return {"tech": None, "demand": None, "core": None, "tech_risk": None, "confidence": None}
    earned = _mean([lin(f.get("om"), 0.0, 0.30), lin(f.get("gm"), 0.2, 0.7), lin(f.get("rev_cagr"), 0.0, 0.25)])
    total = sum(m["s"]) + (8.0 if earned is None else earned / 5)
    if m["c"] == "low":                                    # 근거 부족 종목은 관문 통과 불가
        total = min(total, 59.0)
    return {"tech": round(total, 1), "demand": m["s"][2], "core": m["core"], "tech_risk": m["risk"], "confidence": m["c"]}
