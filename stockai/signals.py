"""모든 분석을 하나의 판단으로 묶는다.

판단 순서
1. 위험 차단: 이상거래, 심각한 뉴스·공시
2. 좋은 회사 관문: 기술 → 시장 수요 → 수익성
3. 좋은 가격 관문: 주가에 반영된 성장 vs 실제 성장 능력
4. 시점: 추세, 과열, 시장 환경
"""
from __future__ import annotations

import pandas as pd

from .config import settings

BUY, WAIT, HOLD, SELL, AVOID, UNKNOWN = "매수 후보", "매수 대기", "관망", "매도 검토", "회피", "판단 보류"


def gates(c: dict) -> dict:
    th = settings()["thresholds"]

    def check(v, line):
        return None if v is None else bool(v >= line)

    demand = check(c.get("demand"), th["demand"])
    rev_ok = None if c.get("rev_g") is None else c["rev_g"] >= 0
    return {"기술력": check(c.get("tech"), th["tech"]),
            "시장 수요": None if demand is None else bool(demand and rev_ok is not False),
            "수익성": check(c.get("fin"), th["fin"]),
            "가격 적정성": check(c.get("value"), th["value"])}


def view_scores(c: dict) -> dict:
    """단기·중기·장기 점수. 없는 항목은 빼고 재계산하며 근거 비율을 함께 준다."""
    out = {}
    for name, w in settings()["view_weights"].items():
        have = {k: v for k, v in w.items() if c.get(k) is not None}
        total = sum(have.values())
        out[name] = {"score": round(sum(c[k] * v for k, v in have.items()) / total, 1) if total else None,
                     "coverage": round(total / sum(w.values()), 2)}
    return out


def trade_plan(close: float, atr_pct: float, ma20_gap: float) -> dict:
    r = settings()["risk"]
    stop = close * (1 - r["atr_stop_mult"] * atr_pct)
    risk_share = max(close - stop, 1e-9) / close
    return {"stop": stop, "entry_low": close / (1 + max(ma20_gap, 0)) if ma20_gap > 0 else close * (1 - atr_pct),
            "entry_high": close,
            "position_pct": round(min(r["risk_per_trade"] / risk_share, r["max_position"]) * 100, 1)}


def overextended(feat: pd.Series) -> bool:
    return bool(feat.get("ma20_gap", 0) > 3 * feat.get("atr_pct", 1))


def crashed(feat: pd.Series) -> bool:
    """5일 수익률이 평소 변동폭의 3.5배 넘게 하락."""
    sig, r5 = feat.get("vol_63"), feat.get("ret_5")
    return bool(pd.notna(sig) and pd.notna(r5) and sig > 0 and r5 / (sig * 5 ** 0.5) <= -3.5)


def decide(c: dict, feat: pd.Series, regime_stage: str) -> dict:
    th = settings()["thresholds"]
    g = gates(c)
    reasons: list[str] = []
    timing = c.get("timing")
    broken = timing is not None and timing <= th["timing_sell"] and feat.get("ma200_gap", 0) < 0
    extended = overextended(feat)
    company_ok = all(g[k] is True for k in ("기술력", "시장 수요", "수익성"))
    all_ok = company_ok and g["가격 적정성"] is True
    crash = crashed(feat)
    news_flag = bool(c.get("red_flags"))                   # 제목 키워드라 오탐이 있어 매수만 막는다
    earnings_soon = c.get("earnings_days") is not None and 0 <= c["earnings_days"] <= 3

    if timing is None:
        return {"action": UNKNOWN, "reasons": ["시세 이력 부족"], "gates": g}
    if c.get("filing_flags"):
        reasons.append("중대 공시 발생: " + c["filing_flags"][0])
    if (c.get("abnormal") or 0) >= th["abnormal_veto"]:
        reasons.append("이상거래 위험 높음")
    if reasons:
        action = SELL if broken else AVOID
    elif broken and not company_ok:                        # 차트와 기업 평가가 모두 나쁠 때만 매도 검토
        action = SELL
        reasons.append("200일선 아래 추세 훼손, 기업 관문도 미달")
    elif all_ok and timing >= th["timing_buy"] and regime_stage != "위험 회피" and not extended \
            and not (crash or news_flag or earnings_soon):
        action = BUY
        reasons.append("4개 관문 통과, 추세 양호")
    elif company_ok:
        action = WAIT
        if crash:
            reasons.append(f"최근 5일 {feat['ret_5']:+.0%} 급락, 원인 확인 전 진입 금지")
        if news_flag:
            reasons.append(f"'{c['red_flags'][0]['word']}' 뉴스 발생, 원문 확인 필요")
        if earnings_soon:
            reasons.append("3일 내 실적 발표, 결과 확인 후 판단")
        if broken:
            reasons.append("200일선 아래 추세 훼손, 보유 시 손절선 점검")
        if g["가격 적정성"] is not True:
            reasons.append("좋은 회사지만 가격 부담 또는 가치 산정 불가")
        if timing < th["timing_buy"]:
            reasons.append(f"시점 점수 {timing:.0f}점 미달")
        if extended:
            reasons.append("단기 이격 과대, 눌림 대기")
        if regime_stage == "위험 회피":
            reasons.append("시장이 위험 회피 국면")
    else:
        action = HOLD
        reasons += [f"{k} 관문 {'미달' if v is False else '자료 없음'}" for k, v in g.items() if v is not True][:3]
    return {"action": action, "reasons": reasons, "gates": g}
