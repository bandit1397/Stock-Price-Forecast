"""이상거래 위험 탐지. 패턴이 비슷하다는 뜻일 뿐 주가조작을 판정하지 않는다."""
from __future__ import annotations

import numpy as np
import pandas as pd

ILLIQUID = {"KR": 1e9, "US": 2e6}       # 20일 중앙 거래대금 기준 (원, 달러)


def abnormal(df: pd.DataFrame, feat: pd.Series, market: str, news_recent: int | None = None) -> dict:
    d = df.tail(60)
    close, reasons, score = d["close"], [], 0
    value = (d["close"] * d["volume"]).tail(20).median()
    if value < ILLIQUID[market]:
        score += 20
        reasons.append("거래대금이 작아 소액으로 가격이 움직일 수 있음")
    vz = feat.get("volume_z")
    if pd.notna(vz) and vz >= 3:
        score += 20
        reasons.append(f"거래량 폭증 (평소 대비 {vz:.1f}표준편차)")
    sig = feat.get("vol_63")
    r5 = feat.get("ret_5")
    if pd.notna(sig) and pd.notna(r5) and sig > 0 and abs(r5) / (sig * np.sqrt(5)) >= 3.5:
        score += 20
        reasons.append(f"5일 수익률 {r5:+.1%}로 평소 변동폭 초과")
        if news_recent == 0:
            score += 15
            reasons.append("뚜렷한 뉴스 없이 급변")
    peak = close.tail(20).max()
    trough = close.loc[:close.tail(20).idxmax()].min()
    if trough > 0 and peak / trough - 1 >= 0.6 and close.iloc[-1] / peak - 1 <= -0.2:
        score += 30
        reasons.append(f"급등({peak / trough - 1:+.0%}) 후 고점 대비 {close.iloc[-1] / peak - 1:.0%} 하락")
    r21 = feat.get("ret_21")
    if pd.notna(r21) and r21 >= 0.5:
        score += 15
        reasons.append(f"1개월 수익률 {np.expm1(r21):+.0%} 과열")
    t = d.tail(10)
    rng = (t["high"] - t["low"]).replace(0, np.nan)
    wick = ((t["high"] - t[["open", "close"]].max(axis=1)) / rng > 0.6) & (rng / t["close"] > 0.05)
    if wick.sum() >= 3:
        score += 10
        reasons.append("긴 윗꼬리 반복 (고가 매도 압력)")
    big = (close.pct_change().tail(20).abs() >= 0.25).sum()
    if market == "KR" and big >= 2:
        score += 20
        reasons.append(f"최근 20일 ±25% 이상 급변 {big}회")
    score = min(score, 100)
    return {"abnormal": score, "level": "높음" if score >= 70 else "주의" if score >= 30 else "낮음", "reasons": reasons}
