"""거시 환경 판단. 시장 추세, 변동성, 신용 스프레드, 금리, 환율, 원자재."""
from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd
import requests

from .config import universe
from .features import last_moves

log = logging.getLogger(__name__)


def fred(series_id: str) -> pd.Series | None:
    try:
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", timeout=20)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))
        s = pd.to_numeric(d.iloc[:, 1], errors="coerce")
        s.index = pd.to_datetime(d.iloc[:, 0])
        return s.dropna()
    except (requests.RequestException, ValueError, IndexError) as e:
        log.warning("FRED 실패 %s: %s", series_id, e)
        return None


def fred_snapshot() -> dict[str, dict]:
    out = {}
    for sid, name in universe()["fred_series"].items():
        s = fred(sid)
        if s is None or len(s) < 14:
            continue
        if sid == "CPIAUCSL":                               # 물가는 전년 대비 상승률로 표시
            yoy = (s / s.shift(12) - 1) * 100
            out[sid] = {"name": "소비자물가 전년비", "value": round(float(yoy.iloc[-1]), 2),
                        "prev": round(float(yoy.iloc[-4]), 2), "date": str(yoy.index[-1].date())}
        else:
            back = 63 if len(s) > 500 and sid != "FEDFUNDS" and sid != "UNRATE" else 3
            out[sid] = {"name": name, "value": round(float(s.iloc[-1]), 2),
                        "prev": round(float(s.iloc[-1 - back]), 2), "date": str(s.index[-1].date())}
    return out


def _trend(panel: dict, t: str) -> dict | None:
    if t not in panel or len(panel[t]) < 200:
        return None
    c = panel[t]["close"]
    return {"last": float(c.iloc[-1]), "above200": bool(c.iloc[-1] > c.tail(200).mean()),
            "above50": bool(c.iloc[-1] > c.tail(50).mean()), "ret63": float(c.iloc[-1] / c.iloc[-64] - 1),
            **last_moves(c)}


def regime(panel: dict, snapshot: dict) -> dict:
    """시장별 환경 점수 0~100과 위험 선호 단계."""
    reasons, base = [], 50.0
    vix = panel["^VIX"]["close"].iloc[-1] if "^VIX" in panel else None
    if vix is not None:
        base += 10 if vix < 20 else -10 if vix < 30 else -25
        reasons.append(f"VIX {vix:.1f}")
    hy = snapshot.get("BAMLH0A0HYM2")
    if hy:
        widening = hy["value"] - hy["prev"]
        base += -15 if widening > 1 else 5 if widening <= 0 else 0
        reasons.append(f"하이일드 스프레드 {hy['value']:.2f}%p ({widening:+.2f})")
    out = {}
    for market, idx in (("US", "^GSPC"), ("KR", "^KS11")):
        t = _trend(panel, idx)
        s = base
        if t:
            s += 20 if t["above200"] else -25
            s += 5 if t["above50"] else -5
        if market == "KR":
            fx = _trend(panel, "KRW=X")
            if fx and fx["ret63"] > 0.04:
                s -= 10
                reasons.append(f"원화 약세 3개월 {fx['ret63']:+.1%}")
        s = float(np.clip(s, 0, 100))
        out[market] = {"score": s, "stage": "위험 회피" if s < 35 else "중립" if s < 60 else "위험 선호", "index": t}
    out["reasons"] = reasons
    out["commodities"] = {n: _trend(panel, t) for n, t in universe()["commodities"].items()}
    return out


def stock_macro(reg: dict, stock: dict, panel: dict, tilt: dict[str, float]) -> float:
    """시장 환경에 원자재 추세와 이슈 기울기를 더한 종목별 거시 점수."""
    s = reg[stock["market"]]["score"]
    drv = _trend(panel, stock["driver"]) if stock.get("driver") else None
    if drv:
        s += 10 if drv["above50"] and drv["ret63"] > 0 else -10 if not drv["above50"] and drv["ret63"] < 0 else 0
    return float(np.clip(s + 5 * tilt.get(stock["category"], 0.0), 0, 100))
