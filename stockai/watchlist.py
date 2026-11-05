"""지정 감시 종목의 매수·매도 점수와 알림.

점수는 조건이 얼마나 갖춰졌는지를 나타내는 0~100 값이며 상승 확률이 아니다.
기업·뉴스·거시는 일일 분석 값을 쓰고, 시점은 매시간 새 가격으로 다시 계산한다.
"""
from __future__ import annotations

import json
import logging

import numpy as np

from . import flows, macro, signals, store
from .alerts import fmt_price
from .config import REPORTS, benchmark_for, settings, universe
from .features import price_frame

log = logging.getLogger(__name__)
TIMING_TOP = 95                                            # timing_score 가 낼 수 있는 최고점
SLOW = ("tech", "fin", "value", "flow", "news", "macro", "abnormal")
NOT_PROBABILITY = "※ 점수는 조건 충족도이며 상승 확률이 아닙니다. 참고용입니다."


def _blocks(r: dict, stage: str) -> list[str]:
    out = [f"공시: {x}" for x in r.get("filing_flags", [])[:1]]
    if (r.get("abnormal") or 0) >= settings()["thresholds"]["abnormal_veto"]:
        out.append("이상거래 위험 높음")
    if r.get("red_flags"):
        out.append(f"'{r['red_flags'][0]['word']}' 뉴스 확인 필요")
    if r.get("earnings_days") is not None and 0 <= r["earnings_days"] <= 3:
        out.append("3일 내 실적 발표")
    if stage == "위험 회피":
        out.append("시장 위험 회피 국면")
    return out


def build(results: list[dict], reg: dict, tilt: dict, panel: dict) -> list[dict]:
    """일일 분석 결과에서 감시 종목의 느린 지표를 모은다."""
    by = {r["ticker"]: r for r in results}
    rows = []
    for w in universe()["watchlist"]:
        t = w["ticker"]
        if t in by:
            r = by[t]
            rows.append({"ticker": t, "name": r["name"], "market": r["market"],
                         "parts": {k: r.get(k) for k in SLOW}, "blocks": _blocks(r, reg[r["market"]]["stage"])})
        elif t in panel:                                   # 기업 평가가 없는 금·은 ETF
            parts = {"flow": flows.flow_score(panel[t], None, None), "macro": macro.stock_macro(reg, w, panel, tilt)}
            rows.append({"ticker": t, "name": w["name"], "market": w["market"], "parts": parts,
                         "blocks": _blocks({}, reg[w["market"]]["stage"])})
        else:
            log.warning("감시 종목 시세 없음: %s", t)
    return rows


def blend(parts: dict, weights: dict, flip: bool = False) -> float | None:
    """없는 항목을 뺀 가중 평균. flip 이면 낮을수록 나쁜 항목을 뒤집어 매도 점수로 쓴다."""
    have = {k: w for k, w in weights.items() if parts.get(k) is not None}
    if not have:
        return None
    val = lambda k: parts[k] if not flip or k == "abnormal" else 100 - parts[k]
    return round(sum(val(k) * w for k, w in have.items()) / sum(have.values()), 1)


def calibrate(raw: float | None, scale: list) -> float | None:
    """평균 원점수는 90점에 닿지 않으므로 과거 분포 눈금으로 보고 점수로 바꾼다."""
    if raw is None:
        return None
    xs, ys = zip(*scale)
    return round(float(np.interp(raw, xs, ys)), 1)


def score(row: dict, panel: dict) -> dict | None:
    df = panel.get(row["ticker"])
    if df is None or len(df) < 260:
        return None
    f = price_frame(df, panel[benchmark_for(row)]["close"].reindex(df.index, method="ffill")).iloc[-1]
    if f["timing"] != f["timing"]:
        return None
    cfg = settings()["watch"]
    parts = {**row["parts"], "timing": min(float(f["timing"]) / TIMING_TOP * 100, 100.0)}
    held = list(row["blocks"])
    if signals.crashed(f):
        held.append("최근 5일 급락")
    if signals.overextended(f):
        held.append("단기 이격 과대")
    buy = calibrate(blend(parts, cfg["buy"]), cfg["buy_scale"])
    close = float(df["close"].iloc[-1])
    return {**row, "close": close, "day": str(df.index[-1].date()), "timing": parts["timing"], "held": held,
            "buy": min(buy, cfg["blocked_buy_max"]) if held else buy, "sell": calibrate(blend(parts, cfg["sell"], flip=True), cfg["sell_scale"]),
            "stop": signals.trade_plan(close, float(f["atr_pct"]), float(f["ma20_gap"]))["stop"]}


def score_all(rows: list[dict], panel: dict) -> list[dict]:
    return [s for s in (score(r, panel) for r in rows) if s]


def _message(s: dict, side: str, level: int) -> str:
    strong = level >= settings()["watch"]["strong"]
    word = "매수" if side == "buy" else "매도"
    head = f'{"🚨 강한" if strong else "🔔"} {word} 조건 {s[side]:.0f}점: {s["name"]}'
    line = f'현재가 {fmt_price(s["close"], s["market"])}'
    if side == "buy":
        line += f' | 손절 참고 {fmt_price(s["stop"], s["market"])}'
    return head + "\n" + line


def alerts(scored: list[dict]) -> list[str]:
    """기준을 넘은 항목을 고른다. 같은 기준은 며칠간 다시 보내지 않고, 95점을 보내면 90점은 함께 처리한다."""
    cfg = settings()["watch"]
    out = []
    for s in scored:
        for side in ("buy", "sell"):
            hit = [x for x in (cfg["strong"], cfg["alert"]) if (s[side] or 0) >= x]
            fresh = [store.should_send(f'watch|{s["ticker"]}|{side}|{x}', cfg["dedup_days"]) for x in hit]
            if fresh and fresh[0]:
                out.append(_message(s, side, hit[0]))
                store.log_watch(s["day"], s["ticker"], side, s[side], s["close"])
    return out


def saved_rows() -> list[dict]:
    """장중 점검이 쓸 마지막 일일 분석 값."""
    path = REPORTS / "latest.json"
    if not path.exists():
        log.warning("일일 분석 결과가 없어 감시 종목 점검을 건너뜁니다")
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("watch", [])
