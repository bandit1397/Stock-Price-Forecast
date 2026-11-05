"""수급 분석. 한국은 네이버 금융의 외국인·기관 순매수(비공식 경로이므로 실패해도 계속 진행)."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from .fundamentals import _mean, lin

log = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0"}


def _num(x) -> float:
    return float(str(x).replace(",", "").replace("+", "") or 0)


def naver_flow(code: str, days: int = 20) -> dict | None:
    try:
        r = requests.get(f"https://m.stock.naver.com/api/stock/{code}/trend?pageSize={days}", headers=UA, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return None
        f = [_num(x["foreignerPureBuyQuant"]) for x in rows]
        o = [_num(x["organPureBuyQuant"]) for x in rows]
        v = [_num(x["accumulatedTradingVolume"]) for x in rows]
        return {"foreign5": sum(f[:5]) / max(sum(v[:5]), 1), "organ5": sum(o[:5]) / max(sum(v[:5]), 1),
                "foreign20": sum(f) / max(sum(v), 1), "organ20": sum(o) / max(sum(v), 1),
                "foreign_ratio": rows[0].get("foreignerHoldRatio")}
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        log.warning("수급 수집 실패 %s: %s", code, e)
        return None


def korean_flows(tickers: list[str]) -> dict[str, dict]:
    codes = [t for t in tickers if t.endswith((".KS", ".KQ"))]
    with ThreadPoolExecutor(4) as ex:
        return dict(zip(codes, ex.map(lambda t: naver_flow(t.split(".")[0]), codes)))


def up_volume_share(df: pd.DataFrame, n: int = 20) -> float | None:
    """상승일 거래량 비중. 모든 시장에 쓸 수 있는 매집 대리 지표."""
    d = df.tail(n + 1)
    chg = d["close"].diff().iloc[1:]
    v = d["volume"].iloc[1:].astype(float)
    return float(v[chg > 0].sum() / v.sum()) if v.sum() > 0 else None


def flow_score(df: pd.DataFrame, kr: dict | None, short_pct: float | None) -> float | None:
    parts = [lin(up_volume_share(df), 0.35, 0.65)]
    if kr:
        parts.append(lin(kr["foreign5"] + kr["organ5"], -0.15, 0.15))
        parts.append(lin(kr["foreign20"] + kr["organ20"], -0.10, 0.10))
    if short_pct is not None:
        parts.append(lin(short_pct, 0.20, 0.02))
    return _mean(parts)
