"""시점 규칙 백테스트.

대상은 가격만으로 재현 가능한 '시점 점수' 규칙이다. 기술·재무 점수는 과거 시점의 값이 없어 검증하지 못한다.
종목을 현재 시점에 골랐으므로(생존 편향) 두 전략의 절대 수익률은 부풀려져 있다. 비교만 의미가 있다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import settings, stocks

ENTER, EXIT = 60, 45


def positions(timing: pd.Series, enter: float = ENTER, exit_: float = EXIT) -> pd.Series:
    """진입 60 이상, 이탈 45 미만. 잦은 매매를 막기 위해 구간을 둔다."""
    state, out = 0, []
    for v in timing.values:
        if np.isnan(v):
            state = 0
        elif state == 0 and v >= enter:
            state = 1
        elif state == 1 and v < exit_:
            state = 0
        out.append(state)
    return pd.Series(out, index=timing.index, dtype=float)


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / 252
    vol = r.std() * np.sqrt(252)
    return {"cagr": round(float(eq.iloc[-1] ** (1 / years) - 1), 4), "vol": round(float(vol), 4),
            "sharpe": round(float(r.mean() * 252 / vol), 3) if vol > 0 else None,
            "mdd": round(float((eq / eq.cummax() - 1).min()), 4)}


def run(panel: dict, table: pd.DataFrame, market: str | None = None, start: str = "2012-01-01") -> dict:
    cost = settings()["costs"]
    sel = [s for s in stocks() if s["ticker"] in panel and (market is None or s["market"] == market)]
    rets, pos, costs_b, costs_s = {}, {}, {}, {}
    for s in sel:
        t = s["ticker"]
        tm = table.loc[table["ticker"] == t].set_index("date")["timing"]
        if tm.empty:
            continue
        rets[t] = panel[t]["close"].pct_change().reindex(tm.index)
        pos[t] = positions(tm).shift(1).fillna(0)          # t일 종가로 판단, t+1일 종가에 체결
        costs_b[t], costs_s[t] = cost[s["market"]]["buy"] / 100, cost[s["market"]]["sell"] / 100
    R = pd.DataFrame(rets).loc[start:]
    # 한국·미국 휴장일이 달라 날짜가 비는 칸은 '미보유'가 아니라 '직전 포지션 유지'로 처리한다
    P = pd.DataFrame(pos).reindex(R.index).ffill().fillna(0)
    listed = R.notna().cummax()
    n = listed.sum(axis=1).replace(0, np.nan)
    W = P.where(listed, 0).div(n, axis=0)                  # 종목당 1/N, 나머지는 현금
    held = W.shift(1).fillna(0)
    gross = (held * R.fillna(0)).sum(axis=1)
    dw = W.diff().fillna(0)
    fee = dw.clip(lower=0).mul(pd.Series(costs_b)).sum(axis=1) + (-dw.clip(upper=0)).mul(pd.Series(costs_s)).sum(axis=1)
    strat = gross - fee
    bench = R.fillna(0).where(listed).sum(axis=1) / n      # 동일가중 매수 후 보유 (비용 없음)
    mid = R.index[len(R) // 2]
    return {"market": market or "전체", "start": str(R.index[0].date()), "end": str(R.index[-1].date()), "stocks": len(R.columns),
            "strategy": metrics(strat), "buy_hold": metrics(bench),
            "first_half": {"strategy": metrics(strat.loc[:mid]), "buy_hold": metrics(bench.loc[:mid])},
            "second_half": {"strategy": metrics(strat.loc[mid:]), "buy_hold": metrics(bench.loc[mid:])},
            "exposure": round(float(W.sum(axis=1).mean()), 3),
            "turnover_year": round(float(dw.abs().sum(axis=1).mean() * 252), 2),
            "cost_year": round(float(fee.mean() * 252), 4)}
