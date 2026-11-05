"""기술적 특성 계산. 모든 값은 해당 시점까지의 데이터만 사용한다."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import benchmark_for, settings, stocks

MACRO_COLS = {"^VIX": "vix", "^TNX": "tnx", "DX-Y.NYB": "dxy", "KRW=X": "usdkrw",
              "CL=F": "oil", "GC=F": "gold", "HG=F": "copper"}


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n).mean()


def last_moves(close: pd.Series) -> dict:
    """최근 등락과, 직전 63일 변동폭 대비 마지막 하루 움직임의 크기."""
    c = close.tail(70)
    r = np.log(c).diff()
    sig = r.iloc[-64:-1].std()
    return {"ret1": float(c.iloc[-1] / c.iloc[-2] - 1),
            "ret21": float(c.iloc[-1] / c.iloc[-22] - 1) if len(c) >= 22 else None,
            "z1": float(r.iloc[-1] / sig) if sig > 0 else None}


def stock_features(df: pd.DataFrame) -> pd.DataFrame:
    c, v = df["close"], df["volume"].astype(float)
    logc = np.log(c)
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 21, 63, 126, 252):
        f[f"ret_{n}"] = logc.diff(n)
    r = logc.diff()
    f["vol_21"] = r.rolling(21).std()
    f["vol_63"] = r.rolling(63).std()
    f["vol_long"] = r.rolling(756, min_periods=252).std()
    f["vol_ratio"] = f["vol_21"] / r.rolling(252, min_periods=126).std()
    for n in (20, 50, 200):
        f[f"ma{n}_gap"] = c / c.rolling(n).mean() - 1
    f["ma50_over_200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1
    f["rsi14"] = rsi(c)
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c
    f["atr_pct"] = atr(df) / c
    lv = np.log(v.where(v > 0))
    f["volume_z"] = (lv - lv.rolling(60).mean()) / lv.rolling(60).std()
    f["dd_252"] = c / c.rolling(252, min_periods=60).max() - 1
    return f


def timing_score(f: pd.DataFrame) -> pd.Series:
    """매수 시점 점수 0~100. 실시간 판단과 백테스트가 같은 함수를 쓴다."""
    s = pd.Series(50.0, index=f.index)
    s += np.where(f["ma200_gap"] > 0, 12, -12)
    s += np.where(f["ma50_over_200"] > 0, 8, -8)
    s += np.where(f["ma50_gap"] > 0, 5, -5)
    s += (f["ret_126"] - f["ret_21"]).clip(-0.4, 0.4) / 0.4 * 12     # 최근 1개월 제외 6개월 모멘텀
    s += np.where(f["rs_63"] > 0, 8, -8)
    s -= ((f["rsi14"] - 75).clip(lower=0) * 1.5).fillna(0)           # 과열 감점
    stretch = f["ma20_gap"] / (f["vol_21"] * 4.5)                    # 20일선 이격을 변동성으로 표준화
    s -= (stretch.clip(lower=1) - 1).fillna(0) * 10
    return s.clip(0, 100).where(f["ma200_gap"].notna())


def price_frame(df: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    """가격 특성, 시장 대비 수익률, 시점 점수. bench 는 df 날짜에 맞춘 지수 종가."""
    f = stock_features(df)
    lb = np.log(bench)
    for n in (21, 63):
        f[f"rs_{n}"] = f[f"ret_{n}"] - lb.diff(n)
    f["timing"] = timing_score(f)
    return f


def _lagged_market(panel: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    """시장 지표. 한국장은 같은 날 미국 종가를 볼 수 없으므로 전 종목 1거래일 지연 적용."""
    m = pd.DataFrame(index=index)
    for t, name in MACRO_COLS.items():
        if t not in panel:
            continue
        s = panel[t]["close"].reindex(index, method="ffill").shift(1)
        if name in ("vix", "tnx"):
            m[name] = s
            m[f"{name}_chg21"] = s.diff(21)
        else:
            m[f"{name}_ret21"] = np.log(s.where(s > 0)).diff(21)   # 2020년 원유 선물 음수 가격 제외
    return m


def build_panel(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """전 종목 특성 + 미래 수익률 라벨을 (date, ticker) 긴 표로 만든다."""
    horizons = settings()["horizons"]
    cats = sorted({s["category"] for s in stocks()})
    frames = []
    for s in stocks():
        t = s["ticker"]
        if t not in panel or len(panel[t]) < 260:
            continue
        df = panel[t]
        bench = panel[benchmark_for(s)]["close"].reindex(df.index, method="ffill")
        f = price_frame(df, bench)
        lb = np.log(bench)
        f["bench_ret21"] = lb.diff(21)
        f["bench_ma200_gap"] = bench / bench.rolling(200).mean() - 1
        f = f.join(_lagged_market(panel, df.index))
        f["is_kr"] = int(s["market"] == "KR")
        f["category"] = cats.index(s["category"])
        f["ticker"] = t
        logc = np.log(df["close"])
        dates = pd.Series(df.index, index=df.index)
        for h in horizons.values():
            f[f"y_{h}"] = logc.shift(-h) - logc
            f[f"x_{h}"] = f[f"y_{h}"] - (lb.shift(-h) - lb)      # 시장 초과수익
            f[f"end_{h}"] = dates.shift(-h)                      # 라벨 확정일
        frames.append(f.rename_axis("date").reset_index())
    out = pd.concat(frames, ignore_index=True)
    for col in ("ret_21", "ret_126", "vol_63"):                  # 같은 날 종목 간 순위
        out[f"{col}_rank"] = out.groupby(["date", "is_kr"])[col].rank(pct=True)
    return out


def feature_columns(table: pd.DataFrame) -> list[str]:
    skip = ("y_", "x_", "end_")
    return [c for c in table.columns if c not in ("date", "ticker") and not c.startswith(skip)]
