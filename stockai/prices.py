"""시세 수집과 저장. 무료 yfinance 사용 (개인 연구용, 실시간 보장 없음)."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from .config import DATA, all_tickers, settings

log = logging.getLogger(__name__)
PATH = DATA / "prices.parquet"
COLS = ["open", "high", "low", "close", "volume"]


def update(tickers: list[str] | None = None, retries: int = 3) -> pd.DataFrame:
    tickers = tickers or all_tickers()
    start = settings()["history_start"]
    raw = None
    for attempt in range(retries):
        try:
            raw = yf.download(tickers, start=start, auto_adjust=True, group_by="ticker",
                              threads=True, progress=False)
            if raw is not None and not raw.empty:
                break
        except Exception as e:  # 네트워크 일시 오류
            log.warning("시세 다운로드 재시도 %s: %s", attempt + 1, e)
        time.sleep(5 * (attempt + 1))
    if raw is None or raw.empty:
        raise RuntimeError("시세를 받지 못했습니다. 인터넷 연결이나 Yahoo 접속을 확인하세요.")
    frames = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            log.warning("시세 없음: %s", t)
            continue
        df = raw[t].rename(columns=str.lower)[COLS].dropna(subset=["close"]).copy()
        if df.empty:
            log.warning("시세 없음: %s", t)
            continue
        df["ticker"] = t
        frames.append(df.rename_axis("date").reset_index())
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    DATA.mkdir(exist_ok=True)
    out.to_parquet(PATH, index=False)
    return out


def drop_incomplete(df: pd.DataFrame, ticker: str, now: datetime | None = None) -> pd.DataFrame:
    """장중에 받은 미완성 일봉을 제거한다. 거래량이 작게 잡혀 지표를 왜곡하기 때문."""
    if df.empty:
        return df
    kr = ticker.endswith((".KS", ".KQ")) or ticker in ("^KS11", "^KQ11")
    tz, h, m = ("Asia/Seoul", 15, 45) if kr else ("America/New_York", 16, 15)
    now = (now or datetime.now(ZoneInfo("UTC"))).astimezone(ZoneInfo(tz))
    last = df.index[-1].date()
    if last == now.date() and (now.hour, now.minute) < (h, m):
        return df.iloc[:-1]
    return df


def load(complete_only: bool = True) -> dict[str, pd.DataFrame]:
    """티커별 일봉 DataFrame 딕셔너리."""
    if not PATH.exists():
        raise FileNotFoundError("시세 파일이 없습니다. 먼저 update 를 실행하세요.")
    data = pd.read_parquet(PATH)
    out = {}
    for t, g in data.groupby("ticker"):
        df = g.set_index("date").sort_index()[COLS]
        df = df[~df.index.duplicated(keep="last")]
        out[t] = drop_incomplete(df, t) if complete_only else df
    return out


def quality_issues(panel: dict[str, pd.DataFrame], tickers: list[str], max_stale_days: int = 5) -> list[str]:
    """오래된 시세, 짧은 이력 등 판단을 믿기 어려운 데이터를 찾는다."""
    issues = []
    today = pd.Timestamp.now().normalize()
    for t in tickers:
        df = panel.get(t)
        if df is None or df.empty:
            issues.append(f"{t}: 시세 없음")
            continue
        if (today - df.index[-1]).days > max_stale_days:
            issues.append(f"{t}: 마지막 시세 {df.index[-1].date()} (오래됨)")
        if len(df) < 252:
            issues.append(f"{t}: 이력 {len(df)}일로 짧음")
    return issues
