"""분석 이력 저장. 실제 운영 성과를 나중에 채점하기 위해 모든 전망을 기록한다."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta

import pandas as pd

from .config import DATA

DB = DATA / "history.db"
SCHEMA = """
create table if not exists signals(day text, ticker text, action text, timing real, tech real, fin real, value real,
  abnormal real, close real, primary key(day, ticker));
create table if not exists forecasts(day text, ticker text, horizon text, days int, close real, low real, mid real,
  high real, p_up real, p_ref real, primary key(day, ticker, horizon));
create table if not exists alerts(key text primary key, sent_at text);
"""


def connect() -> sqlite3.Connection:
    DATA.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    return con


def previous_actions(day: str) -> dict[str, str]:
    with closing(connect()) as con:
        rows = con.execute("select ticker, action from signals where day=(select max(day) from signals where day<?)", (day,))
        return dict(rows.fetchall())


def save(day: str, results: list[dict]) -> None:
    with closing(connect()) as con, con:
        for r in results:
            con.execute("insert or replace into signals values(?,?,?,?,?,?,?,?,?)",
                        (day, r["ticker"], r["action"], r.get("timing"), r.get("tech"), r.get("fin"), r.get("value"),
                         r.get("abnormal"), r["close"]))
            for f in r.get("forecast", []):
                con.execute("insert or replace into forecasts values(?,?,?,?,?,?,?,?,?,?)",
                            (day, r["ticker"], f["horizon"], f["days"], r["close"], f["low"], f["mid"], f["high"],
                             f["p_up"], f["p_ref"]))


def should_send(key: str, dedup_days: int) -> bool:
    """같은 알림을 반복해서 보내지 않는다."""
    with closing(connect()) as con, con:
        row = con.execute("select sent_at from alerts where key=?", (key,)).fetchone()
        if row and datetime.fromisoformat(row[0]) > datetime.now() - timedelta(days=dedup_days):
            return False
        con.execute("insert or replace into alerts values(?,?)", (key, datetime.now().isoformat()))
        return True


def live_scorecard(panel: dict) -> list[dict]:
    """기간이 지난 과거 전망을 실제 가격과 비교해 채점한다 (실시간 모의운용 기록)."""
    with closing(connect()) as con:
        d = pd.read_sql("select * from forecasts", con)
    out = []
    for horizon, g in d.groupby("horizon", sort=False):
        inside = total = 0
        for r in g.itertuples():
            df = panel.get(r.ticker)
            if df is None:
                continue
            after = df.loc[df.index > pd.Timestamp(r.day), "close"]
            if len(after) < r.days:
                continue
            actual = after.iloc[r.days - 1]
            total += 1
            inside += int(r.low <= actual <= r.high)
        if total:
            out.append({"horizon": horizon, "n": total, "in_band": round(inside / total, 3)})
    return out


def today() -> str:
    return str(date.today())
