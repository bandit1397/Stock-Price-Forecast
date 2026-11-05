"""명령줄 실행.

python -m stockai run        시세 갱신 + 분석 + 일일 보고 + 변경 알림
python -m stockai scan       장중 빠른 점검 (급변, 손절선)
python -m stockai train      예측 모델 재검증 (주 1회 권장)
python -m stockai backtest   시점 규칙 백테스트
python -m stockai test-telegram
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np

from . import alerts, analyze, backtest, forecast, prices, report, store, telegram_setup
from .config import REPORTS, portfolio, settings, stocks
from .features import build_panel

log = logging.getLogger("stockai")
RETRAIN_DAYS = 7


def cmd_train() -> None:
    prices.update()
    b = forecast.train_all(build_panel(prices.load()))
    passed = [k for k, v in b["gate"].items() if v.get("passed")]
    print(f"검증 통과 {len(passed)}/{len(b['gate'])}: {passed}")


def cmd_backtest() -> None:
    panel = prices.load()
    table = build_panel(panel)
    res = [backtest.run(panel, table, m) for m in (None, "KR", "US")]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "backtest.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in res:
        print(r["market"], "전략", r["strategy"], "보유", r["buy_hold"], "노출", r["exposure"], "연비용", r["cost_year"])


def cmd_run(args) -> None:
    prices.update()
    model = forecast.MODEL_PATH
    if not model.exists() or time.time() - model.stat().st_mtime > RETRAIN_DAYS * 86400:
        log.info("예측 모델 재검증")
        forecast.train_all(build_panel(prices.load()))
    if not (REPORTS / "backtest.json").exists():
        cmd_backtest()
    rep = analyze.run()
    report.save(rep)
    prev = store.previous_actions(rep["day"])
    store.save(rep["day"], rep["stocks"])
    urgent = alerts.signal_changes(rep["stocks"], prev) + alerts.theme_alerts(rep["themes"])
    if urgent:
        alerts.send("🔔 신호 변경\n\n" + "\n\n".join(urgent) + alerts.FOOT)
    if not args.no_daily:
        alerts.send(report.telegram_summary(rep))
    print(f"완료: 종목 {len(rep['stocks'])}개, 즉시 알림 {len(urgent)}건")


def cmd_scan() -> None:
    """장중 점검. 미완성 일봉을 그대로 쓰며 가격 급변과 보유 종목 손절선만 본다."""
    prices.update()
    panel = prices.load(complete_only=False)
    msgs = []
    days = settings()["alerts"]["dedup_days"]
    for s in stocks():
        df = panel.get(s["ticker"])
        if df is None or len(df) < 70:
            continue
        r = np.log(df["close"]).diff()
        sig, move = r.iloc[-64:-1].std(), df["close"].iloc[-1] / df["close"].iloc[-2] - 1
        if sig > 0 and abs(r.iloc[-1]) >= 3.5 * sig and store.should_send(f'scan|{s["ticker"]}|{df.index[-1].date()}', days):
            msgs.append(f'[급변] {s["name"]} {move:+.1%} ({alerts.fmt_price(df["close"].iloc[-1], s["market"])})')
    for h in portfolio():
        df = panel.get(h["ticker"])
        stop = h.get("stop") or h["avg_price"] * 0.85
        if df is not None and df["close"].iloc[-1] <= stop and store.should_send(f'stop|{h["ticker"]}', 1):
            msgs.append(f'[손절선] {h["ticker"]} 현재 {df["close"].iloc[-1]:,.2f} ≤ {stop:,.2f}')
    if msgs:
        alerts.send("⚡ 장중 알림\n\n" + "\n".join(msgs) + alerts.FOOT)
    print(f"점검 완료: 알림 {len(msgs)}건")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("yfinance", "peewee", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    p = argparse.ArgumentParser(prog="stockai")
    p.add_argument("command", choices=["run", "scan", "train", "backtest", "test-telegram", "connect-telegram"])
    p.add_argument("--no-daily", action="store_true", help="일일 보고는 보내지 않고 변경 알림만 전송")
    a = p.parse_args()
    if a.command == "run":
        cmd_run(a)
    elif a.command == "scan":
        cmd_scan()
    elif a.command == "train":
        cmd_train()
    elif a.command == "backtest":
        cmd_backtest()
    elif a.command == "connect-telegram":
        sys.exit(telegram_setup.run())
    else:
        print("전송 성공" if alerts.send("✅ 주식 알림 연결 테스트") else "전송 실패 또는 토큰 미설정")
