"""텔레그램 알림. 토큰이 없으면 콘솔 출력만 한다."""
from __future__ import annotations

import logging

import requests

from . import signals as sg
from .config import env, settings
from .store import should_send

log = logging.getLogger(__name__)
LIMIT = 3800
FOOT = "\n\n※ 투자 판단 참고용이며 수익을 보장하지 않습니다."


def fmt_price(x: float, market: str) -> str:
    return f"{x:,.0f}원" if market == "KR" else f"${x:,.2f}"


def chunks(text: str, limit: int = LIMIT) -> list[str]:
    out, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit and cur:
            out.append(cur)
            cur = ""
        cur += line + "\n"
    return out + [cur] if cur.strip() else out


def send(text: str) -> bool:
    token, chat = env("TELEGRAM_BOT_TOKEN"), env("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print(text)
        return False
    ok = True
    for part in chunks(text):
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat, "text": part, "disable_web_page_preview": True}, timeout=20)
            if r.status_code != 200:
                log.error("텔레그램 실패 %s: %s", r.status_code, r.json().get("description", ""))
                ok = False
        except requests.RequestException as e:   # 예외 문구에 토큰 URL이 포함될 수 있어 기록하지 않음
            log.error("텔레그램 연결 실패: %s", type(e).__name__)
            ok = False
    return ok


def signal_changes(results: list[dict], prev: dict[str, str]) -> list[str]:
    """판단이 바뀐 종목만 즉시 알림 대상으로 만든다."""
    days = settings()["alerts"]["dedup_days"]
    msgs = []
    for r in results:
        old, new = prev.get(r["ticker"]), r["action"]
        if not old or old == new or new not in (sg.BUY, sg.SELL, sg.AVOID):
            continue
        if not should_send(f'{r["ticker"]}|{new}', days):
            continue
        m = r["market"]
        plan = r.get("plan") or {}
        line = f'[{new}] {r["name"]} ({r["ticker"]})\n{old} → {new} | 현재 {fmt_price(r["close"], m)}\n' + " / ".join(r["reasons"])
        if plan and new == sg.BUY:
            line += f'\n손절 기준 {fmt_price(plan["stop"], m)} | 권장 비중 최대 {plan["position_pct"]}%'
        msgs.append(line)
    return msgs


def theme_alerts(themes: list[dict]) -> list[str]:
    days = settings()["alerts"]["dedup_days"]
    return [f'[급부상 산업] {t["name"]}: {t["stage"]} (점수 {t["score"]:.0f}, ETF {t["etf"]})' for t in themes[:5]
            if t["stage"] in ("조기 발견", "과열 경고") and should_send(f'theme|{t["id"]}|{t["stage"]}', days)]
