"""텔레그램 연결 도우미. 토큰은 사용자 PC 안에서만 다루며 화면이나 로그에 다시 출력하지 않는다."""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

import requests

from .config import ROOT

REPO = "bandit1397/Stock-Price-Forecast"
TOKEN_RE = re.compile(r"^\d{6,12}:[A-Za-z0-9_-]{30,}$")
API = "https://api.telegram.org/bot{}/{}"


def call(token: str, method: str, **params) -> dict:
    """예외 문구에 토큰이 든 URL이 포함되므로 원문을 밖으로 내보내지 않는다."""
    try:
        r = requests.post(API.format(token, method), json=params, timeout=40)
        return r.json()
    except (requests.RequestException, ValueError) as e:
        return {"ok": False, "description": f"연결 실패 ({type(e).__name__})"}


def latest_chat_id(updates: list[dict]) -> int | None:
    """가장 최근 개인 대화의 ID."""
    for u in reversed(updates):
        chat = (u.get("message") or {}).get("chat") or {}
        if chat.get("type") == "private":
            return chat["id"]
    return None


def wait_for_chat(token: str, seconds: int = 120) -> int | None:
    end = time.time() + seconds
    while time.time() < end:
        res = call(token, "getUpdates", timeout=10)
        chat = latest_chat_id(res.get("result") or [])
        if chat is not None:
            return chat
        time.sleep(1)
    return None


def write_env(values: dict[str, str], path: Path | None = None) -> None:
    """기존 설정은 유지하고 지정한 값만 바꾼다."""
    path = path or ROOT / ".env"
    source = path if path.exists() else ROOT / ".env.example"
    lines = source.read_text(encoding="utf-8").splitlines() if source.exists() else []
    left = dict(values)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if key in left and not line.lstrip().startswith("#"):
            lines[i] = f"{key}={left.pop(key)}"
    lines += [f"{k}={v}" for k, v in left.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def push_secrets(values: dict[str, str]) -> bool:
    """GitHub 비밀값 등록. 값은 명령줄이 아니라 표준입력으로 넘긴다."""
    gh = shutil.which("gh")
    if not gh:
        return False
    for name, value in values.items():
        done = subprocess.run([gh, "secret", "set", name, "-R", REPO], input=value, text=True, capture_output=True)
        if done.returncode != 0:
            return False
    return True


def run() -> int:
    print("\n=== 텔레그램 알림 연결 ===\n")
    print("BotFather에서 받은 토큰을 붙여 넣고 Enter를 누르세요 (마우스 오른쪽 클릭 또는 Ctrl+V).")
    token = input("토큰: ").strip()
    if not TOKEN_RE.match(token):
        print("\n토큰 형식이 아닙니다. 123456789:ABC... 형태의 전체 문자열을 복사했는지 확인하세요.")
        return 1
    me = call(token, "getMe")
    if not me.get("ok"):
        print(f'\n토큰을 확인하지 못했습니다: {me.get("description", "알 수 없는 오류")}')
        return 1
    bot = me["result"]["username"]
    print(f"\n확인됨: @{bot}")
    print(f"\n휴대전화에서 @{bot} 를 열고 [시작]을 누른 뒤 '안녕'을 보내세요. 2분간 기다립니다...")
    chat = wait_for_chat(token)
    if chat is None:
        print("\n메시지를 받지 못했습니다. 봇에게 메시지를 보낸 뒤 다시 실행하세요.")
        return 1
    sent = call(token, "sendMessage", chat_id=chat, text="✅ 주식 알림이 연결되었습니다.")
    if not sent.get("ok"):
        print(f'\n시험 메시지 전송 실패: {sent.get("description", "알 수 없는 오류")}')
        return 1
    values = {"TELEGRAM_BOT_TOKEN": token, "TELEGRAM_CHAT_ID": str(chat)}
    write_env(values)
    print("\n1/2 PC 연결 완료. 휴대전화에 시험 메시지를 보냈습니다.")
    if push_secrets(values):
        print("2/2 GitHub 등록 완료. PC가 꺼져 있어도 평일에 자동으로 알림이 옵니다.")
    else:
        print("2/2 GitHub 등록 실패. PC 알림은 동작합니다. README의 GitHub Actions 항목대로 직접 등록하세요.")
    return 0
