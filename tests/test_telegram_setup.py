"""연결 도우미 테스트. 실제 텔레그램에 접속하지 않는다."""
import requests

from stockai import telegram_setup as ts

TOKEN = "123456789:" + "A" * 35


def test_token_format():
    assert ts.TOKEN_RE.match(TOKEN)
    assert not ts.TOKEN_RE.match("hello")
    assert not ts.TOKEN_RE.match("123456789:short")


def test_latest_private_chat_wins():
    ups = [{"message": {"chat": {"id": 1, "type": "private"}}}, {"message": {"chat": {"id": 2, "type": "private"}}},
           {"message": {"chat": {"id": -5, "type": "group"}}}, {"edited_message": {}}]
    assert ts.latest_chat_id(ups) == 2
    assert ts.latest_chat_id([]) is None


def test_write_env_keeps_other_values(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# 설명\nTELEGRAM_BOT_TOKEN=old\nDART_API_KEY=keep\n", encoding="utf-8")
    ts.write_env({"TELEGRAM_BOT_TOKEN": "new", "TELEGRAM_CHAT_ID": "7"}, p)
    text = p.read_text(encoding="utf-8")
    assert "TELEGRAM_BOT_TOKEN=new" in text and "old" not in text
    assert "DART_API_KEY=keep" in text and "TELEGRAM_CHAT_ID=7" in text and "# 설명" in text


def test_network_error_never_exposes_token(monkeypatch):
    def boom(url, **kw):
        raise requests.ConnectionError(f"failed {url}")

    monkeypatch.setattr(ts.requests, "post", boom)
    assert TOKEN not in str(ts.call(TOKEN, "getMe"))


def _fake_call(sent):
    def call(token, method, **params):
        if method == "getMe":
            return {"ok": True, "result": {"username": "my_bot"}}
        if method == "getUpdates":
            return {"ok": True, "result": [{"message": {"chat": {"id": 42, "type": "private"}}}]}
        sent.append(params)
        return {"ok": True}
    return call


def test_full_flow_saves_and_never_prints_token(monkeypatch, capsys):
    sent, saved, pushed = [], {}, {}
    monkeypatch.setattr("builtins.input", lambda _: f" {TOKEN} ")
    monkeypatch.setattr(ts, "call", _fake_call(sent))
    monkeypatch.setattr(ts, "write_env", saved.update)
    monkeypatch.setattr(ts, "push_secrets", lambda v: pushed.update(v) or True)
    assert ts.run() == 0
    assert saved == {"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": "42"} == pushed
    assert sent[0]["chat_id"] == 42
    assert TOKEN not in capsys.readouterr().out


def test_bad_token_stops_before_saving(monkeypatch):
    saved = {}
    monkeypatch.setattr("builtins.input", lambda _: "nonsense")
    monkeypatch.setattr(ts, "write_env", saved.update)
    assert ts.run() == 1 and not saved


def test_secret_value_goes_through_stdin(monkeypatch):
    calls = []

    class Done:
        returncode = 0

    monkeypatch.setattr(ts.shutil, "which", lambda _: "gh")
    monkeypatch.setattr(ts.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)) or Done())
    assert ts.push_secrets({"TELEGRAM_BOT_TOKEN": TOKEN})
    cmd, kw = calls[0]
    assert TOKEN not in " ".join(cmd) and kw["input"] == TOKEN
