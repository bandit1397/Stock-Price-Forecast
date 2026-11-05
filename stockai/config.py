"""설정 파일과 비밀값 로딩."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def _yaml(name: str) -> dict:
    with open(ROOT / "config" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def settings() -> dict:
    return _yaml("settings.yaml")


@lru_cache
def universe() -> dict:
    return _yaml("universe.yaml")


@lru_cache
def moat() -> dict:
    return _yaml("tech_moat.yaml")


@lru_cache
def macro_themes() -> dict:
    return _yaml("macro_themes.yaml")


def portfolio() -> list[dict]:
    path = ROOT / "config" / "portfolio.yaml"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("holdings", []) or []


def env(key: str) -> str:
    """환경변수 우선, 없으면 .env 파일에서 읽는다."""
    if os.environ.get(key):
        return os.environ[key]
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


def stocks() -> list[dict]:
    return universe()["stocks"]


def benchmark_for(stock: dict) -> str:
    b = universe()["benchmarks"]
    if stock["market"] == "US":
        return b["US"]
    return b["KRQ"] if stock["ticker"].endswith(".KQ") else b["KR"]


def all_tickers() -> list[str]:
    u = universe()
    etfs = [t["etf"] for t in u["themes"].values()]
    watch = [w["ticker"] for w in u["watchlist"]]
    return sorted({s["ticker"] for s in u["stocks"]} | set(etfs) | set(u["market_series"]) | set(watch))
