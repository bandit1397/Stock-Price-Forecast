"""공시 위험 탐지. 미국 SEC EDGAR(키 불필요), 한국 OpenDART(무료 키 필요)."""
from __future__ import annotations

import io
import json
import logging
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, timedelta

import requests

from .config import DATA, env

log = logging.getLogger(__name__)
SEC_ITEMS = {"1.03": "파산·회생", "3.01": "상장 유지 기준 미달", "4.01": "감사인 교체", "4.02": "재무제표 신뢰 불가 공시"}
SEC_FORMS = {"NT 10-Q": "분기보고서 지연", "NT 10-K": "연간보고서 지연", "424B5": "신주 발행(희석 가능)"}
DART_WORDS = ["유상증자", "전환사채", "신주인수권부사채", "감자", "횡령", "배임", "불성실공시", "감사의견", "관리종목",
              "상장폐지", "소송", "회생절차", "영업정지", "최대주주변경"]


def _sec_headers() -> dict:
    contact = env("SEC_CONTACT")
    return {"User-Agent": f"StockPriceForecast personal research {contact}".strip()}


def _sec_cik_map() -> dict[str, int]:
    path = DATA / "sec_tickers.json"
    if not path.exists() or time.time() - path.stat().st_mtime > 30 * 86400:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=_sec_headers(), timeout=30)
        r.raise_for_status()
        DATA.mkdir(exist_ok=True)
        path.write_text(r.text, encoding="utf-8")
    return {v["ticker"]: v["cik_str"] for v in json.loads(path.read_text(encoding="utf-8")).values()}


def sec_flags(recent: dict, since: date) -> list[str]:
    out = []
    for form, filed, items in zip(recent["form"], recent["filingDate"], recent.get("items", [""] * len(recent["form"]))):
        if date.fromisoformat(filed) < since:
            continue
        if form in SEC_FORMS:
            out.append(f"{filed} {SEC_FORMS[form]}")
        if form == "8-K":
            out += [f"{filed} {label}" for code, label in SEC_ITEMS.items() if code in str(items).split(",")]
    return sorted(set(out), reverse=True)


def sec_recent(tickers: list[str], days: int = 30) -> dict[str, list[str]]:
    if not env("SEC_CONTACT"):                             # SEC는 연락처 없는 요청을 403으로 거부한다
        return {}
    try:
        ciks = _sec_cik_map()
    except (requests.RequestException, ValueError) as e:
        log.warning("SEC 목록 실패: %s", e)
        return {}
    out, since = {}, date.today() - timedelta(days=days)
    for t in tickers:
        if t not in ciks:
            continue
        try:
            r = requests.get(f"https://data.sec.gov/submissions/CIK{ciks[t]:010d}.json", headers=_sec_headers(), timeout=20)
            r.raise_for_status()
            out[t] = sec_flags(r.json()["filings"]["recent"], since)
        except (requests.RequestException, ValueError, KeyError) as e:
            log.warning("SEC 공시 실패 %s: %s", t, e)
        time.sleep(0.15)                                   # SEC 제한: 초당 10회
    return out


def _dart_corp_map(key: str) -> dict[str, str]:
    path = DATA / "dart_corp.json"
    if path.exists() and time.time() - path.stat().st_mtime < 30 * 86400:
        return json.loads(path.read_text(encoding="utf-8"))
    r = requests.get("https://opendart.fss.or.kr/api/corpCode.xml", params={"crtfc_key": key}, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        root = ET.fromstring(z.read(z.namelist()[0]))
    out = {x.findtext("stock_code").strip(): x.findtext("corp_code") for x in root.iter("list")
           if (x.findtext("stock_code") or "").strip()}
    DATA.mkdir(exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


def dart_recent(tickers: list[str], days: int = 30) -> dict[str, list[str]]:
    key = env("DART_API_KEY")
    if not key:
        return {}
    try:
        corp = _dart_corp_map(key)
    except (requests.RequestException, zipfile.BadZipFile, ET.ParseError) as e:
        log.warning("DART 목록 실패: %s", e)
        return {}
    start = (date.today() - timedelta(days=days)).strftime("%Y%m%d")
    out = {}
    for t in tickers:
        code = corp.get(t.split(".")[0])
        if not code:
            continue
        try:
            r = requests.get("https://opendart.fss.or.kr/api/list.json", timeout=20,
                             params={"crtfc_key": key, "corp_code": code, "bgn_de": start, "page_count": 100})
            r.raise_for_status()
            rows = r.json().get("list", [])
            out[t] = [f'{x["rcept_dt"]} {x["report_nm"].strip()}' for x in rows if any(w in x["report_nm"] for w in DART_WORDS)]
        except (requests.RequestException, ValueError) as e:
            log.warning("DART 공시 실패 %s: %s", t, e)
    return out
