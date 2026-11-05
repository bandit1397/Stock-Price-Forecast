"""뉴스·이슈 수집 (Google News RSS, 무료).

제목의 키워드만 보는 단순 방식이다. 문맥이나 부정문은 이해하지 못하므로 점수 비중을 낮게 두고,
심각한 사건어는 사람이 확인하도록 경고로 전달한다.
"""
from __future__ import annotations

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import feedparser
import requests

log = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0 (StockPriceForecast personal research)"}

POS = ["수주", "공급계약", "흑자전환", "어닝 서프라이즈", "깜짝 실적", "최대 실적", "목표주가 상향", "자사주", "증설", "품목허가",
       "beats", "beat estimates", "raises guidance", "raised guidance", "upgrade", "upgraded", "wins contract",
       "record revenue", "fda approval", "fda approves", "buyback", "price target raised"]
NEG = ["어닝쇼크", "실적 부진", "목표주가 하향", "적자전환", "소송", "리콜", "하향 조정", "매도 리포트",
       "misses", "missed estimates", "cuts guidance", "lowers guidance", "downgrade", "downgraded",
       "lawsuit", "recall", "probe", "price target cut", "layoffs"]
RED = ["유상증자", "전환사채", "횡령", "배임", "상장폐지", "감사의견", "압수수색", "불성실공시", "거래정지", "관리종목", "분식",
       "fraud", "bankruptcy", "delisting", "restatement", "sec investigation", "short seller report", "accounting irregularities"]


def _hits(title: str, words: list[str]) -> list[str]:
    low = title.lower()
    out = []
    for w in words:
        if w.isascii():
            if re.search(rf"\b{re.escape(w)}\b", low):
                out.append(w)
        elif w in low:
            out.append(w)
    return out


def fetch_rss(query: str, lang: str = "en", days: int = 30) -> list[dict] | None:
    """기사 목록. 수집 실패는 None으로 돌려 '뉴스 없음'과 구분한다."""
    loc = "hl=ko&gl=KR&ceid=KR:ko" if lang == "ko" else "hl=en-US&gl=US&ceid=US:en"
    r = None
    for q in (query, query.replace('"', "")):              # 따옴표 검색이 간혹 404를 반환
        try:
            r = requests.get(f"https://news.google.com/rss/search?q={quote(q + f' when:{days}d')}&{loc}",
                             headers=UA, timeout=15)
            r.raise_for_status()
            break
        except requests.RequestException:
            r = None
    if r is None:
        log.warning("뉴스 수집 실패: %s", query)
        return None
    items = []
    for e in feedparser.parse(r.content).entries:
        ts = e.get("published_parsed")
        when = datetime(*ts[:6], tzinfo=timezone.utc) if ts else None
        items.append({"title": e.get("title", ""), "when": when, "link": e.get("link", "")})
    return items


def score_company(items: list[dict], names: list[str], now: datetime | None = None) -> dict:
    """회사 뉴스 점수. 기사가 3건 미만이면 판단하지 않는다."""
    now = now or datetime.now(timezone.utc)
    seen, rel = set(), []
    for it in items:
        key = re.sub(r"\W+", "", it["title"].rsplit(" - ", 1)[0].lower())[:60]
        if key in seen or not any(n.lower() in it["title"].lower() for n in names):
            continue
        seen.add(key)
        rel.append(it)
    net, red, heads = 0.0, [], []
    for it in rel:
        age = (now - it["when"]).days if it["when"] else 15
        w = math.exp(-max(age, 0) / 10)
        p, n = _hits(it["title"], POS), _hits(it["title"], NEG)
        net += w * (len(p) - len(n))
        r = _hits(it["title"], RED)
        if r and age <= 14:
            red.append({"word": r[0], "title": it["title"][:100], "link": it["link"]})
        if p or n or r:
            heads.append(it["title"][:100])
    recent = sum(1 for it in rel if it["when"] and (now - it["when"]) <= timedelta(days=7))
    score = 50 + 50 * math.tanh(net / 3) if len(rel) >= 3 else None
    return {"news": score, "count": len(rel), "recent7": recent, "red_flags": red[:3], "headlines": heads[:4]}


def intensity(items: list[dict] | None, now: datetime | None = None) -> float | None:
    """최근 7일 기사 속도 / 이전 23일 기사 속도. 검색이 최근 기사를 선호해 절대값은 부풀려진다."""
    now = now or datetime.now(timezone.utc)
    dated = [i["when"] for i in items or [] if i["when"]]
    if len(dated) < 5:
        return None
    new = sum(1 for d in dated if now - d <= timedelta(days=7))
    old = len(dated) - new
    return round((new / 7) / max(old / 23, 0.1), 2)


def company_news(stock_list: list[dict]) -> dict[str, dict]:
    def job(s: dict):
        kr = s["market"] == "KR"
        q = f'"{s["name"]}" 주식' if kr else f'"{s["en"]}" stock'
        items = fetch_rss(q, "ko" if kr else "en")
        return s["ticker"], None if items is None else score_company(items, [s["name"], s["en"]])

    with ThreadPoolExecutor(5) as ex:
        return dict(ex.map(job, stock_list))


def issue_scan(issues: list[dict]) -> list[dict]:
    def job(it: dict):
        items = fetch_rss(it["q"], it.get("lang", "en"))
        return {**it, "raw": intensity(items), "titles": [i["title"][:90] for i in (items or [])[:3]]}

    with ThreadPoolExecutor(5) as ex:
        rows = list(ex.map(job, issues))
    return relative_intensity(rows)


def relative_intensity(rows: list[dict]) -> list[dict]:
    """공통 쏠림을 없애기 위해 이슈 중앙값 대비 강도로 바꾼다. 1.0이면 다른 이슈와 비슷함."""
    vals = sorted(r["raw"] for r in rows if r.get("raw") is not None)
    mid = vals[len(vals) // 2] if vals else None
    for r in rows:
        r["intensity"] = round(r["raw"] / mid, 2) if r.get("raw") is not None and mid else None
    return rows


def sector_tilt(issues: list[dict]) -> dict[str, float]:
    """이슈 강도가 평소보다 높을 때만 업종별 기울기를 만든다."""
    tilt: dict[str, float] = {}
    for it in issues:
        excess = min(max((it.get("intensity") or 1) - 1, 0), 2)
        for sector, sign in (it.get("exposure") or {}).items():
            tilt[sector] = tilt.get(sector, 0.0) + sign * excess
    return tilt
