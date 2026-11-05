"""사람이 읽는 보고서."""
from __future__ import annotations

from . import signals as sg
from .alerts import FOOT, fmt_price
from .config import REPORTS, settings

ORDER = {sg.BUY: 0, sg.WAIT: 1, sg.HOLD: 2, sg.SELL: 3, sg.AVOID: 4, sg.UNKNOWN: 5}


def _long(r: dict) -> float:
    return r["views"]["장기"]["score"] or 0


def ranked(rep: dict, category: str | None = None) -> list[dict]:
    xs = [r for r in rep["stocks"] if category is None or r["category"] == category]
    return sorted(xs, key=lambda r: (ORDER[r["action"]], -_long(r)))


def model_status(rep: dict) -> str:
    m = rep.get("model")
    if not m:
        return "모델 미학습"
    passed = [k for k, v in m["gate"].items() if v.get("passed")]
    return f'검증 통과 {len(passed)}/{len(m["gate"])}개' + (f' ({", ".join(passed)})' if passed else " → 상승 확률은 표시하지 않음")


def telegram_summary(rep: dict) -> str:
    n = settings()["alerts"]["daily_top"]
    reg = rep["regime"]
    lines = [f'📊 일일 주식 보고 {rep["day"]}',
             f'시장: 미국 {reg["US"]["stage"]} · 한국 {reg["KR"]["stage"]}',
             f'예측 모델: {model_status(rep)}', ""]
    buys = [r for r in ranked(rep) if r["action"] == sg.BUY][:n]
    lines.append("■ 매수 후보" if buys else "■ 매수 후보 없음")
    for r in buys:
        p = r["plan"]
        lines.append(f'· {r["name"]} {fmt_price(r["close"], r["market"])} | 손절 {fmt_price(p["stop"], r["market"])} | 비중≤{p["position_pct"]}%')
    waits = [r for r in ranked(rep) if r["action"] == sg.WAIT][:n]
    if waits:
        lines.append("\n■ 좋은 회사, 시점 대기")
        lines += [f'· {r["name"]}: {r["reasons"][0]}' for r in waits]
    bad = [r for r in rep["stocks"] if r["action"] in (sg.SELL, sg.AVOID)]
    if bad:
        lines.append("\n■ 매도 검토·회피")
        lines += [f'· {r["name"]} [{r["action"]}]: {r["reasons"][0]}' for r in bad[:n]]
    lines.append("\n■ 급부상 산업 상위 5")
    lines += [f'· {t["name"]} {t["score"]:.0f}점 ({t["stage"]})' for t in rep["themes"][:5]]
    hot = [i for i in rep["issues"] if (i["intensity"] or 0) >= 1.5][:3]
    if hot:
        lines.append("\n■ 뉴스가 집중된 이슈")
        lines += [f'· {i["name"]} (다른 이슈의 {i["intensity"]:.1f}배)' for i in hot]
    if rep["events"]:
        lines.append("\n■ 3주 내 일정")
        lines += [f'· {e["date"]} {e["name"]}' for e in rep["events"][:6]]
    if rep["data_issues"]:
        lines.append("\n■ 데이터 경고")
        lines += [f"· {x}" for x in rep["data_issues"][:4]]
    hits = [p for p in rep["portfolio"] if p.get("stop_hit")]
    if hits:
        lines.append("\n⚠ 보유 종목 손절선 도달: " + ", ".join(p["name"] for p in hits))
    return "\n".join(lines) + FOOT


def _pct(x, d=0):
    return "-" if x is None else f"{x:.{d}%}"


def _n(x):
    return "-" if x is None else f"{x:.0f}"


def markdown(rep: dict) -> str:
    L = [f'# 주식 종합 분석 {rep["day"]}', "", f'생성 {rep["generated"]} · 예측 모델 {model_status(rep)}', ""]
    L += ["## 카테고리별 판단", ""]
    for cat in sorted({r["category"] for r in rep["stocks"]}):
        L += [f"### {cat}", "", "| 판단 | 종목 | 현재가 | 기술 | 재무 | 가치 | 시점 | 이상거래 | 사유 |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for r in ranked(rep, cat):
            L.append(f'| {r["action"]} | {r["name"]} | {fmt_price(r["close"], r["market"])} | {_n(r.get("tech"))} | {_n(r.get("fin"))} | '
                     f'{_n(r.get("value"))} | {_n(r.get("timing"))} | {r["abnormal"]} | {"; ".join(r["reasons"][:2])} |')
        L.append("")
    L += ["## 급부상 산업", "", "| 순위 | 산업 | 점수 | 단계 | ETF |", "|---:|---|---:|---|---|"]
    L += [f'| {i + 1} | {t["name"]} | {t["score"]:.0f} | {t["stage"]} | {t["etf"]} |' for i, t in enumerate(rep["themes"])]
    L += ["", "## 데이터 경고", ""] + ([f"- {x}" for x in rep["data_issues"]] or ["- 없음"])
    L += ["", FOOT.strip()]
    return "\n".join(L)


def save(rep: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f'report_{rep["day"]}.md').write_text(markdown(rep), encoding="utf-8")
