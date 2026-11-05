"""PC 분석 화면. 실행: streamlit run app.py"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stockai import prices
from stockai import report as rp
from stockai.alerts import fmt_price
from stockai.config import REPORTS

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ACTIONS = ["매수 후보", "매수 대기", "관망", "매도 검토", "회피", "판단 보류"]
GATE_MARK = {True: "✅ 통과", False: "❌ 미달", None: "❔ 자료 없음"}
TARGET = {"up": "상승 여부", "beat": "시장 초과 여부"}

st.set_page_config(page_title="주식 종합 분석", page_icon="📈", layout="wide")


@st.cache_data(ttl=300)
def load_report() -> dict | None:
    path = REPORTS / "latest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(ttl=300)
def load_prices() -> dict:
    return prices.load() if prices.PATH.exists() else {}


def pct(x, digits=1) -> str:
    return "-" if x is None else f"{x:+.{digits}%}"


def num(x) -> str:
    return "-" if x is None else f"{x:.0f}"


def layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), hovermode="closest")
    fig.update_xaxes(showgrid=True, gridwidth=1, zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


def range_chart(s: dict) -> go.Figure:
    """현재가 대비 80% 가격 범위. 기간이 길수록 불확실성이 커지는 모습을 보여준다."""
    rows = list(reversed(s["forecast"]))
    y = [r["horizon"] for r in rows]
    lo = [(r["low"] / s["close"] - 1) * 100 for r in rows]
    hi = [(r["high"] / s["close"] - 1) * 100 for r in rows]
    mid = [(r["mid"] / s["close"] - 1) * 100 for r in rows]
    text = [f'{r["horizon"]}<br>범위 {fmt_price(r["low"], s["market"])} ~ {fmt_price(r["high"], s["market"])}'
            f'<br>중앙 {fmt_price(r["mid"], s["market"])}<br>{r["status"]}' for r in rows]
    fig = go.Figure(go.Bar(y=y, x=[h - l for h, l in zip(hi, lo)], base=lo, orientation="h", marker_color=BLUE,
                           marker_cornerradius=4, width=0.45, hovertext=text, hoverinfo="text"))
    fig.add_trace(go.Scatter(y=y, x=mid, mode="markers", hovertext=text, hoverinfo="text",
                             marker=dict(size=9, color="white", line=dict(color=BLUE, width=2))))
    fig.add_vline(x=0, line_width=1, line_color="#8a8984")
    fig.update_layout(showlegend=False, bargap=0.4)
    fig.update_xaxes(ticksuffix="%", title="현재가 대비 변동")
    return layout(fig, 340)


def price_chart(df: pd.DataFrame) -> go.Figure:
    d = df.tail(504)
    full = df["close"]
    fig = go.Figure()
    for name, series, color, width in (("종가", d["close"], BLUE, 2), ("50일선", full.rolling(50).mean().reindex(d.index), ORANGE, 2),
                                       ("200일선", full.rolling(200).mean().reindex(d.index), AQUA, 2)):
        fig.add_trace(go.Scatter(x=d.index, y=series, name=name, mode="lines", line=dict(color=color, width=width)))
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified")
    layout(fig, 320).update_layout(hovermode="x unified")
    return fig


def page_today(rep: dict) -> None:
    reg = rep["regime"]
    counts = {a: sum(r["action"] == a for r in rep["stocks"]) for a in ACTIONS}
    c = st.columns(4)
    c[0].metric("미국 시장", reg["US"]["stage"], f'{reg["US"]["score"]:.0f}점', delta_color="off", delta_arrow="off")
    c[1].metric("한국 시장", reg["KR"]["stage"], f'{reg["KR"]["score"]:.0f}점', delta_color="off", delta_arrow="off")
    c[2].metric("매수 후보", f'{counts["매수 후보"]}개')
    c[3].metric("매도 검토·회피", f'{counts["매도 검토"] + counts["회피"]}개')
    st.info(f'예측 모델: {rp.model_status(rep)}. 매수 후보는 기업·가격·추세 조건을 통과했다는 뜻이며, 상승을 예측했다는 뜻이 아닙니다.')
    for x in rep["data_issues"]:
        st.warning(x)
    cats = ["전체"] + sorted({r["category"] for r in rep["stocks"]})
    for tab, cat in zip(st.tabs(cats), cats):
        rows = [{"판단": r["action"], "종목": r["name"], "현재가": fmt_price(r["close"], r["market"]),
                 "기술": r.get("tech"), "재무": r.get("fin"), "가치": r.get("value"), "시점": r.get("timing"),
                 "수급": r.get("flow"), "이상거래": r["abnormal"], "장기 점수": r["views"]["장기"]["score"],
                 "사유": " / ".join(r["reasons"][:2])} for r in rp.ranked(rep, None if cat == "전체" else cat)]
        fmt = {k: st.column_config.NumberColumn(format="%.0f") for k in ("기술", "재무", "가치", "시점", "수급", "장기 점수")}
        tab.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config=fmt)
    if rep["portfolio"]:
        st.subheader("보유 종목")
        st.dataframe(pd.DataFrame(rep["portfolio"]), hide_index=True)


def page_stock(rep: dict) -> None:
    names = {f'{r["name"]} ({r["ticker"]})': r for r in rp.ranked(rep)}
    s = names[st.selectbox("종목", list(names))]
    c = st.columns(4)
    c[0].metric("판단", s["action"])
    c[1].metric("현재가", fmt_price(s["close"], s["market"]), s["date"], delta_color="off", delta_arrow="off")
    c[2].metric("이상거래 위험", s["abnormal_level"], f'{s["abnormal"]}점', delta_color="off", delta_arrow="off")
    if s.get("plan"):
        c[3].metric("손절 기준", fmt_price(s["plan"]["stop"], s["market"]), f'비중 최대 {s["plan"]["position_pct"]}%', delta_color="off", delta_arrow="off")
    for x in s["reasons"]:
        st.write("• " + x)

    left, right = st.columns(2)
    left.subheader("8개 기간 가격 범위 (80%)")
    left.plotly_chart(range_chart(s), width="stretch", theme="streamlit")
    left.caption("막대는 과거 변동성으로 계산한 80% 범위, 점은 중앙값입니다. 방향을 맞히는 예측이 아닙니다.")
    table = pd.DataFrame([{"기간": r["horizon"], "하단": fmt_price(r["low"], s["market"]), "중앙": fmt_price(r["mid"], s["market"]),
                           "상단": fmt_price(r["high"], s["market"]), "상승 확률": "미표시" if r["p_up"] is None else f'{r["p_up"]:.0%}',
                           "상태": r["status"]} for r in s["forecast"]])
    left.dataframe(table, hide_index=True, width="stretch")
    right.subheader("가격 추세")
    df = load_prices().get(s["ticker"])
    if df is not None:
        right.plotly_chart(price_chart(df), width="stretch", theme="streamlit")
    right.subheader("투자 판단 관문")
    right.dataframe(pd.DataFrame([{"관문": k, "결과": GATE_MARK[v]} for k, v in s["gates"].items()]), hide_index=True, width="stretch")

    st.subheader("기간별 종합 점수")
    v = st.columns(3)
    for col, (name, x) in zip(v, s["views"].items()):
        col.metric(name, num(x["score"]), f'근거 비율 {x["coverage"]:.0%}', delta_color="off", delta_arrow="off")
    st.caption("점수 가중치는 사람이 정한 값이며 통계로 검증되지 않았습니다.")

    a, b = st.columns(2)
    a.subheader("기술 경쟁력")
    a.write(f'**점수** {num(s.get("tech"))} / 100 · 근거 수준 {s.get("confidence") or "-"}')
    a.write(f'**핵심 기술** {s.get("core") or "-"}')
    a.write(f'**위험** {s.get("tech_risk") or "-"}')
    a.caption("80점은 초기 의견(config/tech_moat.yaml, 직접 수정 가능), 20점은 재무 데이터로 자동 계산합니다.")
    b.subheader("주가에 반영된 성장")
    f = s["fundamentals"]
    b.write(f'현재 주가가 요구하는 이익 성장 **연 {pct(s.get("implied_g"))}** · 추정 성장 능력 **연 {pct(s.get("capable_g"))}**')
    b.write(f'매출 성장 {pct(f.get("rev_g"))} · 영업이익률 {pct(f.get("om"))} · 평균 {pct(f.get("om_avg"))} · 선행 PER {num(f.get("fwd_pe"))}배')
    for w in s["value_warn"]:
        b.warning(w)
    if s.get("kr_flow"):
        k = s["kr_flow"]
        st.write(f'**최근 5일 수급 (거래량 대비)** 외국인 {pct(k["foreign5"])} · 기관 {pct(k["organ5"])} · 외국인 지분 {k.get("foreign_ratio")}')
    for x in s["abnormal_reasons"]:
        st.warning("이상 패턴: " + x)
    for x in s["red_flags"]:
        st.error(f'확인 필요 뉴스: {x["title"]}')
    for x in s["filing_flags"]:
        st.error("공시: " + x)
    if s["headlines"]:
        st.subheader("주요 뉴스")
        for x in s["headlines"]:
            st.write("• " + x)


def page_themes(rep: dict) -> None:
    st.subheader("급부상 산업 순위")
    t = list(reversed(rep["themes"]))
    fig = go.Figure(go.Bar(y=[x["name"] for x in t], x=[x["score"] for x in t], orientation="h", marker_color=BLUE, marker_cornerradius=4,
                           text=[f'{x["score"]:.0f}' for x in t], textposition="outside", cliponaxis=False,
                           hovertext=[f'{x["name"]}: {x["stage"]}<br>ETF {x["etf"]}' for x in t], hoverinfo="text"))
    fig.update_xaxes(range=[0, 100], title="점수 (100점 만점)")
    st.plotly_chart(layout(fig, 430), width="stretch", theme="streamlit")
    st.dataframe(pd.DataFrame([{"산업": x["name"], "단계": x["stage"], "ETF": x["etf"], "1개월 시장 대비": pct(x["rs21"]),
                                "3개월 시장 대비": pct(x["rs63"]), "거래대금 변화": f'{x["volume_ratio"]:.2f}배',
                                "구성종목 50일선 위": "-" if x["breadth"] is None else f'{x["breadth"]:.0%}'} for x in rep["themes"]]),
                 hide_index=True, width="stretch")
    st.caption("상대강도 25, 거래대금 20, 뉴스 20, 실적 15, 추세 10, 안전성 10점. 가중치는 검증되지 않은 초기값입니다.")


def page_macro(rep: dict) -> None:
    st.subheader("경제 지표")
    st.dataframe(pd.DataFrame([{"지표": v["name"], "현재": v["value"], "이전": v["prev"], "기준일": v["date"]} for v in rep["fred"].values()]),
                 hide_index=True, width="stretch")
    st.subheader("원자재")
    st.dataframe(pd.DataFrame([{"품목": k, "가격": v["last"], "1개월": pct(v["ret21"]), "3개월": pct(v["ret63"]),
                                "200일선 위": "예" if v["above200"] else "아니오"} for k, v in rep["regime"]["commodities"].items() if v]),
                 hide_index=True, width="stretch")
    st.subheader("정치·경제·사회 이슈")
    st.dataframe(pd.DataFrame([{"이슈": i["name"], "뉴스 집중도": i["intensity"], "최근 기사": (i["titles"] or ["-"])[0]} for i in rep["issues"]]),
                 hide_index=True, width="stretch")
    st.caption("뉴스 집중도 1.0은 다른 이슈와 비슷한 수준입니다. 기사량이 곧 주가 영향을 뜻하지는 않습니다.")
    if rep["events"]:
        st.subheader("3주 내 일정")
        st.dataframe(pd.DataFrame(rep["events"]).rename(columns={"date": "날짜", "name": "일정"}), hide_index=True, width="stretch")


def page_validation(rep: dict) -> None:
    m = rep.get("model")
    st.subheader("예측력 검증")
    st.write("과거 평균 확률보다 낫다는 증거가 있어야 통과합니다. 과거 데이터를 시간순으로 나눠 학습에 쓰지 않은 기간으로 시험했습니다.")
    if m:
        rows = []
        for k, v in m["gate"].items():
            h, target = k.split("|")
            rows.append({"기간": h, "대상": TARGET[target], "결과": "✅ 통과" if v.get("passed") else "❌ 미통과", "AUC": v.get("auc"),
                         "모델 적중률": v.get("hit_rate"), "단순 기준(항상 상승)": v.get("base_rate"), "개선도": v.get("skill"), "독립 표본": v.get("n_indep")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.subheader("가격 범위 정확도")
        st.dataframe(pd.DataFrame([{"기간": k, "80% 범위의 실제 포함률": "-" if v is None else f"{v:.1%}"} for k, v in m["coverage"].items()]),
                     hide_index=True, width="stretch")
    st.subheader("시점 규칙 백테스트 (비용 포함)")
    if rep.get("backtest"):
        st.dataframe(pd.DataFrame([{"시장": b["market"], "규칙 연수익": pct(b["strategy"]["cagr"]), "단순 보유 연수익": pct(b["buy_hold"]["cagr"]),
                                    "규칙 샤프": b["strategy"]["sharpe"], "보유 샤프": b["buy_hold"]["sharpe"], "규칙 최대낙폭": pct(b["strategy"]["mdd"]),
                                    "보유 최대낙폭": pct(b["buy_hold"]["mdd"]), "평균 투자비중": f'{b["exposure"]:.0%}'} for b in rep["backtest"]]),
                     hide_index=True, width="stretch")
        st.warning("현재 살아남은 유망 종목으로 계산해 절대 수익률은 실제보다 부풀려져 있습니다. 기술·재무 점수는 과거 시점 값이 없어 검증되지 않았습니다.")
    st.subheader("실제 운영 성적")
    if rep["scorecard"]:
        st.dataframe(pd.DataFrame(rep["scorecard"]).rename(columns={"horizon": "기간", "n": "채점 수", "in_band": "범위 적중률"}), hide_index=True)
    else:
        st.write("아직 기간이 지난 전망이 없습니다. 매일 실행하면 자동으로 쌓입니다.")


def block_auto_translate() -> None:
    """Streamlit이 페이지를 영어로 선언해 브라우저 번역이 판단 문구를 왜곡하는 것을 막는다."""
    st.html("<script>const d=document.documentElement;d.lang='ko';d.setAttribute('translate','no');"
            "d.classList.add('notranslate');</script>", unsafe_allow_javascript=True)


def main() -> None:
    block_auto_translate()
    rep = load_report()
    st.title("주식 종합 분석")
    if not rep:
        st.error("분석 결과가 없습니다. 먼저 run_daily.bat 을 실행하세요.")
        return
    st.caption(f'생성 {rep["generated"]} · 투자 판단 참고용이며 수익을 보장하지 않습니다. '
               '문구가 이상하면 브라우저 자동 번역을 끄세요 (이 페이지는 이미 한국어입니다).')
    pages = {"오늘의 판단": page_today, "종목 상세": page_stock, "급부상 산업": page_themes, "거시·이슈": page_macro, "검증 결과": page_validation}
    for tab, (name, fn) in zip(st.tabs(list(pages)), pages.items()):
        with tab:
            fn(rep)


main()
