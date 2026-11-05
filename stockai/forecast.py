"""기간별 예측과 예측력 검증.

원칙
- 모든 기간의 가격 범위는 변동성 기반 기준분포에서 나온다.
- ML 확률은 워크포워드 검증(시간순, 라벨 겹침 제거) 관문을 통과한 기간에만 표시한다.
- 2년, 3년은 독립 표본이 부족해 검증이 불가능하므로 시나리오로만 제공한다.
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import DATA, settings
from .features import feature_columns

log = logging.getLogger(__name__)
MODEL_PATH = DATA / "models.joblib"
Z80 = 1.2816
TARGETS = {"up": "y", "beat": "x"}     # up: 상승 여부, beat: 시장 초과 여부


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120,
                                          min_samples_leaf=300, l2_regularization=1.0, random_state=7)


def _thin(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """겹치는 라벨이 많은 긴 기간은 학습 날짜를 솎아낸다."""
    stride = max(1, h // 5)
    if stride == 1:
        return df
    keep = np.sort(df["date"].unique())[::stride]
    return df[df["date"].isin(keep)]


def walk_forward(table: pd.DataFrame, h: int, target: str, cols: list[str]) -> pd.DataFrame:
    """연도별 시간순 검증. 학습 라벨은 시험 시작일 전에 확정된 것만 쓴다."""
    ycol = f"{TARGETS[target]}_{h}"
    d = table.dropna(subset=[ycol, "timing"]).copy()
    d["label"] = (d[ycol] > 0).astype(int)
    out = []
    for year in range(settings()["first_test_year"], int(d["date"].dt.year.max()) + 1):
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        train = d[d[f"end_{h}"] < start]
        test = d[(d["date"] >= start) & (d["date"] < end)]
        if train["date"].nunique() < 500 or test.empty or train["label"].nunique() < 2:
            continue
        tr = _thin(train, h)
        m = _model().fit(tr[cols], tr["label"])
        out.append(pd.DataFrame({"date": test["date"].values, "label": test["label"].values,
                                 "p": m.predict_proba(test[cols])[:, 1],
                                 "base": train["label"].mean(), "fold": year}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def gate(oos: pd.DataFrame, h: int, n_boot: int = 2000) -> dict:
    """모델이 '과거 평균 확률'보다 낫다는 증거가 충분한지 판정한다."""
    g = settings()["gate"]
    if oos.empty or oos["label"].nunique() < 2:
        return {"passed": False, "reason": "표본 부족"}
    gain = (oos["base"] - oos["label"]) ** 2 - (oos["p"] - oos["label"]) ** 2
    daily = gain.groupby(oos["date"]).mean().iloc[::h]             # 겹치지 않는 날짜만
    rng = np.random.default_rng(7)
    boots = rng.choice(daily.values, size=(n_boot, len(daily)), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, g["ci_quantile"]))
    fold_win = float((gain.groupby(oos["fold"]).mean() > 0).mean())
    auc = float(roc_auc_score(oos["label"], oos["p"]))
    base_brier = float(((oos["base"] - oos["label"]) ** 2).mean())
    skill = float(gain.mean() / base_brier)
    passed = lo > 0 and fold_win >= g["min_fold_win"] and auc >= g["min_auc"]
    return {"passed": bool(passed), "auc": round(auc, 4), "skill": round(skill, 4),
            "ci_low": round(lo, 6), "fold_win": round(fold_win, 2), "n_indep": int(len(daily)),
            "hit_rate": round(float(((oos["p"] > 0.5) == (oos["label"] == 1)).mean()), 4),
            "base_rate": round(float(oos["label"].mean()), 4)}


def base_params(cur_d, long_d, h: int, mu: float | None = None):
    """기간 h의 로그수익 중앙값과 표준편차. 변동성은 장기 평균으로 회귀한다."""
    mu = settings()["long_run_return"] if mu is None else mu
    w = np.exp(-h / 63)
    sig_d = w * cur_d + (1 - w) * long_d
    med = (mu - 0.5 * (sig_d ** 2) * 252) * h / 252
    return med, sig_d * np.sqrt(h)


def band_coverage(table: pd.DataFrame, h: int) -> float | None:
    """과거에 80% 가격 범위가 실제로 몇 %를 담았는지 측정한다."""
    d = table.dropna(subset=[f"y_{h}", "vol_63", "vol_long"])
    if d.empty:
        return None
    med, sig = base_params(d["vol_63"], d["vol_long"], h)
    inside = (d[f"y_{h}"] >= med - Z80 * sig) & (d[f"y_{h}"] <= med + Z80 * sig)
    return round(float(inside.mean()), 3)


def train_all(table: pd.DataFrame) -> dict:
    cols = feature_columns(table)
    s = settings()
    bundle: dict = {"cols": cols, "models": {}, "gate": {}, "coverage": {},
                    "trained": str(pd.Timestamp.now().date())}
    for label, h in s["horizons"].items():
        bundle["coverage"][label] = band_coverage(table, h)
        if h > s["ml_max_horizon"]:
            continue
        for target, prefix in TARGETS.items():
            oos = walk_forward(table, h, target, cols)
            res = gate(oos, h)
            bundle["gate"][f"{label}|{target}"] = res
            log.info("%s %s: %s", label, target, res)
            if not res["passed"]:
                continue
            d = table.dropna(subset=[f"{prefix}_{h}", "timing"])
            d = _thin(d, h)
            m = _model().fit(d[cols], (d[f"{prefix}_{h}"] > 0).astype(int))
            logit = np.log(oos["p"].clip(1e-4, 1 - 1e-4) / (1 - oos["p"].clip(1e-4, 1 - 1e-4)))
            cal = LogisticRegression().fit(logit.to_frame("l"), oos["label"])
            bundle["models"][f"{label}|{target}"] = {"model": m, "cal": cal}
    DATA.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_bundle() -> dict | None:
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def _prob(entry: dict, row: pd.DataFrame) -> float:
    p = float(np.clip(entry["model"].predict_proba(row)[0, 1], 1e-4, 1 - 1e-4))
    logit = pd.DataFrame({"l": [np.log(p / (1 - p))]})
    return float(entry["cal"].predict_proba(logit)[0, 1])


def predict(latest: pd.Series, close: float, bundle: dict | None) -> list[dict]:
    """한 종목의 8개 기간 전망."""
    s = settings()
    cur = latest.get("vol_63")
    long_ = latest.get("vol_long")
    if pd.isna(cur):
        return []
    long_ = cur if pd.isna(long_) else long_
    rows = []
    frame = latest.to_frame().T[bundle["cols"]].astype(float) if bundle else None
    for label, h in s["horizons"].items():
        med, sig = base_params(cur, long_, h)
        models = bundle["models"] if bundle else {}
        p_up = _prob(models[f"{label}|up"], frame) if f"{label}|up" in models else None
        p_beat = _prob(models[f"{label}|beat"], frame) if f"{label}|beat" in models else None
        center = sig * norm.ppf(p_up) if p_up is not None else med
        status = "시나리오" if h > s["ml_max_horizon"] else ("검증 통과" if (p_up or p_beat) else "예측력 미입증")
        rows.append({"horizon": label, "days": h, "status": status,
                     "p_up": None if p_up is None else round(p_up, 3),
                     "p_beat": None if p_beat is None else round(p_beat, 3),
                     "p_ref": round(float(norm.cdf(med / sig)), 3),
                     "low": float(close * np.exp(center - Z80 * sig)),
                     "mid": float(close * np.exp(center)),
                     "high": float(close * np.exp(center + Z80 * sig))})
    return rows
