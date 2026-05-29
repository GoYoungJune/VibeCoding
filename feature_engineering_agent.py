"""
Kaggle Binary Classification Feature Engineering Agent
Usage: python feature_engineering_agent.py <train_processed_csv> <target_col>
Example: python feature_engineering_agent.py train_processed.csv Survived
"""

import sys
import os
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from itertools import combinations
from statsmodels.stats.outliers_influence import variance_inflation_factor

REPORT_LINES: list[str] = []
FIG_DIR = "fe_figures"
os.makedirs(FIG_DIR, exist_ok=True)

HIGH_CORR_THRESHOLD = 0.95
VIF_THRESHOLD       = 10.0
LGBM_LOW_IMP_PCT    = 0.10   # 하위 10% 제거
CV_FOLDS            = 5
RANDOM_STATE        = 42


# ── 유틸 ─────────────────────────────────────────────────────
def log(text: str = "") -> None:
    print(text)
    REPORT_LINES.append(text)


def section(title: str) -> None:
    bar = "─" * 60
    log(f"\n{bar}")
    log(f"  {title}")
    log(bar)


def save_fig(name: str) -> str:
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    return path


def cv_auc(X: pd.DataFrame, y: pd.Series, label: str) -> float:
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05,
        num_leaves=31, random_state=RANDOM_STATE,
        verbose=-1, n_jobs=-1,
    )
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    mean, std = scores.mean(), scores.std()
    log(f"  {label:<40} AUC = {mean:.4f} ± {std:.4f}")
    return mean


# ══════════════════════════════════════════════════════════════
# STEP 1: 로드
# ══════════════════════════════════════════════════════════════
def step1_load(csv_path: str, target: str) -> tuple[pd.DataFrame, pd.Series]:
    section("STEP 1 · 데이터 로드")

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        sys.exit(f"[ERROR] '{target}' 컬럼 없음. 사용 가능: {list(df.columns)}")

    y = df[target].copy()
    X = df.drop(columns=[target])

    # 수치형만 유지
    X = X.select_dtypes(include=np.number)
    log(f"  입력 shape : {X.shape[0]:,} rows × {X.shape[1]} cols")
    log(f"  수치형 피처: {list(X.columns)}")
    return X, y


# ══════════════════════════════════════════════════════════════
# STEP 2: 수치형 조합 피처
# ══════════════════════════════════════════════════════════════
def step2_combination_features(X: pd.DataFrame) -> pd.DataFrame:
    section("STEP 2 · 수치형 조합 피처 생성 (비율·곱·차이)")

    cols = X.columns.tolist()
    new_feats: dict[str, pd.Series] = {}

    for c1, c2 in combinations(cols, 2):
        s1, s2 = X[c1], X[c2]

        # 차이
        new_feats[f"{c1}_minus_{c2}"] = s1 - s2

        # 합
        new_feats[f"{c1}_plus_{c2}"] = s1 + s2

        # 곱
        new_feats[f"{c1}_mul_{c2}"] = s1 * s2

        # 비율 (분모 0 방지)
        denom = s2.replace(0, np.nan)
        new_feats[f"{c1}_div_{c2}"] = s1 / denom

    new_df = pd.DataFrame(new_feats, index=X.index)

    # inf / -inf → NaN → 0
    new_df = new_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    log(f"  생성된 조합 피처 수: {new_df.shape[1]}")
    return pd.concat([X, new_df], axis=1)


# ══════════════════════════════════════════════════════════════
# STEP 3: 통계 기반 피처
# ══════════════════════════════════════════════════════════════
def step3_stat_features(X: pd.DataFrame) -> pd.DataFrame:
    section("STEP 3 · 행 단위 통계 피처")

    base_cols = [c for c in X.columns if "_minus_" not in c
                 and "_plus_" not in c
                 and "_mul_" not in c
                 and "_div_" not in c]
    base = X[base_cols]

    X = X.copy()
    X["row_mean"]   = base.mean(axis=1)
    X["row_std"]    = base.std(axis=1).fillna(0)
    X["row_max"]    = base.max(axis=1)
    X["row_min"]    = base.min(axis=1)
    X["row_range"]  = X["row_max"] - X["row_min"]
    X["row_skew"]   = base.skew(axis=1).fillna(0)

    log(f"  추가된 통계 피처: row_mean, row_std, row_max, row_min, row_range, row_skew")
    log(f"  현재 총 피처 수: {X.shape[1]}")
    return X


# ══════════════════════════════════════════════════════════════
# STEP 4: 고상관 피처 제거
# ══════════════════════════════════════════════════════════════
def step4_remove_high_corr(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    section(f"STEP 4 · 고상관 피처 제거 (|r| ≥ {HIGH_CORR_THRESHOLD})")

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    drop_cols = [col for col in upper.columns if any(upper[col] >= HIGH_CORR_THRESHOLD)]
    X_out = X.drop(columns=drop_cols, errors="ignore")

    log(f"  제거된 고상관 피처 수: {len(drop_cols)}")
    if drop_cols:
        log(f"  제거 목록 (앞 10개): {drop_cols[:10]}")
    log(f"  잔여 피처 수: {X_out.shape[1]}")
    return X_out, drop_cols


# ══════════════════════════════════════════════════════════════
# STEP 5: LightGBM feature importance → 하위 제거
# ══════════════════════════════════════════════════════════════
def step5_lgbm_importance(
        X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    section(f"STEP 5 · LightGBM Feature Importance (하위 {LGBM_LOW_IMP_PCT:.0%} 제거)")

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        num_leaves=31, random_state=RANDOM_STATE,
        verbose=-1, n_jobs=-1,
    )
    model.fit(X, y)

    imp_df = pd.DataFrame({
        "feature":    X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # 하위 LGBM_LOW_IMP_PCT 컷오프
    cutoff_idx = int(len(imp_df) * (1 - LGBM_LOW_IMP_PCT))
    keep_feats  = imp_df.iloc[:cutoff_idx]["feature"].tolist()
    drop_feats  = imp_df.iloc[cutoff_idx:]["feature"].tolist()

    log(f"  전체 피처: {len(imp_df)} → 유지: {len(keep_feats)} / 제거: {len(drop_feats)}")
    log(f"  중요도 컷오프 값: {imp_df.iloc[cutoff_idx-1]['importance']:.1f}")
    log(f"\n  Top-10 피처:")
    for _, row in imp_df.head(10).iterrows():
        bar = "█" * min(int(row["importance"] / max(imp_df["importance"]) * 30), 30)
        log(f"    {row['feature']:<35} {row['importance']:>6.0f}  {bar}")

    # 시각화
    top_n = min(30, len(imp_df))
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    colors = ["#DD8452" if f in drop_feats else "#4C72B0" for f in imp_df.head(top_n)["feature"]]
    ax.barh(imp_df.head(top_n)["feature"][::-1],
            imp_df.head(top_n)["importance"][::-1],
            color=colors[::-1], edgecolor="white")
    ax.set_title(f"LightGBM Feature Importance (Top {top_n})\n"
                 f"Blue=kept  Orange=removed (bottom {LGBM_LOW_IMP_PCT:.0%})", fontsize=12)
    ax.set_xlabel("Importance (gain)")
    plt.tight_layout()
    path = save_fig("step5_lgbm_importance")
    log(f"\n  → 저장: {path}")

    X_out = X[keep_feats].copy()
    return X_out, imp_df, drop_feats


# ══════════════════════════════════════════════════════════════
# STEP 6: VIF 다중공선성 탐지
# ══════════════════════════════════════════════════════════════
def step6_vif(X: pd.DataFrame) -> pd.DataFrame:
    section(f"STEP 6 · VIF 다중공선성 탐지 (임계값 {VIF_THRESHOLD})")

    # VIF 계산은 컬럼 수가 많으면 느리므로 샘플링
    sample = X.sample(min(500, len(X)), random_state=RANDOM_STATE).fillna(0)

    vif_records = []
    for i, col in enumerate(sample.columns):
        try:
            v = variance_inflation_factor(sample.values, i)
        except Exception:
            v = np.nan
        vif_records.append({"feature": col, "VIF": round(v, 2)})

    vif_df = pd.DataFrame(vif_records).sort_values("VIF", ascending=False).reset_index(drop=True)

    high_vif = vif_df[vif_df["VIF"] > VIF_THRESHOLD]
    log(f"  VIF > {VIF_THRESHOLD} 피처 수: {len(high_vif)}")

    if not high_vif.empty:
        log(f"\n  {'피처':<35} {'VIF':>8}")
        log(f"  {'─'*35} {'─'*8}")
        for _, row in high_vif.head(15).iterrows():
            flag = "⚠" if row["VIF"] > 20 else " "
            log(f"  {flag} {row['feature']:<33} {row['VIF']:>8.2f}")
        log(f"\n  ℹ  VIF > {VIF_THRESHOLD} 피처는 참고용입니다.")
        log(f"     트리 계열 모델은 다중공선성에 강해 제거 없이 진행합니다.")
    else:
        log(f"  ✅ 심각한 다중공선성 없음 (모든 VIF ≤ {VIF_THRESHOLD})")

    # VIF 시각화 (상위 20개)
    plot_df = vif_df.head(20).copy()
    plot_df["VIF"] = plot_df["VIF"].clip(upper=50)  # inf 방지
    fig, ax = plt.subplots(figsize=(9, max(5, len(plot_df) * 0.35)))
    colors = ["#DD8452" if v > VIF_THRESHOLD else "#4C72B0" for v in plot_df["VIF"]]
    ax.barh(plot_df["feature"][::-1], plot_df["VIF"][::-1],
            color=colors[::-1], edgecolor="white")
    ax.axvline(VIF_THRESHOLD, color="red", linestyle="--", alpha=0.7, label=f"VIF={VIF_THRESHOLD}")
    ax.set_title("VIF — Top 20 Features", fontsize=12)
    ax.set_xlabel("VIF (capped at 50)")
    ax.legend()
    plt.tight_layout()
    path = save_fig("step6_vif")
    log(f"  → 저장: {path}")

    return vif_df


# ══════════════════════════════════════════════════════════════
# STEP 7: CV 스코어 비교
# ══════════════════════════════════════════════════════════════
def step7_cv_comparison(
        X_orig: pd.DataFrame, X_fe: pd.DataFrame, y: pd.Series
) -> tuple[float, float]:
    section(f"STEP 7 · CV AUC 비교 ({CV_FOLDS}-Fold StratifiedKFold)")

    log(f"  {'모델':<40} {'AUC':>12}")
    log(f"  {'─'*40} {'─'*12}")
    auc_before = cv_auc(X_orig, y, "원본 피처")
    auc_after  = cv_auc(X_fe,   y, "피처 엔지니어링 후")

    diff = auc_after - auc_before
    sign = "▲" if diff > 0 else "▼"
    log(f"\n  변화: {sign} {abs(diff):.4f}  "
        f"({'개선' if diff > 0 else '하락'} {abs(diff)*100:.2f}%p)")

    # 시각화
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["원본 피처", "FE 후"], [auc_before, auc_after],
                  color=["#4C72B0", "#DD8452" if diff > 0 else "#999"],
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, [auc_before, auc_after]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(min(auc_before, auc_after) - 0.05, max(auc_before, auc_after) + 0.05)
    ax.set_ylabel("CV AUC (mean)")
    ax.set_title(f"CV Score Comparison ({CV_FOLDS}-Fold)\nΔ = {diff:+.4f}")
    plt.tight_layout()
    path = save_fig("step7_cv_comparison")
    log(f"  → 저장: {path}")

    return auc_before, auc_after


# ══════════════════════════════════════════════════════════════
# STEP 8: 저장
# ══════════════════════════════════════════════════════════════
def step8_save(
        X_orig: pd.DataFrame,
        X_fe: pd.DataFrame,
        y: pd.Series,
        target: str,
        drop_corr: list[str],
        drop_imp: list[str],
        auc_before: float,
        auc_after: float,
        vif_df: pd.DataFrame,
) -> None:
    section("STEP 8 · 결과 저장")

    # 최종 데이터 저장
    out_df = X_fe.copy()
    out_df[target] = y.values
    out_df.to_csv("train_fe.csv", index=False)
    log(f"  ✅ train_fe.csv 저장 — {out_df.shape}")

    # selected_features.json
    result = {
        "selected_features": X_fe.columns.tolist(),
        "n_selected": len(X_fe.columns),
        "n_original": len(X_orig.columns),
        "n_generated": len(X_fe.columns) - len(X_orig.columns) + len(drop_corr) + len(drop_imp),
        "dropped_high_corr": drop_corr,
        "dropped_low_importance": drop_imp,
        "cv_auc_before": round(auc_before, 4),
        "cv_auc_after":  round(auc_after,  4),
        "cv_auc_delta":  round(auc_after - auc_before, 4),
        "top10_vif": vif_df.head(10)[["feature", "VIF"]].to_dict(orient="records"),
    }
    with open("selected_features.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"  ✅ selected_features.json 저장 완료")

    # 리포트 요약
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    md_lines = [
        f"# Feature Engineering Report",
        f"*생성일시: {ts}*\n",
        "---",
        "## 요약",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| 원본 피처 수 | {len(X_orig.columns)} |",
        f"| 최종 피처 수 | {len(X_fe.columns)} |",
        f"| 고상관 제거 | {len(drop_corr)} |",
        f"| 낮은 중요도 제거 | {len(drop_imp)} |",
        f"| CV AUC (원본) | {auc_before:.4f} |",
        f"| CV AUC (FE 후) | {auc_after:.4f} |",
        f"| AUC 변화 | {auc_after - auc_before:+.4f} |",
        "",
        "## 선택된 피처",
        "```",
        "\n".join(X_fe.columns.tolist()),
        "```",
        "",
        "## 시각화",
        f"![importance](fe_figures/step5_lgbm_importance.png)",
        f"![vif](fe_figures/step6_vif.png)",
        f"![cv](fe_figures/step7_cv_comparison.png)",
        "",
        "---",
        "*Generated by Kaggle Feature Engineering Agent*",
    ]
    with open("fe_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    log(f"  ✅ fe_report.md 저장 완료")

    # 콘솔 요약
    log(f"\n  ┌─ 피처 엔지니어링 요약 ──────────────────────────┐")
    log(f"  │ 원본 피처       : {len(X_orig.columns):>4}개                          │")
    log(f"  │ 조합+통계 생성  : {result['n_generated']:>4}개 생성됨                   │")
    log(f"  │ 고상관 제거     : {len(drop_corr):>4}개 제거됨                   │")
    log(f"  │ 낮은 중요도 제거: {len(drop_imp):>4}개 제거됨                   │")
    log(f"  │ 최종 피처       : {len(X_fe.columns):>4}개                          │")
    log(f"  │ CV AUC 변화     : {auc_before:.4f} → {auc_after:.4f}  ({auc_after-auc_before:+.4f}) │")
    log(f"  └─────────────────────────────────────────────────┘")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def run(csv_path: str, target: str) -> None:
    log("=" * 60)
    log("  Kaggle Binary Classification Feature Engineering Agent")
    log("=" * 60)

    X_orig, y = step1_load(csv_path, target)

    # CV 기준점 (원본)
    log("\n  [원본 피처 CV 기준 측정 중...]")
    auc_before_placeholder = None  # step7에서 측정

    X = step2_combination_features(X_orig)
    X = step3_stat_features(X)
    X, drop_corr = step4_remove_high_corr(X)
    X, imp_df, drop_imp = step5_lgbm_importance(X, y)
    vif_df = step6_vif(X)
    auc_before, auc_after = step7_cv_comparison(X_orig, X, y)
    step8_save(X_orig, X, y, target, drop_corr, drop_imp, auc_before, auc_after, vif_df)

    log("\n" + "=" * 60)
    log("  피처 엔지니어링 완료!")
    log("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python feature_engineering_agent.py <train_csv> <target_col>")
        print("Example: python feature_engineering_agent.py train_processed.csv Survived")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
