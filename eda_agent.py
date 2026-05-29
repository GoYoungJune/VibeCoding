"""
Kaggle Binary Classification EDA Agent
Usage: python eda_agent.py <csv_path> <target_column>
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
from datetime import datetime

# ── 스타일 설정 ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
PALETTE = ["#4C72B0", "#DD8452"]
sns.set_palette(PALETTE)

REPORT_LINES: list[str] = []
FIGURES_DIR = "eda_figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


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
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def insight(text: str) -> None:
    log(f"  💡 {text}")


# ══════════════════════════════════════════════════════════════
# STEP 1: 기본 정보
# ══════════════════════════════════════════════════════════════
def step1_basic_info(df: pd.DataFrame, target: str) -> None:
    section("STEP 1 · 데이터 기본 정보")

    log(f"  Shape       : {df.shape[0]:,} rows × {df.shape[1]} cols")
    log(f"  Target      : {target}")
    log(f"  Memory      : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    if target in cat_cols:
        cat_cols.remove(target)

    log(f"  수치형 피처 ({len(num_cols)}) : {num_cols}")
    log(f"  범주형 피처 ({len(cat_cols)}) : {cat_cols}\n")

    log("  [describe – 수치형]")
    log(df[num_cols + [target]].describe().round(3).to_string(max_cols=20))

    if cat_cols:
        log("\n  [describe – 범주형]")
        log(df[cat_cols].describe().to_string())

    insight("컬럼 수 대비 행 수가 적으면 과적합 위험 → 정규화/드롭아웃 고려")
    if df.shape[0] < 1000:
        insight("⚠ 샘플 수 1,000 미만 — 교차검증(CV) 폴드 수를 줄이거나 LOOCV 권장")


# ══════════════════════════════════════════════════════════════
# STEP 2: 결측치
# ══════════════════════════════════════════════════════════════
def step2_missing(df: pd.DataFrame) -> dict:
    section("STEP 2 · 결측치 분석")

    miss = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    miss = miss[miss > 0]

    if miss.empty:
        log("  결측치 없음 ✓")
        return {}

    log("  컬럼별 결측 비율 (%):\n")
    for col, pct in miss.items():
        bar = "█" * int(pct / 2)
        log(f"    {col:<20} {pct:5.1f}%  {bar}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(miss) * 0.5 + 2)))

    miss.plot.barh(ax=axes[0], color="#DD8452")
    axes[0].set_title("Missing Value Rate (%)")
    axes[0].set_xlabel("Missing %")
    axes[0].axvline(50, color="red", linestyle="--", alpha=0.6, label="50% threshold")
    axes[0].legend()

    # 결측 패턴 히트맵 (샘플 200행)
    sample = df[miss.index].isnull().head(200)
    sns.heatmap(sample.T, cbar=False, ax=axes[1],
                cmap=["#4C72B0", "#FFCCBC"], linewidths=0.3)
    axes[1].set_title("Missing Pattern (first 200 rows)")
    axes[1].set_xlabel("Row index")

    plt.tight_layout()
    path = save_fig("step2_missing")
    log(f"\n  → 저장: {path}")

    high_miss = miss[miss > 50].index.tolist()
    mid_miss  = miss[(miss > 20) & (miss <= 50)].index.tolist()
    low_miss  = miss[miss <= 20].index.tolist()

    if high_miss:
        insight(f"결측 50% 초과 → 삭제 검토: {high_miss}")
    if mid_miss:
        insight(f"결측 20~50% → 모델 기반 대체(KNN/MICE) 권장: {mid_miss}")
    if low_miss:
        insight(f"결측 20% 이하 → 중앙값/최빈값 대체 가능: {low_miss}")

    return miss.to_dict()


# ══════════════════════════════════════════════════════════════
# STEP 3: 타겟 불균형
# ══════════════════════════════════════════════════════════════
def step3_target_balance(df: pd.DataFrame, target: str) -> None:
    section("STEP 3 · 타겟 클래스 불균형")

    vc = df[target].value_counts()
    ratio = vc.min() / vc.max()
    log(f"  클래스 분포:\n{vc.to_string()}\n")
    log(f"  소수:다수 비율 = {ratio:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    vc.plot.bar(ax=axes[0], color=PALETTE, edgecolor="white")
    axes[0].set_title("Target Class Count")
    axes[0].set_xlabel(target)
    axes[0].tick_params(axis="x", rotation=0)

    vc.plot.pie(ax=axes[1], autopct="%1.1f%%", colors=PALETTE,
                startangle=90, wedgeprops={"edgecolor": "white"})
    axes[1].set_title("Target Class Ratio")
    axes[1].set_ylabel("")

    plt.tight_layout()
    path = save_fig("step3_target")
    log(f"\n  → 저장: {path}")

    if ratio < 0.1:
        insight("⚠ 심각한 불균형(< 10%) → SMOTE, 클래스 가중치, threshold 조정 필수")
    elif ratio < 0.3:
        insight("⚠ 불균형(< 30%) → class_weight='balanced' 또는 ADASYN 권장")
    elif ratio < 0.5:
        insight("경미한 불균형 → 평가지표로 F1/AUC-ROC 사용 권장 (Accuracy 지양)")
    else:
        insight("균형 잡힌 타겟 ✓")


# ══════════════════════════════════════════════════════════════
# STEP 4: 피처 분포
# ══════════════════════════════════════════════════════════════
def step4_distributions(df: pd.DataFrame, target: str) -> None:
    section("STEP 4 · 피처 분포 시각화")

    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # ── 수치형 ──
    if num_cols:
        n = len(num_cols)
        cols_per_row = 3
        rows = (n + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row,
                                 figsize=(cols_per_row * 4, rows * 3.5))
        axes = np.array(axes).flatten()

        for i, col in enumerate(num_cols):
            ax = axes[i]
            for cls, grp in df.groupby(target)[col]:
                grp.dropna().plot.kde(ax=ax, label=f"{target}={cls}", alpha=0.75)
            ax.set_title(col)
            ax.set_xlabel("")
            ax.legend(fontsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Numerical Features — KDE by Target", y=1.01, fontsize=14)
        plt.tight_layout()
        path = save_fig("step4_num_dist")
        log(f"  수치형 KDE → 저장: {path}")

    # ── 범주형 ──
    if cat_cols:
        n = len(cat_cols)
        cols_per_row = 3
        rows = (n + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row,
                                 figsize=(cols_per_row * 4, rows * 3.5))
        axes = np.array(axes).flatten()

        for i, col in enumerate(cat_cols):
            ax = axes[i]
            ct = pd.crosstab(df[col], df[target], normalize="index") * 100
            ct.plot.bar(ax=ax, color=PALETTE, edgecolor="white", width=0.7)
            ax.set_title(col)
            ax.set_xlabel("")
            ax.set_ylabel("Positive rate (%)")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(title=target, fontsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Categorical Features — Positive Rate by Category", y=1.01, fontsize=14)
        plt.tight_layout()
        path = save_fig("step4_cat_dist")
        log(f"  범주형 분포 → 저장: {path}")

    insight("타겟별 KDE가 크게 분리된 피처 → 단독으로도 강력한 예측 신호")
    insight("범주형 양성률이 균일한 피처 → 예측력 낮음, 삭제 검토")


# ══════════════════════════════════════════════════════════════
# STEP 5: 상관관계
# ══════════════════════════════════════════════════════════════
def step5_correlation(df: pd.DataFrame, target: str) -> None:
    section("STEP 5 · 상관관계 분석")

    num_df = df.select_dtypes(include=np.number)

    # 수치형 컬럼만 있을 때
    if num_df.shape[1] < 2:
        log("  수치형 피처 부족 — 상관관계 분석 생략")
        return

    corr = num_df.corr()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 전체 히트맵
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", ax=axes[0],
                cmap="coolwarm", center=0, linewidths=0.5,
                annot_kws={"size": 8})
    axes[0].set_title("Correlation Heatmap (lower triangle)")

    # 타겟 상관계수 막대
    if target in corr.columns:
        target_corr = corr[target].drop(target).sort_values()
        colors = ["#DD8452" if v > 0 else "#4C72B0" for v in target_corr]
        target_corr.plot.barh(ax=axes[1], color=colors, edgecolor="white")
        axes[1].set_title(f"Feature Correlation with '{target}'")
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_xlabel("Pearson r")

    plt.tight_layout()
    path = save_fig("step5_correlation")
    log(f"  → 저장: {path}")

    if target in corr.columns:
        tc = corr[target].drop(target).abs().sort_values(ascending=False)
        top = tc.head(3)
        log(f"\n  타겟 상관 Top-3:\n{top.round(3).to_string()}")
        insight(f"상관도 높은 피처 → 우선 활용: {top.index.tolist()}")

    # 다중공선성 탐지
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            c1, c2 = corr.columns[i], corr.columns[j]
            if c1 == target or c2 == target:
                continue
            if abs(corr.loc[c1, c2]) > 0.85:
                high_corr_pairs.append((c1, c2, corr.loc[c1, c2]))

    if high_corr_pairs:
        log("\n  다중공선성 의심 쌍 (|r| > 0.85):")
        for c1, c2, r in high_corr_pairs:
            log(f"    {c1} ↔ {c2} : r={r:.3f}")
        insight("다중공선성 쌍 발견 → 하나 삭제 또는 PCA 변환 검토")


# ══════════════════════════════════════════════════════════════
# STEP 6: 이상치 탐지
# ══════════════════════════════════════════════════════════════
def step6_outliers(df: pd.DataFrame, target: str) -> dict:
    section("STEP 6 · 이상치 탐지 (IQR 기준)")

    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]
    outlier_summary = {}

    if not num_cols:
        log("  수치형 피처 없음 — 이상치 분석 생략")
        return {}

    n = len(num_cols)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row,
                             figsize=(cols_per_row * 4, rows * 3))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = n_out / len(df) * 100
        outlier_summary[col] = {"count": int(n_out), "pct": round(pct, 2),
                                 "lower": round(lower, 3), "upper": round(upper, 3)}

        df.boxplot(column=col, by=target, ax=ax,
                   boxprops=dict(color="#4C72B0"),
                   medianprops=dict(color="#DD8452", linewidth=2),
                   whiskerprops=dict(color="#4C72B0"),
                   capprops=dict(color="#4C72B0"),
                   flierprops=dict(marker="o", markerfacecolor="#DD8452",
                                   markersize=3, alpha=0.4))
        ax.set_title(f"{col}\n(outliers: {n_out}, {pct:.1f}%)", fontsize=10)
        ax.set_xlabel(target)
        fig.suptitle("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Boxplot by Target (IQR Outlier Detection)", y=1.01, fontsize=13)
    plt.tight_layout()
    path = save_fig("step6_outliers")
    log(f"  → 저장: {path}\n")

    log("  이상치 요약:")
    for col, info in outlier_summary.items():
        flag = "⚠" if info["pct"] > 5 else " "
        log(f"    {flag} {col:<20} {info['count']:>5}건 ({info['pct']:5.1f}%)"
            f"  [허용범위: {info['lower']} ~ {info['upper']}]")

    severe = [c for c, v in outlier_summary.items() if v["pct"] > 10]
    mild   = [c for c, v in outlier_summary.items() if 2 < v["pct"] <= 10]

    if severe:
        insight(f"이상치 10% 초과 → 로그변환 또는 Winsorizing 강력 권장: {severe}")
    if mild:
        insight(f"이상치 2~10% → 클리핑(clip) 또는 RobustScaler 고려: {mild}")

    return outlier_summary


# ══════════════════════════════════════════════════════════════
# STEP 7: 리포트 저장
# ══════════════════════════════════════════════════════════════
def step7_save_report(df: pd.DataFrame, target: str, missing: dict,
                      outliers: dict) -> None:
    section("STEP 7 · 전처리 주의사항 요약 및 리포트 저장")

    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    vc = df[target].value_counts()
    ratio = vc.min() / vc.max()

    warnings_list = []

    if missing:
        high = [c for c, v in missing.items() if v > 50]
        mid  = [c for c, v in missing.items() if 20 < v <= 50]
        low  = [c for c, v in missing.items() if v <= 20]
        if high:
            warnings_list.append(f"**결측치 50%+** 컬럼 삭제 검토: `{high}`")
        if mid:
            warnings_list.append(f"**결측치 20~50%** MICE/KNN 대체: `{mid}`")
        if low:
            warnings_list.append(f"**결측치 ≤20%** 중앙값/최빈값 대체: `{low}`")

    if ratio < 0.3:
        warnings_list.append(f"**클래스 불균형** (비율 {ratio:.2f}) → SMOTE / class_weight")

    severe_out = [c for c, v in outliers.items() if v["pct"] > 10]
    mild_out   = [c for c, v in outliers.items() if 2 < v["pct"] <= 10]
    if severe_out:
        warnings_list.append(f"**이상치 10%+** 로그변환/Winsorizing: `{severe_out}`")
    if mild_out:
        warnings_list.append(f"**이상치 2~10%** RobustScaler/clip: `{mild_out}`")

    if num_cols:
        skewed = []
        for col in num_cols:
            sk = df[col].dropna().skew()
            if abs(sk) > 1:
                skewed.append(f"{col}(skew={sk:.2f})")
        if skewed:
            warnings_list.append(f"**고왜도 피처** 로그/박스콕스 변환: `{skewed}`")

    if cat_cols:
        high_card = [c for c in cat_cols if df[c].nunique() > 20]
        if high_card:
            warnings_list.append(f"**고카디널리티** 범주형 → Target Encoding / 임베딩: `{high_card}`")

    log("\n  ⚙ 전처리 체크리스트:")
    for w in warnings_list:
        log(f"    - {w}")

    # ── 마크다운 작성 ──────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [
        f"# EDA Report — `{target}` 이진 분류",
        f"*생성일시: {ts}*\n",
        "---",
        "## 1. 데이터 개요",
        f"- Shape: **{df.shape[0]:,} rows × {df.shape[1]} cols**",
        f"- 수치형: {num_cols}",
        f"- 범주형: {cat_cols}",
        "",
        "## 2. 결측치",
    ]
    if missing:
        md += [f"| 컬럼 | 결측률(%) |", "|---|---|"]
        for col, pct in sorted(missing.items(), key=lambda x: -x[1]):
            md.append(f"| {col} | {pct:.1f} |")
    else:
        md.append("- 결측치 없음 ✓")

    md += [
        "",
        "## 3. 타겟 불균형",
        f"- 소수:다수 비율: **{ratio:.3f}**",
        f"- 분포: {vc.to_dict()}",
        "",
        "## 4–6. 시각화",
        f"![결측치](eda_figures/step2_missing.png)",
        f"![타겟분포](eda_figures/step3_target.png)",
        f"![수치분포](eda_figures/step4_num_dist.png)",
        f"![범주분포](eda_figures/step4_cat_dist.png)",
        f"![상관관계](eda_figures/step5_correlation.png)",
        f"![이상치](eda_figures/step6_outliers.png)",
        "",
        "## 7. 전처리 주의사항",
    ]
    for w in warnings_list:
        md.append(f"- {w}")

    md += [
        "",
        "---",
        "*Generated by Kaggle EDA Agent*",
    ]

    with open("eda_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    log("\n  ✅ eda_report.md 저장 완료")
    log(f"  ✅ 시각화 이미지 {len(os.listdir(FIGURES_DIR))}개 → {FIGURES_DIR}/")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def run(csv_path: str, target: str) -> None:
    log("=" * 60)
    log("  Kaggle Binary Classification EDA Agent")
    log("=" * 60)

    df = pd.read_csv(csv_path)
    log(f"\n  파일 로드: {csv_path}  ({df.shape[0]:,} × {df.shape[1]})")

    if target not in df.columns:
        sys.exit(f"[ERROR] '{target}' 컬럼이 데이터에 없습니다.\n"
                 f"  사용 가능한 컬럼: {list(df.columns)}")

    step1_basic_info(df, target)
    missing  = step2_missing(df)
    step3_target_balance(df, target)
    step4_distributions(df, target)
    step5_correlation(df, target)
    outliers = step6_outliers(df, target)
    step7_save_report(df, target, missing, outliers)

    log("\n" + "=" * 60)
    log("  EDA 완료!")
    log("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eda_agent.py <csv_path> <target_column>")
        print("Example: python eda_agent.py titanic.csv Survived")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
