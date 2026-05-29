"""
Kaggle Binary Classification Model Experiment Agent
Usage: python model_experiment_agent.py <train_csv> <target_col>
Example: python model_experiment_agent.py train_processed.csv Survived
         python model_experiment_agent.py train_fe.csv Survived
"""

import sys
import os
import csv
import time
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ── 상수 ─────────────────────────────────────────────────────
RANDOM_STATE   = 42
CV_FOLDS       = 5
LOG_FILE       = "experiment_log.csv"
FIG_DIR        = "experiment_figures"
TOP_N          = 3          # 튜닝 에이전트에 전달할 상위 모델 수
os.makedirs(FIG_DIR, exist_ok=True)

LOG_FIELDNAMES = [
    "timestamp", "model", "cv_auc_mean", "cv_auc_std",
    "cv_f1_mean", "cv_f1_std",
    "cv_precision_mean", "cv_recall_mean",
    "train_time_sec", "n_features", "data_path",
]


# ── 유틸 ─────────────────────────────────────────────────────
def log(text: str = "") -> None:
    print(text)


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


def append_log(row: dict) -> None:
    """실험 결과를 CSV에 즉시 기록 — 중단돼도 보존"""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ══════════════════════════════════════════════════════════════
# 모델 정의
# ══════════════════════════════════════════════════════════════
def get_models() -> dict:
    return {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ══════════════════════════════════════════════════════════════
# STEP 1: 데이터 로드
# ══════════════════════════════════════════════════════════════
def step1_load(csv_path: str, target: str) -> tuple[pd.DataFrame, pd.Series]:
    section("STEP 1 · 데이터 로드")

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        sys.exit(f"[ERROR] '{target}' 컬럼 없음. 사용 가능: {list(df.columns)}")

    y = df[target].copy()
    X = df.drop(columns=[target]).select_dtypes(include=np.number).fillna(0)

    log(f"  데이터  : {csv_path}")
    log(f"  Shape   : {X.shape[0]:,} rows × {X.shape[1]} features")
    log(f"  Target  : {target}  →  {dict(y.value_counts().sort_index())}")
    return X, y


# ══════════════════════════════════════════════════════════════
# STEP 2: 단일 모델 CV 실험
# ══════════════════════════════════════════════════════════════
def run_cv(
        name: str,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        csv_path: str,
) -> dict:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    auc_scores, f1_scores, prec_scores, rec_scores = [], [], [], []
    oof_pred  = np.zeros(len(y))
    oof_prob  = np.zeros(len(y))

    t0 = time.perf_counter()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        import copy
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)

        prob = m.predict_proba(X_val)[:, 1]
        pred = (prob >= 0.5).astype(int)

        oof_prob[val_idx] = prob
        oof_pred[val_idx] = pred

        auc_scores.append(roc_auc_score(y_val, prob))
        f1_scores.append(f1_score(y_val, pred, zero_division=0))
        prec_scores.append(precision_score(y_val, pred, zero_division=0))
        rec_scores.append(recall_score(y_val, pred, zero_division=0))

    elapsed = time.perf_counter() - t0

    result = {
        "model":             name,
        "cv_auc_mean":       float(np.mean(auc_scores)),
        "cv_auc_std":        float(np.std(auc_scores)),
        "cv_f1_mean":        float(np.mean(f1_scores)),
        "cv_f1_std":         float(np.std(f1_scores)),
        "cv_precision_mean": float(np.mean(prec_scores)),
        "cv_recall_mean":    float(np.mean(rec_scores)),
        "train_time_sec":    round(elapsed, 2),
        "n_features":        X.shape[1],
        "data_path":         csv_path,
        "oof_prob":          oof_prob,
        "oof_pred":          oof_pred,
    }
    return result


# ══════════════════════════════════════════════════════════════
# STEP 3: 혼동행렬 저장
# ══════════════════════════════════════════════════════════════
def save_confusion_matrix(name: str, y_true: pd.Series,
                          y_pred: np.ndarray) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(f"{name}\nConfusion Matrix (OOF)", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"], rotation=0)

    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    ax.set_xlabel(
        f"Predicted\n\nTN={tn}  FP={fp}  FN={fn}  TP={tp}  "
        f"(Acc={acc:.3f})",
        fontsize=9,
    )
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    path = save_fig(f"cm_{fname}")
    return path


# ══════════════════════════════════════════════════════════════
# STEP 4: 전체 실험 루프
# ══════════════════════════════════════════════════════════════
def step4_run_all(X: pd.DataFrame, y: pd.Series,
                  csv_path: str) -> list[dict]:
    section("STEP 4 · 모델 실험 (5-Fold CV)")

    models   = get_models()
    results  = []
    header   = f"  {'모델':<20} {'AUC':>8} {'±':>6} {'F1':>7} {'Prec':>7} {'Rec':>7} {'시간(s)':>8}"
    sep      = "  " + "─" * 68
    log(header)
    log(sep)

    for name, model in models.items():
        log(f"  {name:<20} 실험 중...", )
        try:
            res = run_cv(name, model, X, y, csv_path)
            res["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 즉시 로그 저장 (중단 대비)
            log_row = {k: res[k] for k in LOG_FIELDNAMES}
            append_log(log_row)

            # 혼동행렬 저장
            cm_path = save_confusion_matrix(name, y, res["oof_pred"])

            results.append(res)

            # 결과 출력 (이전 줄 덮어쓰기 효과를 위해 새 줄로 출력)
            log(f"\r  {name:<20} "
                f"{res['cv_auc_mean']:>8.4f} "
                f"{res['cv_auc_std']:>6.4f} "
                f"{res['cv_f1_mean']:>7.4f} "
                f"{res['cv_precision_mean']:>7.4f} "
                f"{res['cv_recall_mean']:>7.4f} "
                f"{res['train_time_sec']:>7.1f}s"
                f"  → CM: {os.path.basename(cm_path)}")

        except Exception as e:
            log(f"\r  {name:<20} ❌ 오류: {e}")

    log(sep)
    return results


# ══════════════════════════════════════════════════════════════
# STEP 5: 결과 시각화 및 TOP-N 선정
# ══════════════════════════════════════════════════════════════
def step5_visualize_and_rank(results: list[dict]) -> list[dict]:
    section("STEP 5 · 결과 시각화 및 Top-3 모델 선정")

    df = pd.DataFrame([{
        "model":        r["model"],
        "auc":          r["cv_auc_mean"],
        "auc_std":      r["cv_auc_std"],
        "f1":           r["cv_f1_mean"],
        "precision":    r["cv_precision_mean"],
        "recall":       r["cv_recall_mean"],
        "time":         r["train_time_sec"],
    } for r in results]).sort_values("auc", ascending=False).reset_index(drop=True)

    log(f"\n  ━━ 모델 성능 순위 ━━")
    log(f"  {'순위':<4} {'모델':<22} {'AUC':>8} {'±':>6} {'F1':>7} {'시간':>7}")
    log(f"  " + "─" * 58)
    for i, row in df.iterrows():
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"  {i+1}."
        log(f"  {medal}  {row['model']:<20} {row['auc']:>8.4f} {row['auc_std']:>6.4f} "
            f"{row['f1']:>7.4f} {row['time']:>6.1f}s")

    # ── 종합 성능 바 차트 ──────────────────────────────────────
    metrics = ["auc", "f1", "precision", "recall"]
    n_metrics = len(metrics)
    x = np.arange(len(df))
    width = 0.18
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, df[metric], width,
                      label=metric.upper(), color=color, alpha=0.85, edgecolor="white")
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — AUC / F1 / Precision / Recall", fontsize=13)
    ax.legend(loc="lower right")
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    plt.tight_layout()
    save_fig("step5_model_comparison")

    # ── AUC ± std 에러바 ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = ["#4C72B0"] * len(df)
    palette[0] = "#DD8452"  # 1위 강조
    ax.barh(df["model"][::-1], df["auc"][::-1],
            xerr=df["auc_std"][::-1],
            color=palette[::-1], edgecolor="white",
            error_kw=dict(ecolor="black", capsize=4, capthick=1.5))
    ax.set_title("CV AUC ± Std (5-Fold)", fontsize=12)
    ax.set_xlabel("AUC-ROC")
    ax.axvline(0.8, color="red", linestyle="--", alpha=0.5)
    for i, (auc, std) in enumerate(zip(df["auc"][::-1], df["auc_std"][::-1])):
        ax.text(auc + std + 0.003, i, f"{auc:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    save_fig("step5_auc_errorbar")

    log(f"\n  → 저장: experiment_figures/step5_model_comparison.png")
    log(f"  → 저장: experiment_figures/step5_auc_errorbar.png")

    # TOP-N 선정
    top_models = df.head(TOP_N)["model"].tolist()
    top_results = [r for r in results if r["model"] in top_models]
    top_results.sort(key=lambda x: x["cv_auc_mean"], reverse=True)

    log(f"\n  🏆 튜닝 대상 Top-{TOP_N} 모델:")
    for i, r in enumerate(top_results, 1):
        log(f"     {i}. {r['model']:<22} AUC={r['cv_auc_mean']:.4f}")

    return top_results


# ══════════════════════════════════════════════════════════════
# STEP 6: 혼동행렬 합본 & 저장
# ══════════════════════════════════════════════════════════════
def step6_combined_cm(results: list[dict], y: pd.Series) -> None:
    section("STEP 6 · 혼동행렬 합본 저장")

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = confusion_matrix(y, res["oof_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 13, "weight": "bold"},
        )
        tn, fp, fn, tp = cm.ravel()
        ax.set_title(
            f"{res['model']}\n"
            f"AUC={res['cv_auc_mean']:.4f}  F1={res['cv_f1_mean']:.4f}",
            fontsize=10,
        )
        ax.set_xlabel(f"Predicted\nTN={tn} FP={fp} FN={fn} TP={tp}")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices — OOF Predictions (All Models)", fontsize=13)
    plt.tight_layout()
    path = save_fig("step6_confusion_matrices_all")
    log(f"  → 저장: {path}")


# ══════════════════════════════════════════════════════════════
# STEP 7: Top-N 요약 저장
# ══════════════════════════════════════════════════════════════
def step7_save_summary(
        top_results: list[dict],
        all_results: list[dict],
        csv_path: str,
) -> None:
    section("STEP 7 · 실험 요약 저장")

    # top_models.json — 튜닝 에이전트 입력용
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": csv_path,
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "top_models": [
            {
                "rank": i + 1,
                "model": r["model"],
                "cv_auc_mean": round(r["cv_auc_mean"], 4),
                "cv_auc_std":  round(r["cv_auc_std"], 4),
                "cv_f1_mean":  round(r["cv_f1_mean"], 4),
                "cv_precision_mean": round(r["cv_precision_mean"], 4),
                "cv_recall_mean":    round(r["cv_recall_mean"], 4),
                "train_time_sec":    r["train_time_sec"],
            }
            for i, r in enumerate(top_results)
        ],
        "all_models": [
            {
                "model": r["model"],
                "cv_auc_mean": round(r["cv_auc_mean"], 4),
                "cv_f1_mean":  round(r["cv_f1_mean"], 4),
            }
            for r in sorted(all_results, key=lambda x: x["cv_auc_mean"], reverse=True)
        ],
    }
    with open("top_models.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"  ✅ top_models.json 저장 — 튜닝 에이전트 입력용")
    log(f"  ✅ experiment_log.csv 누적 기록 완료")

    # 콘솔 최종 요약
    log(f"\n  ┌─ 실험 요약 ──────────────────────────────────────┐")
    log(f"  │ 실험 모델 수   : {len(all_results)}개                              │")
    log(f"  │ 튜닝 대상      : {', '.join(r['model'] for r in top_results)}")
    best = top_results[0]
    log(f"  │ 최고 모델      : {best['model']} (AUC={best['cv_auc_mean']:.4f})           │")
    log(f"  │ 로그 파일      : {LOG_FILE}                     │")
    log(f"  └─────────────────────────────────────────────────┘")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def run(csv_path: str, target: str) -> None:
    log("=" * 60)
    log("  Kaggle Binary Classification Model Experiment Agent")
    log("=" * 60)

    X, y = step1_load(csv_path, target)

    section("STEP 2 · 실험 모델 목록")
    models = get_models()
    for name, m in models.items():
        log(f"  • {name}")

    section("STEP 3 · 기본 하이퍼파라미터 확인")
    log("  모든 모델 random_state=42 고정, 기본 파라미터로 실험")
    log(f"  평가: {CV_FOLDS}-Fold StratifiedKFold  |  지표: AUC-ROC, F1, Precision, Recall")

    all_results = step4_run_all(X, y, csv_path)
    top_results = step5_visualize_and_rank(all_results)
    step6_combined_cm(all_results, y)
    step7_save_summary(top_results, all_results, csv_path)

    log("\n" + "=" * 60)
    log("  모델 실험 완료!")
    log("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python model_experiment_agent.py <train_csv> <target_col>")
        print("Example: python model_experiment_agent.py train_processed.csv Survived")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
