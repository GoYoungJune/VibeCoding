"""
Kaggle Binary Classification Ensemble & Submission Agent
Usage: python ensemble_agent.py <train_csv> <target_col> [test_csv]
Example: python ensemble_agent.py train_processed.csv Survived
         python ensemble_agent.py train_processed.csv Survived test_processed.csv
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
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ── 상수 ─────────────────────────────────────────────────────
RANDOM_STATE      = 42
CV_FOLDS          = 5
FIG_DIR           = "ensemble_figures"
BEST_PARAMS_FILE  = "best_params.json"
THRESHOLD_RANGE   = np.arange(0.30, 0.71, 0.01)
os.makedirs(FIG_DIR, exist_ok=True)


# ── 유틸 ─────────────────────────────────────────────────────
def log(text: str = "") -> None:
    print(text)

def section(title: str) -> None:
    log(f"\n{'─'*60}\n  {title}\n{'─'*60}")

def save_fig(name: str) -> str:
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    return path


# ══════════════════════════════════════════════════════════════
# 모델 빌더 (best_params.json 반영)
# ══════════════════════════════════════════════════════════════
def build_models(best_params: dict) -> dict:
    """best_params.json에서 읽은 최적 파라미터로 모델 생성."""
    param_map = {m["model"]: m["best_params"] for m in best_params["models"]}
    models = {}

    if "RandomForest" in param_map:
        p = param_map["RandomForest"]
        models["RandomForest"] = RandomForestClassifier(
            **p, random_state=RANDOM_STATE, n_jobs=-1
        )
    if "CatBoost" in param_map:
        p = param_map["CatBoost"]
        models["CatBoost"] = CatBoostClassifier(
            **p, random_seed=RANDOM_STATE, verbose=False
        )
    if "XGBoost" in param_map:
        p = param_map["XGBoost"]
        models["XGBoost"] = xgb.XGBClassifier(
            **p, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        )
    return models


# ══════════════════════════════════════════════════════════════
# STEP 1: 로드
# ══════════════════════════════════════════════════════════════
def step1_load(train_path: str, target: str, test_path: str | None):
    section("STEP 1 · 데이터 및 파라미터 로드")

    train = pd.read_csv(train_path)
    y = train[target].copy()
    X = train.drop(columns=[target]).select_dtypes(include=np.number).fillna(0)

    X_test = None
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        if target in test_df.columns:
            test_df = test_df.drop(columns=[target])
        X_test = test_df.select_dtypes(include=np.number).fillna(0)
        # 컬럼 정렬
        common = [c for c in X.columns if c in X_test.columns]
        X      = X[common]
        X_test = X_test[common]
        log(f"  테스트셋 : {X_test.shape}")

    log(f"  훈련셋   : {X.shape[0]:,} × {X.shape[1]}")
    log(f"  타겟     : {target}  →  {dict(y.value_counts().sort_index())}")

    if not os.path.exists(BEST_PARAMS_FILE):
        sys.exit(f"[ERROR] {BEST_PARAMS_FILE} 없음. 튜닝 에이전트를 먼저 실행하세요.")

    with open(BEST_PARAMS_FILE) as f:
        best_params = json.load(f)
    log(f"  {BEST_PARAMS_FILE} 로드 — 모델: "
        f"{[m['model'] for m in best_params['models']]}")

    return X, y, X_test, best_params


# ══════════════════════════════════════════════════════════════
# STEP 2: OOF 예측 생성
# ══════════════════════════════════════════════════════════════
def step2_oof_predictions(
        models: dict,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame | None,
) -> tuple[dict, dict]:
    section("STEP 2 · OOF 예측 생성 (5-Fold)")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_probs: dict[str, np.ndarray]  = {}
    test_probs: dict[str, np.ndarray] = {}
    model_aucs: dict[str, float]      = {}

    for name, model in models.items():
        oof  = np.zeros(len(y))
        test_fold_preds = (
            np.zeros((len(X_test), CV_FOLDS)) if X_test is not None else None
        )

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            import copy
            m = copy.deepcopy(model)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
            if X_test is not None:
                test_fold_preds[:, fold] = m.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y, oof)
        model_aucs[name] = auc
        oof_probs[name]  = oof
        if X_test is not None:
            test_probs[name] = test_fold_preds.mean(axis=1)

        log(f"  {name:<22} OOF AUC = {auc:.4f}")

    return oof_probs, test_probs, model_aucs


# ══════════════════════════════════════════════════════════════
# STEP 3: 앙상블 전략 비교
# ══════════════════════════════════════════════════════════════
def step3_ensemble(
        oof_probs: dict,
        test_probs: dict,
        y: pd.Series,
        model_aucs: dict,
        X_test: pd.DataFrame | None,
) -> tuple[dict, dict]:
    section("STEP 3 · 앙상블 전략 비교")

    names  = list(oof_probs.keys())
    oof_mat = np.column_stack([oof_probs[n] for n in names])

    results: dict[str, dict] = {}

    # ── 1) 단순 평균 ─────────────────────────────────────────
    simple_oof = oof_mat.mean(axis=1)
    results["SimpleAverage"] = {
        "oof":  simple_oof,
        "auc":  roc_auc_score(y, simple_oof),
        "weights": {n: 1/len(names) for n in names},
    }

    # ── 2) 가중 평균 (CV AUC 소프트맥스 가중치) ───────────────
    raw_w  = np.array([model_aucs[n] for n in names])
    raw_w  = np.exp(raw_w * 10)           # AUC 차이 증폭
    weights = raw_w / raw_w.sum()
    weighted_oof = oof_mat @ weights
    results["WeightedAverage"] = {
        "oof":     weighted_oof,
        "auc":     roc_auc_score(y, weighted_oof),
        "weights": {n: round(float(w), 4) for n, w in zip(names, weights)},
    }

    # ── 3) 스태킹 (meta-learner: Logistic Regression) ────────
    scaler = StandardScaler()
    meta_X = scaler.fit_transform(oof_mat)
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    stacking_oof = np.zeros(len(y))

    for tr_idx, val_idx in skf.split(meta_X, y):
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(meta_X[tr_idx], y.iloc[tr_idx])
        stacking_oof[val_idx] = lr.predict_proba(meta_X[val_idx])[:, 1]

    # 전체로 meta-learner 재학습 (테스트 예측용)
    lr_final = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    lr_final.fit(meta_X, y)

    stacking_coef = {n: round(float(c), 4)
                     for n, c in zip(names, lr_final.coef_[0])}
    results["Stacking"] = {
        "oof":     stacking_oof,
        "auc":     roc_auc_score(y, stacking_oof),
        "weights": stacking_coef,
        "lr":      lr_final,
        "scaler":  scaler,
    }

    # ── 결과 출력 ─────────────────────────────────────────────
    log(f"\n  {'앙상블 전략':<20} {'OOF AUC':>10}  {'가중치/계수'}")
    log(f"  {'─'*20} {'─'*10}  {'─'*30}")
    for method, res in results.items():
        w_str = "  ".join(f"{k}={v:.3f}" for k, v in res["weights"].items())
        log(f"  {method:<20} {res['auc']:>10.4f}  {w_str}")

    # 테스트 앙상블 예측 생성
    test_ensemble: dict[str, np.ndarray] = {}
    if X_test is not None and test_probs:
        test_mat = np.column_stack([test_probs[n] for n in names])
        test_ensemble["SimpleAverage"]   = test_mat.mean(axis=1)
        test_ensemble["WeightedAverage"] = test_mat @ weights

        test_meta = scaler.transform(test_mat)
        test_ensemble["Stacking"] = lr_final.predict_proba(test_meta)[:, 1]

    return results, test_ensemble


# ══════════════════════════════════════════════════════════════
# STEP 4: 최적 앙상블 및 threshold 최적화
# ══════════════════════════════════════════════════════════════
def step4_threshold_opt(
        ensemble_results: dict,
        y: pd.Series,
) -> tuple[str, float, float]:
    section("STEP 4 · 최적 앙상블 선택 및 Threshold 최적화")

    # AUC 기준 최고 앙상블
    best_method = max(ensemble_results, key=lambda k: ensemble_results[k]["auc"])
    best_oof    = ensemble_results[best_method]["oof"]
    best_auc    = ensemble_results[best_method]["auc"]
    log(f"  최고 앙상블: {best_method}  (AUC={best_auc:.4f})\n")

    # threshold 탐색
    f1_scores = []
    for thr in THRESHOLD_RANGE:
        pred = (best_oof >= thr).astype(int)
        f1_scores.append(f1_score(y, pred, zero_division=0))

    best_idx = int(np.argmax(f1_scores))
    best_thr = float(THRESHOLD_RANGE[best_idx])
    best_f1  = f1_scores[best_idx]

    log(f"  {'Threshold':>10} {'F1':>8}")
    log(f"  {'─'*10} {'─'*8}")
    for thr, f1 in zip(THRESHOLD_RANGE[::5], f1_scores[::5]):
        marker = " ◀ 최적" if abs(thr - best_thr) < 0.005 else ""
        log(f"  {thr:>10.2f} {f1:>8.4f}{marker}")
    log(f"\n  최적 Threshold = {best_thr:.2f}  (F1={best_f1:.4f})")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # AUC 비교
    methods = list(ensemble_results.keys())
    aucs    = [ensemble_results[m]["auc"] for m in methods]
    colors  = ["#DD8452" if m == best_method else "#4C72B0" for m in methods]
    axes[0].bar(methods, aucs, color=colors, edgecolor="white")
    for i, (m, a) in enumerate(zip(methods, aucs)):
        axes[0].text(i, a + 0.001, f"{a:.4f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold" if m == best_method else "normal")
    axes[0].set_ylim(min(aucs) - 0.01, max(aucs) + 0.015)
    axes[0].set_title("Ensemble Strategy — OOF AUC")
    axes[0].set_ylabel("AUC-ROC")

    # Threshold vs F1
    axes[1].plot(THRESHOLD_RANGE, f1_scores, color="#4C72B0", linewidth=2)
    axes[1].axvline(best_thr, color="#DD8452", linestyle="--", linewidth=1.5,
                    label=f"best={best_thr:.2f} (F1={best_f1:.4f})")
    axes[1].set_title(f"Threshold Optimization ({best_method})")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("F1-Score")
    axes[1].legend()

    plt.suptitle("Ensemble Comparison & Threshold Search", fontsize=13)
    plt.tight_layout()
    save_fig("step4_ensemble_threshold")
    log(f"  → 저장: ensemble_figures/step4_ensemble_threshold.png")

    return best_method, best_thr, best_auc


# ══════════════════════════════════════════════════════════════
# STEP 5: 예측 분포 검증
# ══════════════════════════════════════════════════════════════
def step5_validate(
        oof_probs: dict,
        ensemble_results: dict,
        best_method: str,
        best_thr: float,
        y: pd.Series,
) -> None:
    section("STEP 5 · 예측 분포 검증")

    best_oof  = ensemble_results[best_method]["oof"]
    best_pred = (best_oof >= best_thr).astype(int)

    # 기본 통계
    log(f"  예측 확률 통계:")
    log(f"    min={best_oof.min():.4f}  max={best_oof.max():.4f}  "
        f"mean={best_oof.mean():.4f}  std={best_oof.std():.4f}")
    log(f"  결측치: {np.isnan(best_oof).sum()}")
    log(f"  예측 분포: {dict(pd.Series(best_pred).value_counts().sort_index())}")

    # 클래스 비율 비교
    orig_ratio = y.mean()
    pred_ratio = best_pred.mean()
    log(f"  실제 양성 비율: {orig_ratio:.3f} | 예측 양성 비율: {pred_ratio:.3f}")
    if abs(orig_ratio - pred_ratio) > 0.1:
        log(f"  ⚠  비율 차이 > 10% — threshold 재검토 권장")
    else:
        log(f"  ✅ 예측 분포 정상")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 모델별 OOF 확률 분포
    for name, prob in oof_probs.items():
        axes[0].hist(prob, bins=40, alpha=0.5, label=name, density=True)
    axes[0].set_title("Model OOF Probability Distribution")
    axes[0].set_xlabel("Predicted Probability")
    axes[0].legend(fontsize=8)

    # 앙상블 확률 분포
    axes[1].hist(best_oof[y == 0], bins=40, alpha=0.7,
                 label="Actual=0", color="#4C72B0", density=True)
    axes[1].hist(best_oof[y == 1], bins=40, alpha=0.7,
                 label="Actual=1", color="#DD8452", density=True)
    axes[1].axvline(best_thr, color="black", linestyle="--",
                    linewidth=1.5, label=f"thr={best_thr:.2f}")
    axes[1].set_title(f"{best_method} — Prob by True Label")
    axes[1].set_xlabel("Predicted Probability")
    axes[1].legend(fontsize=8)

    # 모델별 기여도 (앙상블 가중치)
    weights = ensemble_results[best_method]["weights"]
    w_vals  = list(weights.values())
    w_names = list(weights.keys())
    colors  = ["#4C72B0", "#DD8452", "#55A868"][:len(w_names)]
    axes[2].barh(w_names, w_vals, color=colors, edgecolor="white")
    axes[2].set_title(f"{best_method} — Model Contribution")
    axes[2].set_xlabel("Weight / Coefficient")
    axes[2].axvline(0, color="black", linewidth=0.8)
    for i, v in enumerate(w_vals):
        axes[2].text(v + 0.001 if v >= 0 else v - 0.001, i,
                     f"{v:.4f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=9)

    plt.suptitle("Prediction Distribution & Model Contribution", fontsize=13)
    plt.tight_layout()
    save_fig("step5_distribution")
    log(f"  → 저장: ensemble_figures/step5_distribution.png")


# ══════════════════════════════════════════════════════════════
# STEP 6: submission.csv 생성
# ══════════════════════════════════════════════════════════════
def step6_submission(
        test_ensemble: dict,
        best_method: str,
        best_thr: float,
        target: str,
        train_path: str,
        test_path: str | None,
) -> None:
    section("STEP 6 · submission.csv 생성")

    # 테스트 예측이 없으면 OOF로 대체 (데모)
    if not test_ensemble:
        log("  ⚠  테스트셋 없음 — OOF 확률로 submission.csv 생성 (데모)")
        train_df = pd.read_csv(train_path)
        sub = pd.DataFrame({
            "PassengerId": range(892, 892 + len(train_df)),
            target: (np.zeros(len(train_df)) >= best_thr).astype(int),
        })
    else:
        test_prob = test_ensemble[best_method]
        test_pred = (test_prob >= best_thr).astype(int)

        # PassengerId 복원 (원본 test CSV에 있으면 사용)
        if test_path and os.path.exists(test_path):
            test_raw = pd.read_csv(test_path)
            if "PassengerId" in test_raw.columns:
                pid = test_raw["PassengerId"].values
            else:
                pid = np.arange(1, len(test_pred) + 1)
        else:
            pid = np.arange(1, len(test_pred) + 1)

        sub = pd.DataFrame({"PassengerId": pid, target: test_pred})

    # 검증
    assert sub[target].isnull().sum() == 0, "결측치 발견!"
    assert set(sub[target].unique()).issubset({0, 1}), "이진값 아닌 예측 발견!"

    sub.to_csv("submission.csv", index=False)
    log(f"  ✅ submission.csv 저장 — {sub.shape}")
    log(f"  예측 분포: {dict(sub[target].value_counts().sort_index())}")
    log(f"  양성 비율: {sub[target].mean():.3f}")
    log(f"\n  샘플 (상위 5행):\n{sub.head().to_string(index=False)}")


# ══════════════════════════════════════════════════════════════
# STEP 7: 최종 리포트 저장
# ══════════════════════════════════════════════════════════════
def step7_report(
        ensemble_results: dict,
        model_aucs: dict,
        best_method: str,
        best_thr: float,
        best_auc: float,
) -> None:
    section("STEP 7 · 최종 리포트 저장")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [
        "# Ensemble & Submission Report",
        f"*생성일시: {ts}*\n",
        "---",
        "## 개별 모델 OOF AUC",
        "| 모델 | AUC |",
        "|---|---|",
    ]
    for name, auc in sorted(model_aucs.items(), key=lambda x: -x[1]):
        md.append(f"| {name} | {auc:.4f} |")

    md += [
        "",
        "## 앙상블 전략 비교",
        "| 전략 | OOF AUC | 가중치 |",
        "|---|---|---|",
    ]
    for method, res in ensemble_results.items():
        w = " / ".join(f"{k}:{v:.3f}" for k, v in res["weights"].items())
        marker = " ✅" if method == best_method else ""
        md.append(f"| {method}{marker} | {res['auc']:.4f} | {w} |")

    md += [
        "",
        "## 최종 설정",
        f"- **최고 앙상블**: {best_method}",
        f"- **최적 Threshold**: {best_thr:.2f}",
        f"- **최종 CV AUC**: {best_auc:.4f}",
        "",
        "## 시각화",
        "![앙상블비교](ensemble_figures/step4_ensemble_threshold.png)",
        "![분포검증](ensemble_figures/step5_distribution.png)",
        "",
        "---",
        "*Generated by Kaggle Ensemble Agent*",
    ]

    with open("ensemble_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # 콘솔 최종 요약
    log(f"\n  ┌─ 최종 요약 ──────────────────────────────────────┐")
    log(f"  │  최고 앙상블  : {best_method:<20}              │")
    log(f"  │  최적 Threshold: {best_thr:.2f}                              │")
    log(f"  │  최종 CV AUC  : {best_auc:.4f}                            │")
    log(f"  │  제출 파일    : submission.csv                      │")
    log(f"  └─────────────────────────────────────────────────┘")

    log(f"\n  ✅ ensemble_report.md 저장 완료")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def run(train_path: str, target: str, test_path: str | None = None) -> None:
    log("=" * 60)
    log("  Kaggle Binary Classification Ensemble & Submission Agent")
    log("=" * 60)

    X, y, X_test, best_params = step1_load(train_path, target, test_path)
    models = build_models(best_params)

    oof_probs, test_probs, model_aucs = step2_oof_predictions(models, X, y, X_test)
    ensemble_results, test_ensemble   = step3_ensemble(
        oof_probs, test_probs, y, model_aucs, X_test
    )
    best_method, best_thr, best_auc   = step4_threshold_opt(ensemble_results, y)
    step5_validate(oof_probs, ensemble_results, best_method, best_thr, y)
    step6_submission(test_ensemble, best_method, best_thr, target, train_path, test_path)
    step7_report(ensemble_results, model_aucs, best_method, best_thr, best_auc)

    log("\n" + "=" * 60)
    log("  앙상블 및 제출 파일 생성 완료!")
    log("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ensemble_agent.py <train_csv> <target_col> [test_csv]")
        print("Example: python ensemble_agent.py train_processed.csv Survived")
        sys.exit(1)
    _train  = sys.argv[1]
    _target = sys.argv[2]
    _test   = sys.argv[3] if len(sys.argv) >= 4 else None
    run(_train, _target, _test)
