"""
Kaggle Binary Classification Hyperparameter Tuning Agent
Usage: python tuning_agent.py <train_csv> <target_col> [n_trials]
Example: python tuning_agent.py train_processed.csv Survived 50
"""

import sys
import os
import json
import time
import copy
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ── 상수 ─────────────────────────────────────────────────────
RANDOM_STATE = 42
CV_FOLDS     = 5
FIG_DIR      = "tuning_figures"
os.makedirs(FIG_DIR, exist_ok=True)


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

def cv_auc(model, X: pd.DataFrame, y: pd.Series) -> float:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1
    )
    return float(scores.mean())


# ══════════════════════════════════════════════════════════════
# Optuna objective 함수 (모델별)
# ══════════════════════════════════════════════════════════════
def make_lgbm_objective(X, y):
    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
            num_leaves        = trial.suggest_int("num_leaves", 20, 300),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples = trial.suggest_int("min_child_samples", 10, 100),
            feature_fraction  = trial.suggest_float("feature_fraction", 0.5, 1.0),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            random_state      = RANDOM_STATE,
            verbose           = -1,
            n_jobs            = -1,
        )
        model = lgb.LGBMClassifier(**params)
        return cv_auc(model, X, y)
    return objective


def make_xgb_objective(X, y):
    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
            max_depth         = trial.suggest_int("max_depth", 3, 12),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 20),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            eval_metric       = "logloss",
            random_state      = RANDOM_STATE,
            verbosity         = 0,
            n_jobs            = -1,
        )
        model = xgb.XGBClassifier(**params)
        return cv_auc(model, X, y)
    return objective


def make_catboost_objective(X, y):
    def objective(trial):
        params = dict(
            iterations         = trial.suggest_int("iterations", 100, 1000),
            depth              = trial.suggest_int("depth", 3, 10),
            learning_rate      = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg        = trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            bagging_temperature= trial.suggest_float("bagging_temperature", 0.0, 1.0),
            border_count       = trial.suggest_int("border_count", 32, 255),
            random_seed        = RANDOM_STATE,
            verbose            = False,
        )
        model = CatBoostClassifier(**params)
        return cv_auc(model, X, y)
    return objective


def make_rf_objective(X, y):
    def objective(trial):
        params = dict(
            n_estimators    = trial.suggest_int("n_estimators", 100, 800),
            max_depth       = trial.suggest_int("max_depth", 3, 30),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 20),
            max_features    = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
            random_state    = RANDOM_STATE,
            n_jobs          = -1,
        )
        model = RandomForestClassifier(**params)
        return cv_auc(model, X, y)
    return objective


OBJECTIVE_MAP = {
    "LightGBM":         make_lgbm_objective,
    "XGBoost":          make_xgb_objective,
    "CatBoost":         make_catboost_objective,
    "RandomForest":     make_rf_objective,
}

BASELINE_MAP = {
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
    ),
    "CatBoost": CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        random_seed=RANDOM_STATE, verbose=False,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
}


# ══════════════════════════════════════════════════════════════
# STEP 1: 로드
# ══════════════════════════════════════════════════════════════
def step1_load(csv_path: str, target: str,
               top_models_path: str = "top_models.json"):
    section("STEP 1 · 데이터 및 실험 결과 로드")

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        sys.exit(f"[ERROR] '{target}' 없음.")
    y = df[target].copy()
    X = df.drop(columns=[target]).select_dtypes(include=np.number).fillna(0)
    log(f"  데이터 : {csv_path}  ({X.shape[0]:,} × {X.shape[1]})")

    # top_models.json에서 튜닝 대상 모델 읽기
    if os.path.exists(top_models_path):
        with open(top_models_path) as f:
            meta = json.load(f)
        top_models = [m["model"] for m in meta["top_models"]]
        baseline_aucs = {m["model"]: m["cv_auc_mean"] for m in meta["top_models"]}
        log(f"  top_models.json 로드 — 튜닝 대상: {top_models}")
    else:
        top_models = list(OBJECTIVE_MAP.keys())
        baseline_aucs = {}
        log(f"  top_models.json 없음 → 전체 모델 튜닝: {top_models}")

    return X, y, top_models, baseline_aucs


# ══════════════════════════════════════════════════════════════
# STEP 2: Optuna 튜닝 (모델별)
# ══════════════════════════════════════════════════════════════
def step2_tune(
        name: str,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int,
        baseline_auc: float,
) -> dict:
    section(f"STEP 2-{name} · Optuna 최적화  (n_trials={n_trials})")
    log(f"  기준 AUC (기본 파라미터): {baseline_auc:.4f}")

    if name not in OBJECTIVE_MAP:
        log(f"  ⚠  {name} — 탐색 공간 미정의, 건너뜀")
        return {}

    objective_fn = OBJECTIVE_MAP[name](X, y)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # 진행 상황 출력 콜백
    best_so_far = [0.0]
    def callback(study, trial):
        if trial.value > best_so_far[0]:
            best_so_far[0] = trial.value
        if trial.number % 10 == 0 or trial.value == study.best_value:
            log(f"  Trial {trial.number:>3}  AUC={trial.value:.4f}  "
                f"best={study.best_value:.4f}")

    t0 = time.perf_counter()
    study.optimize(objective_fn, n_trials=n_trials,
                   callbacks=[callback], show_progress_bar=False)
    elapsed = time.perf_counter() - t0

    best_auc    = study.best_value
    best_params = study.best_params
    delta       = best_auc - baseline_auc
    sign        = "▲" if delta > 0 else "▼"

    log(f"\n  ── 결과 ──────────────────────────────────────────")
    log(f"  기준 AUC   : {baseline_auc:.4f}")
    log(f"  최적 AUC   : {best_auc:.4f}  ({sign} {abs(delta):.4f})")
    log(f"  소요 시간  : {elapsed:.1f}s  ({n_trials} trials)")
    log(f"  최적 파라미터:")
    for k, v in best_params.items():
        log(f"    {k:<25} = {v}")

    # 시각화 저장
    _save_optuna_plots(study, name)

    return {
        "model":         name,
        "baseline_auc":  baseline_auc,
        "tuned_auc":     best_auc,
        "delta":         delta,
        "best_params":   best_params,
        "n_trials":      n_trials,
        "elapsed_sec":   round(elapsed, 1),
        "study":         study,
    }


def _save_optuna_plots(study: optuna.Study, name: str) -> None:
    fname = name.lower().replace(" ", "_")

    # 최적화 이력
    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title(f"{name} — Optimization History")
        plt.tight_layout()
        save_fig(f"{fname}_opt_history")
    except Exception:
        _plot_history_manual(study, name, fname)

    # 파라미터 중요도
    try:
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title(f"{name} — Parameter Importance")
        plt.tight_layout()
        save_fig(f"{fname}_param_importance")
    except Exception:
        pass

    log(f"  → 저장: tuning_figures/{fname}_opt_history.png")
    log(f"  → 저장: tuning_figures/{fname}_param_importance.png")


def _plot_history_manual(study, name, fname):
    trials = [t for t in study.trials if t.value is not None]
    if not trials:
        return
    vals  = [t.value for t in trials]
    bests = pd.Series(vals).cummax().tolist()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(range(len(vals)), vals, s=12, alpha=0.5, color="#4C72B0", label="Trial")
    ax.plot(range(len(bests)), bests, color="#DD8452", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"{name} — Optimization History")
    ax.legend()
    plt.tight_layout()
    save_fig(f"{fname}_opt_history")


# ══════════════════════════════════════════════════════════════
# STEP 3: 최적 모델로 최종 CV 검증
# ══════════════════════════════════════════════════════════════
def step3_final_cv(results: list[dict], X: pd.DataFrame, y: pd.Series) -> list[dict]:
    section("STEP 3 · 최적 파라미터로 최종 CV 검증")

    for res in results:
        name   = res["model"]
        params = res["best_params"]

        if name == "LightGBM":
            model = lgb.LGBMClassifier(
                **params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
            )
        elif name == "XGBoost":
            model = xgb.XGBClassifier(
                **params, eval_metric="logloss",
                random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
            )
        elif name == "CatBoost":
            model = CatBoostClassifier(
                **params, random_seed=RANDOM_STATE, verbose=False
            )
        elif name == "RandomForest":
            model = RandomForestClassifier(
                **params, random_state=RANDOM_STATE, n_jobs=-1
            )
        else:
            continue

        auc = cv_auc(model, X, y)
        res["final_auc"] = auc
        log(f"  {name:<22} 최종 AUC = {auc:.4f}  "
            f"(기준 {res['baseline_auc']:.4f}  "
            f"{'▲' if auc > res['baseline_auc'] else '▼'}"
            f"{abs(auc - res['baseline_auc']):.4f})")

    return results


# ══════════════════════════════════════════════════════════════
# STEP 4: 비교 시각화 & 저장
# ══════════════════════════════════════════════════════════════
def step4_compare_and_save(results: list[dict]) -> None:
    section("STEP 4 · 튜닝 전후 비교 및 저장")

    # 비교 차트
    n = len(results)
    names    = [r["model"] for r in results]
    baseline = [r["baseline_auc"] for r in results]
    tuned    = [r.get("final_auc", r["tuned_auc"]) for r in results]

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, n * 2.5), 5))
    b1 = ax.bar(x - width/2, baseline, width, label="기본 파라미터",
                color="#4C72B0", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width/2, tuned,    width, label="Optuna 튜닝 후",
                color="#DD8452", alpha=0.85, edgecolor="white")

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(min(baseline + tuned) - 0.03, max(baseline + tuned) + 0.04)
    ax.set_ylabel("CV AUC-ROC")
    ax.set_title("Hyperparameter Tuning — Before vs After (Optuna)", fontsize=13)
    ax.legend()
    plt.tight_layout()
    path = save_fig("step4_tuning_comparison")
    log(f"  → 저장: {path}")

    # best_params.json 저장
    output = {
        "generated_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cv_folds":      CV_FOLDS,
        "random_state":  RANDOM_STATE,
        "models": []
    }
    for res in results:
        output["models"].append({
            "model":        res["model"],
            "n_trials":     res["n_trials"],
            "baseline_auc": round(res["baseline_auc"], 4),
            "tuned_auc":    round(res.get("final_auc", res["tuned_auc"]), 4),
            "delta":        round(res.get("final_auc", res["tuned_auc"]) - res["baseline_auc"], 4),
            "elapsed_sec":  res["elapsed_sec"],
            "best_params":  res["best_params"],
        })

    # 최고 모델 (튜닝 후 AUC 기준)
    best = max(output["models"], key=lambda x: x["tuned_auc"])
    output["overall_best"] = best["model"]
    output["overall_best_auc"] = best["tuned_auc"]

    with open("best_params.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log(f"  ✅ best_params.json 저장 완료")

    # 콘솔 요약
    log(f"\n  ┌─ 튜닝 요약 ──────────────────────────────────────┐")
    for m in output["models"]:
        sign = "▲" if m["delta"] > 0 else "▼"
        log(f"  │  {m['model']:<20} "
            f"{m['baseline_auc']:.4f} → {m['tuned_auc']:.4f}  "
            f"({sign}{abs(m['delta']):.4f})  {m['elapsed_sec']}s")
    log(f"  │")
    log(f"  │  🏆 최고 모델: {output['overall_best']}  "
        f"AUC={output['overall_best_auc']:.4f}")
    log(f"  └─────────────────────────────────────────────────┘")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def run(csv_path: str, target: str, n_trials: int = 50) -> None:
    log("=" * 60)
    log("  Kaggle Binary Classification Tuning Agent (Optuna)")
    log("=" * 60)
    log(f"  n_trials = {n_trials}  |  CV = {CV_FOLDS}-Fold  |  seed = {RANDOM_STATE}")

    X, y, top_models, baseline_aucs = step1_load(csv_path, target)

    # 베이스라인 AUC가 없는 모델은 직접 계산
    for name in top_models:
        if name not in baseline_aucs and name in BASELINE_MAP:
            log(f"\n  [{name}] 기본 파라미터 AUC 측정 중...")
            baseline_aucs[name] = cv_auc(BASELINE_MAP[name], X, y)

    results = []
    for name in top_models:
        baseline = baseline_aucs.get(name, 0.0)
        res = step2_tune(name, X, y, n_trials, baseline)
        if res:
            results.append(res)

    if results:
        results = step3_final_cv(results, X, y)
        step4_compare_and_save(results)

    log("\n" + "=" * 60)
    log("  하이퍼파라미터 튜닝 완료!")
    log("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tuning_agent.py <train_csv> <target_col> [n_trials]")
        print("Example: python tuning_agent.py train_processed.csv Survived 50")
        sys.exit(1)

    _csv    = sys.argv[1]
    _target = sys.argv[2]
    _trials = int(sys.argv[3]) if len(sys.argv) >= 4 else 50

    run(_csv, _target, _trials)
