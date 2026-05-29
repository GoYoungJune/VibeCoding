"""
Titanic Survival Predictor
지금까지 만든 전처리 · 튜닝 · 앙상블 에이전트 파이프라인을 통합한 예측기.

사용법:
  # 1) 모델 학습 & 저장 (최초 1회)
  python titanic_predictor.py --train

  # 2) CSV 파일로 예측
  python titanic_predictor.py --predict titanic_test.csv

  # 3) 단일 승객 예측 (대화형)
  python titanic_predictor.py --interactive

출력: 1(생존), 0(사망)
"""

import sys
import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

# CrossValTargetEncoder가 target_encoder.pkl에 직렬화돼 있으므로
# preprocessing_agent에서 클래스를 임포트해 pickle이 참조할 수 있게 함
from preprocessing_agent import CrossValTargetEncoder  # noqa: F401

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ── 경로 상수 ─────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
PREPROCESSOR_PKT = os.path.join(BASE_DIR, "preprocessor.pkl")
TARGET_ENC_PKL   = os.path.join(BASE_DIR, "target_encoder.pkl")
BEST_PARAMS_JSON = os.path.join(BASE_DIR, "best_params.json")
TRAINED_MODELS   = os.path.join(BASE_DIR, "trained_models.pkl")
TRAIN_CSV        = os.path.join(BASE_DIR, "train_processed.csv")
TARGET_COL       = "Survived"
THRESHOLD        = 0.47   # 앙상블 에이전트에서 F1 최적화로 결정된 값
RANDOM_STATE     = 42
CV_FOLDS         = 5


# ══════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════
def _build_models(best_params: dict) -> dict:
    param_map = {m["model"]: m["best_params"] for m in best_params["models"]}
    models = {}
    if "RandomForest" in param_map:
        models["RandomForest"] = RandomForestClassifier(
            **param_map["RandomForest"], random_state=RANDOM_STATE, n_jobs=-1
        )
    if "CatBoost" in param_map:
        models["CatBoost"] = CatBoostClassifier(
            **param_map["CatBoost"], random_seed=RANDOM_STATE, verbose=False
        )
    if "XGBoost" in param_map:
        models["XGBoost"] = xgb.XGBClassifier(
            **param_map["XGBoost"], eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        )
    return models


# ══════════════════════════════════════════════════════════════
# 전처리 파이프라인 (원시 데이터 → 피처 행렬)
# ══════════════════════════════════════════════════════════════
def preprocess(raw_df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """
    preprocessing_agent가 저장한 preprocessor.pkl / target_encoder.pkl 로
    원시 타이타닉 데이터를 숫자 행렬로 변환한다.
    fit=False → transform만 (테스트/추론 시 사용)
    """
    preprocessor  = joblib.load(PREPROCESSOR_PKT)
    te_exists      = os.path.exists(TARGET_ENC_PKL)
    target_encoder = joblib.load(TARGET_ENC_PKL) if te_exists else None

    df = raw_df.copy()

    # 타겟 열이 있으면 분리
    y = None
    if TARGET_COL in df.columns:
        y = df[TARGET_COL].copy()
        df = df.drop(columns=[TARGET_COL])

    # 고카디널리티(Ticket) Target Encoding — transform only
    if target_encoder is not None:
        high_card_cols = target_encoder.cols
        present = [c for c in high_card_cols if c in df.columns]
        if present:
            encoded = target_encoder.transform(df[present])
            for col in present:
                df[col] = encoded[col].values

    # sklearn ColumnTransformer
    X_np = preprocessor.transform(df)

    # 컬럼명 복원
    def _get_feature_names(ct):
        names = []
        for tname, trans, cols in ct.transformers_:
            if tname == "remainder":
                continue
            if hasattr(trans, "get_feature_names_out"):
                names.extend(trans.get_feature_names_out())
            elif hasattr(trans, "named_steps"):
                last = list(trans.named_steps.values())[-1]
                if hasattr(last, "get_feature_names_out"):
                    names.extend(last.get_feature_names_out())
                else:
                    names.extend(cols if isinstance(cols, list) else [cols])
            else:
                names.extend(cols if isinstance(cols, list) else [cols])
        return names

    feature_names = _get_feature_names(preprocessor)
    X_df = pd.DataFrame(X_np, columns=feature_names, index=df.index)
    return X_df, y


# ══════════════════════════════════════════════════════════════
# TRAIN: 모델 학습 & 저장
# ══════════════════════════════════════════════════════════════
def train():
    print("=" * 55)
    print("  Titanic Predictor — 모델 학습")
    print("=" * 55)

    # 데이터 로드
    train_df = pd.read_csv(TRAIN_CSV)
    y = train_df[TARGET_COL]
    X = train_df.drop(columns=[TARGET_COL]).select_dtypes(include=np.number).fillna(0)
    print(f"\n  훈련 데이터: {X.shape[0]:,} rows × {X.shape[1]} features")

    # 파라미터 로드
    with open(BEST_PARAMS_JSON) as f:
        best_params = json.load(f)
    models = _build_models(best_params)
    print(f"  학습 모델  : {list(models.keys())}")

    # CV AUC 확인
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"\n  {'모델':<22} {'CV AUC':>8}")
    print(f"  {'─'*22} {'─'*8}")

    import copy
    trained = {}
    for name, model in models.items():
        oof = np.zeros(len(y))
        for tr_idx, val_idx in skf.split(X, y):
            m = copy.deepcopy(model)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y, oof)
        print(f"  {name:<22} {auc:.4f}")

        # 전체 데이터로 최종 학습
        final_model = copy.deepcopy(model)
        final_model.fit(X, y)
        trained[name] = final_model

    # 저장
    joblib.dump({"models": trained, "feature_cols": list(X.columns)}, TRAINED_MODELS)
    print(f"\n  ✅ trained_models.pkl 저장 완료")
    print(f"  → 이제 --predict 또는 --interactive 로 예측 가능합니다.")


# ══════════════════════════════════════════════════════════════
# PREDICT: CSV → 예측
# ══════════════════════════════════════════════════════════════
def predict_csv(csv_path: str) -> pd.DataFrame:
    print("=" * 55)
    print("  Titanic Predictor — CSV 예측")
    print("=" * 55)

    if not os.path.exists(TRAINED_MODELS):
        sys.exit("[ERROR] trained_models.pkl 없음. 먼저 --train 을 실행하세요.")

    raw = pd.read_csv(csv_path)
    print(f"\n  입력: {csv_path}  ({len(raw)} rows)")

    # PassengerId 보존
    pid = raw["PassengerId"].values if "PassengerId" in raw.columns \
          else np.arange(1, len(raw) + 1)

    # 전처리
    X, y_true = preprocess(raw)

    # 모델 로드
    bundle  = joblib.load(TRAINED_MODELS)
    trained = bundle["models"]
    feat_cols = bundle["feature_cols"]

    # 컬럼 정렬 (훈련 피처와 맞춤)
    for col in feat_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feat_cols]

    # 앙상블 예측 (단순 평균)
    probs = np.column_stack([
        m.predict_proba(X)[:, 1] for m in trained.values()
    ]).mean(axis=1)
    preds = (probs >= THRESHOLD).astype(int)

    result = pd.DataFrame({
        "PassengerId": pid,
        "Survived":    preds,
        "Probability": probs.round(4),
    })

    # 검증
    assert result["Survived"].isnull().sum() == 0
    print(f"\n  예측 분포: {dict(result['Survived'].value_counts().sort_index())}")
    print(f"  생존 비율 : {result['Survived'].mean():.3f}")

    if y_true is not None:
        auc = roc_auc_score(y_true, probs)
        acc = (preds == y_true.values).mean()
        print(f"  AUC      : {auc:.4f}")
        print(f"  Accuracy : {acc:.4f}")

    out_path = "predictions.csv"
    result[["PassengerId", "Survived"]].to_csv(out_path, index=False)
    print(f"\n  ✅ {out_path} 저장 완료")
    print(f"\n  샘플 (상위 10행):")
    print(result.head(10).to_string(index=False))

    return result


# ══════════════════════════════════════════════════════════════
# INTERACTIVE: 단일 승객 대화형 예측
# ══════════════════════════════════════════════════════════════
def interactive():
    print("=" * 55)
    print("  Titanic Predictor — 승객 정보 입력")
    print("=" * 55)

    if not os.path.exists(TRAINED_MODELS):
        sys.exit("[ERROR] trained_models.pkl 없음. 먼저 --train 을 실행하세요.")

    print("\n  승객 정보를 입력하세요 (Enter = 기본값 사용)\n")

    def ask(prompt, default, cast=str):
        val = input(f"  {prompt} [{default}]: ").strip()
        return cast(val) if val else cast(default)

    pclass   = ask("객실 등급 (1·2·3)",     3,   int)
    sex      = ask("성별 (male/female)",     "male", str).lower()
    age      = ask("나이",                   28,  float)
    sibsp    = ask("형제자매/배우자 수",       0,   int)
    parch    = ask("부모/자녀 수",            0,   int)
    fare     = ask("운임",                   14.0, float)
    embarked = ask("탑승 항구 (S·C·Q)",     "S", str).upper()
    ticket   = ask("티켓 번호",             "A/5 21171", str)
    cabin    = ask("객실 번호 (없으면 Enter)", "",  str)

    row = pd.DataFrame([{
        "PassengerId": 9999,
        "Pclass":   pclass,
        "Name":     "Test, Mr. Passenger",
        "Sex":      sex,
        "Age":      age,
        "SibSp":    sibsp,
        "Parch":    parch,
        "Ticket":   ticket,
        "Fare":     fare,
        "Cabin":    cabin if cabin else np.nan,
        "Embarked": embarked,
    }])

    X, _ = preprocess(row)

    bundle    = joblib.load(TRAINED_MODELS)
    trained   = bundle["models"]
    feat_cols = bundle["feature_cols"]

    for col in feat_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feat_cols]

    probs = np.column_stack([
        m.predict_proba(X)[:, 1] for m in trained.values()
    ]).mean(axis=1)
    prob   = float(probs[0])
    pred   = int(prob >= THRESHOLD)
    label  = "✅ 생존 (1)" if pred == 1 else "❌ 사망 (0)"

    print(f"\n  {'─'*40}")
    print(f"  예측 결과  : {label}")
    print(f"  생존 확률  : {prob:.1%}")
    print(f"  임계값     : {THRESHOLD}")
    print(f"  {'─'*40}")

    # 모델별 확률 출력
    print(f"\n  모델별 생존 확률:")
    for name, model in trained.items():
        p = model.predict_proba(X)[0, 1]
        bar = "█" * int(p * 20)
        print(f"    {name:<18} {p:.1%}  {bar}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Titanic Survival Predictor",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train",       action="store_true",
                       help="모델 학습 & trained_models.pkl 저장")
    group.add_argument("--predict",     metavar="CSV",
                       help="CSV 파일 예측 → predictions.csv 생성")
    group.add_argument("--interactive", action="store_true",
                       help="승객 정보 직접 입력하여 예측")

    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        predict_csv(args.predict)
    elif args.interactive:
        interactive()
