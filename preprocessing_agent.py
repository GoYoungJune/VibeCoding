"""
Kaggle Binary Classification Preprocessing Agent
Usage: python preprocessing_agent.py <train_csv> <target_col> [test_csv]
Example: python preprocessing_agent.py titanic.csv Survived
         python preprocessing_agent.py train.csv Survived test.csv
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

import category_encoders as ce

REPORT_LINES: list[str] = []
IMBALANCE_THRESHOLD = 10.0   # 10:1 이상이면 SMOTE 권장
HIGH_CARD_THRESHOLD = 10     # 유니크 값 10개 초과면 고카디널리티
MISSING_DROP_THRESHOLD = 0.30  # 결측률 30% 초과 컬럼 제거


# ── 유틸 ─────────────────────────────────────────────────────
def log(text: str = "") -> None:
    print(text)
    REPORT_LINES.append(text)


def section(title: str) -> None:
    bar = "─" * 60
    log(f"\n{bar}")
    log(f"  {title}")
    log(bar)


def warn(text: str) -> None:
    log(f"  ⚠  {text}")


def ok(text: str) -> None:
    log(f"  ✅ {text}")


def info(text: str) -> None:
    log(f"  ℹ  {text}")


# ── 커스텀 트랜스포머 ──────────────────────────────────────────
class ColumnDropper(BaseEstimator, TransformerMixin):
    """fit 시 지정 컬럼 제거, transform 시 동일 컬럼 제거 (없으면 무시)."""
    def __init__(self, cols: list[str]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.cols if c in X.columns], errors="ignore")


class CrossValTargetEncoder(BaseEstimator, TransformerMixin):
    """
    타겟 누수 방지용 CV 기반 Target Encoding.
    fit: 전체 데이터로 인코더 학습 (예측/추론용)
    fit_transform: KFold OOF 방식으로 훈련 데이터 인코딩 (누수 방지)
    """
    def __init__(self, cols: list[str], n_splits: int = 5, smoothing: float = 1.0):
        self.cols = cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self._encoder = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("CrossValTargetEncoder requires y.")
        self._encoder = ce.TargetEncoder(cols=self.cols, smoothing=self.smoothing)
        self._encoder.fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        if y is None:
            return self.fit(X).transform(X)

        self.fit(X, y)

        X = X.copy().reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        # float 결과를 담을 빈 배열 (string → float 변환 문제 방지)
        result = np.zeros((len(X), len(self.cols)), dtype=float)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(X):
            enc = ce.TargetEncoder(cols=self.cols, smoothing=self.smoothing)
            enc.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            encoded = enc.transform(X.iloc[val_idx])
            result[val_idx] = encoded[self.cols].values

        return pd.DataFrame(result, columns=self.cols)

    def transform(self, X):
        return self._encoder.transform(X)


# ══════════════════════════════════════════════════════════════
# STEP 1: 데이터 로드 및 기초 점검
# ══════════════════════════════════════════════════════════════
def step1_load(train_path: str, target: str,
               test_path: str | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    section("STEP 1 · 데이터 로드 및 기초 점검")

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path) if test_path else None

    log(f"  Train : {train.shape[0]:,} rows × {train.shape[1]} cols")
    if test is not None:
        log(f"  Test  : {test.shape[0]:,} rows × {test.shape[1]} cols")

    if target not in train.columns:
        sys.exit(f"[ERROR] 타겟 컬럼 '{target}' 없음. 사용 가능: {list(train.columns)}")

    unique_vals = sorted(train[target].dropna().unique())
    if len(unique_vals) != 2:
        warn(f"타겟 고유값이 2개가 아닙니다: {unique_vals}")
    else:
        ok(f"이진 분류 확인 — 타겟 값: {unique_vals}")

    return train, test


# ══════════════════════════════════════════════════════════════
# STEP 2: 결측치 처리 계획
# ══════════════════════════════════════════════════════════════
def step2_missing_plan(train: pd.DataFrame,
                       target: str) -> tuple[list, list, list]:
    section("STEP 2 · 결측치 처리 계획")

    drop_cols: list[str] = []
    miss_rate = train.drop(columns=[target]).isnull().mean()
    miss_rate = miss_rate[miss_rate > 0].sort_values(ascending=False)

    if miss_rate.empty:
        ok("결측치 없음 — imputation 생략")
        return [], [], []

    log(f"\n  {'컬럼':<22} {'결측률':>8}  {'처리 방법'}")
    log(f"  {'─'*22} {'─'*8}  {'─'*20}")

    for col, rate in miss_rate.items():
        if rate > MISSING_DROP_THRESHOLD:
            action = f"제거 (결측률 {rate:.0%} > {MISSING_DROP_THRESHOLD:.0%})"
            drop_cols.append(col)
        elif train[col].dtype == object:
            action = "mode imputation 또는 'Unknown' 추가"
        else:
            action = "median imputation"
        log(f"  {col:<22} {rate:>7.1%}  {action}")

    if drop_cols:
        warn(f"결측률 {MISSING_DROP_THRESHOLD:.0%} 초과 → 제거 예정 컬럼: {drop_cols}")

    return drop_cols, [], []


# ══════════════════════════════════════════════════════════════
# STEP 3: 피처 분류
# ══════════════════════════════════════════════════════════════
def step3_classify_features(
        train: pd.DataFrame,
        target: str,
        drop_cols: list[str],
        id_cols: list[str] | None = None,
) -> tuple[list, list, list, list]:
    section("STEP 3 · 피처 분류")

    df = train.drop(columns=[target] + drop_cols, errors="ignore")

    # 자동 ID 컬럼 감지: 이름에 'id'가 들어가거나 고유값 = 행 수
    auto_id = [c for c in df.columns
               if ("id" in c.lower() or df[c].nunique() == len(df))
               and c not in (id_cols or [])]
    all_id = list(set((id_cols or []) + auto_id))

    df = df.drop(columns=all_id, errors="ignore")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    low_card  = [c for c in cat_cols if df[c].nunique() <= HIGH_CARD_THRESHOLD]
    high_card = [c for c in cat_cols if df[c].nunique() >  HIGH_CARD_THRESHOLD]

    log(f"  제거 (ID/고중복) : {all_id}")
    log(f"  수치형 피처      : {num_cols}")
    log(f"  저카디널리티     : {low_card}  → One-Hot Encoding")
    log(f"  고카디널리티     : {high_card}  → Target Encoding (OOF)")

    return num_cols, low_card, high_card, all_id


# ══════════════════════════════════════════════════════════════
# STEP 4: 클래스 불균형
# ══════════════════════════════════════════════════════════════
def step4_imbalance(train: pd.DataFrame,
                    target: str) -> tuple[float, bool]:
    section("STEP 4 · 클래스 불균형 분석")

    vc = train[target].value_counts()
    ratio = vc.max() / vc.min()
    log(f"  클래스 분포: {vc.to_dict()}")
    log(f"  다수:소수 비율 = {ratio:.2f}:1")

    use_smote = False
    if ratio >= IMBALANCE_THRESHOLD:
        warn(f"심각한 불균형 ({ratio:.1f}:1 ≥ {IMBALANCE_THRESHOLD}:1)")
        warn("SMOTE 적용 예정 (전처리 후 훈련 데이터에만 적용)")
        use_smote = True
    elif ratio >= 3:
        info(f"불균형 감지 ({ratio:.1f}:1) — class_weight='balanced' 권장")
        info("모델 학습 시 class_weight 파라미터를 설정하세요")
    else:
        ok(f"균형 잡힌 타겟 ({ratio:.1f}:1) ✓")

    return ratio, use_smote


# ══════════════════════════════════════════════════════════════
# STEP 5: 파이프라인 구성 및 실행
# ══════════════════════════════════════════════════════════════
def step5_build_and_run(
        train: pd.DataFrame,
        test: pd.DataFrame | None,
        target: str,
        drop_cols: list[str],
        id_cols: list[str],
        num_cols: list[str],
        low_card: list[str],
        high_card: list[str],
        use_smote: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    section("STEP 5 · 파이프라인 구성 및 실행")

    X_train = train.drop(columns=[target] + drop_cols + id_cols, errors="ignore")
    y_train = train[target].copy()

    X_test = None
    if test is not None:
        X_test = test.drop(columns=drop_cols + id_cols, errors="ignore")
        # 테스트에 타겟이 있으면 제거
        if target in X_test.columns:
            X_test = X_test.drop(columns=[target])

    # ── 수치형 파이프라인 ─────────────────────────────────────
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # ── 저카디널리티 파이프라인 ───────────────────────────────
    low_cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # ── ColumnTransformer (수치 + 저카디널리티) ───────────────
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
    if low_card:
        transformers.append(("low_cat", low_cat_pipeline, low_card))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",   # ID컬럼 등 나머지 제거
    )

    # ── 고카디널리티: OOF Target Encoding (누수 방지) ─────────
    # ColumnTransformer 적용 전에 별도 처리 (원본 컬럼 필요)
    if high_card:
        log(f"\n  고카디널리티 Target Encoding (OOF, {5}-Fold):")
        te = CrossValTargetEncoder(cols=high_card)

        # 훈련: OOF 방식
        X_train_hc = te.fit_transform(X_train[high_card], y_train)
        for col in high_card:
            X_train[col] = X_train_hc[col].values
            log(f"    {col}: encoded (train OOF)")

        # 테스트: full-fit transform
        if X_test is not None:
            X_test_hc = te.transform(X_test[high_card])
            for col in high_card:
                X_test[col] = X_test_hc[col].values
                log(f"    {col}: encoded (test)")

        # 이제 high_card 컬럼은 수치형으로 변환됐으므로 num_cols에 추가
        num_cols_extended = num_cols + high_card
        transformers_ext = []
        if num_cols_extended:
            transformers_ext.append(("num", num_pipeline, num_cols_extended))
        if low_card:
            transformers_ext.append(("low_cat", low_cat_pipeline, low_card))
        preprocessor = ColumnTransformer(
            transformers=transformers_ext,
            remainder="drop",
        )

        joblib.dump(te, "target_encoder.pkl")
        ok("target_encoder.pkl 저장 완료")

    # ── fit_transform (train) ─────────────────────────────────
    log("\n  ColumnTransformer fit_transform (train)...")
    X_proc = preprocessor.fit_transform(X_train)

    # ── 컬럼명 복원 ───────────────────────────────────────────
    feature_names = _get_feature_names(preprocessor)
    X_proc_df = pd.DataFrame(X_proc, columns=feature_names)
    X_proc_df[target] = y_train.values

    log(f"  전처리 완료 — train: {X_proc_df.shape}")

    # ── transform only (test) ─────────────────────────────────
    X_test_df = None
    if X_test is not None:
        X_test_proc = preprocessor.transform(X_test)
        X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)
        log(f"  전처리 완료 — test : {X_test_df.shape}")

    # ── SMOTE ────────────────────────────────────────────────
    if use_smote:
        log("\n  SMOTE 적용 중...")
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_sm, y_sm = sm.fit_resample(
            X_proc_df.drop(columns=[target]), X_proc_df[target]
        )
        X_proc_df = pd.DataFrame(X_sm, columns=feature_names)
        X_proc_df[target] = y_sm.values
        log(f"  SMOTE 후 train: {X_proc_df.shape}")
        vc_after = y_sm.value_counts()
        log(f"  클래스 분포 (SMOTE 후): {vc_after.to_dict()}")

    # ── Pipeline 저장 ─────────────────────────────────────────
    joblib.dump(preprocessor, "preprocessor.pkl")
    ok("preprocessor.pkl 저장 완료")

    return X_proc_df, X_test_df


def _get_feature_names(ct: ColumnTransformer) -> list[str]:
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder":
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


# ══════════════════════════════════════════════════════════════
# STEP 6: 저장 및 리포트
# ══════════════════════════════════════════════════════════════
def step6_save(X_train_df: pd.DataFrame,
               X_test_df: pd.DataFrame | None,
               target: str,
               ratio: float,
               use_smote: bool,
               drop_cols: list[str],
               id_cols: list[str],
               num_cols: list[str],
               low_card: list[str],
               high_card: list[str]) -> None:
    section("STEP 6 · 저장 및 전처리 요약 리포트")

    X_train_df.to_csv("train_processed.csv", index=False)
    ok(f"train_processed.csv 저장 — {X_train_df.shape}")

    if X_test_df is not None:
        X_test_df.to_csv("test_processed.csv", index=False)
        ok(f"test_processed.csv 저장 — {X_test_df.shape}")

    # 마크다운 리포트
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = [
        f"# Preprocessing Report — `{target}` 이진 분류",
        f"*생성일시: {ts}*\n",
        "---",
        "## 처리 요약",
        f"| 항목 | 내용 |",
        f"|---|---|",
        f"| 제거 컬럼 (결측률 >{MISSING_DROP_THRESHOLD:.0%}) | `{drop_cols}` |",
        f"| 제거 컬럼 (ID) | `{id_cols}` |",
        f"| 수치형 피처 (median impute) | `{num_cols}` |",
        f"| 저카디널리티 (OHE) | `{low_card}` |",
        f"| 고카디널리티 (Target Enc.) | `{high_card}` |",
        f"| 클래스 불균형 비율 | {ratio:.2f}:1 |",
        f"| SMOTE 적용 | {'예' if use_smote else '아니오'} |",
        "",
        "## 산출물",
        "| 파일 | 설명 |",
        "|---|---|",
        "| `preprocessor.pkl` | sklearn ColumnTransformer (fit 완료) |",
    ]
    if high_card:
        md.append("| `target_encoder.pkl` | CrossVal TargetEncoder (누수 방지) |")
    md += [
        "| `train_processed.csv` | 전처리된 훈련 데이터 |",
    ]
    if X_test_df is not None:
        md.append("| `test_processed.csv` | 전처리된 테스트 데이터 |")

    md += [
        "",
        "## 다음 단계 (모델링) 주의사항",
        f"- `preprocessor.pkl`을 `joblib.load`해서 새 데이터에 `.transform()`만 사용",
        f"- 불균형 비율 {ratio:.1f}:1 → " + (
            "SMOTE 적용 완료, 평가지표는 F1/AUC-ROC 사용"
            if use_smote else
            ("class_weight='balanced' 설정 권장" if ratio >= 3 else "Accuracy 사용 가능")
        ),
        "- Cross-validation 시 전처리(fit)를 fold 내부에서 수행하거나 이 파이프라인 재사용",
        "- Target Encoding은 OOF 방식으로 누수 방지됨 — 별도 재학습 불필요",
        "",
        "---",
        "*Generated by Kaggle Preprocessing Agent*",
    ]

    with open("preprocessing_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    ok("preprocessing_report.md 저장 완료")

    # 콘솔 요약
    log("\n  ⚙ 다음 단계 주의사항:")
    log(f"    - preprocessor.pkl → .transform() only (fit 금지)")
    if ratio >= 3 and not use_smote:
        log(f"    - class_weight='balanced' 설정 권장 (비율 {ratio:.1f}:1)")
    if use_smote:
        log(f"    - SMOTE 완료 — 평가지표는 F1 / AUC-ROC 사용")
    if high_card:
        log(f"    - Target Encoding OOF 적용 완료 (누수 없음)")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def run(train_path: str, target: str, test_path: str | None = None) -> None:
    log("=" * 60)
    log("  Kaggle Binary Classification Preprocessing Agent")
    log("=" * 60)

    train, test = step1_load(train_path, target, test_path)
    drop_cols, _, _ = step2_missing_plan(train, target)
    num_cols, low_card, high_card, id_cols = step3_classify_features(
        train, target, drop_cols
    )
    ratio, use_smote = step4_imbalance(train, target)
    X_train_df, X_test_df = step5_build_and_run(
        train, test, target,
        drop_cols, id_cols,
        num_cols, low_card, high_card,
        use_smote,
    )
    step6_save(
        X_train_df, X_test_df, target,
        ratio, use_smote,
        drop_cols, id_cols,
        num_cols, low_card, high_card,
    )

    log("\n" + "=" * 60)
    log("  전처리 완료!")
    log("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocessing_agent.py <train_csv> <target_col> [test_csv]")
        print("Example: python preprocessing_agent.py titanic.csv Survived")
        sys.exit(1)

    _train  = sys.argv[1]
    _target = sys.argv[2]
    _test   = sys.argv[3] if len(sys.argv) >= 4 else None

    run(_train, _target, _test)
