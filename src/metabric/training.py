"""Leakage-free training entry point for the METABRIC 5-year model."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
RAW_PATHS = [ROOT / "data" / "METABRIC_RNA_Mutation.csv", ROOT / "data" / "METABRIC_RNA_Mutation"]

NUMERIC = [
    "age_at_diagnosis", "tumor_size", "mutation_count",
    "lymph_nodes_examined_positive",
]
ORDINAL = ["cellularity", "neoplasm_histologic_grade"]
ORDINAL_CATEGORIES = [["Low", "Moderate", "High"], [1.0, 2.0, 3.0]]
CATEGORICAL = [
    "er_status", "pr_status", "her2_status", "cancer_type",
    "type_of_breast_surgery", "inferred_menopausal_state",
    "cancer_type_detailed", "tumor_other_histologic_subtype",
    "pam50_+_claudin-low_subtype", "integrative_cluster",
]
BINARY = ["chemotherapy", "hormone_therapy", "radio_therapy"]
NON_FEATURES = {
    "patient_id", "overall_survival_months", "overall_survival", "death_from_cancer",
    "tumor_stage", "3-gene_classifier_subtype", "primary_tumor_laterality",
    "er_status_measured_by_ihc", "her2_status_measured_by_snp6", "oncotree_code",
    # Cohort is a batch/period identifier, not an ordered biological quantity.
    # NPI is derived from tumor size, grade and positive lymph nodes, which are
    # already represented directly; excluding it avoids deterministic overlap.
    "cohort", "nottingham_prognostic_index",
}


def load_raw() -> pd.DataFrame:
    try:
        path = next(path for path in RAW_PATHS if path.exists())
    except StopIteration as exc:
        raise FileNotFoundError("Place METABRIC_RNA_Mutation.csv in data/") from exc
    return pd.read_csv(path, low_memory=False)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()
    data["tumor_size"] = np.log1p(data["tumor_size"])
    data["mutation_count"] = np.log1p(data["mutation_count"])
    mutation_cols = [c for c in data if c.endswith("_mut")]
    data[mutation_cols] = data[mutation_cols].apply(
        lambda col: col.map(lambda value: 0 if pd.isna(value) or str(value) == "0" else 1)
    )
    clinical = NUMERIC + ORDINAL + CATEGORICAL + BINARY
    genomic = [c for c in data if c not in NON_FEATURES and c not in clinical]
    return data[clinical + genomic], genomic


def build_pipeline(genomic: list[str]) -> Pipeline:
    clinical = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
            ("ordinal", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OrdinalEncoder(categories=ORDINAL_CATEGORIES, handle_unknown="use_encoded_value", unknown_value=-1))]), ORDINAL),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
            ("binary", SimpleImputer(strategy="most_frequent"), BINARY),
            ("genomic", Pipeline([("imputer", SimpleImputer(strategy="median")), ("variance", VarianceThreshold(threshold=0.01)), ("select", SelectKBest(f_classif, k=50))]), genomic),
        ],
        verbose_feature_names_out=True,
    )
    return Pipeline([
        ("preprocessor", clinical),
        ("classifier", RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)),
    ])


def _logit(probability) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_platt(raw_probability, y) -> LogisticRegression:
    """Platt scaling: logistic regression of the outcome on the logit of the raw probability.

    The map is strictly monotone, so ranking metrics such as ROC-AUC are unchanged.
    """
    return LogisticRegression(C=1e6).fit(_logit(raw_probability), y)


def apply_platt(calibrator: LogisticRegression, raw_probability) -> np.ndarray:
    return calibrator.predict_proba(_logit(raw_probability))[:, 1]


def calibration_slope_intercept(y_true, survival_probability) -> tuple[float, float]:
    """Slope and intercept of the outcome (death) on the logit of the predicted death probability."""
    death = (np.asarray(y_true) == 0).astype(int)
    fit = LogisticRegression(C=1e6).fit(_logit(1 - np.asarray(survival_probability)), death)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def bootstrap_auc(y_true, probability, n_bootstraps: int = 2000) -> list[float]:
    """Return a reproducible percentile 95% bootstrap interval for ROC-AUC."""
    rng = np.random.default_rng(42)
    y_array, p_array = np.asarray(y_true), np.asarray(probability)
    scores = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(y_array), len(y_array))
        if np.unique(y_array[indices]).size == 2:
            scores.append(roc_auc_score(y_array[indices], p_array[indices]))
    return np.quantile(scores, [0.025, 0.975]).tolist()


def main() -> None:
    raw = load_raw()
    known = ~((raw["overall_survival_months"] < 60) & (raw["overall_survival"] == 1))
    cohort = raw.loc[known].copy()
    y = (cohort["overall_survival_months"] >= 60).astype(int)
    X, genomic = prepare_features(cohort)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_model = build_pipeline(genomic)

    # Nested CV: the inner loop selects the RF regularisation while the outer
    # loop estimates generalisation. Every candidate contains preprocessing,
    # so all transformations are refitted without seeing the validation fold.
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=41)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        base_model,
        {"classifier__min_samples_leaf": [1, 3]},
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )
    cv_scores = cross_validate(
        search, X_train, y_train, cv=outer_cv,
        scoring={"roc_auc": "roc_auc", "brier": "neg_brier_score"}, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    probability = model.predict_proba(X_test)[:, 1]
    prediction = model.predict(X_test)

    # The class-weighted forest ranks patients well but its raw probabilities are
    # biased (deaths over-predicted). A Platt map is fitted on out-of-fold training
    # predictions only, so the test set stays untouched for the final evaluation.
    oof_probability = cross_val_predict(
        clone(model), X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=43),
        method="predict_proba", n_jobs=-1,
    )[:, 1]
    calibrator = fit_platt(oof_probability, y_train)
    calibrated = apply_platt(calibrator, probability)
    raw_slope, raw_intercept = calibration_slope_intercept(y_test, probability)
    cal_slope, cal_intercept = calibration_slope_intercept(y_test, calibrated)

    metrics = {
        "test_patients": int(len(y_test)),
        # AUC is identical for raw and calibrated probabilities (monotone map).
        "roc_auc": float(roc_auc_score(y_test, calibrated)),
        "roc_auc_95_ci": bootstrap_auc(y_test, calibrated),
        # Brier score of the probability shown to users (calibrated).
        "brier_score": float(brier_score_loss(y_test, calibrated)),
        "brier_score_uncalibrated": float(brier_score_loss(y_test, probability)),
        "calibration_slope": cal_slope,
        "calibration_intercept": cal_intercept,
        "calibration_slope_uncalibrated": raw_slope,
        "calibration_intercept_uncalibrated": raw_intercept,
        "mean_predicted_survival": float(calibrated.mean()),
        "mean_predicted_survival_uncalibrated": float(probability.mean()),
        "observed_survival": float(y_test.mean()),
        "cv_roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
        "cv_roc_auc_std": float(cv_scores["test_roc_auc"].std()),
        "cv_brier_mean": float(-cv_scores["test_brier"].mean()),
        "selected_parameters": search.best_params_,
        "accuracy": float(accuracy_score(y_test, prediction)),
        # Class predictions come from the uncalibrated class-weighted forest (default 0.5 rule).
        "classification_report": classification_report(y_test, prediction, output_dict=True),
    }

    names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    defaults = {
        column: (X_train[column].median() if pd.api.types.is_numeric_dtype(X_train[column])
                 else X_train[column].mode(dropna=True).iloc[0])
        for column in X_train
    }
    bundle = {
        "model": model,
        "calibrator": calibrator,
        "raw_feature_cols": X.columns.tolist(),
        "defaults": defaults,
        "feature_importances": pd.Series(importances, index=names),
        "metrics": metrics,
    }
    joblib.dump(bundle, ROOT / "streamlit_app" / "rf_model.joblib")
    joblib.dump(X.columns.tolist(), ROOT / "streamlit_app" / "feature_cols.joblib")
    (ROOT / "streamlit_app" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
