"""Leakage-free training entry point for the METABRIC 5-year mortality model."""

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
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_PATHS = [
    ROOT / "data" / "METABRIC_RNA_Mutation.csv",
    ROOT / "data" / "METABRIC_RNA_Mutation",
]


# ============================================================
# Feature groups
# ============================================================

NUMERIC = [
    "age_at_diagnosis",
    "tumor_size",
    "mutation_count",
    "lymph_nodes_examined_positive",
]

ORDINAL = [
    "cellularity",
    "neoplasm_histologic_grade",
]

ORDINAL_CATEGORIES = [
    ["Low", "Moderate", "High"],
    [1.0, 2.0, 3.0],
]

CATEGORICAL = [
    "er_status",
    "pr_status",
    "her2_status",
    "cancer_type",
    "type_of_breast_surgery",
    "inferred_menopausal_state",
    "cancer_type_detailed",
    "tumor_other_histologic_subtype",
    "pam50_+_claudin-low_subtype",
    "integrative_cluster",
]

BINARY = [
    "chemotherapy",
    "hormone_therapy",
    "radio_therapy",
]

NON_FEATURES = {
    "patient_id",
    "overall_survival_months",
    "overall_survival",
    "death_from_cancer",
    "tumor_stage",
    "3-gene_classifier_subtype",
    "primary_tumor_laterality",
    "er_status_measured_by_ihc",
    "her2_status_measured_by_snp6",
    "oncotree_code",

    # Cohort is a batch/period identifier rather than
    # an ordered biological quantity.
    "cohort",

    # NPI is derived from tumor size, grade and positive
    # lymph nodes, which are already represented directly.
    "nottingham_prognostic_index",
}


# ============================================================
# Data loading
# ============================================================

def load_raw() -> pd.DataFrame:
    """Load the raw METABRIC dataset."""

    try:
        path = next(
            path
            for path in RAW_PATHS
            if path.exists()
        )
    except StopIteration as exc:
        raise FileNotFoundError(
            "Place METABRIC_RNA_Mutation.csv in data/"
        ) from exc

    return pd.read_csv(
        path,
        low_memory=False,
    )


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare deterministic feature transformations.

    No statistics are learned from the complete dataset here.
    Data-dependent preprocessing is performed inside the
    scikit-learn pipeline.
    """

    data = df.copy()

    # Deterministic transformations
    data["tumor_size"] = np.log1p(
        data["tumor_size"]
    )

    data["mutation_count"] = np.log1p(
        data["mutation_count"]
    )

    # Mutation columns are converted to binary indicators
    mutation_cols = [
        c for c in data
        if c.endswith("_mut")
    ]

    data[mutation_cols] = data[mutation_cols].apply(
        lambda col: col.map(
            lambda value:
            0
            if pd.isna(value) or str(value) == "0"
            else 1
        )
    )

    clinical = (
        NUMERIC
        + ORDINAL
        + CATEGORICAL
        + BINARY
    )

    genomic = [
        c
        for c in data
        if c not in NON_FEATURES
        and c not in clinical
    ]

    return (
        data[clinical + genomic],
        genomic,
    )


# ============================================================
# Modeling pipeline
# ============================================================

def build_pipeline(
    genomic: list[str],
) -> Pipeline:
    """
    Build the complete preprocessing and Random Forest pipeline.

    All learned transformations are contained inside the
    pipeline so that they are refitted within each CV fold.
    """

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scale",
                            StandardScaler(),
                        ),
                    ]
                ),
                NUMERIC,
            ),
            (
                "ordinal",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "encode",
                            OrdinalEncoder(
                                categories=ORDINAL_CATEGORIES,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                ORDINAL,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
            (
                "binary",
                SimpleImputer(
                    strategy="most_frequent"
                ),
                BINARY,
            ),
            (
                "genomic",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "variance",
                            VarianceThreshold(
                                threshold=0.01
                            ),
                        ),
                        (
                            "select",
                            SelectKBest(
                                f_classif,
                                k=50,
                            ),
                        ),
                    ]
                ),
                genomic,
            ),
        ],
        verbose_feature_names_out=True,
    )

    return Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,

                    # Weighting is selected later by CV.
                    class_weight=None,

                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ============================================================
# Probability calibration utilities
# ============================================================

def _logit(probability) -> np.ndarray:
    """Convert probabilities to log-odds."""

    p = np.clip(
        np.asarray(
            probability,
            dtype=float,
        ),
        1e-4,
        1 - 1e-4,
    )

    return np.log(
        p / (1 - p)
    ).reshape(-1, 1)


def fit_platt(
    raw_probability,
    y,
) -> LogisticRegression:
    """
    Fit Platt scaling to predicted mortality probabilities.

    The calibration model is a logistic regression of the
    observed 5-year mortality outcome on the logit of the
    predicted mortality probability.
    """

    return LogisticRegression(
        C=1e6
    ).fit(
        _logit(raw_probability),
        y,
    )


def apply_platt(
    calibrator: LogisticRegression,
    raw_probability,
) -> np.ndarray:
    """Apply a fitted Platt calibration model."""

    return calibrator.predict_proba(
        _logit(raw_probability)
    )[:, 1]


def calibration_slope_intercept(
    y_true,
    mortality_probability,
) -> tuple[float, float]:
    """
    Estimate calibration slope and intercept for
    5-year mortality probabilities.

    Target coding:
    1 = death before 5 years
    0 = survived at least 5 years
    """

    y_array = np.asarray(y_true)

    fit = LogisticRegression(
        C=1e6
    ).fit(
        _logit(mortality_probability),
        y_array,
    )

    return (
        float(fit.coef_[0, 0]),
        float(fit.intercept_[0]),
    )


# ============================================================
# Bootstrap ROC-AUC
# ============================================================

def bootstrap_auc(
    y_true,
    probability,
    n_bootstraps: int = 2000,
) -> list[float]:
    """
    Return a reproducible percentile 95% bootstrap
    confidence interval for ROC-AUC.
    """

    rng = np.random.default_rng(42)

    y_array = np.asarray(y_true)
    p_array = np.asarray(probability)

    scores = []

    for _ in range(n_bootstraps):

        indices = rng.integers(
            0,
            len(y_array),
            len(y_array),
        )

        # ROC-AUC requires both classes
        if np.unique(
            y_array[indices]
        ).size == 2:

            scores.append(
                roc_auc_score(
                    y_array[indices],
                    p_array[indices],
                )
            )

    return np.quantile(
        scores,
        [0.025, 0.975],
    ).tolist()


# ============================================================
# Training
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    raw = load_raw()


    # --------------------------------------------------------
    # Define the fixed 5-year outcome
    #
    # Patients censored before 60 months have an unknown
    # 5-year outcome and are excluded.
    # --------------------------------------------------------

    unknown_5y = (
        (raw["overall_survival_months"] < 60)
        & (raw["overall_survival"] == 1)
    )

    cohort = raw.loc[
        ~unknown_5y
    ].copy()


    # --------------------------------------------------------
    # Target
    #
    # 1 = death observed before 5 years
    # 0 = survived at least 5 years
    # --------------------------------------------------------

    y = (
        (cohort["overall_survival_months"] < 60)
        & (cohort["overall_survival"] == 0)
    ).astype(int)


    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X, genomic = prepare_features(
        cohort
    )


    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )


    # --------------------------------------------------------
    # Base pipeline
    # --------------------------------------------------------

    base_model = build_pipeline(
        genomic
    )


    # --------------------------------------------------------
    # Nested cross-validation
    #
    # Inner CV:
    #   hyperparameter / class-weight selection
    #
    # Outer CV:
    #   performance estimation
    # --------------------------------------------------------

    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=41,
    )

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )


    search = GridSearchCV(
        base_model,
        {
            "classifier__min_samples_leaf": [
                1,
                3,
            ],
            "classifier__class_weight": [
                None,
                "balanced",
            ],
        },
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )


    # --------------------------------------------------------
    # Outer CV performance
    # --------------------------------------------------------

    cv_scores = cross_validate(
        search,
        X_train,
        y_train,
        cv=outer_cv,
        scoring={
            "roc_auc": "roc_auc",
            "brier": "neg_brier_score",
        },
        n_jobs=-1,
    )


    # --------------------------------------------------------
    # Fit selected Random Forest on complete training set
    # --------------------------------------------------------

    search.fit(
        X_train,
        y_train,
    )

    model = search.best_estimator_


    # --------------------------------------------------------
    # Held-out test predictions
    # --------------------------------------------------------

    mortality_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    prediction = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Optional Platt calibration
    #
    # The calibrator is learned exclusively from out-of-fold
    # predictions generated within the training set.
    # The test set is not used to fit the calibrator.
    # --------------------------------------------------------

    oof_mortality_probability = (
        cross_val_predict(
            clone(model),
            X_train,
            y_train,
            cv=StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=43,
            ),
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
    )

    calibrator = fit_platt(
        oof_mortality_probability,
        y_train,
    )

    calibrated_mortality_probability = (
        apply_platt(
            calibrator,
            mortality_probability,
        )
    )


    # --------------------------------------------------------
    # Calibration diagnostics
    # --------------------------------------------------------

    raw_slope, raw_intercept = (
        calibration_slope_intercept(
            y_test,
            mortality_probability,
        )
    )

    cal_slope, cal_intercept = (
        calibration_slope_intercept(
            y_test,
            calibrated_mortality_probability,
        )
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        prediction,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "target_definition": (
            "1 = death before 5 years; "
            "0 = survived at least 5 years"
        ),

        "test_patients": int(
            len(y_test)
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                mortality_probability,
            )
        ),

        "roc_auc_95_ci": bootstrap_auc(
            y_test,
            mortality_probability,
        ),

        "brier_score_uncalibrated": float(
            brier_score_loss(
                y_test,
                mortality_probability,
            )
        ),

        "brier_score_calibrated": float(
            brier_score_loss(
                y_test,
                calibrated_mortality_probability,
            )
        ),

        "calibration_slope_uncalibrated": (
            raw_slope
        ),

        "calibration_intercept_uncalibrated": (
            raw_intercept
        ),

        "calibration_slope_calibrated": (
            cal_slope
        ),

        "calibration_intercept_calibrated": (
            cal_intercept
        ),

        "mean_predicted_mortality": float(
            mortality_probability.mean()
        ),

        "mean_predicted_mortality_calibrated": float(
            calibrated_mortality_probability.mean()
        ),

        "observed_mortality": float(
            y_test.mean()
        ),

        "cv_roc_auc_mean": float(
            cv_scores[
                "test_roc_auc"
            ].mean()
        ),

        "cv_roc_auc_std": float(
            cv_scores[
                "test_roc_auc"
            ].std()
        ),

        "cv_brier_mean": float(
            -cv_scores[
                "test_brier"
            ].mean()
        ),

        "selected_parameters": (
            search.best_params_
        ),

        "accuracy": float(
            accuracy_score(
                y_test,
                prediction,
            )
        ),

        "death_recall": float(
            report["1"]["recall"]
        ),

        "death_precision": float(
            report["1"]["precision"]
        ),

        "survivor_recall": float(
            report["0"]["recall"]
        ),

        "classification_report": report,
    }


    # --------------------------------------------------------
    # Feature names and importances
    # --------------------------------------------------------

    names = (
        model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = (
        model
        .named_steps["classifier"]
        .feature_importances_
    )


    # --------------------------------------------------------
    # Default values used by the Streamlit application
    # --------------------------------------------------------

    defaults = {
        column: (
            X_train[column].median()
            if pd.api.types.is_numeric_dtype(
                X_train[column]
            )
            else X_train[column]
            .mode(dropna=True)
            .iloc[0]
        )
        for column in X_train
    }


    # --------------------------------------------------------
    # Model bundle
    # --------------------------------------------------------

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "raw_feature_cols": (
            X.columns.tolist()
        ),
        "defaults": defaults,
        "feature_importances": pd.Series(
            importances,
            index=names,
        ),
        "metrics": metrics,
        "target_definition": (
            "1 = death before 5 years; "
            "0 = survived at least 5 years"
        ),
    }


    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    joblib.dump(
        bundle,
        ROOT
        / "streamlit_app"
        / "rf_model.joblib",
    )

    joblib.dump(
        X.columns.tolist(),
        ROOT
        / "streamlit_app"
        / "feature_cols.joblib",
    )

    (
        ROOT
        / "streamlit_app"
        / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )


    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()