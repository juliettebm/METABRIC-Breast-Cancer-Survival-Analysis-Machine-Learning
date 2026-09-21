"""Regenerate ``data/processed`` from the raw METABRIC file (same logic as notebook 02).

Run ``python prepare_data.py`` from the repository root. The processed frames are used by
the descriptive notebooks (03, 04) and by the Kaplan-Meier page of the Streamlit app. The
predictive model does not use them: ``train_model.py`` starts from the raw file.

Note: the cohort-wide preprocessing and the top-25 gene selection below are descriptive
only (the selection uses survival time on the full cohort, so it must not be used for
confirmatory p-values or predictive evaluation).
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .training import ROOT, load_raw

TOP_N_SURVIVAL = 25

COLS_TO_DROP = [
    "tumor_stage", "3-gene_classifier_subtype", "primary_tumor_laterality",
    "er_status_measured_by_ihc", "her2_status_measured_by_snp6", "patient_id",
    "overall_survival", "death_from_cancer", "oncotree_code",
]
TARGET_COLS = ["overall_survival_months", "event"]
TREATMENT_COLS = ["chemotherapy", "hormone_therapy", "radio_therapy"]
BIOMARKER_COLS = ["er_status", "pr_status", "her2_status"]
CLINICAL_COLS = [
    "age_at_diagnosis", "type_of_breast_surgery", "cancer_type",
    "cancer_type_detailed", "cellularity", "cohort",
    "neoplasm_histologic_grade", "tumor_other_histologic_subtype",
    "inferred_menopausal_state", "integrative_cluster",
    "lymph_nodes_examined_positive", "mutation_count",
    "nottingham_prognostic_index", "pam50_+_claudin-low_subtype", "tumor_size",
]

NUM_COLS = ["tumor_size", "mutation_count"]
SCALE_COLS = ["age_at_diagnosis", "nottingham_prognostic_index", "lymph_nodes_examined_positive", "cohort"]
ORD_COLS = ["cellularity", "neoplasm_histologic_grade"]
ORD_CATEGORIES = [["Low", "Moderate", "High"], [1.0, 2.0, 3.0]]
BINARY_COLS = [
    "er_status", "pr_status", "her2_status", "cancer_type",
    "type_of_breast_surgery", "inferred_menopausal_state",
]
OHE_COLS = [
    "cancer_type_detailed", "tumor_other_histologic_subtype",
    "pam50_+_claudin-low_subtype", "integrative_cluster",
]


def build_processed_frames(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return ``(df_survival, df_ml, feature_groups)`` exactly as notebook 02 exports them."""
    df = raw.copy()
    df["event"] = 1 - df["overall_survival"]  # the dataset encodes 1 = alive
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])

    mut_cols = [c for c in df.columns if c.endswith("_mut")]
    assigned = TARGET_COLS + TREATMENT_COLS + BIOMARKER_COLS + mut_cols + CLINICAL_COLS
    expr_cols = [c for c in df.columns if c not in assigned]

    df["mutation_count"] = np.log1p(df["mutation_count"])
    df["tumor_size"] = np.log1p(df["tumor_size"])
    df[mut_cols] = df[mut_cols].apply(lambda col: col.map(lambda x: 0 if x == "0" else 1))

    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUM_COLS),
            ("scale", Pipeline([("scaler", StandardScaler())]), SCALE_COLS),
            ("ord", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OrdinalEncoder(categories=ORD_CATEGORIES))]), ORD_COLS),
            ("binary", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("encoder", OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore"))]), BINARY_COLS),
            ("ohe", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]), OHE_COLS),
        ],
        remainder="passthrough",
    )
    df_clean = pd.DataFrame(
        preprocessor.fit_transform(df), columns=preprocessor.get_feature_names_out(), index=df.index
    )
    df_clean.columns = [c.split("__")[-1] for c in df_clean.columns]

    mut_clean = [c for c in df_clean.columns if c.endswith("_mut")]
    expr_clean = [c for c in df_clean.columns if c in expr_cols]
    clinical_clean = [
        c for c in df_clean.columns
        if c not in TARGET_COLS and c not in mut_clean and c not in expr_clean
    ]
    genomic_clean = mut_clean + expr_clean

    # Label-free filter, then descriptive selection by correlation with survival time.
    selector = VarianceThreshold(threshold=0.01).fit(df_clean[genomic_clean])
    cols_var = df_clean[genomic_clean].columns[selector.get_support()].tolist()
    correlations = df_clean[cols_var].apply(lambda x: x.corr(df_clean["overall_survival_months"]))
    top_genes = correlations.abs().sort_values(ascending=False).head(TOP_N_SURVIVAL).index.tolist()

    df_survival = df_clean[clinical_clean + top_genes + TARGET_COLS].copy()
    df_ml = df_clean[clinical_clean + cols_var + TARGET_COLS].copy()
    return df_survival, df_ml, {"clinical": clinical_clean, "genes": cols_var}


def main(output_dir: Path | None = None) -> None:
    output_dir = output_dir or ROOT / "data" / "processed"
    df_survival, df_ml, groups = build_processed_frames(load_raw())
    output_dir.mkdir(parents=True, exist_ok=True)
    df_survival.to_csv(output_dir / "df_survival.csv", index=False)
    df_ml.to_csv(output_dir / "df_ml.csv", index=False)
    (output_dir / "feature_groups.json").write_text(json.dumps(groups), encoding="utf-8")
    print(f"df_survival {df_survival.shape}, df_ml {df_ml.shape} written to {output_dir}")


if __name__ == "__main__":
    main()
