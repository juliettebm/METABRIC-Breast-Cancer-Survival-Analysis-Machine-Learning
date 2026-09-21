import ast
import json
from pathlib import Path
import subprocess

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_python_and_metadata_files_are_valid():
    """Static smoke test that does not require the restricted Kaggle data."""
    for relative_path in (
        "train_model.py",
        "src/metabric/training.py",
        "streamlit_app/app.py",
    ):
        path = ROOT / relative_path
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    metrics = json.loads((ROOT / "streamlit_app" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_patients"] > 0
    assert 0.5 < metrics["roc_auc"] <= 1
    assert len(metrics["roc_auc_95_ci"]) == 2


def test_model_bundle_predicts_from_raw_schema():
    bundle = joblib.load(ROOT / "streamlit_app" / "rf_model.joblib")
    required = {"model", "raw_feature_cols", "defaults", "feature_importances", "metrics"}
    assert required <= set(bundle)
    assert not {"cohort", "nottingham_prognostic_index"} & set(bundle["raw_feature_cols"])
    row = pd.DataFrame([bundle["defaults"]], columns=bundle["raw_feature_cols"])
    probability = bundle["model"].predict_proba(row)
    assert probability.shape == (1, 2)
    assert probability.sum() == pytest.approx(1.0)
    assert 0 <= probability[0, 1] <= 1
    assert len(bundle["feature_importances"]) == len(
        bundle["model"].named_steps["classifier"].feature_importances_
    )


def test_readme_matches_retrained_metrics():
    """Keep recruiter-facing claims synchronized with the deployed artifact."""
    metrics = json.loads((ROOT / "streamlit_app" / "metrics.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lower, upper = metrics["roc_auc_95_ci"]
    expected = (
        f"AUC is {metrics['roc_auc']:.3f} "
        f"(bootstrap 95% CI {lower:.3f}–{upper:.3f}), "
        f"Brier score is {metrics['brier_score']:.3f}"
    )
    assert expected in readme
    report = metrics["classification_report"]
    table_row = (
        f"| **Random Forest (deployed, tuned)** | **{metrics['roc_auc']:.3f}** | "
        f"**{metrics['accuracy']:.0%}** | **{report['1']['recall']:.0%}** | "
        f"**{report['0']['recall']:.0%}** |"
    )
    assert table_row in readme

    benchmark = json.loads(
        (ROOT / "streamlit_app" / "benchmark_metrics.json").read_text(encoding="utf-8")
    )
    xgb = next(row for row in benchmark if row["model"] == "XGBoost")
    xgb_row = (
        f"| XGBoost (benchmark) | {xgb['test_auc']:.3f} | "
        f"{xgb['accuracy']:.0%} | {xgb['survivor_recall']:.0%} | "
        f"{xgb['death_recall']:.0%} |"
    )
    assert xgb_row in readme


def test_third_party_data_is_not_tracked():
    tracked = subprocess.check_output(
        ["git", "ls-files", "data"], cwd=ROOT, text=True
    ).strip()
    assert tracked == ""
