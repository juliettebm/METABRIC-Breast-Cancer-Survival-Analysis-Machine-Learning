import ast
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_python_and_metadata_files_are_valid():
    """Static smoke test that does not require the restricted Kaggle data."""
    for relative_path in ("train_model.py", "streamlit_app/app.py"):
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
    row = pd.DataFrame([bundle["defaults"]], columns=bundle["raw_feature_cols"])
    probability = bundle["model"].predict_proba(row)
    assert probability.shape == (1, 2)
    assert probability.sum() == pytest.approx(1.0)
    assert 0 <= probability[0, 1] <= 1
    assert len(bundle["feature_importances"]) == len(
        bundle["model"].named_steps["classifier"].feature_importances_
    )
