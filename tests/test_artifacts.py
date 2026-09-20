from pathlib import Path

import joblib
import pandas as pd

import train_model


ROOT = Path(__file__).resolve().parents[1]


def test_raw_data_loads():
    data = train_model.load_raw()
    assert not data.empty
    assert {"overall_survival_months", "overall_survival"} <= set(data.columns)


def test_model_bundle_predicts_from_raw_schema():
    bundle = joblib.load(ROOT / "streamlit_app" / "rf_model.joblib")
    required = {"model", "raw_feature_cols", "defaults", "feature_importances", "metrics"}
    assert required <= set(bundle)
    row = pd.DataFrame([bundle["defaults"]], columns=bundle["raw_feature_cols"])
    probability = bundle["model"].predict_proba(row)
    assert probability.shape == (1, 2)
    assert 0 <= probability[0, 1] <= 1
