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


def _robustness():
    return json.loads((ROOT / "reports" / "robustness_metrics.json").read_text(encoding="utf-8"))


def test_permutation_test_shows_no_leakage():
    """With shuffled labels the pipeline must fall back to chance level."""
    result = _robustness()["permutation_test"]
    assert 0.40 < result["permuted_auc_mean"] < 0.60
    assert result["real_auc"] > result["permuted_auc_max"]


def test_readme_matches_robustness_metrics():
    robustness = _robustness()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    perm = robustness["permutation_test"]
    assert f"Permuted AUC averages {perm['permuted_auc_mean']:.3f}" in readme
    split = robustness["split_variability"]
    assert f"AUC {split['auc_mean']:.3f} ± {split['auc_std']:.3f}" in readme
    cox = robustness["cox"]
    assert f"Apparent {cox['apparent_c_index']:.3f}; 5-fold cross-validated {cox['cv_c_index_mean']:.3f}" in readme
    raw_rf = robustness["calibration_comparison"]["models"]["random_forest"]
    assert f"calibration intercept {raw_rf['intercept_raw']['mean']:.2f}" in readme
    calibration = robustness["threshold_and_calibration"]
    for row in calibration["thresholds"]:
        assert f"{row['death_recall']:.1%}" in readme


def test_readme_matches_model_comparison():
    comparison = _robustness()["model_comparison"]["metrics"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    auc = comparison["auc"]
    assert (
        f"| ROC-AUC | {auc['random_forest_mean']:.3f} | {auc['xgboost_mean']:.3f} |" in readme
    )
    brier = comparison["brier"]
    assert f"| Brier score (lower is better) | {brier['random_forest_mean']:.3f} | {brier['xgboost_mean']:.3f} |" in readme


def test_deployed_probabilities_are_calibrated():
    """The app shows a probability: it must be calibrated, not the raw class-weighted output."""
    bundle = joblib.load(ROOT / "streamlit_app" / "rf_model.joblib")
    assert "calibrator" in bundle
    metrics = bundle["metrics"]
    assert abs(metrics["mean_predicted_survival"] - metrics["observed_survival"]) < 0.03
    assert 0.8 < metrics["calibration_slope"] < 1.25
    assert abs(metrics["calibration_intercept"]) < 0.25
    assert metrics["brier_score"] < metrics["brier_score_uncalibrated"]


def test_readme_matches_calibration_comparison():
    calibration = _robustness()["calibration_comparison"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    rf, xgb = calibration["models"]["random_forest"], calibration["models"]["xgboost"]
    row = (
        f"| Platt | {rf['brier_platt']['mean']:.3f} ± {rf['brier_platt']['std']:.3f} | "
        f"{xgb['brier_platt']['mean']:.3f} ± {xgb['brier_platt']['std']:.3f} |"
    )
    assert row in readme


def test_readme_matches_survival_model_comparison():
    survival = _robustness()["survival_model_comparison"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    auc, cindex = survival["auc_5y"], survival["cindex"]
    assert f"{auc['classifier_mean']:.3f} ± {auc['classifier_std']:.3f} | {auc['cox_mean']:.3f} ± {auc['cox_std']:.3f}" in readme
    assert f"{cindex['classifier_mean']:.3f} ± {cindex['classifier_std']:.3f} | {cindex['cox_mean']:.3f} ± {cindex['cox_std']:.3f}" in readme
