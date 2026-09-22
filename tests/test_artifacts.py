import ast
import sys
import json
from pathlib import Path
import subprocess

import numpy as np
import joblib
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))


def test_python_and_metadata_files_are_valid():
    """Static smoke test that does not require the restricted Kaggle data."""

    for relative_path in (
        "train_model.py",
        "prepare_data.py",
        "robustness_checks.py",
        "src/metabric/training.py",
        "src/metabric/prepare_data.py",
        "src/metabric/robustness.py",
        "streamlit_app/app.py",
    ):
        path = ROOT / relative_path
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    metrics = json.loads(
        (ROOT / "streamlit_app" / "metrics.json").read_text(encoding="utf-8")
    )

    assert metrics["test_patients"] > 0
    assert 0.5 < metrics["roc_auc"] <= 1
    assert len(metrics["roc_auc_95_ci"]) == 2


def test_model_bundle_predicts_from_raw_schema():
    bundle = joblib.load(
        ROOT / "streamlit_app" / "rf_model.joblib"
    )

    required = {
        "model",
        "raw_feature_cols",
        "defaults",
        "feature_importances",
        "metrics",
    }

    assert required <= set(bundle)

    assert not {
        "cohort",
        "nottingham_prognostic_index",
    } & set(bundle["raw_feature_cols"])

    row = pd.DataFrame(
        [bundle["defaults"]],
        columns=bundle["raw_feature_cols"],
    )

    probability = bundle["model"].predict_proba(row)

    assert probability.shape == (1, 2)
    assert probability.sum() == pytest.approx(1.0)
    assert 0 <= probability[0, 1] <= 1

    assert len(bundle["feature_importances"]) == len(
        bundle["model"].named_steps[
            "classifier"
        ].feature_importances_
    )


def test_readme_matches_retrained_metrics():
    """Keep recruiter-facing claims synchronized with the deployed artifact."""

    metrics = json.loads(
        (ROOT / "streamlit_app" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )

    readme = (ROOT / "README.md").read_text(
        encoding="utf-8"
    )

    lower, upper = metrics["roc_auc_95_ci"]

    expected = (
        f"AUC is {metrics['roc_auc']:.3f} "
        f"(bootstrap 95% CI {lower:.3f}–{upper:.3f}), "
        f"Brier score is "
        f"{metrics['brier_score_calibrated']:.3f}"
    )

    assert expected in readme

    report = metrics["classification_report"]

    table_row = (
        f"| **Random Forest (deployed, tuned)** | "
        f"**{metrics['roc_auc']:.3f}** | "
        f"**{metrics['accuracy']:.0%}** | "
        f"**{report['1']['recall']:.0%}** | "
        f"**{report['0']['recall']:.0%}** |"
    )

    assert table_row in readme

    benchmark = json.loads(
        (
            ROOT
            / "streamlit_app"
            / "benchmark_metrics.json"
        ).read_text(encoding="utf-8")
    )

    xgb = next(
        row
        for row in benchmark
        if row["model"] == "XGBoost"
    )

    # README columns:
    # Model | ROC-AUC | Accuracy | Death recall | Survivor recall
    xgb_row = (
        f"| XGBoost (benchmark) | "
        f"{xgb['test_auc']:.3f} | "
        f"{xgb['accuracy']:.0%} | "
        f"{xgb['death_recall']:.0%} | "
        f"{xgb['survivor_recall']:.0%} |"
    )

    assert xgb_row in readme


def test_third_party_data_is_not_tracked():
    tracked = subprocess.check_output(
        ["git", "ls-files", "data"],
        cwd=ROOT,
        text=True,
    ).strip()

    assert tracked == ""


def _robustness():
    return json.loads(
        (
            ROOT
            / "reports"
            / "robustness_metrics.json"
        ).read_text(encoding="utf-8")
    )


def test_permutation_test_shows_no_leakage():
    """With shuffled labels the pipeline must fall back to chance level."""

    result = _robustness()["permutation_test"]

    assert 0.40 < result["permuted_auc_mean"] < 0.60
    assert result["real_auc"] > result["permuted_auc_max"]


def test_readme_matches_robustness_metrics():
    robustness = _robustness()

    readme = (ROOT / "README.md").read_text(
        encoding="utf-8"
    )

    perm = robustness["permutation_test"]

    assert f"{perm['real_auc']:.3f}" in readme
    assert f"{perm['permuted_auc_mean']:.3f}" in readme

    split = robustness["split_variability"]

    assert (
        f"{split['auc_mean']:.3f} ± "
        f"{split['auc_std']:.3f}"
    ) in readme

    assert f"{split['auc_min']:.3f}" in readme
    assert f"{split['auc_max']:.3f}" in readme

    cox = robustness["cox"]

    assert f"{cox['apparent_c_index']:.3f}" in readme
    assert f"{cox['cv_c_index_mean']:.3f}" in readme

    calibration = robustness[
        "threshold_and_calibration"
    ]

    for row in calibration["thresholds"]:
        assert f"{row['death_recall']:.1%}" in readme


def test_readme_matches_model_comparison():
    comparison = _robustness()[
        "model_comparison"
    ]["metrics"]

    readme = (ROOT / "README.md").read_text(
        encoding="utf-8"
    )

    auc = comparison["auc"]

    assert (
        f"{auc['random_forest_mean']:.3f}"
    ) in readme

    assert (
        f"{auc['xgboost_mean']:.3f}"
    ) in readme

    brier = comparison["brier"]

    assert (
        f"{brier['random_forest_mean']:.3f}"
    ) in readme

    assert (
        f"{brier['xgboost_mean']:.3f}"
    ) in readme

    matched = comparison[
        "recall_flagging_observed_mortality_share"
    ]

    assert (
        f"{matched['random_forest_mean']:.1%}"
    ) in readme

    assert (
        f"{matched['xgboost_mean']:.1%}"
    ) in readme


def test_deployed_probabilities_are_calibrated():
    """The app shows a calibrated mortality probability."""

    bundle = joblib.load(
        ROOT / "streamlit_app" / "rf_model.joblib"
    )

    assert "calibrator" in bundle

    metrics = bundle["metrics"]

    # Calibration metrics must be valid finite values.
    assert np.isfinite(
        metrics["calibration_slope_calibrated"]
    )

    assert np.isfinite(
        metrics["calibration_intercept_calibrated"]
    )

    # Predicted and observed mortality must be
    # valid probabilities.
    assert (
        0.0
        <= metrics[
            "mean_predicted_mortality_calibrated"
        ]
        <= 1.0
    )

    assert (
        0.0
        <= metrics["observed_mortality"]
        <= 1.0
    )

    # Mean calibrated mortality should remain
    # close to observed mortality.
    assert abs(
        metrics[
            "mean_predicted_mortality_calibrated"
        ]
        - metrics["observed_mortality"]
    ) < 0.03

    # Calibration should improve the Brier score
    # on this artifact.
    assert (
        metrics["brier_score_calibrated"]
        < metrics["brier_score_uncalibrated"]
    )


def test_readme_matches_calibration_comparison():
    calibration = _robustness()[
        "calibration_comparison"
    ]

    readme = (ROOT / "README.md").read_text(
        encoding="utf-8"
    )

    rf = calibration["models"]["random_forest"]
    xgb = calibration["models"]["xgboost"]

    assert (
        f"{rf['brier_raw']['mean']:.4f}"
    ) in readme

    assert (
        f"{rf['brier_platt']['mean']:.4f}"
    ) in readme

    assert (
        f"{rf['slope_raw']['mean']:.3f}"
    ) in readme

    assert (
        f"{rf['slope_platt']['mean']:.3f}"
    ) in readme

    assert (
        f"{xgb['brier_raw']['mean']:.4f}"
    ) in readme

    assert (
        f"{xgb['brier_platt']['mean']:.4f}"
    ) in readme

    assert (
        f"{xgb['slope_raw']['mean']:.3f}"
    ) in readme

    assert (
        f"{xgb['slope_platt']['mean']:.3f}"
    ) in readme


def test_readme_matches_survival_model_comparison():
    survival = _robustness()[
        "survival_model_comparison"
    ]

    readme = (ROOT / "README.md").read_text(
        encoding="utf-8"
    )

    auc = survival["auc_5y"]
    cindex = survival["cindex"]
    n_splits = survival["n_splits"]

    assert (
        f"{auc['classifier_mean']:.3f} ± "
        f"{auc['classifier_std']:.3f}"
    ) in readme

    assert (
        f"{auc['cox_mean']:.3f} ± "
        f"{auc['cox_std']:.3f}"
    ) in readme

    assert (
        f"{cindex['classifier_mean']:.3f} ± "
        f"{cindex['classifier_std']:.3f}"
    ) in readme

    assert (
        f"{cindex['cox_mean']:.3f} ± "
        f"{cindex['cox_std']:.3f}"
    ) in readme

    classifier_better_auc = (
        n_splits - auc["cox_better_in_splits"]
    )

    assert (
        f"{classifier_better_auc} of "
        f"{n_splits} splits"
    ) in readme

    assert (
        f"{cindex['cox_better_in_splits']} of "
        f"{n_splits} splits"
    ) in readme


def test_readme_matches_er_over_time():
    er = _robustness()["er_over_time"]

    readme = (ROOT / "README.md").read_text(
        encoding="utf-8"
    )

    low, high = er[
        "bootstrap_crossing_95_interval"
    ]

    assert (
        f"{er['crossing_months_point_estimate']} "
        f"months"
    ) in readme

    assert (
        f"{low:.0f}–{high:.0f} months"
    ) in readme

    first = er["hazard_ratio_by_period"][0]
    last = er["hazard_ratio_by_period"][-1]

    assert (
        f"{first['hr_adjusted']:.2f}"
    ) in readme

    assert (
        f"{last['hr_adjusted']:.2f}"
    ) in readme

    # Check the time-varying pattern numerically:
    # protective early, adverse late.
    assert (
        first["hr_adjusted"]
        < 1
        < last["hr_adjusted"]
    )


def test_processed_data_can_be_regenerated_from_raw():
    """Local-only: needs the raw Kaggle file, which is not versioned."""

    from metabric.prepare_data import (
        TOP_N_SURVIVAL,
        build_processed_frames,
    )
    from metabric.training import load_raw

    try:
        raw = load_raw()
    except FileNotFoundError:
        pytest.skip(
            "raw METABRIC file not available"
        )

    df_survival, df_ml, groups = (
        build_processed_frames(raw)
    )

    assert (
        len(df_survival)
        == len(df_ml)
        == len(raw)
    )

    assert (
        df_survival.shape[1]
        == len(groups["clinical"])
        + TOP_N_SURVIVAL
        + 2
    )

    assert (
        df_ml.shape[1]
        == len(groups["clinical"])
        + len(groups["genes"])
        + 2
    )


def test_app_starts_and_prediction_page_works():
    """The model page must render whether or not the local data files exist."""

    pytest.importorskip("plotly")
    pytest.importorskip("lifelines")

    app_test = pytest.importorskip(
        "streamlit.testing.v1"
    )

    app = app_test.AppTest.from_file(
        str(
            ROOT
            / "streamlit_app"
            / "app.py"
        ),
        default_timeout=120,
    ).run()

    assert not app.exception

    app.sidebar.radio[0].set_value(
        "🤖 ML Prediction"
    ).run()

    assert not app.exception