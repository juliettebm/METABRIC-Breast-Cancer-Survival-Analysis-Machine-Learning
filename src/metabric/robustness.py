"""Robustness diagnostics answering the usual methodological objections.

Run with ``python -m metabric.robustness`` (from ``src``) or
``python robustness_checks.py``. The deployed model is not modified: every check
refits the same leakage-free pipeline (``min_samples_leaf`` fixed to the value
selected by ``training.py``) and results are written to
``reports/robustness_metrics.json`` plus two figures.

Checks
1. Label-permutation test: with shuffled labels the whole pipeline must fall to
   AUC ~ 0.5, otherwise some information leaks into the evaluation.
2. Decision-threshold table (recall / precision of deaths) and calibration curve.
3. Selection bias: patients censored before 60 months (excluded) vs kept.
4. Split variability: AUC over repeated stratified train/test splits.
5. Horizon sensitivity: same protocol with a 120-month horizon.
6. Cox: cross-validated C-index and ER-stratified fit (ER violates proportional hazards).
7. Model comparison: Random Forest vs XGBoost at matched recall / matched flagged share
   over repeated splits (the default-threshold death recall alone is threshold-driven).
8. Calibration: slope/intercept and Brier of both models before and after Platt / isotonic
   recalibration fitted on a calibration fold held out from the training data.
9. Survival model: penalised Cox trained on ALL patients (censored ones included) compared with
   the 5-year classifier on the same test patients (AUC at 5 years, C-index).
"""

from pathlib import Path
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .training import ROOT, apply_platt, build_pipeline, fit_platt, load_raw, prepare_features

SEED = 42
MIN_SAMPLES_LEAF = 3  # value selected by the nested search in training.py
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6)


def horizon_cohort(raw: pd.DataFrame, months: int):
    """Patients whose status at ``months`` is observable, with 1 = alive at horizon."""
    known = ~((raw["overall_survival_months"] < months) & (raw["overall_survival"] == 1))
    cohort = raw.loc[known].copy()
    y = (cohort["overall_survival_months"] >= months).astype(int)
    X, genomic = prepare_features(cohort)
    return X, y, genomic


def fresh_pipeline(genomic, n_estimators: int = 500):
    model = build_pipeline(genomic)
    model.set_params(classifier__min_samples_leaf=MIN_SAMPLES_LEAF, classifier__n_estimators=n_estimators)
    return model


def permutation_test(X_train, X_test, y_train, y_test, genomic, n_permutations: int = 20) -> dict:
    rng = np.random.default_rng(SEED)
    real = clone(fresh_pipeline(genomic)).fit(X_train, y_train)
    real_auc = roc_auc_score(y_test, real.predict_proba(X_test)[:, 1])
    null = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(y_train.to_numpy())
        model = clone(fresh_pipeline(genomic, n_estimators=200)).fit(X_train, shuffled)
        null.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    null = np.asarray(null)
    return {
        "real_auc": float(real_auc),
        "n_permutations": n_permutations,
        "permuted_auc_mean": float(null.mean()),
        "permuted_auc_min": float(null.min()),
        "permuted_auc_max": float(null.max()),
        "empirical_p_value": float((1 + (null >= real_auc).sum()) / (1 + n_permutations)),
    }, real


def threshold_and_calibration(model, X_train, y_train, X_test, y_test, figure_dir: Path) -> dict:
    """Threshold table and calibration of the raw forest, plus the deployed Platt recalibration."""
    p_survive = model.predict_proba(X_test)[:, 1]
    death = (y_test.to_numpy() == 0).astype(int)
    p_death = 1 - p_survive
    rows = []
    for threshold in THRESHOLDS:
        flagged = (p_death >= threshold).astype(int)  # predicted death
        rows.append({
            "death_probability_threshold": threshold,
            "death_recall": float(recall_score(death, flagged)),
            "death_precision": float(precision_score(death, flagged, zero_division=0)),
            "flagged_share": float(flagged.mean()),
        })

    frac_pos, mean_pred = calibration_curve(death, p_death, n_bins=8, strategy="quantile")
    logit = np.log(np.clip(p_death, 1e-4, 1 - 1e-4) / (1 - np.clip(p_death, 1e-4, 1 - 1e-4)))
    # Calibration slope/intercept via a logistic fit of the outcome on the logit.
    from sklearn.linear_model import LogisticRegression
    fit = LogisticRegression(C=1e6).fit(logit.reshape(-1, 1), death)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", label="Random Forest, raw")
    oof = cross_val_predict(
        clone(model), X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=43),
        method="predict_proba", n_jobs=-1,
    )[:, 1]
    calibrated = apply_platt(fit_platt(oof, y_train), p_survive)
    frac_cal, mean_cal = calibration_curve(death, 1 - calibrated, n_bins=8, strategy="quantile")
    ax.plot(mean_cal, frac_cal, "s-", label="Random Forest, Platt-recalibrated")
    ax.set_xlabel("Predicted probability of death before 5 years")
    ax.set_ylabel("Observed share of deaths")
    ax.set_title("Calibration curve (369 test patients)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "06_calibration.png", dpi=150)
    plt.close(fig)
    return {
        "thresholds": rows,
        "calibration_slope": float(fit.coef_[0, 0]),
        "calibration_intercept": float(fit.intercept_[0]),
        "recalibrated_brier": float(brier_score_loss(y_test, calibrated)),
    }


def selection_bias(raw: pd.DataFrame, months: int = 60) -> dict:
    excluded = (raw["overall_survival_months"] < months) & (raw["overall_survival"] == 1)
    grade = pd.to_numeric(raw["neoplasm_histologic_grade"], errors="coerce")
    out = {"excluded_patients": int(excluded.sum()), "kept_patients": int((~excluded).sum())}
    for label, mask in (("excluded", excluded), ("kept", ~excluded)):
        out[label] = {
            "mean_age_at_diagnosis": float(raw.loc[mask, "age_at_diagnosis"].mean()),
            "share_grade_3": float((grade[mask] == 3).mean()),
            "share_er_positive": float((raw.loc[mask, "er_status"] == "Positive").mean()),
            "mean_tumor_size": float(raw.loc[mask, "tumor_size"].mean()),
        }
    return out


def split_variability(X, y, genomic, n_splits: int = 20) -> dict:
    aucs = []
    for seed in range(n_splits):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        model = clone(fresh_pipeline(genomic, n_estimators=200)).fit(X_tr, y_tr)
        aucs.append(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    aucs = np.asarray(aucs)
    return {
        "n_splits": n_splits,
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std(ddof=1)),
        "auc_min": float(aucs.min()),
        "auc_max": float(aucs.max()),
    }


def horizon_sensitivity(raw: pd.DataFrame, months: int = 120) -> dict:
    X, y, genomic = horizon_cohort(raw, months)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    model = clone(fresh_pipeline(genomic)).fit(X_tr, y_tr)
    return {
        "horizon_months": months,
        "patients": int(len(y)),
        "alive_share": float(y.mean()),
        "test_auc": float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])),
    }


def xgboost_pipeline(genomic):
    from xgboost import XGBClassifier  # optional dependency, only needed for this check

    model = build_pipeline(genomic)
    model.set_params(classifier=XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        random_state=SEED, eval_metric="logloss", n_jobs=-1,
    ))
    return model


def _death_operating_points(y_test, p_survive) -> dict:
    """Death-class recall/precision at the default threshold and at matched recall."""
    death = (np.asarray(y_test) == 0).astype(int)
    p_death = 1 - np.asarray(p_survive)
    default = (p_death >= 0.5).astype(int)
    precision, recall, _ = precision_recall_curve(death, p_death)
    out = {
        "auc": roc_auc_score(death, p_death),
        "brier": brier_score_loss(y_test, p_survive),
        "recall_at_0.5": recall_score(death, default),
        "flagged_share_at_0.5": default.mean(),
    }
    for target in (0.5, 0.8):
        out[f"precision_at_recall_{target}"] = float(precision[recall >= target].max())
    # Recall when flagging the same share of patients as the Random Forest does at 0.5.
    top_k = int(round(0.2222 * len(death)))
    flagged = np.zeros(len(death), dtype=int)
    flagged[np.argsort(-p_death)[:top_k]] = 1
    out["recall_flagging_22pct"] = recall_score(death, flagged)
    return {k: float(v) for k, v in out.items()}


def model_comparison(X, y, genomic, n_splits: int = 20) -> dict:
    rows = {"random_forest": [], "xgboost": []}
    for seed in range(n_splits):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        for name, pipeline in (("random_forest", fresh_pipeline(genomic, n_estimators=200)),
                               ("xgboost", xgboost_pipeline(genomic))):
            fitted = clone(pipeline).fit(X_tr, y_tr)
            rows[name].append(_death_operating_points(y_te, fitted.predict_proba(X_te)[:, 1]))

    def column(name, key):
        return np.array([row[key] for row in rows[name]])

    summary = {}
    for key in rows["random_forest"][0]:
        rf, xgb = column("random_forest", key), column("xgboost", key)
        diff = xgb - rf
        summary[key] = {
            "random_forest_mean": float(rf.mean()),
            "xgboost_mean": float(xgb.mean()),
            "paired_diff_mean": float(diff.mean()),
            "paired_diff_std": float(diff.std(ddof=1)),
            "xgboost_better_in_splits": int((diff < 0).sum() if key == "brier" else (diff > 0).sum()),
        }
    return {"n_splits": n_splits, "metrics": summary}


def _calibration_slope_intercept(y_test, p_survive) -> tuple[float, float]:
    """Slope and intercept of the outcome on the logit of the predicted death probability."""
    from sklearn.linear_model import LogisticRegression

    death = (np.asarray(y_test) == 0).astype(int)
    p = np.clip(1 - np.asarray(p_survive), 1e-4, 1 - 1e-4)
    fit = LogisticRegression(C=1e6).fit(np.log(p / (1 - p)).reshape(-1, 1), death)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def calibration_comparison(X, y, genomic, n_splits: int = 20) -> dict:
    """Recalibrate both models identically and compare Brier score on untouched test data.

    Each split: train (80%) is divided into a fit part (75%) and a calibration part (25%).
    Raw and recalibrated predictions come from the same fitted model, so the only
    difference is the calibration map (Platt = logistic on the logit; isotonic).
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)

    records = {"random_forest": [], "xgboost": []}
    for seed in range(n_splits):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_tr, y_tr, test_size=0.25, random_state=seed, stratify=y_tr
        )
        for name, pipeline in (("random_forest", fresh_pipeline(genomic, n_estimators=200)),
                               ("xgboost", xgboost_pipeline(genomic))):
            model = clone(pipeline).fit(X_fit, y_fit)
            p_cal = model.predict_proba(X_cal)[:, 1]
            p_raw = model.predict_proba(X_te)[:, 1]
            platt = LogisticRegression(C=1e6).fit(logit(p_cal), y_cal)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, y_cal)
            p_platt = platt.predict_proba(logit(p_raw))[:, 1]
            p_iso = iso.predict(p_raw)
            slope, intercept = _calibration_slope_intercept(y_te, p_raw)
            slope_p, intercept_p = _calibration_slope_intercept(y_te, p_platt)
            records[name].append({
                "brier_raw": brier_score_loss(y_te, p_raw),
                "brier_platt": brier_score_loss(y_te, p_platt),
                "brier_isotonic": brier_score_loss(y_te, p_iso),
                "auc_raw": roc_auc_score(y_te, p_raw),
                "auc_platt": roc_auc_score(y_te, p_platt),
                "slope_raw": slope, "intercept_raw": intercept,
                "slope_platt": slope_p, "intercept_platt": intercept_p,
                "mean_predicted_survival_raw": float(np.mean(p_raw)),
                "mean_predicted_survival_platt": float(np.mean(p_platt)),
                "observed_survival": float(np.mean(y_te)),
            })

    summary = {}
    for name, rows in records.items():
        summary[name] = {
            key: {"mean": float(np.mean([r[key] for r in rows])),
                  "std": float(np.std([r[key] for r in rows], ddof=1))}
            for key in rows[0]
        }
    paired = {}
    for key in ("brier_raw", "brier_platt", "brier_isotonic"):
        diff = np.array([x[key] - r[key] for r, x in zip(records["random_forest"], records["xgboost"])])
        paired[key] = {
            "xgboost_minus_rf_mean": float(diff.mean()),
            "xgboost_minus_rf_std": float(diff.std(ddof=1)),
            "xgboost_better_in_splits": int((diff < 0).sum()),
        }
    return {"n_splits": n_splits, "models": summary, "paired_brier": paired}


def survival_model_comparison(raw: pd.DataFrame, n_splits: int = 20, horizon: int = 60) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lifelines convergence / overflow warnings on rare columns
        return _survival_model_comparison(raw, n_splits, horizon)


def _survival_model_comparison(raw: pd.DataFrame, n_splits: int, horizon: int) -> dict:
    """Penalised Cox (keeps censored patients) vs the 5-year classifier, same test patients.

    Splits are stratified on the death indicator over the full cohort. The classifier is
    trained on patients whose 5-year status is known (as in ``training.py``); the Cox model
    is trained on everyone, using duration and censoring. Both use the same leakage-free
    preprocessing (gene selection is fitted on the training part only; the Cox model
    selects genes against the death indicator because the 5-year label is unavailable for
    censored patients). Evaluation:
    - AUC at 5 years on test patients whose 5-year status is known (score = predicted
      5-year survival);
    - C-index on ALL test patients, censored included (score = predicted survival).
    """
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index, k_fold_cross_validation
    from sklearn.preprocessing import StandardScaler

    X_all, genomic = prepare_features(raw)
    time = raw["overall_survival_months"].to_numpy(dtype=float)
    event = (1 - raw["overall_survival"]).to_numpy()  # dataset encodes 1 = alive
    known = ~((time < horizon) & (event == 0))
    alive_at_horizon = (time >= horizon).astype(int)
    penalizers = (0.05, 0.2, 1.0)

    rows = []
    for seed in range(n_splits):
        idx_tr, idx_te = train_test_split(
            np.arange(len(raw)), test_size=0.2, random_state=seed, stratify=event
        )
        tr_known = idx_tr[known[idx_tr]]
        te_known_mask = known[idx_te]

        classifier = clone(fresh_pipeline(genomic, n_estimators=200)).fit(
            X_all.iloc[tr_known], alive_at_horizon[tr_known]
        )
        p_classifier = classifier.predict_proba(X_all.iloc[idx_te])[:, 1]

        preprocessor = clone(build_pipeline(genomic).named_steps["preprocessor"]).fit(
            X_all.iloc[idx_tr], event[idx_tr]
        )
        Z_tr = preprocessor.transform(X_all.iloc[idx_tr])
        Z_te = preprocessor.transform(X_all.iloc[idx_te])
        keep = Z_tr.std(axis=0) > 1e-8
        scaler = StandardScaler().fit(Z_tr[:, keep])
        columns = [f"f{i}" for i in range(int(keep.sum()))]
        train_df = pd.DataFrame(scaler.transform(Z_tr[:, keep]), columns=columns)
        train_df["T"], train_df["E"] = time[idx_tr], event[idx_tr]
        test_df = pd.DataFrame(scaler.transform(Z_te[:, keep]), columns=columns)

        best = max(
            penalizers,
            key=lambda pen: np.mean(k_fold_cross_validation(
                CoxPHFitter(penalizer=pen), train_df, "T", "E", k=3,
                scoring_method="concordance_index", seed=seed,
            )),
        )
        cox = CoxPHFitter(penalizer=best).fit(train_df, "T", "E")
        s_cox = cox.predict_survival_function(test_df, times=[horizon]).T.iloc[:, 0].to_numpy()

        te_time, te_event = time[idx_te], event[idx_te]
        rows.append({
            "auc_5y_classifier": roc_auc_score(alive_at_horizon[idx_te][te_known_mask], p_classifier[te_known_mask]),
            "auc_5y_cox": roc_auc_score(alive_at_horizon[idx_te][te_known_mask], s_cox[te_known_mask]),
            "cindex_classifier": concordance_index(te_time, p_classifier, te_event),
            "cindex_cox": concordance_index(te_time, s_cox, te_event),
            "cox_penalizer": best,
            "train_censored_before_horizon_excluded_by_classifier": int((~known[idx_tr]).sum()),
        })

    def col(key):
        return np.array([r[key] for r in rows], dtype=float)

    out = {"n_splits": n_splits, "horizon_months": horizon,
           "penalizers_tried": list(penalizers),
           "cox_penalizer_mean": float(col("cox_penalizer").mean())}
    for metric in ("auc_5y", "cindex"):
        clf, cox = col(f"{metric}_classifier"), col(f"{metric}_cox")
        diff = cox - clf
        out[metric] = {
            "classifier_mean": float(clf.mean()), "classifier_std": float(clf.std(ddof=1)),
            "cox_mean": float(cox.mean()), "cox_std": float(cox.std(ddof=1)),
            "paired_diff_mean": float(diff.mean()), "paired_diff_std": float(diff.std(ddof=1)),
            "cox_better_in_splits": int((diff > 0).sum()),
        }
    return out


def cox_checks(raw: pd.DataFrame) -> dict:
    """Cross-validated C-index of the clinical Cox model and an ER-stratified fit."""
    # lifelines warns about non-unique indices it rebuilds internally when splitting folds.
    warnings.filterwarnings("ignore", message="DataFrame Index is not unique", category=UserWarning)
    from lifelines import CoxPHFitter
    from lifelines.utils import k_fold_cross_validation

    df = pd.DataFrame({
        "overall_survival_months": raw["overall_survival_months"],
        "event": 1 - raw["overall_survival"],  # dataset encodes 1 = alive
        "age_at_diagnosis": raw["age_at_diagnosis"],
        "tumor_size": np.log1p(raw["tumor_size"]),
        "lymph_nodes_examined_positive": raw["lymph_nodes_examined_positive"],
        "neoplasm_histologic_grade": pd.to_numeric(raw["neoplasm_histologic_grade"], errors="coerce"),
        "her2_positive": (raw["her2_status"] == "Positive").astype(int),
        "pr_positive": (raw["pr_status"] == "Positive").astype(int),
        "chemotherapy": raw["chemotherapy"],
        "hormone_therapy": raw["hormone_therapy"],
        "radio_therapy": raw["radio_therapy"],
        "er_positive": (raw["er_status"] == "Positive").astype(int),
    }).dropna()
    df = df[df["overall_survival_months"] > 0].reset_index(drop=True)

    covariates = [c for c in df if c not in ("overall_survival_months", "event", "er_positive")]
    full = df[covariates + ["er_positive", "overall_survival_months", "event"]]

    apparent = CoxPHFitter(penalizer=0.0).fit(full, "overall_survival_months", "event")
    scores = k_fold_cross_validation(
        CoxPHFitter(penalizer=0.0), full, "overall_survival_months", "event",
        k=5, scoring_method="concordance_index", seed=SEED,
    )
    stratified = CoxPHFitter(penalizer=0.0).fit(
        df[covariates + ["er_positive", "overall_survival_months", "event"]],
        "overall_survival_months", "event", strata=["er_positive"],
    )
    strat_scores = k_fold_cross_validation(
        CoxPHFitter(penalizer=0.0), full, "overall_survival_months", "event",
        k=5, scoring_method="concordance_index", seed=SEED, fitter_kwargs={"strata": ["er_positive"]},
    )
    return {
        "patients": int(len(df)),
        "apparent_c_index": float(apparent.concordance_index_),
        "cv_c_index_mean": float(np.mean(scores)),
        "cv_c_index_std": float(np.std(scores, ddof=1)),
        "er_stratified_apparent_c_index": float(stratified.concordance_index_),
        "er_stratified_cv_c_index_mean": float(np.mean(strat_scores)),
        "er_stratified_hazard_ratios": {k: float(v) for k, v in np.exp(stratified.params_).items()},
    }


def main() -> None:
    raw = load_raw()
    report_dir, figure_dir = ROOT / "reports", ROOT / "figures"
    report_dir.mkdir(exist_ok=True)

    X, y, genomic = horizon_cohort(raw, 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    permutation, model = permutation_test(X_train, X_test, y_train, y_test, genomic)
    results = {
        "permutation_test": permutation,
        "threshold_and_calibration": threshold_and_calibration(model, X_train, y_train, X_test, y_test, figure_dir),
        "selection_bias_60_months": selection_bias(raw),
        "split_variability": split_variability(X, y, genomic),
        "horizon_sensitivity": horizon_sensitivity(raw),
        "cox": cox_checks(raw),
        "model_comparison": model_comparison(X, y, genomic),
        "calibration_comparison": calibration_comparison(X, y, genomic),
        "survival_model_comparison": survival_model_comparison(raw),
    }
    (report_dir / "robustness_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
