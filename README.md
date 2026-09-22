# METABRIC Breast Cancer Survival Analysis & Machine Learning

[![Dataset](https://img.shields.io/badge/Dataset-METABRIC%20(Kaggle)-blue?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric)
[![CI](https://github.com/juliettebm/METABRIC-Breast-Cancer-Survival-Analysis-Machine-Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/juliettebm/METABRIC-Breast-Cancer-Survival-Analysis-Machine-Learning/actions/workflows/ci.yml)
[![Reproducibility](https://img.shields.io/badge/reproducibility-pinned%20environment-success)](requirements.txt)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest%20%7C%20XGBoost-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://streamlit.io/)

Survival analysis and machine-learning prediction on breast cancer using the METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) cohort.

The project combines exploratory analysis, Kaplan–Meier estimation, multivariable Cox regression and fixed-horizon machine learning while explicitly accounting for censoring, leakage prevention and internal validation.

---

## Objective

This project addresses two complementary questions:

1. **Survival analysis:** Which clinical and tumor characteristics are associated with overall survival in METABRIC?
2. **Machine learning:** Can death before 5 years be predicted from multimodal clinical and genomic information?

For the fixed-horizon ML task:

- `1` = death observed before 60 months
- `0` = survived at least 60 months
- patients censored before 60 months are excluded because their 5-year outcome cannot be determined directly.

This results in **1,844 patients**, with approximately **22% deaths before 5 years**.

The classification analysis complements rather than replaces the censoring-aware survival analyses.

---

## Dataset

- **Name:** Breast Cancer Gene Expression Profiles (METABRIC)
- **Source:** [Kaggle: raghadalharbi/breast-cancer-gene-expression-profiles-metabric](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric)
- **Raw cohort:** 1,904 patients
- **Raw variables:** 693
- **Data types:** clinical variables, biomarkers, treatments, gene expression and somatic mutations

The raw CSV and processed dataframes are not versioned in the repository.

Download `METABRIC_RNA_Mutation.csv` from Kaggle and place it in:

```text
data/METABRIC_RNA_Mutation.csv
```

This keeps the third-party dataset separate from the source code while preserving clear data provenance.

---

## Project Structure

```text
.
├── data/
│   ├── METABRIC_RNA_Mutation.csv        # raw dataset, downloaded separately
│   └── processed/
│       ├── df_survival.csv              # descriptive survival-analysis dataset
│       ├── df_ml.csv                    # dataset prepared for predictive workflow
│       └── feature_groups.json          # feature-group metadata
│
├── notebooks/
│   ├── 01_exploration.ipynb             # exploratory data analysis
│   ├── 02_preprocessing.ipynb           # deterministic preprocessing and exports
│   ├── 03_kaplan_meier.ipynb            # Kaplan–Meier curves and log-rank tests
│   ├── 04_cox_model.ipynb               # multivariable survival modeling
│   └── 05_ml_prediction.ipynb           # 5-year mortality prediction
│
├── figures/                              # generated figures
├── reports/                              # additional robustness outputs
│
├── src/
│   └── metabric/
│       └── training.py                   # reusable predictive training pipeline
│
├── streamlit_app/
│   ├── app.py                            # interactive educational dashboard
│   ├── rf_model.joblib                   # deployed Random Forest bundle
│   ├── feature_cols.joblib               # expected feature schema
│   ├── metrics.json                      # deployed-model metrics
│   └── benchmark_metrics.json            # benchmark results
│
├── train_model.py                        # canonical model-training entry point
├── prepare_data.py                       # regenerates processed datasets
├── requirements.txt
├── .gitignore
└── README.md
```

`train_model.py` is the canonical entry point for rebuilding the Streamlit prediction artifact. Reusable preparation, pipeline construction and evaluation code is implemented in `src/metabric/training.py`.

---

# Analysis Workflow

## 1. Exploratory Data Analysis

`01_exploration.ipynb`

The exploratory analysis covers:

- dataset structure and feature blocks;
- clinical and genomic variable distributions;
- survival-target encoding;
- missing-data patterns;
- outlier inspection;
- clinical, biomarker and treatment distributions;
- gene-expression and mutation characteristics.

The original survival variable uses:

```text
overall_survival = 0 → death observed
overall_survival = 1 → alive/censored
```

For survival analysis, it is converted to the `lifelines` convention:

```python
event = 1 - overall_survival
```

so that:

```text
event = 1 → death observed
event = 0 → censored
```

The median observed time in the cohort is approximately **115.6 months (9.6 years)**. This is an observed-time summary and is not interpreted as a formal median follow-up estimate.

---

## 2. Preprocessing

`02_preprocessing.ipynb`

Two analysis datasets are created:

- `df_survival` for descriptive survival analyses;
- `df_ml` for the machine-learning workflow.

The preprocessing notebook performs deterministic operations such as:

- target recoding;
- removal of identifiers, duplicated variables and leakage-prone outcome information;
- separation of survival and ML datasets;
- mutation binarization;
- feature-group definition.

The exported datasets contain:

```text
df_survival: 1,904 × 685
df_ml:       1,904 × 688
```

The raw genomic feature space contains:

- **489 gene-expression variables**
- **173 mutation variables**

An exploratory label-free variance filter reduces the mutation set from 173 to 115 variables, producing an exploratory genomic pool of **604 variables**.

Importantly, this cohort-wide exploratory filtering is not used as supervised feature selection for predictive evaluation.

For machine learning, learned operations such as imputation, encoding, scaling, variance filtering and supervised genomic feature selection are fitted inside the training pipeline so that information from validation or test observations is not used to learn transformations.

---

## 3. Kaplan–Meier Survival Analysis

`03_kaplan_meier.ipynb`

Kaplan–Meier curves and log-rank tests are used to describe unadjusted survival differences across clinically relevant groups.

| Variable | Log-rank p-value | KM pattern |
|---|---:|---|
| ER Status | 0.0216 | Curves converge and cross |
| HER2 Status | 2.22e-05 | No obvious crossing |
| PR Status | 7.45e-05 | Curves converge over time |
| Histological Grade | 5.39e-06 | Clear Grade 1 > Grade 2 > Grade 3 survival gradient |
| Chemotherapy | 7.26e-03 | Curves converge and cross |
| Hormone Therapy | 1.13e-04 | Curves cross |

For histological grade, the global multi-group log-rank test gives:

```text
p = 5.39e-06
```

with estimated median overall survival of:

```text
Grade 1: 208.0 months
Grade 2: 169.0 months
Grade 3: 128.5 months
```

All six subgroup analyses are exploratory and unadjusted.

Visual crossing or convergence of Kaplan–Meier curves can suggest time-varying associations, but it does **not** formally establish whether the proportional-hazards assumption holds. This is evaluated in the Cox analysis.

Treatment comparisons are observational and must not be interpreted causally because treatment allocation is related to patient and tumor characteristics.

---

## 4. Cox Survival Modeling

`04_cox_model.ipynb`

The Cox analysis evaluates clinical associations with mortality while accounting for event timing and censoring.

The prespecified primary model focuses on:

- age at diagnosis;
- Nottingham Prognostic Index (NPI);
- ER status;
- PR status;
- HER2 status;
- chemotherapy;
- hormone therapy;
- radiotherapy.

Histological grade, tumor size and positive lymph nodes are not included alongside NPI in the primary model because these variables contribute to the NPI itself.

### Initial multivariable model

In the conventional multivariable Cox model, important associations include:

- increasing age;
- increasing NPI;
- HER2-positive status;
- chemotherapy status;
- radiotherapy status.

The conventional model reaches a concordance of approximately **0.66**.

Formal proportional-hazards diagnostics identify substantial evidence of non-proportionality for several predictors, including age, NPI, ER status and hormone therapy.

### Final Cox specification

The final specification therefore:

- models **age** and **NPI** flexibly using spline terms;
- stratifies on **ER status**, **PR status** and **hormone therapy**;
- estimates conventional hazard ratios for **HER2 status**, **chemotherapy** and **radiotherapy**.

In this model:

| Predictor | Hazard Ratio | 95% CI |
|---|---:|---:|
| HER2 positive | 1.35 | 1.12–1.63 |
| Chemotherapy | 1.55 | 1.25–1.92 |
| Radiotherapy | 0.77 | 0.68–0.87 |

Concordance is approximately **0.68**.

These are adjusted observational associations, not causal treatment effects.

---

## 5. Five-Year Mortality Prediction

`05_ml_prediction.ipynb`

The secondary machine-learning analysis reframes the problem as fixed-horizon binary classification.

### Target

```text
1 = death observed before 5 years
0 = survived at least 5 years
```

Patients censored before 60 months are excluded.

Final classification cohort:

```text
1,844 patients
```

A stratified 80/20 split gives:

```text
Training set: 1,475 patients
Test set:       369 patients
```

The held-out test set contains:

```text
82 deaths before 5 years
287 5-year survivors
```

### Leakage control

The predictive pipeline fits learned transformations using training data only.

This includes:

- numerical imputation;
- categorical imputation and encoding;
- scaling where required;
- genomic variance filtering;
- supervised genomic feature selection.

Genomic selection uses `SelectKBest` within the pipeline, retaining 50 genomic predictors for each fitted training fold.

### Model selection

Random Forest and XGBoost are evaluated using nested cross-validation.

Mean outer cross-validation performance:

| Model | ROC-AUC |
|---|---:|
| Random Forest | **0.760 ± 0.018** |
| XGBoost | **0.747 ± 0.019** |
| Dummy classifier | 0.500 |

For the Random Forest, model selection retained:

```text
class_weight = "balanced"
min_samples_leaf = 3
```

XGBoost retained:

```text
scale_pos_weight = 1.0
```

Class weighting was therefore treated as a model-selection choice rather than assumed to be necessary from class imbalance alone.

---

# Held-Out Test Results

| Model | ROC-AUC | Brier score | Death recall | Death precision | Survivor recall |
|---|---:|---:|---:|---:|---:|
| Dummy | 0.500 | 0.173 | 0.0% | — | 100.0% |
| Random Forest | **0.694** | **0.178** | **40.2%** | **40.7%** | **83.3%** |
| XGBoost | **0.722** | **0.154** | **22.0%** | **47.4%** | **93.0%** |

Bootstrap 95% confidence intervals for ROC-AUC:

```text
Random Forest: 0.694 [0.627–0.755]
XGBoost:       0.722 [0.656–0.783]
```

The confidence intervals overlap substantially, so the observed difference does not establish that one algorithm is statistically superior.

XGBoost has the highest observed held-out ROC-AUC and lowest uncalibrated Brier score.

At the default threshold of 0.5, the class-weighted Random Forest identifies a larger proportion of deaths, but this comes at the cost of more false-positive mortality predictions.

This illustrates the distinction between:

- **discrimination** — how well patients are ranked by risk;
- **calibration** — how closely predicted probabilities correspond to observed frequencies;
- **classification performance** — which additionally depends on the selected decision threshold.

Accuracy is not used as the primary performance measure because approximately 78% of the classification cohort survives at least five years. A majority-class dummy classifier already reaches an accuracy of approximately 0.778 while detecting no deaths.

`streamlit_app/metrics.json` is the source of truth for the deployed artifact: on the untouched test set, its AUC is 0.694 (bootstrap 95% CI 0.627–0.755), Brier score is 0.160 after Platt recalibration (0.178 before recalibration), while executed notebook 05 is the source for the side-by-side benchmark below.

| Model | ROC-AUC | Accuracy | Death recall | Survivor recall |
|---|---:|---:|---:|---:|
| **Random Forest (deployed, tuned)** | **0.694** | **74%** | **40%** | **83%** |
| XGBoost (benchmark) | 0.722 | 77% | 93% | 22% |

---

# Robustness Checks

`python robustness_checks.py` (code in `src/metabric/robustness.py`) refits the same leakage-free pipeline and writes `reports/robustness_metrics.json` and `figures/06_calibration.png`. The deployed model is not modified. These checks answer the usual objections to the numbers above; all remain internal to one cohort.

| Check | Result |
|---|---|
| **Label-permutation test** (20 permutations of the training labels, whole pipeline refitted) | Permuted AUC averages 0.494 (range 0.387–0.602) against 0.758 for the real labels. The empirical p-value is 0.048, which is the minimum achievable with 20 permutations. No sign of leakage. |
| **Split variability** (20 stratified 80/20 splits, 200 trees) | AUC 0.755 ± 0.024 (min 0.710, max 0.791): the single-split bootstrap interval understates the split-to-split spread, and explains why the deployed 369-patient test split above (0.694) differs from this mean — the label was recoded from "survived" to "died" for clarity, which shifted which patients scikit-learn's stratified split assigns to the held-out fold, without changing typical model performance. |
| **Horizon sensitivity** (120 months) | 1,648 patients with an observable 10-year status (55.2% alive); test AUC 0.718 against 0.758 at 5 years. |
| **Selection bias of the 5-year cohort** | The 60 patients censored before 60 months (excluded) have a mean age of 58.1 vs 61.2 years, 43.3% vs 48.9% grade 3, 73.3% vs 76.7% ER-positive. Descriptive only (no test, small group): no large imbalance, but independence of censoring is not proven. |
| **Cox C-index** | Apparent 0.674; 5-fold cross-validated 0.673 ± 0.016. The 0.67 reported above is therefore not inflated by overfitting. |
| **Cox stratified on ER** (handles the proportional-hazards violation) | Apparent C-index 0.680, cross-validated 0.669: stratification fixes the assumption without improving discrimination. |

**Model comparison, Random Forest vs XGBoost** (same leakage-free preprocessing, 20 stratified splits, paired differences; XGBoost uses the notebook's untuned settings, the forest uses 200 trees here):

| Metric (mean over 20 splits) | Random Forest | XGBoost | Paired difference (XGBoost − RF) | Splits won by XGBoost |
|---|---|---|---|---|
| ROC-AUC | 0.755 | 0.745 | -0.010 ± 0.017 | 4/20 |
| Brier score (lower is better) | 0.166 | 0.151 | -0.016 ± 0.005 | 20/20 |
| Death recall at threshold 0.5 | 44.6% | 23.4% | -21.3% | 0/20 |
| Share of patients flagged at 0.5 | 20.9% | 9.8% | -11.1% | 0/20 |
| Death recall when flagging the top 22% | 46.8% | 46.0% | -0.8% | 4/20 |
| Death precision at 80% recall | 36.0% | 34.8% | -1.2% | 8/20 |

Averaged over 20 splits, discrimination is essentially a tie between the two algorithms (0.755 vs 0.745, within one standard deviation). At the default threshold XGBoost flags about half as many patients, which explains its lower death recall; when both flag the same share of patients, recalls are equivalent. XGBoost's only consistent advantage is the *raw* Brier score, and it disappears once both models are recalibrated (see Calibration below). Neither model was tuned exhaustively, so these are comparisons of two reasonable configurations, not of the best achievable models.

**Decision threshold** (test set, death = positive class; deployed default is 0.5):

| Death-probability threshold | Death recall | Death precision | Patients flagged |
|---|---|---|---|
| 0.2 | 92.7% | 26.1% | 78.9% |
| 0.3 | 81.7% | 33.0% | 55.0% |
| 0.4 | 69.5% | 40.4% | 38.2% |
| **0.5 (default)** | **51.2%** | **51.2%** | **22.2%** |
| 0.6 | 22.0% | 56.3% | 8.7% |

The 51% death recall shown in this table is a property of the threshold, not of the ranking (AUC is threshold-free): lowering it to 0.3 detects 82% of deaths at the price of 33% precision. The right threshold depends on the cost of a missed death versus a false alarm and has not been chosen here.

**Calibration** (`figures/06_calibration.png`). The raw class-weighted forest is miscalibrated: over 20 splits it predicts death with calibration intercept -0.67, slope 1.23 on the death logit (raw XGBoost is closer to intercept -0.20 but over-confident at slope 0.73). This is consistent with `class_weight="balanced"` shifting probabilities towards the minority class (not tested in isolation).

Both models were recalibrated identically (Platt scaling and isotonic regression fitted on a calibration fold held out from the training data; the test fold is untouched), over 20 splits:

| Brier score (mean ± sd over 20 splits) | Random Forest | XGBoost | Paired difference (XGBoost − RF) | Splits won by XGBoost |
|---|---|---|---|---|
| Raw | 0.169 ± 0.006 | 0.152 ± 0.007 | -0.017 ± 0.006 | 20/20 |
| Platt | 0.151 ± 0.004 | 0.150 ± 0.006 | -0.000 ± 0.005 | 10/20 |
| Isotonic | 0.155 ± 0.006 | 0.154 ± 0.007 | -0.001 ± 0.005 | 13/20 |

After recalibration the two models are indistinguishable; the raw XGBoost advantage came entirely from the forest's miscalibration. Platt scaling is strictly monotone, so AUC is unchanged. **The deployed model is therefore the Random Forest with a Platt calibrator** fitted on out-of-fold training predictions (`train_model.py`). On the specific 369-patient test split behind the deployed artifact, this brings the Brier score to 0.160 (raw 0.178); calibration on that single split is weaker than the 20-split average above (slope 0.67, intercept -0.37 vs the ~1.0/0.0 target), which is consistent with the split-variability finding above rather than a new miscalibration mechanism. Calibration is internal to one cohort and one split: it must be re-checked on any external data.

**Survival model vs 5-year classifier** (penalised Cox, 20 stratified splits of the full cohort, paired differences). The Cox model is trained on all patients, censored ones included, using duration and censoring; the classifier is trained on the patients whose 5-year status is known (the 60 patients censored before 60 months are excluded, as in `train_model.py`). Both are evaluated on the same test patients, with the same leakage-free preprocessing (gene selection fitted on the training part only; the Cox model selects genes against the death indicator because the 5-year label does not exist for censored patients). The Cox penalty (0.05, 0.2, 1.0) is chosen by 3-fold cross-validated C-index inside each training set.

| Metric (mean ± sd over 20 splits) | 5-year classifier | Penalised Cox | Paired difference (Cox − classifier) | Splits won by Cox |
|---|---|---|---|---|
| AUC at 5 years (test patients with a known 5-year status) | 0.747 ± 0.023 | 0.709 ± 0.033 | -0.038 ± 0.027 | 2/20 |
| C-index (all test patients, censored included) | 0.656 ± 0.019 | 0.669 ± 0.020 | +0.014 ± 0.017 | 15/20 |

Keeping the censored patients does **not** improve 5-year discrimination: the Cox model is clearly worse at the 5-year AUC (it models the whole follow-up rather than the 5-year outcome). It ranks patients slightly better over the full follow-up (C-index), which is what a survival model is designed for. This gives no evidence that excluding the 60 censored patients biases the classifier, but it does not prove that censoring is independent of the outcome. The Cox model was not tuned beyond its penalty, and its gene selection is disadvantaged by the missing 5-year label, so it is a fair benchmark, not an upper bound.

**ER over time** (`er_over_time` in `robustness.py`; 1,815 patients with complete covariates). ER violates the proportional-hazards assumption (Schoenfeld test on the adjusted Cox model, p = 7.6e-12). The Kaplan-Meier curves of ER-positive and ER-negative patients cross at **175 months** (bootstrap 95% interval 146–224, 1,000 resamples; crossings before 24 months are ignored because the curves are then almost identical). At that time 389 ER-positive and 113 ER-negative patients are still at risk, which explains the wide interval. Hazard ratios of ER-positive vs ER-negative by follow-up period (patients still alive at the start of each window; adjusted for age, tumor size, positive nodes, grade and HER2):

| Follow-up period | Patients at start | Deaths | HR unadjusted (95% CI) | HR adjusted (95% CI) | p (adjusted) |
|---|---|---|---|---|---|
| 0–60 months | 1815 | 398 | 0.38 (0.31–0.46) | 0.42 (0.33–0.53) | <0.001 |
| 60–120 months | 1361 | 306 | 1.39 (1.01–1.91) | 1.20 (0.85–1.70) | 0.311 |
| After 120 months | 861 | 337 | 2.23 (1.58–3.15) | 1.58 (1.09–2.29) | 0.016 |

ER-positive status is strongly protective during the first five years and adverse after ten years (late relapses). Because these effects have opposite signs, the single hazard ratio of the overall multivariate model is close to 1 and not significant. This is a descriptive analysis: each window conditions on survival to its start, so hazard ratios in later windows compare survivors and are not causal.

---

## Streamlit Demonstrator

The repository includes an educational Streamlit application with three pages:

### Home

Overview of the METABRIC cohort, project workflow and main results.

### Survival Analysis

Interactive Kaplan–Meier visualization by:

- ER status;
- HER2 status;
- PR status;
- histological grade;
- chemotherapy;
- hormone therapy.

### ML Prediction

Interactive estimation of the predicted probability of **death before 5 years** from a custom patient profile.

The deployed model is the tuned Random Forest produced by the canonical training pipeline.

Probability recalibration is applied to improve agreement between predicted mortality probabilities and observed outcome frequencies.

The application is an educational demonstrator and **must not be interpreted as a clinical risk calculator**.

---

# Reproduce the Project

## 1. Clone the repository

```bash
git clone https://github.com/juliettebm/METABRIC-Breast-Cancer-Survival-Analysis-Machine-Learning.git
cd METABRIC-Breast-Cancer-Survival-Analysis-Machine-Learning
```

## 2. Create the environment

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## 3. Download the dataset

Download:

```text
METABRIC_RNA_Mutation.csv
```

from the Kaggle dataset and place it in:

```text
data/
```

## 4. Generate processed datasets

```bash
python prepare_data.py
```

This regenerates the processed datasets required by the descriptive survival analyses.

## 5. Run the notebooks

The recommended order is:

```text
01_exploration.ipynb
        ↓
02_preprocessing.ipynb
        ↓
03_kaplan_meier.ipynb
04_cox_model.ipynb
        ↓
05_ml_prediction.ipynb
```

Kaplan–Meier and Cox analyses are complementary survival analyses after preprocessing.

## 6. Retrain the Streamlit model

From the repository root:

```bash
python train_model.py
```

This creates the exact model artifact used by the Streamlit demonstrator.

## 7. Launch Streamlit

```bash
cd streamlit_app
streamlit run app.py
```

## 8. Run tests

```bash
python -m pytest -q
```

---

# Methodological Notes

### Why exclude patients censored before five years from the classification task?

For a patient censored before 60 months, it is unknown whether death would have occurred before the five-year horizon after the last observed follow-up.

Assigning such patients to the survivor class would therefore create incorrect labels.

The fixed-horizon classifier excludes these observations, while the Cox analysis retains censored patients and uses their available follow-up information.

### Why use both survival analysis and classification?

They answer different questions.

Kaplan–Meier and Cox models use event timing and censoring.

The classifier instead asks a simpler fixed-horizon question:

> Can death before five years be predicted among patients whose five-year outcome is observable?

The two approaches should therefore be interpreted as complementary rather than interchangeable.

### Why is NPI handled differently in the Cox and ML analyses?

NPI is a clinically established composite index incorporating tumor size, histological grade and lymph-node involvement.

It is used directly in the primary Cox analysis.

For the ML workflow, the underlying component variables are available as predictors, so NPI is excluded to avoid duplicating the same prognostic information in composite and component form.

### Why is feature selection inside the ML pipeline?

Supervised feature selection performed before the train/test split would allow information from held-out patients to influence predictor selection.

For predictive evaluation, feature-selection parameters are therefore learned using training data only and are refitted inside cross-validation folds.

---

# Limitations

This project is intended for educational and methodological purposes and is not a clinical decision-support system.

Important limitations include:

- METABRIC is a historical cohort and may not represent contemporary breast-cancer populations or treatment strategies.
- The predictive feature space is large relative to the sample size, creating a risk of overfitting despite fold-safe feature selection and nested cross-validation.
- Fixed-horizon classification excludes patients censored before 60 months and therefore does not use censoring as fully as dedicated survival models.
- The held-out test set contains only 369 patients, including 82 deaths before five years, resulting in uncertainty around performance estimates.
- Calibration remains imperfect and requires explicit assessment.
- Classification metrics depend on the decision threshold; the default threshold of 0.5 is not a clinically validated operating point.
- Treatment variables are observational predictors and their coefficients or associations must not be interpreted as causal treatment effects.
- All validation remains internal to METABRIC.
- No independent temporal or external validation has been performed.

Independent validation would be required before any translational or clinical interpretation.

---

# Conclusion

This project combines censoring-aware survival analysis with a leakage-controlled fixed-horizon machine-learning analysis of METABRIC.

Kaplan–Meier analysis identifies substantial unadjusted survival differences across several clinical subgroups, including a clear histological-grade gradient.

The Cox analysis demonstrates the importance of formally evaluating the proportional-hazards assumption rather than inferring it from Kaplan–Meier curves alone. Flexible and stratified modeling is used when the conventional proportional-hazards specification is inadequate.

For five-year mortality prediction, both Random Forest and XGBoost demonstrate moderate discrimination. Random Forest achieves the highest mean nested cross-validation ROC-AUC, whereas XGBoost achieves the highest observed held-out ROC-AUC and lowest uncalibrated Brier score. Their held-out ROC-AUC confidence intervals overlap.

At the default threshold, the class-weighted Random Forest detects a larger proportion of deaths but produces more false-positive mortality predictions, illustrating the difference between discrimination and threshold-dependent classification performance.

The Random Forest is retained as the reproducible Streamlit demonstration model. Neither model is clinically validated.

More broadly, fixed-horizon classification provides an intuitive prediction target, while dedicated survival models remain preferable when event timing and censoring are central to the scientific question.

---

## Disclaimer

This repository is for **educational and methodological purposes only**.

The Streamlit application and trained model are **not clinical decision-making tools** and must not be used for patient care.

The versioned `.joblib` artifact is generated by this repository's training code. Joblib/pickle artifacts can execute code while loading and should never be replaced with artifacts from untrusted sources.

---

## License

Released under the [MIT License](LICENSE).

The METABRIC dataset is distributed separately by its source and is not redistributed by this repository.

---

## Stack

Python · pandas · NumPy · scikit-learn · XGBoost · lifelines · matplotlib · Plotly · Streamlit

---

## Author

**Juliette Bouli-Mengue**  
Clinical Research → Data Science