# METABRIC Survival Dashboard

Survival analysis and machine learning prediction on breast cancer, based on the
METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) dataset.

## Research Questions

1. **Survival analysis** — Which clinical and genomic factors are associated with
   the overall survival of breast cancer patients?
2. **Machine learning** — Can 5-year survival be predicted from multimodal data
   (clinical + genomic)?

## Dataset

- **Name**: Breast Cancer Gene Expression Profiles (METABRIC)
- **Source**: [Kaggle — raghadalharbi/breast-cancer-gene-expression-profiles-metabric](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric)
- **Size**: 1,904 patients, 693 variables (clinical, biomarkers, treatments, gene
  expression, gene mutations)

The raw CSV is not included in this repository (see [`.gitignore`](.gitignore)).
Download `METABRIC_RNA_Mutation.csv` from Kaggle and place it in `data/` before
running the notebooks.

## Project Structure

```
.
├── data/
│   ├── METABRIC_RNA_Mutation.csv        # raw dataset (download separately, not versioned)
│   └── processed/
│       ├── df_survival.csv              # clinical vars + biomarkers + top 25 genes (KM / Cox)
│       └── df_ml.csv                    # clinical vars + biomarkers + top 50 genes (ML)
├── notebooks/
│   ├── 01_exploration.ipynb             # EDA: types, outliers, missing values, distributions
│   ├── 02_preprocessing.ipynb           # cleaning, encoding, imputation, feature selection
│   ├── 03_kaplan_meier.ipynb            # KM curves by subgroup, log-rank tests
│   ├── 04_cox_model.ipynb               # univariate / multivariate Cox model, hazard ratios
│   └── 05_ml_prediction.ipynb           # 5-year survival classification (RF, XGBoost)
├── figures/
├── streamlit_app/
│   ├── app.py                           # interactive dashboard
│   ├── rf_model.joblib                  # trained Random Forest model
│   └── feature_cols.joblib              # feature column order expected by the model
├── requirements.txt
├── .gitignore
└── README.md
```

## Methodology

### 1. Exploration (`01_exploration.ipynb`)
General overview, variable-block separation (clinical / biomarkers / treatments /
gene expression / mutations), target-variable encoding check, outlier inspection,
and missing-value mechanism analysis (MCAR / MAR / MNAR).

### 2. Preprocessing (`02_preprocessing.ipynb`)
- Recoding of the target: `event = 1 - overall_survival` to match the `lifelines`
  convention.
- Exclusion of unusable variables (data leakage, MNAR, duplicated information).
- `log1p` transform on skewed variables (`tumor_size`, `mutation_count`).
- Binarization of mutation columns (variant name → presence/absence).
- Imputation, encoding, and scaling via a `scikit-learn` `ColumnTransformer` pipeline.
- Genomic feature selection (`VarianceThreshold` + correlation with survival) to
  reduce ~660 genomic variables down to the top 25 (survival analysis) / top 50
  (ML) genes.
- Export of `df_survival.csv` and `df_ml.csv`.

### 3. Kaplan-Meier survival analysis (`03_kaplan_meier.ipynb`)
Survival curves and log-rank tests stratified by ER status, HER2 status, PR
status, histologic grade, chemotherapy, and hormone therapy.

### 4. Cox Proportional Hazards model (`04_cox_model.ipynb`)
Univariate and multivariate Cox regression to identify factors independently
associated with survival, hazard-ratio forest plot, and a proportional-hazards
assumption check via Schoenfeld residuals.

### 5. ML prediction (`05_ml_prediction.ipynb`)
Binary classification of 5-year survival (`survived_5y`) using a Random Forest
and XGBoost, benchmarked against a `DummyClassifier` baseline.

## Key Results

| Model | ROC-AUC | Accuracy | Survivor Recall | Deceased Recall |
|---|---|---|---|---|
| Baseline (Dummy) | 0.500 | 60% | 0% | 100% |
| **Random Forest** | **0.759** | **73%** | **63%** | **79%** |
| XGBoost | 0.731 | 69% | 61% | 74% |

Random Forest is retained as the production model in the Streamlit app for its
robustness out-of-the-box.

| Variable (Kaplan-Meier) | Log-rank p-value | Proportional hazards |
|---|---|---|
| ER Status | 0.02 | ⚠️ curves cross (late relapse) |
| HER2 Status | 2.22e-05 | ✅ |
| PR Status | 7.45e-05 | ✅ |
| Histologic Grade | 1.14e-03 | ✅ |
| Chemotherapy | 7.26e-03 | ⚠️ indication bias |
| Hormone Therapy | 1.13e-04 | ⚠️ reverse indication bias |

## Installation

```bash
git clone <repo-url>
cd <repo-name>
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `METABRIC_RNA_Mutation.csv` from Kaggle and place it in `data/`.

## Usage

### Run the notebooks

Notebooks must be run in order (each one depends on the outputs of the previous
one):

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

Run `01` → `02` → `03` / `04` (independent) → `05`.

### Run the dashboard

```bash
cd streamlit_app
streamlit run app.py
```

The dashboard has three pages: **Home** (cohort overview), **Survival Analysis**
(interactive Kaplan-Meier curves), and **ML Prediction** (5-year survival
estimator for a custom patient profile).

## Disclaimer

⚠️ This project is for **educational purposes only**. The model is trained on
historical data (METABRIC, 2000–2010) and is **not a clinical decision-making
tool**.

## Author

**Juliette Bouli-Mengue**
Clinical Research Associate → Healthcare Data Science
