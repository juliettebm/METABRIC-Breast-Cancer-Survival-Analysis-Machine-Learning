# METABRIC Breast Cancer Survival Analysis & Machine Learning

This repository contains an end-to-end data science pipeline analyzing the **METABRIC** (Molecular Taxonomy of Breast Cancer International Consortium) dataset. The project blends traditional biostatistics (survival analysis) with modern machine learning to uncover key clinical and genomic factors associated with breast cancer survival and to predict 5-year patient outcomes.

## Project Structure

```text
├── data/
│   └── METABRIC_RNA_Mutation.csv       # Raw dataset (1,904 patients, 693 variables)
├── notebooks/
│   ├── 01_exploration.ipynb            # Exploratory Data Analysis & target recoding strategy
│   ├── 02_preprocessing.ipynb          # Data cleaning, log1p transform, pipeline & feature selection
│   ├── 03_kaplan_meier.ipynb           # Stratified survival curves (ER, HER2, treatments, etc.)
│   ├── 04_cox_model.ipynb              # Univariate & Multivariate Cox Proportional Hazards modeling
│   └── 05_ml_prediction.ipynb          # 5-Year survival prediction (Random Forest & XGBoost)
├── outputs/
│   ├── df_survival.csv                 # Cleaned clinical data + top 25 genes for survival models
│   └── df_ml.csv                       # Fully imputed, encoded matrix + top 50 genes for ML models
├── streamlit_app/
│   ├── rf_model.joblib                 # Serialized Random Forest classifier
│   └── feature_cols.joblib             # Saved list of matching feature columns
├── README.md
└── .gitignore

Notebooks Workflow & Insights
01 — Exploratory Data Analysis (01_exploration.ipynb)
Target Assessment: Identified that overall_survival uses an inverse encoding convention (0 = Deceased, 1 = Alive/Censored). Planned a recoding strategy (event = 1 - overall_survival) to fit standard survival libraries.

Missing Not At Random (MNAR): Detected that tumor_stage has a 26.3% missing rate with heavy bias towards deceased patients. Decision made to exclude it and use the Nottingham Prognostic Index (NPI) as a proxy.

Outlier Strategy: Plotted and identified right-skewed distributions in tumor_size and mutation_count (hypermutator phenotypes), leading to a planned log-transformation.

02 — Preprocessing (02_preprocessing.ipynb)
Feature Engineering: Implemented the event mapping and binarized 173 _mut columns to track the simple presence or absence of mutations.

Transformation Pipeline: Applied np.log1p to highly skewed features. Built an automated ColumnTransformer handling numerical imputation (SimpleImputer(strategy='median') + StandardScaler) and categorical encoding (OneHotEncoder).

Genomic Feature Selection: Compressed the heavy genomic space from 662 initial molecular attributes down to the top 25 genes (for survival data) and top 50 genes (for ML matrices) using variance thresholds and correlation filtering.

03 — Kaplan-Meier Survival Analysis (03_kaplan_meier.ipynb)
Stratified Logs: Analyzed non-parametric curves across major clinical subsets.

Key Findings: Validated that ER+, PR+, and lower histologic grades are linked to significantly higher long-term survival. Noted that ER status curves cross later in time, signaling a potential violation of the proportional hazards assumption. Noted indication bias in chemotherapy lines, necessitating multivariate adjustment.

04 — Cox Proportional Hazards Model (04_cox_model.ipynb)
Multivariate Adjustment: Built univariate models followed by a comprehensive multivariate Cox model to estimate adjusted Hazard Ratios (HR).

Hypothesis Testing: Performed Schoenfeld residual tests to evaluate the proportional hazards assumption. Confirmed that classic parameters (tumor_size, positive lymph nodes) remain dominant independent hazards.

05 — Machine Learning Prediction (05_ml_prediction.ipynb)
Classification Framework: Converted the continuous timeline into a binary classification task: 5-Year Survival (survived_5y), preserving a robust, balanced cohort (58% vs 42%).

Model Benchmark: Trained a baseline dummy classifier, an optimized Random Forest, and an XGBoost algorithm.

Deployment Prep: Evaluated feature importances and saved the best-performing model (rf_model.joblib) along with its feature names to feed an interactive Streamlit application.

Requirements & Environment
To reproduce the analysis and run the notebooks locally, install the required packages:

pip install pandas numpy scikit-learn matplotlib seaborn lifelines joblib
