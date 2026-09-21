import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

# ── Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="METABRIC Survival Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0f1117;
        color: #e8e8e8;
    }

    .stApp {
        background-color: #0f1117;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #1e2d40;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #161b27 0%, #1a2235 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #4a9eff;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 6px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c8d6e8;
        border-left: 3px solid #4a9eff;
        padding-left: 12px;
        margin: 28px 0 16px 0;
    }

    /* Insight box */
    .insight-box {
        background: #0d1f35;
        border: 1px solid #1e3a5f;
        border-left: 4px solid #4a9eff;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.88rem;
        color: #a8b8cc;
        margin-top: 12px;
    }

    /* Warning box */
    .warning-box {
        background: #1f1a0d;
        border: 1px solid #3a2e1e;
        border-left: 4px solid #f0a030;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.88rem;
        color: #c8a878;
        margin-top: 12px;
    }

    /* Prediction result */
    .pred-high {
        background: linear-gradient(135deg, #0d2b1a 0%, #0f3020 100%);
        border: 1px solid #1a5c30;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }

    .pred-low {
        background: linear-gradient(135deg, #2b0d0d 0%, #301010 100%);
        border: 1px solid #5c1a1a;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }

    .pred-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
    }

    h1, h2, h3 { color: #e8e8e8; }

    .stSelectbox label, .stSlider label { color: #8892a4; font-size: 0.85rem; }

    hr { border-color: #1e2d40; }
</style>
""", unsafe_allow_html=True)

# ── Helper ────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha=0.12):
    """Convert a '#rrggbb' hex color string into an 'rgba(r,g,b,a)' string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

# ── Data loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Return (df_survival, df_raw, df_ml), or (None, None, None) if the local data is missing.

    The model page only needs the versioned model bundle, so missing data must not stop the app.
    """
    raw_candidates = [
        ROOT_DIR / "data" / "METABRIC_RNA_Mutation.csv",
        ROOT_DIR / "data" / "METABRIC_RNA_Mutation",
    ]
    processed = ROOT_DIR / "data" / "processed"
    raw_path = next((path for path in raw_candidates if path.exists()), None)
    needed = [processed / "df_survival.csv", processed / "df_ml.csv"]
    if raw_path is None or not all(path.exists() for path in needed):
        return None, None, None
    return (
        pd.read_csv(needed[0]),
        pd.read_csv(raw_path, low_memory=False),
        pd.read_csv(needed[1]),
    )

@st.cache_resource
def load_model():
    bundle = joblib.load(APP_DIR / "rf_model.joblib")
    return bundle

df, df_raw, df_ml = load_data()
model_bundle = load_model()
rf = model_bundle["model"]
feature_cols = model_bundle["raw_feature_cols"]

DATA_AVAILABLE = df is not None

# Original labels
if DATA_AVAILABLE:
    df_plot = df.copy()
    for col in ["er_status", "pr_status", "her2_status", "neoplasm_histologic_grade"]:
        df_plot[col + "_label"] = df_raw[col].values


def require_data():
    """Stop the current page with instructions when the local data files are absent."""
    if not DATA_AVAILABLE:
        st.warning(
            "This page needs the local dataset, which is not versioned. Download "
            "`METABRIC_RNA_Mutation.csv` from Kaggle into `data/`, then run "
            "`python prepare_data.py` from the repository root and restart the app. "
            "The **ML Prediction** page works without it."
        )
        st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 METABRIC")
    st.markdown("**Breast Cancer Survival**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📈 Survival Analysis", "🤖 ML Prediction"],
        index=0 if DATA_AVAILABLE else 2,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#5a6a7a; line-height:1.6'>
    <b style='color:#8892a4'>Dataset</b><br>
    METABRIC — 1,904 patients<br>
    UK/Canada breast cancer<br><br>
    <b style='color:#8892a4'>Source</b><br>
    Kaggle — raghadalharbi<br><br>
    <b style='color:#8892a4'>Author</b><br>
    Juliette Bouli-Mengue<br>
    Clinical Research Associate → Healthcare Data Science
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    require_data()
    st.markdown("# METABRIC Survival Dashboard")
    st.markdown("**Survival analysis and ML prediction on breast cancer**")
    st.markdown("---")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>1,904</div>
            <div class='metric-label'>Patients</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>57.9%</div>
            <div class='metric-label'>Event rate</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>154 months</div>
            <div class='metric-label'>Median survival</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{model_bundle['metrics']['roc_auc']:.2f}</div>
            <div class='metric-label'>ROC-AUC (RF)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("<div class='section-header'>Research questions</div>", unsafe_allow_html=True)
        st.markdown("""
        **1. Survival Analysis**
        Which clinical and genomic factors are associated with overall survival?

        **2. Machine Learning**
        Can 5-year survival be predicted from multimodal data (clinical + genomic)?
        """)

        st.markdown("<div class='section-header'>Pipeline</div>", unsafe_allow_html=True)
        steps = {
            "01 EDA": "Exploration, missing values, distributions",
            "02 Preprocessing": "Imputation, encoding, feature selection",
            "03 Kaplan-Meier": "Survival curves by subgroup",
            "04 Cox": "Multivariate model, hazard ratios",
            "05 ML": "Random Forest, XGBoost, feature importances"
        }
        for k, v in steps.items():
            st.markdown(f"**`{k}`** — {v}")

    with col_right:
        st.markdown("<div class='section-header'>Cohort distribution</div>", unsafe_allow_html=True)

        # Overall survival distribution
        er_counts = df_raw["er_status"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=er_counts.index,
            values=er_counts.values,
            hole=0.55,
            marker_colors=["#4a9eff", "#e05555"],
            textfont_size=12
        ))
        fig_pie.update_layout(
            title="ER Status",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8d6e8",
            height=280,
            margin=dict(t=40, b=0, l=0, r=0),
            legend=dict(font=dict(color="#8892a4"))
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("""
        <div class='insight-box'>
        ER+ is the majority (~77%) — consistent with breast cancer epidemiology.
        This is a mature cohort: median follow-up 9.6 years, max 29.6 years.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — SURVIVAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Survival Analysis":
    require_data()
    st.markdown("# Survival Analysis — Kaplan-Meier")
    st.markdown("Compare survival curves between clinical subgroups.")
    st.markdown("---")

    # Selector
    variable = st.selectbox(
        "Stratification variable",
        ["ER Status", "HER2 Status", "PR Status", "Histologic Grade",
         "Chemotherapy", "Hormone Therapy"]
    )

    # Variable → data mapping
    config = {
        "ER Status": {
            "col": "er_status_label", "groups": ["Positive", "Negative"],
            "colors": ["#4a9eff", "#e05555"], "labels": ["ER+", "ER-"]
        },
        "HER2 Status": {
            "col": "her2_status_label", "groups": ["Positive", "Negative"],
            "colors": ["#f0a030", "#4a9eff"], "labels": ["HER2+", "HER2-"]
        },
        "PR Status": {
            "col": "pr_status_label", "groups": ["Positive", "Negative"],
            "colors": ["#50c878", "#e05555"], "labels": ["PR+", "PR-"]
        },
        "Histologic Grade": {
            "col": "neoplasm_histologic_grade_label",
            "groups": [1.0, 2.0, 3.0],
            "colors": ["#50c878", "#f0a030", "#e05555"],
            "labels": ["Grade 1", "Grade 2", "Grade 3"]
        },
        "Chemotherapy": {
            "col": "chemotherapy", "groups": [1, 0],
            "colors": ["#8172B2", "#4a9eff"], "labels": ["With chemo", "Without chemo"]
        },
        "Hormone Therapy": {
            "col": "hormone_therapy", "groups": [1, 0],
            "colors": ["#50c878", "#e05555"], "labels": ["With hormone therapy", "Without hormone therapy"]
        }
    }

    cfg = config[variable]

    # KM computation
    kmf = KaplanMeierFitter()
    fig = go.Figure()

    n_groups = []
    for group, color, label in zip(cfg["groups"], cfg["colors"], cfg["labels"]):
        mask = df_plot[cfg["col"]] == group
        n = mask.sum()
        n_groups.append(n)

        kmf.fit(
            durations=df_plot.loc[mask, "overall_survival_months"],
            event_observed=df_plot.loc[mask, "event"]
        )

        t = kmf.survival_function_.index
        s = kmf.survival_function_["KM_estimate"]
        ci_low = kmf.confidence_interval_["KM_estimate_lower_0.95"]
        ci_high = kmf.confidence_interval_["KM_estimate_upper_0.95"]

        # Confidence interval band
        fig.add_trace(go.Scatter(
            x=list(t) + list(t[::-1]),
            y=list(ci_high) + list(ci_low[::-1]),
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.12),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter(
            x=t, y=s,
            mode="lines",
            name=f"{label} (n={n})",
            line=dict(color=color, width=2.5)
        ))

    # Log-rank test (2 groups only)
    p_text = ""
    if len(cfg["groups"]) == 2:
        mask_a = df_plot[cfg["col"]] == cfg["groups"][0]
        mask_b = df_plot[cfg["col"]] == cfg["groups"][1]
        r = logrank_test(
            df_plot.loc[mask_a, "overall_survival_months"],
            df_plot.loc[mask_b, "overall_survival_months"],
            event_observed_A=df_plot.loc[mask_a, "event"],
            event_observed_B=df_plot.loc[mask_b, "event"]
        )
        p_val = r.p_value
        p_text = f"p = {p_val:.2e}"

    fig.update_layout(
        title=f"KM Curves — {variable}  |  Log-rank {p_text}",
        xaxis_title="Time (months)",
        yaxis_title="Survival probability",
        yaxis_range=[0, 1],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c8d6e8",
        xaxis=dict(gridcolor="#1e2d40", zerolinecolor="#1e2d40"),
        yaxis=dict(gridcolor="#1e2d40", zerolinecolor="#1e2d40"),
        legend=dict(font=dict(color="#8892a4"), bgcolor="rgba(0,0,0,0)"),
        height=480
    )

    st.plotly_chart(fig, use_container_width=True)

    # Dynamic interpretations
    interpretations = {
        "ER Status": """
        <b>p=0.02</b> — Significant difference. The curves converge/cross late
        (roughly 150–200 months; the visual crossing time is imprecise):
        a late-relapse phenomenon in ER+ patients, well documented in METABRIC.
        The proportional hazards assumption is violated for this variable.
        """,
        "HER2 Status": """
        <b>p=2.22e-05</b> — Highly significant difference. HER2+ (n=236):
        markedly worse prognosis. Consistent with a pre-trastuzumab cohort
        where anti-HER2 therapies were not yet standardized.
        """,
        "PR Status": """
        <b>p=7.45e-05</b> — Highly significant difference. PR+ patients survive
        better in the short and medium term. The curves converge without crossing →
        proportional hazards holds → good candidate for Cox regression.
        """,
        "Histologic Grade": """
        Survival gradient Grade 1 > Grade 2 > Grade 3 — consistent with clinical expectations.
        Grade 3 (poorly differentiated tumors): early and rapid decline.
        """,
        "Chemotherapy": """
        <b>Indication bias</b>: chemotherapy is prescribed for the most aggressive tumors
        → treated patients have a worse baseline prognosis. This result
        cannot be interpreted causally without multivariate adjustment (Cox).
        """,
        "Hormone Therapy": """
        <b>Reverse indication bias</b>: hormone therapy is prescribed to ER+ patients
        who have a better baseline prognosis. The treatment's true effect
        can only be isolated using the multivariate Cox model.
        """
    }

    st.markdown(f"""
    <div class='insight-box'>
    {interpretations[variable]}
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML PREDICTION
# ══════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Prediction":
    st.markdown("# Prediction — 5-Year Survival")
    st.markdown("Enter a patient's profile to estimate their 5-year survival probability.")
    st.markdown("---")

    col_inputs, col_result = st.columns([1, 1])

    with col_inputs:
        st.markdown("<div class='section-header'>Clinical profile</div>", unsafe_allow_html=True)

        age = st.slider("Age at diagnosis", 20, 95, 55)
        tumor_size = st.slider("Tumor size (mm)", 1, 150, 25)
        lymph_nodes = st.slider("Positive lymph nodes", 0, 20, 0)
        grade = st.selectbox("Histologic grade", [1, 2, 3], index=1)
        mutation_count = st.slider("Mutation count", 0, 30, 5)

        st.markdown("<div class='section-header'>Biomarkers</div>", unsafe_allow_html=True)
        er = st.radio("ER Status", ["Positive", "Negative"], horizontal=True)
        pr = st.radio("PR Status", ["Positive", "Negative"], horizontal=True)
        her2 = st.radio("HER2 Status", ["Negative", "Positive"], horizontal=True)

        st.markdown("<div class='section-header'>Treatments</div>", unsafe_allow_html=True)
        chemo = st.checkbox("Chemotherapy")
        hormone = st.checkbox("Hormone therapy")
        radio = st.checkbox("Radiotherapy")

    with col_result:
        st.markdown("<div class='section-header'>Result</div>", unsafe_allow_html=True)

        # The saved sklearn pipeline owns imputation, encoding, scaling and
        # gene selection. Defaults were learned from the training fold only.
        input_data = pd.DataFrame([model_bundle["defaults"]], columns=feature_cols)
        input_data.loc[0, "age_at_diagnosis"] = age
        input_data.loc[0, "tumor_size"] = np.log1p(tumor_size)
        input_data.loc[0, "lymph_nodes_examined_positive"] = lymph_nodes
        input_data.loc[0, "neoplasm_histologic_grade"] = float(grade)
        input_data.loc[0, "mutation_count"] = np.log1p(mutation_count)
        input_data.loc[0, "er_status"] = er
        input_data.loc[0, "pr_status"] = pr
        input_data.loc[0, "her2_status"] = her2
        input_data.loc[0, "chemotherapy"] = int(chemo)
        input_data.loc[0, "hormone_therapy"] = int(hormone)
        input_data.loc[0, "radio_therapy"] = int(radio)

        # Prediction
        # The forest is class-weighted, so its raw probabilities under-estimate survival
        # (mean about 64% vs 78% observed). A Platt map fitted on out-of-fold training
        # predictions recalibrates them before they are read as a risk.
        proba = rf.predict_proba(input_data)[0][1]
        calibrator = model_bundle.get("calibrator")
        if calibrator is not None:
            clipped = float(np.clip(proba, 1e-4, 1 - 1e-4))
            proba = calibrator.predict_proba([[np.log(clipped / (1 - clipped))]])[0][1]
        pct = int(proba * 100)

        # 5-year survival is the majority outcome (~78% of the cohort): compare with that base rate
        BASE_RATE = 0.777
        favorable = proba >= BASE_RATE
        css_class = "pred-high" if favorable else "pred-low"
        color = "#50c878" if favorable else "#e05555"
        verdict = "Above cohort average (78%)" if favorable else "Below cohort average (78%)"

        st.markdown(f"""
        <div class='{css_class}'>
            <div class='pred-value' style='color:{color}'>{pct}%</div>
            <div style='color:{color}; font-size:0.9rem; margin-top:8px; font-weight:500'>
                {verdict}
            </div>
            <div style='color:#5a6a7a; font-size:0.78rem; margin-top:6px'>
                5-year survival probability
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            domain={"x": [0, 1], "y": [0, 1]},
            number={"suffix": "%", "font": {"color": color, "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#5a6a7a"},
                "bar": {"color": color},
                "bgcolor": "#161b27",
                "bordercolor": "#1e2d40",
                "steps": [
                    {"range": [0, 40], "color": "#2b0d0d"},
                    {"range": [40, 60], "color": "#1f1a0d"},
                    {"range": [60, 100], "color": "#0d2b1a"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.8,
                    "value": 50
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c8d6e8",
            height=220,
            margin=dict(t=20, b=0, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div class='warning-box'>
        ⚠️ <b>Educational use only.</b> This model is trained on historical
        data (METABRIC, 2000-2010) and is not a clinical decision-making tool.
        ROC-AUC = {model_bundle['metrics']['roc_auc']:.3f}
        (bootstrap 95% CI {model_bundle['metrics']['roc_auc_95_ci'][0]:.3f}–{model_bundle['metrics']['roc_auc_95_ci'][1]:.3f})
        on the held-out test set.
        </div>
        """, unsafe_allow_html=True)

        # Top 5 feature importances
        st.markdown("<div class='section-header'>Top 5 factors (global model)</div>", unsafe_allow_html=True)
        importances = model_bundle["feature_importances"]
        top5 = importances.sort_values(ascending=False).head(5)

        fig_imp = go.Figure(go.Bar(
            x=top5.values[::-1],
            y=top5.index[::-1],
            orientation="h",
            marker_color=["#4a9eff"] * 5
        ))
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#8892a4",
            xaxis=dict(gridcolor="#1e2d40"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            height=220,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_imp, use_container_width=True)
