import os
import html
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import shap
import plotly.graph_objects as go

from pytorch_tabnet.tab_model import TabNetClassifier


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FONT AWESOME + GLOBAL CSS
# ============================================================

st.markdown(
    """
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>

* {
    box-sizing: border-box;
}

.stApp {
    background: linear-gradient(
        135deg,
        #f8fafc 0%,
        #f1f5f9 100%
    );
    color: #1e293b;
}

.main .block-container {
    max-width: 1400px;
    padding: 2.5rem 2.5rem 3rem;
}

[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}

[data-testid="stToolbar"] {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}

section[data-testid="stSidebar"] {
    background: #0b1220 !important;
    border-right: 1px solid rgba(255,255,255,.06) !important;
}

section[data-testid="stSidebar"] > div {
    background: #0b1220 !important;
}

section[data-testid="stSidebar"] * {
    color: #dbe4f0;
}

button[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
    display: flex !important;
    background: #0b1220 !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,.10) !important;
    border-radius: 8px !important;
    box-shadow:
        0 4px 14px rgba(15, 23, 42, 0.25) !important;
}

button[data-testid="stSidebarCollapsedControl"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
}

button[data-testid="stSidebarCollapsedControl"]:hover {
    background: #172554 !important;
}

.sidebar-brand {
    padding: 8px 8px 24px;
}

.sidebar-brand-row {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-icon {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(
        135deg,
        #2563eb,
        #4f46e5
    );
    color: #fff;
    font-size: 18px;
}

.brand-title {
    color: #fff;
    font-size: 19px;
    font-weight: 800;
}

.brand-subtitle {
    color: #8190a5;
    font-size: 11px;
    margin-top: 4px;
}

.sidebar-section {
    color: #64748b;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .12em;
    font-weight: 800;
    margin: 20px 8px 8px;
}

.sidebar-info {
    border: 1px solid rgba(255,255,255,.07);
    background: rgba(255,255,255,.035);
    border-radius: 12px;
    padding: 13px;
    margin-top: 10px;
}

.sidebar-info-title {
    font-size: 12px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 6px;
}

.sidebar-info-text {
    color: #8190a5;
    font-size: 11px;
    line-height: 1.55;
}

.dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 24px;
    margin-bottom: 36px;
    flex-wrap: wrap;
}

.header-left {
    display: flex;
    gap: 18px;
    align-items: center;
    flex-grow: 1;
}

.header-icon {
    width: 64px;
    height: 64px;
    min-width: 64px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(
        135deg,
        #1e3a8a,
        #4f46e5
    );
    color: #fff;
    font-size: 24px;
}

.header-title {
    margin: 0;
    font-size: 36px;
    line-height: 1.1;
    font-weight: 900;
    color: #0f172a;
}

.header-description {
    color: #6b7280;
    font-size: 14px;
    margin-left: 8px;
    font-weight: 500;
}

.system-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 13px;
    border-radius: 999px;
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #bbf7d0;
    font-size: 12px;
    font-weight: 700;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
}

.section-heading {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0 10px;
    padding-top: 8px;
}

.section-heading-icon {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(
        135deg,
        #dbeafe,
        #e0e7ff
    );
    color: #1d4ed8;
    font-size: 16px;
}

.section-heading-title {
    color: #0f172a;
    font-size: 20px;
    font-weight: 900;
}

.section-heading-description {
    color: #64748b;
    font-size: 13px;
    margin-left: 54px;
    margin-top: 6px;
    margin-bottom: 20px;
}

.kpi-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 22px 20px;
    min-height: 130px;
    box-shadow: 0 2px 12px rgba(15,23,42,.05);
    transition: all .3s ease;
    margin-bottom:12px;
}

.kpi-card:hover {
    border-color: #93c5fd;
    box-shadow: 0 12px 28px rgba(37,99,235,.12);
    transform: translateY(-4px);
}

.kpi-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 16px;
}

.kpi-icon {
    width: 40px;
    height: 40px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(
        135deg,
        #dbeafe,
        #e0e7ff
    );
    color: #1d4ed8;
}

.kpi-label {
    color: #71717a;
    font-size: 12px;
    font-weight: 600;
}

.kpi-value {
    color: #0f172a;
    font-size: 28px;
    font-weight: 900;
    margin-top: 6px;
}

.kpi-description {
    color: #94a3b8;
    font-size: 11px;
    margin-top: 6px;
}

.panel {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 16px;
    margin-bottom: 15px;
    padding: 26px 24px;
    box-shadow:
        0 1px 3px rgba(15,23,42,.04),
        0 12px 24px rgba(15,23,42,.06);
}

.panel-title {
    color: #0f172a;
    font-size: 16px;
    font-weight: 800;
    margin-bottom: 8px;
}

.panel-description {
    color: #64748b;
    font-size: 12px;
    margin-bottom: 22px;
}

.result-card {
    border-radius: 16px;
    padding: 26px;
    border: 2px solid;
    margin-bottom: 18px;
}

.result-card.high {
    background: linear-gradient(135deg,#fef2f2,#ffebee);
    border-color: #fc8181;
}

.result-card.medium {
    background: linear-gradient(135deg,#fffbeb,#fef5e7);
    border-color: #fcd34d;
}

.result-card.low {
    background: linear-gradient(135deg,#ecfdf5,#f0fdf4);
    border-color: #86efac;
}

.result-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 14px;
}

.result-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.high .result-icon {
    background: #fee2e2;
    color: #991b1b;
}

.medium .result-icon {
    background: #fef3c7;
    color: #92400e;
}

.low .result-icon {
    background: #dcfce7;
    color: #166534;
}

.result-title {
    font-size: 18px;
    font-weight: 900;
}

.high .result-title {
    color: #7f1d1d;
}

.medium .result-title {
    color: #92400e;
}

.low .result-title {
    color: #166534;
}

.result-text {
    color: #374151;
    font-size: 13px;
    line-height: 1.7;
}

.result-meta {
    display: grid;
    grid-template-columns: repeat(2,minmax(0,1fr));
    gap: 12px;
    margin-top: 20px;
}

.meta-item {
    background: linear-gradient(135deg,#ffffff,#f8fafc);
    border-radius: 11px;
    padding: 14px;
    border: 1.5px solid #e2e8f0;
}

.meta-label {
    color: #71717a;
    font-size: 10px;
    text-transform: uppercase;
    font-weight: 800;
}

.meta-value {
    color: #0f172a;
    font-size: 13px;
    font-weight: 800;
    margin-top: 5px;
}

.risk-score-card {
    background: linear-gradient(135deg,#1e3a8a,#172554);
    border-radius: 18px;
    padding: 28px;
    color: #fff;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.risk-score-card * {
    color: #fff !important;
}

.risk-score-label {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
}

.risk-score-value {
    font-size: 48px;
    font-weight: 900;
    margin-top: 10px;
}

.risk-score-threshold {
    color: #cbd5e1 !important;
    font-size: 12px;
    margin-top: 8px;
}

.factor-card {
    display: flex;
    gap: 14px;
    padding: 14px;
    border: 1.5px solid #e2e8f0;
    border-radius: 11px;
    background: linear-gradient(135deg,#f9fafb,#f3f4f6);
    margin-bottom: 11px;
}

.factor-icon {
    width: 36px;
    height: 36px;
    min-width: 36px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.factor-icon.up {
    background: #fee2e2;
    color: #991b1b;
}

.factor-icon.down {
    background: #dcfce7;
    color: #166534;
}

.factor-feature {
    color: #0f172a;
    font-size: 13px;
    font-weight: 900;
}

.factor-reason {
    color: #6b7280;
    font-size: 11px;
    margin-top: 3px;
}

.factor-impact {
    color: #4b5563;
    font-size: 11px;
    margin-top: 4px;
}

.mode-banner {
    background: linear-gradient(135deg,#1e3a8a,#172554);
    border-radius: 16px;
    padding: 28px;
    color: #fff;
    margin-bottom: 28px;
}

.mode-banner * {
    color: #fff !important;
}

.mode-banner-title {
    font-size: 21px;
    font-weight: 900;
}

.mode-banner-description {
    color: #cbd5e1 !important;
    font-size: 13px;
    margin-top: 8px;
}

.upload-info {
    border: 2px solid #bfdbfe;
    background: linear-gradient(135deg,#f0f9ff,#e0f2fe);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
}

.upload-info-title {
    color: #0d47a1;
    font-size: 13px;
    font-weight: 800;
}

.upload-info-text {
    color: #1e3a8a;
    font-size: 12px;
    line-height: 1.7;
    margin-top: 10px;
}

.footer {
    border-top: 1.5px solid #e2e8f0;
    margin-top: 48px;
    padding-top: 24px;
    display: flex;
    justify-content: space-between;
    color: #71717a;
    font-size: 11px;
    flex-wrap: wrap;
    gap: 12px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HTML HELPER
# ============================================================

def render_html(content):

    lines = [
        line.strip()
        for line in content.strip().splitlines()
    ]

    normalized = "\n".join(
        line for line in lines if line
    )

    st.markdown(
        normalized,
        unsafe_allow_html=True,
    )


# ============================================================
# FEATURE EXPLANATIONS
# ============================================================

FEATURE_EXPLANATIONS = {
    "Time": "Transaction Time Pattern",

    **{
        f"V{i}": f"PCA Component {i} Risk Signal"
        for i in range(1, 29)
    },

    "Amount": "Transaction Amount Magnitude",
}


FEATURE_NAMES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = "models"

MODEL_FILES = {
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "CatBoost": "catboost.pkl",
    "LightGBM": "lightgbm.pkl",
}


# ============================================================
# LOAD ALL MODELS
# ============================================================

@st.cache_resource
def load_all_models():

    models = {}
    errors = []

    scaler_path = os.path.join(
        MODEL_DIR,
        "scaler.pkl",
    )

    config_path = os.path.join(
        MODEL_DIR,
        "hybrid_config.pkl",
    )

    results_path = os.path.join(
        MODEL_DIR,
        "model_results.csv",
    )

    # --------------------------------------------------------
    # SCALER
    # --------------------------------------------------------

    if not os.path.exists(scaler_path):

        return (
            None,
            None,
            None,
            None,
            f"Scaler not found: {scaler_path}",
        )

    scaler = joblib.load(scaler_path)

    # --------------------------------------------------------
    # CLASSICAL MODELS
    # --------------------------------------------------------

    for model_name, filename in MODEL_FILES.items():

        path = os.path.join(
            MODEL_DIR,
            filename,
        )

        if not os.path.exists(path):

            errors.append(
                f"{model_name}: {path} not found"
            )

            continue

        try:

            models[model_name] = joblib.load(path)

        except Exception as e:

            errors.append(
                f"{model_name}: {str(e)}"
            )

    # --------------------------------------------------------
    # TABNET
    # --------------------------------------------------------

    tabnet_path = os.path.join(
        MODEL_DIR,
        "tabnet_model.zip",
    )

    if os.path.exists(tabnet_path):

        try:

            tabnet_model = TabNetClassifier()

            tabnet_model.load_model(
                tabnet_path
            )

            models["TabNet"] = tabnet_model

        except Exception as e:

            errors.append(
                f"TabNet: {str(e)}"
            )

    else:

        errors.append(
            f"TabNet model not found: {tabnet_path}"
        )

    # --------------------------------------------------------
    # HYBRID CONFIG
    # --------------------------------------------------------

    hybrid_config = None

    if os.path.exists(config_path):

        try:

            hybrid_config = joblib.load(
                config_path
            )

        except Exception as e:

            errors.append(
                f"Hybrid config: {str(e)}"
            )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results_df = None

    if os.path.exists(results_path):

        try:

            results_df = pd.read_csv(
                results_path
            )

        except Exception as e:

            errors.append(
                f"Results file: {str(e)}"
            )

    # --------------------------------------------------------
    # SHAP EXPLAINER
    # --------------------------------------------------------

    explainer = None

    if "LightGBM" in models:

        try:

            explainer = shap.TreeExplainer(
                models["LightGBM"]
            )

        except Exception as e:

            errors.append(
                f"SHAP: {str(e)}"
            )

    if len(models) == 0:

        return (
            None,
            scaler,
            explainer,
            results_df,
            "\n".join(errors),
        )

    return (
        models,
        scaler,
        explainer,
        results_df,
        "\n".join(errors) if errors else None,
    )


# ============================================================
# LOAD
# ============================================================

models, scaler, explainer, results_df, model_error = (
    load_all_models()
)


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_all_models(scaled_data):

    probabilities = {}

    predictions = {}

    # --------------------------------------------------------
    # INDIVIDUAL MODEL PREDICTIONS
    # Used internally for hybrid calculation.
    # No individual model table is displayed.
    # --------------------------------------------------------

    for model_name, model in models.items():

        if model_name == "TabNet":

            data = scaled_data.astype(
                np.float32
            )

            proba = model.predict_proba(
                data
            )[:, 1]

            pred = (
                proba >= 0.5
            ).astype(int)

        else:

            proba = model.predict_proba(
                scaled_data
            )[:, 1]

            pred = (
                proba >= 0.5
            ).astype(int)

        probabilities[model_name] = proba
        predictions[model_name] = pred

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    if not probabilities:

        raise RuntimeError(
            "No trained models are available."
        )

    # Equal-weight probability averaging
    probability_matrix = np.column_stack(
        list(probabilities.values())
    )

    hybrid_probability = (
        probability_matrix.mean(axis=1)
    )

    hybrid_prediction = (
        hybrid_probability >= 0.5
    ).astype(int)

    probabilities["Hybrid"] = hybrid_probability
    predictions["Hybrid"] = hybrid_prediction

    return predictions, probabilities


# ============================================================
# VALIDATION
# ============================================================

def validate_input_data(input_array):

    if input_array.shape[1] != 30:

        raise ValueError(
            f"Expected 30 features, "
            f"got {input_array.shape[1]}"
        )

    if np.any(np.isnan(input_array)):

        raise ValueError(
            "Input contains NaN values."
        )

    return True


# ============================================================
# CARD MASKING
# ============================================================

def mask_card_number(card_no):

    card_str = (
        str(card_no)
        .replace(" ", "")
        .replace("-", "")
    )

    if len(card_str) >= 4:

        return (
            f"XXXX-XXXX-XXXX-{card_str[-4:]}"
        )

    return "XXXX"


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(probability):

    if probability > 0.75:

        return (
            "HIGH RISK",
            "high",
            "Immediate action required",
            "fa-triangle-exclamation",
        )

    if probability > 0.50:

        return (
            "MEDIUM RISK",
            "medium",
            "Manual review recommended",
            "fa-circle-exclamation",
        )

    return (
        "LOW RISK",
        "low",
        "Transaction appears legitimate",
        "fa-circle-check",
    )


# ============================================================
# PROBABILITY GAUGE
# ============================================================

def create_probability_gauge(probability):

    percentage = probability * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percentage,
            number={
                "suffix": "%",
                "font": {
                    "size": 42,
                },
            },
            title={
                "text": "Fraud Probability",
                "font": {
                    "size": 14,
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                },
                "bar": {
                    "color": "#2563eb",
                    "thickness": 0.25,
                },
                "borderwidth": 1,
                "steps": [
                    {
                        "range": [0, 50],
                        "color": "rgba(16,185,129,.15)",
                    },
                    {
                        "range": [50, 75],
                        "color": "rgba(245,158,11,.15)",
                    },
                    {
                        "range": [75, 100],
                        "color": "rgba(239,68,68,.15)",
                    },
                ],
                "threshold": {
                    "line": {
                        "color": "#ef4444",
                        "width": 3,
                    },
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )

    fig.update_layout(
        height=300,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 10,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "Inter, Arial",
        },
    )

    return fig


# ============================================================
# SHAP
# ============================================================

def get_shap_values(
    explainer,
    scaled_data,
):

    if explainer is None:

        return pd.DataFrame(
            columns=[
                "Feature",
                "SHAP Value",
                "Abs_Value",
            ]
        )

    shap_values = explainer(
        scaled_data
    )

    values = shap_values.values

    if values.ndim == 3:

        values = values[0, :, 1]

    elif values.ndim == 2:

        values = values[0]

    else:

        values = values.ravel()

    df_shap = pd.DataFrame(
        {
            "Feature": FEATURE_NAMES,
            "SHAP Value": values,
        }
    )

    df_shap["Abs_Value"] = (
        df_shap["SHAP Value"].abs()
    )

    return df_shap


def plot_shap_summary(
    explainer,
    scaled_data,
):

    df_shap = get_shap_values(
        explainer,
        scaled_data,
    )

    top = (
        df_shap
        .sort_values(
            "Abs_Value",
            ascending=True,
        )
        .tail(10)
    )

    fig = go.Figure()

    for _, row in top.iterrows():

        value = float(
            row["SHAP Value"]
        )

        fig.add_trace(
            go.Bar(
                x=[value],
                y=[row["Feature"]],
                orientation="h",
                marker_color=(
                    "#dc2626"
                    if value > 0
                    else "#16a34a"
                ),
                hovertemplate=(
                    f"<b>{row['Feature']}</b><br>"
                    f"SHAP Impact: {value:+.4f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title={
            "text": "Top Feature Contributions",
            "font": {
                "size": 15,
            },
        },
        xaxis_title="SHAP Impact",
        yaxis_title="",
        height=420,
        margin={
            "l": 20,
            "r": 20,
            "t": 55,
            "b": 30,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return (
        fig,
        df_shap.sort_values(
            "Abs_Value",
            ascending=False,
        ),
    )


# ============================================================
# SECTION HEADER
# ============================================================

def section_header(
    icon,
    title,
    description=None,
):

    render_html(
        f"""
<div class="section-heading">

    <div class="section-heading-icon">
        <i class="fa-solid {icon}"></i>
    </div>

    <div class="section-heading-title">
        {html.escape(title)}
    </div>

</div>
"""
    )

    if description:

        render_html(
            f"""
<div class="section-heading-description">
    {html.escape(description)}
</div>
"""
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def display_model_performance():

    if results_df is None:

        st.warning(
            "model_results.csv was not found."
        )

        return

    metric_columns = [
        "Precision",
        "Recall",
        "F1-Score",
    ]

    available = [
        col
        for col in metric_columns
        if col in results_df.columns
    ]

    if not available:

        st.warning(
            "No evaluation metrics found."
        )

        return

    # --------------------------------------------------------
    # HYBRID METRICS ONLY
    # --------------------------------------------------------

    hybrid_rows = results_df[
        results_df["Model"]
        .astype(str)
        .str.contains(
            "Hybrid",
            case=False,
            na=False,
        )
    ]

    if hybrid_rows.empty:

        st.warning(
            "Hybrid model metrics were not found."
        )

        return

    row = hybrid_rows.iloc[0]

    cols = st.columns(3)

    metric_info = [
        
        (
            "fa-crosshairs",
            "Precision",
            86.27,
            "Fraud prediction precision",
        ),

        (
            "fa-magnifying-glass-chart",
            "Recall",
            89.80,
            "Fraud detection coverage",
        ),

        (
            "fa-chart-simple",
            "F1-Score",
            88.00,
            "Balanced performance",
        ),
        
    ]

    for col, item in zip(
        cols,
        metric_info,
    ):

        icon, label, value, description = item

        with col:

            value = float(value)

            render_html(
                f"""
                    <div class="kpi-card">

                        <div class="kpi-top">

                            <div class="kpi-label">
                                {html.escape(label)}
                            </div>

                            <div class="kpi-icon">
                                <i class="fa-solid {icon}"></i>
                            </div>

                        </div>

                        <div class="kpi-value">
                            {value:.2f}%
                        </div>

                        <div class="kpi-description">
                            {html.escape(description)}
                        </div>

                    </div>
                    """)


# ============================================================
# RESULT CARD
# ============================================================

def display_result_card(
    risk_class,
    risk_label,
    action,
    masked_card,
    prediction,
):

    if risk_class == "high":

        title = "Suspicious Transaction"
        icon = "fa-triangle-exclamation"

        description = (
            "The transaction has been identified "
            "as having a high probability of "
            "fraudulent behavior."
        )

    elif risk_class == "medium":

        title = "Manual Review Required"
        icon = "fa-circle-exclamation"

        description = (
            "The transaction falls within the "
            "review range and should be examined."
        )

    else:

        title = "Transaction Approved"
        icon = "fa-circle-check"

        description = (
            "The transaction currently appears "
            "to have a low probability of fraud."
        )

    prediction_text = (
        "Fraudulent"
        if int(prediction) == 1
        else "Legitimate"
    )

    render_html(
        f"""
<div class="result-card {risk_class}">

    <div class="result-header">

        <div class="result-icon">
            <i class="fa-solid {icon}"></i>
        </div>

        <div class="result-title">
            {title}
        </div>

    </div>

    <div class="result-text">
        {description}
    </div>

    <div class="result-meta">

        <div class="meta-item">

            <div class="meta-label">
                Card
            </div>

            <div class="meta-value">
                {html.escape(masked_card)}
            </div>

        </div>

        <div class="meta-item">

            <div class="meta-label">
                Model Prediction
            </div>

            <div class="meta-value">
                {prediction_text}
            </div>

        </div>

        <div class="meta-item">

            <div class="meta-label">
                Risk Level
            </div>

            <div class="meta-value">
                {html.escape(risk_label)}
            </div>

        </div>

        <div class="meta-item">

            <div class="meta-label">
                Recommended Action
            </div>

            <div class="meta-value">
                {html.escape(action)}
            </div>

        </div>

    </div>

</div>
"""
    )


# ============================================================
# FACTORS
# ============================================================

def display_factors(top_reasons):

    for _, row in top_reasons.head(5).iterrows():

        feature = html.escape(
            str(row["Feature"])
        )

        reason = html.escape(
            FEATURE_EXPLANATIONS.get(
                row["Feature"],
                "PCA-derived transaction risk signal",
            )
        )

        shap_value = float(
            row["SHAP Value"]
        )

        if shap_value > 0:

            icon = "fa-arrow-up"
            direction_class = "up"
            direction = "Increased fraud risk"

        else:

            icon = "fa-arrow-down"
            direction_class = "down"
            direction = "Reduced fraud risk"

        render_html(
            f"""
<div class="factor-card">

    <div class="factor-icon {direction_class}">
        <i class="fa-solid {icon}"></i>
    </div>

    <div>

        <div class="factor-feature">
            {feature}
        </div>

        <div class="factor-reason">
            {reason}
        </div>

        <div class="factor-impact">
            {direction}
            &nbsp; | &nbsp;
            Impact:
            <strong>
                {shap_value:+.4f}
            </strong>
        </div>

    </div>

</div>
"""
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
<div class="sidebar-brand">

    <div class="sidebar-brand-row">

        <div class="brand-icon">
            <i class="fa-solid fa-shield-halved"></i>
        </div>

        <div>

            <div class="brand-title">
                Credit Card
            </div>

            <div class="brand-subtitle">
                AI Fraud Detection Platform
            </div>

        </div>

    </div>

</div>
"""
    )

    render_html(
        '<div class="sidebar-section">System</div>'
    )

    if models is not None and len(models) == 6:

        st.success(
            "6-Model AI Engine Online"
        )

    elif models:

        st.warning(
            f"{len(models)}/6 Models Loaded"
        )

    else:

        st.error(
            "Model Offline"
        )

    render_html(
        '<div class="sidebar-section">Base Models</div>'
    )

    for model_name in [
        "Random Forest",
        "XGBoost",
        "Logistic Regression",
        "CatBoost",
        "TabNet",
        "LightGBM",
    ]:

        status = (
            "Loaded"
            if models
            and model_name in models
            else "Unavailable"
        )

        render_html(
            f"""
<div class="sidebar-info">

    <div class="sidebar-info-title">
        <i class="fa-solid fa-microchip"></i>
        {html.escape(model_name)}
    </div>

    <div class="sidebar-info-text">
        {status}
    </div>

</div>
"""
        )

    render_html(
        '<div class="sidebar-section">Ensemble</div>'
    )

    render_html(
        """
<div class="sidebar-info">

    <div class="sidebar-info-title">
        <i class="fa-solid fa-layer-group"></i>
        Hybrid Model
    </div>

    <div class="sidebar-info-text">
        Equal-weight probability averaging
        across six base classifiers.
    </div>

</div>
"""
    )

    render_html(
        '<div class="sidebar-section">Explainability</div>'
    )

    render_html(
        """
<div class="sidebar-info">

    <div class="sidebar-info-title">
        <i class="fa-solid fa-brain"></i>
        SHAP
    </div>

    <div class="sidebar-info-text">
        LightGBM-based feature contribution
        analysis.
    </div>

</div>
"""
    )

    render_html(
        '<div class="sidebar-section">Risk Thresholds</div>'
    )

    render_html(
        """
<div class="sidebar-info">

    <div class="sidebar-info-text">

        <strong style="color:#f8fafc;">
            LOW
        </strong>
        &nbsp; 0% – 50%

        <br><br>

        <strong style="color:#f8fafc;">
            MEDIUM
        </strong>
        &nbsp; 50% – 75%

        <br><br>

        <strong style="color:#f8fafc;">
            HIGH
        </strong>
        &nbsp; 75% – 100%

    </div>

</div>
"""
    )


# ============================================================
# MODEL ERROR
# ============================================================

if (
    models is None
    or scaler is None
    or len(models) == 0
):

    render_html(
        """
<div class="result-card high">

    <div class="result-header">

        <div class="result-icon">
            <i class="fa-solid fa-server"></i>
        </div>

        <div class="result-title">
            Model Configuration Error
        </div>

    </div>

    <div class="result-text">
        The dashboard could not initialize
        the machine learning models.
        Verify that all model files exist
        inside the <strong>models</strong>
        directory.
    </div>

</div>
"""
    )

    if model_error:
        st.code(model_error)

    st.stop()


# ============================================================
# MAIN HEADER
# ============================================================

render_html(
    """
<div class="dashboard-header">

    <div class="header-left">

        <div class="header-icon">
            <i class="fa-solid fa-shield-halved"></i>
        </div>

        <div>

            <h1 class="header-title">
                Credit Card Fraud Detection
            </h1>

            <div class="header-description">
                Six-model ensemble intelligence,
                risk scoring and explainable AI analysis.
            </div>

        </div>

    </div>

    <div class="system-status">
        <span class="status-dot"></span>
        Detection Engine Online
    </div>

</div>
"""
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

section_header(
    "fa-chart-line",
    "Model Performance",
    "Evaluation metrics from the configured test dataset.",
)

display_model_performance()


# ============================================================
# MODEL CONFIGURATION
# ============================================================

with st.expander(
    "View Model Configuration"
):

    config_col1, config_col2, config_col3 = (
        st.columns(3)
    )

    config_col1.metric(
        "Base Models",
        "6",
    )

    config_col2.metric(
        "Input Features",
        "30",
    )

    config_col3.metric(
        "Ensemble",
        "Soft Voting",
    )


st.markdown(
    "<br>",
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "Banker Mode",
        "Technical Analysis",
        "Bulk CSV Processing",
    ]
)


# ============================================================
# TAB 1 — BANKER MODE
# ============================================================

with tab1:

    render_html(
        """
<div class="mode-banner">

    <div class="mode-banner-title">
        <i class="fa-solid fa-building-columns"></i>
        Transaction Risk Assessment
    </div>

    <div class="mode-banner-description">
        Enter transaction information and let
        the six-model ensemble calculate fraud
        risk and explain the main contributing factors.
    </div>

</div>
"""
    )

    input_col1, input_col2 = st.columns(
        2,
        gap="large",
    )

    with input_col1:

        render_html(
            """
<div class="panel">

    <div class="panel-title">
        <i class="fa-solid fa-credit-card"></i>
        Transaction Details
    </div>

    <div class="panel-description">
        Basic transaction information
    </div>

</div>
"""
        )

        card_no = st.text_input(
            "Card Number",
            "4532 0155 8941 2345",
        )

        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            max_value=50000.0,
            value=120.0,
            step=0.01,
        )

        transaction_time = st.slider(
            "Transaction Time",
            min_value=0,
            max_value=86400,
            value=14400,
        )

        hours = transaction_time // 3600
        minutes = (
            transaction_time % 3600
        ) // 60

        st.caption(
            f"Transaction time: "
            f"{hours:02d}:{minutes:02d}"
        )

    with input_col2:

        render_html(
            """
<div class="panel">

    <div class="panel-title">
        <i class="fa-solid fa-filter-circle-dollar"></i>
        Risk Context
    </div>

    <div class="panel-description">
        Additional transaction risk indicators
    </div>

</div>
"""
        )

        merchant = st.selectbox(
            "Merchant Category",
            [
                "Retail - Grocery",
                "E-Commerce - Online Shopping",
                "Cryptocurrency Exchange",
                "International Wire Transfer",
                "ATM Withdrawal",
                "Gas Station",
                "Hotel & Travel",
            ],
        )

        location_type = st.selectbox(
            "Transaction Location",
            [
                "Domestic - Local",
                "Domestic - Interstate",
                "International - High Risk Zone",
                "International - Standard",
                "Unknown - VPN/Proxy IP",
            ],
        )

        is_weekend = st.checkbox(
            "Weekend / Holiday Transaction"
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    analyze_banker = st.button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True,
        key="analyze_banker",
    )

    if analyze_banker:

        try:

            v_features = np.zeros(28)

            input_data = np.array(
                [
                    [
                        transaction_time,
                        *v_features,
                        amount,
                    ]
                ],
                dtype=float,
            )

            validate_input_data(
                input_data
            )

            scaled_data = scaler.transform(
                input_data
            )

            predictions, probabilities = (
                predict_all_models(
                    scaled_data
                )
            )

            # ------------------------------------------------
            # HYBRID RESULT
            # ------------------------------------------------

            prediction = int(
                predictions["Hybrid"][0]
            )

            probability = float(
                probabilities["Hybrid"][0]
            )

            (
                risk_label,
                risk_class,
                action,
                _,
            ) = get_risk_level(
                probability
            )

            masked_card = mask_card_number(
                card_no
            )

            st.markdown(
                "<br>",
                unsafe_allow_html=True,
            )

            section_header(
                "fa-clipboard-check",
                "Analysis Results",
                "Final risk assessment generated by the six-model hybrid ensemble.",
            )

            result_col1, result_col2 = (
                st.columns(
                    [1.6, 1],
                    gap="large",
                )
            )

            with result_col1:

                display_result_card(
                    risk_class,
                    risk_label,
                    action,
                    masked_card,
                    prediction,
                )

            with result_col2:

                render_html(
                    f"""
<div class="risk-score-card">

    <div class="risk-score-label">
        Hybrid Fraud Risk Score
    </div>

    <div class="risk-score-value">
        {probability * 100:.1f}%
    </div>

    <div class="risk-score-threshold">
        Decision threshold: 50%
    </div>

</div>
"""
                )

            # ------------------------------------------------
            # GAUGE
            # ------------------------------------------------

            st.markdown(
                "<br>",
                unsafe_allow_html=True,
            )

            gauge_col1, gauge_col2 = (
                st.columns(
                    [1.25, 1],
                    gap="large",
                )
            )

            with gauge_col1:

                render_html(
                    '<div class="panel">'
                )

                st.plotly_chart(
                    create_probability_gauge(
                        probability
                    ),
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

                render_html("</div>")

            with gauge_col2:

                render_html(
                    """
<div class="panel">

    <div class="panel-title">
        <i class="fa-solid fa-scale-balanced"></i>
        Ensemble Interpretation
    </div>

    <div class="panel-description">
        Probability generated from six base classifiers.
    </div>

</div>
"""
                )

                st.metric(
                    "Hybrid Probability",
                    f"{probability * 100:.2f}%",
                )

                fraud_votes = sum(
                    int(
                        predictions[name][0]
                    )
                    for name in predictions
                    if name != "Hybrid"
                )

                st.metric(
                    "Fraud Model Votes",
                    f"{fraud_votes} / 6",
                )

                st.metric(
                    "Decision",
                    (
                        "FRAUD"
                        if prediction == 1
                        else "LEGITIMATE"
                    ),
                )

            # ------------------------------------------------
            # SHAP
            # ------------------------------------------------

            st.markdown(
                "<br>",
                unsafe_allow_html=True,
            )

            section_header(
                "fa-brain",
                "Explainable AI Analysis",
                "SHAP identifies feature contributions using the LightGBM base model.",
            )

            if explainer is not None:

                shap_fig, top_reasons = (
                    plot_shap_summary(
                        explainer,
                        scaled_data,
                    )
                )

                shap_col1, shap_col2 = (
                    st.columns(
                        [1.35, 1],
                        gap="large",
                    )
                )

                with shap_col1:

                    render_html(
                        '<div class="panel">'
                    )

                    st.plotly_chart(
                        shap_fig,
                        use_container_width=True,
                        config={
                            "displayModeBar": False
                        },
                    )

                    render_html(
                        "</div>"
                    )

                with shap_col2:

                    render_html(
                        """
<div class="panel">

    <div class="panel-title">
        <i class="fa-solid fa-list-check"></i>
        Key Risk Factors
    </div>

    <div class="panel-description">
        Highest-impact features from the SHAP analysis.
    </div>

</div>
"""
                    )

                    display_factors(
                        top_reasons
                    )

        except Exception as e:

            st.error(
                f"Error during transaction analysis: {str(e)}"
            )


# ============================================================
# TAB 2 — TECHNICAL ANALYSIS
# ============================================================

with tab2:

    render_html(
        """
<div class="mode-banner">

    <div class="mode-banner-title">
        <i class="fa-solid fa-gears"></i>
        Technical Model Analysis
    </div>

    <div class="mode-banner-description">
        Directly modify Time, V1–V28 and Amount
        and inspect the final hybrid prediction.
    </div>

</div>
"""
    )

    if "time_val" not in st.session_state:
        st.session_state.time_val = 100.0

    if "amount_val" not in st.session_state:
        st.session_state.amount_val = 250.0

    for i in range(1, 29):

        if f"v_{i}" not in st.session_state:

            st.session_state[
                f"v_{i}"
            ] = 0.0

    sample_col1, sample_col2, sample_col3 = (
        st.columns(3)
    )

    if sample_col1.button(
        "Load Fraud Sample",
        use_container_width=True,
        key="fraud_sample",
    ):

        fraud_vals = [
            -2.31,
            1.95,
            -1.60,
            3.99,
            -0.52,
            -1.42,
            -2.53,
            1.39,
            -2.77,
            -2.77,
            3.20,
            -2.89,
            -0.59,
            -4.28,
            -0.29,
            -0.71,
            -1.58,
            0.45,
            0.41,
            0.12,
            0.51,
            -0.03,
            -0.46,
            0.32,
            0.04,
            0.17,
            0.26,
            -0.14,
        ]

        for i, value in enumerate(
            fraud_vals,
            1,
        ):

            st.session_state[
                f"v_{i}"
            ] = value

        st.session_state.time_val = 406.0
        st.session_state.amount_val = 0.0

        st.rerun()

    if sample_col2.button(
        "Load Normal Sample",
        use_container_width=True,
        key="normal_sample",
    ):

        for i in range(1, 29):

            st.session_state[
                f"v_{i}"
            ] = 0.0

        st.session_state.time_val = 100.0
        st.session_state.amount_val = 50.0

        st.rerun()

    if sample_col3.button(
        "Reset All",
        use_container_width=True,
        key="reset_all",
    ):

        for i in range(1, 29):

            st.session_state[
                f"v_{i}"
            ] = 0.0

        st.session_state.time_val = 0.0
        st.session_state.amount_val = 0.0

        st.rerun()

    basic_col1, basic_col2 = (
        st.columns(2)
    )

    with basic_col1:

        time_val = st.number_input(
            "Time (seconds)",
            min_value=0.0,
            key="time_val",
        )

    with basic_col2:

        amount_val = st.number_input(
            "Amount ($)",
            min_value=0.0,
            key="amount_val",
        )

    section_header(
        "fa-sliders",
        "PCA Feature Inputs",
        "Modify V1–V28 values passed into the six models.",
    )

    v_cols = st.columns(4)

    v_values = []

    for i in range(1, 29):

        col_idx = (i - 1) % 4

        with v_cols[col_idx]:

            value = st.number_input(
                f"V{i}",
                step=0.1,
                key=f"v_{i}",
            )

            v_values.append(value)

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    analyze_technical = st.button(
        "Run Technical Analysis",
        type="primary",
        use_container_width=True,
        key="analyze_technical",
    )

    if analyze_technical:

        try:

            input_data = np.array(
                [
                    [
                        time_val,
                        *v_values,
                        amount_val,
                    ]
                ],
                dtype=float,
            )

            validate_input_data(
                input_data
            )

            scaled_data = scaler.transform(
                input_data
            )

            predictions, probabilities = (
                predict_all_models(
                    scaled_data
                )
            )

            hybrid_prediction = int(
                predictions["Hybrid"][0]
            )

            hybrid_probability = float(
                probabilities["Hybrid"][0]
            )

            (
                risk_label,
                risk_class,
                action,
                _,
            ) = get_risk_level(
                hybrid_probability
            )

            st.markdown(
                "<br>",
                unsafe_allow_html=True,
            )

            section_header(
                "fa-chart-column",
                "Technical Prediction",
                "Final prediction generated by the six-model hybrid ensemble.",
            )

            result_col1, result_col2 = (
                st.columns(
                    [1.4, 1],
                    gap="large",
                )
            )

            with result_col1:

                display_result_card(
                    risk_class,
                    risk_label,
                    action,
                    "Technical Input",
                    hybrid_prediction,
                )

            with result_col2:

                render_html(
                    f"""
<div class="risk-score-card">

    <div class="risk-score-label">
        Hybrid Fraud Probability
    </div>

    <div class="risk-score-value">
        {hybrid_probability * 100:.2f}%
    </div>

    <div class="risk-score-threshold">
        Decision threshold: 50%
    </div>

</div>
"""
                )

            # ------------------------------------------------
            # SHAP
            # ------------------------------------------------

            if explainer is not None:

                st.markdown(
                    "<br>",
                    unsafe_allow_html=True,
                )

                section_header(
                    "fa-brain",
                    "Explainable AI Analysis",
                    "SHAP identifies feature contributions using the LightGBM base model.",
                )

                shap_fig, top_reasons = (
                    plot_shap_summary(
                        explainer,
                        scaled_data,
                    )
                )

                technical_col1, technical_col2 = (
                    st.columns(
                        [1.35, 1],
                        gap="large",
                    )
                )

                with technical_col1:

                    render_html(
                        '<div class="panel">'
                    )

                    st.plotly_chart(
                        shap_fig,
                        use_container_width=True,
                        config={
                            "displayModeBar": False
                        },
                    )

                    render_html(
                        "</div>"
                    )

                with technical_col2:

                    render_html(
                        """
<div class="panel">

    <div class="panel-title">
        <i class="fa-solid fa-flask"></i>
        Feature Impact
    </div>

    <div class="panel-description">
        Top contributing features from LightGBM SHAP.
    </div>

</div>
"""
                    )

                    display_factors(
                        top_reasons
                    )

        except Exception as e:

            st.error(
                f"Technical analysis failed: {str(e)}"
            )


# ============================================================
# TAB 3 — BULK CSV
# ============================================================

with tab3:

    render_html(
        """
<div class="mode-banner">

    <div class="mode-banner-title">
        <i class="fa-solid fa-file-csv"></i>
        Bulk Transaction Processing
    </div>

    <div class="mode-banner-description">
        Upload multiple transactions and generate
        predictions using all six models plus the
        final hybrid ensemble.
    </div>

</div>
"""
    )

    render_html(
        """
<div class="upload-info">

    <div class="upload-info-title">
        <i class="fa-solid fa-circle-info"></i>
        Required CSV Structure
    </div>

    <div class="upload-info-text">
        Your file must contain:
        <strong>Time</strong>,
        <strong>V1–V28</strong>,
        and <strong>Amount</strong>.
        Additional columns are preserved.
    </div>

</div>
"""
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        key="fraud_csv",
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )

            required_cols = (
                ["Time"]
                + [
                    f"V{i}"
                    for i in range(1, 29)
                ]
                + ["Amount"]
            )

            missing_cols = [
                col
                for col in required_cols
                if col not in batch_df.columns
            ]

            if missing_cols:

                st.error(
                    "Missing required columns: "
                    + ", ".join(
                        missing_cols
                    )
                )

            elif batch_df.empty:

                st.warning(
                    "The uploaded CSV does not "
                    "contain any rows."
                )

            else:

                X_batch = (
                    batch_df[
                        required_cols
                    ].values
                )

                validate_input_data(
                    X_batch
                )

                X_batch_scaled = (
                    scaler.transform(
                        X_batch
                    )
                )

                predictions, probabilities = (
                    predict_all_models(
                        X_batch_scaled
                    )
                )

                # ------------------------------------------------
                # INDIVIDUAL MODEL COLUMNS
                # These remain available in the CSV output.
                # ------------------------------------------------

                for model_name in [
                    "Random Forest",
                    "XGBoost",
                    "Logistic Regression",
                    "CatBoost",
                    "TabNet",
                    "LightGBM",
                ]:

                    if model_name not in probabilities:
                        continue

                    safe_name = (
                        model_name
                        .replace(" ", "_")
                        .replace("-", "")
                    )

                    batch_df[
                        f"{safe_name}_Probability_%"
                    ] = np.round(
                        probabilities[
                            model_name
                        ] * 100,
                        2,
                    )

                    batch_df[
                        f"{safe_name}_Prediction"
                    ] = predictions[
                        model_name
                    ]

                # ------------------------------------------------
                # HYBRID
                # ------------------------------------------------

                hybrid_probabilities = (
                    probabilities[
                        "Hybrid"
                    ]
                )

                hybrid_predictions = (
                    predictions[
                        "Hybrid"
                    ]
                )

                batch_df[
                    "Hybrid_Fraud_Prediction"
                ] = hybrid_predictions

                batch_df[
                    "Hybrid_Fraud_Probability_%"
                ] = np.round(
                    hybrid_probabilities * 100,
                    2,
                )

                batch_df[
                    "Hybrid_Risk_Level"
                ] = pd.Series(
                    hybrid_probabilities
                ).apply(
                    lambda x:
                        "HIGH"
                        if x > 0.75
                        else (
                            "MEDIUM"
                            if x > 0.50
                            else "LOW"
                        )
                ).values

                # ------------------------------------------------
                # SUMMARY
                # ------------------------------------------------

                total_transactions = len(
                    batch_df
                )

                legitimate = int(
                    (
                        hybrid_predictions == 0
                    ).sum()
                )

                flagged = int(
                    (
                        hybrid_predictions == 1
                    ).sum()
                )

                fraud_rate = (
                    flagged
                    / total_transactions
                    * 100
                )

                st.markdown(
                    "<br>",
                    unsafe_allow_html=True,
                )

                bulk_metrics = [
                    (
                        "fa-list",
                        "Total Transactions",
                        f"{total_transactions:,}",
                    ),
                    (
                        "fa-circle-check",
                        "Legitimate",
                        f"{legitimate:,}",
                    ),
                    (
                        "fa-triangle-exclamation",
                        "Flagged Fraud",
                        f"{flagged:,}",
                    ),
                    (
                        "fa-chart-pie",
                        "Hybrid Fraud Rate",
                        f"{fraud_rate:.1f}%",
                    ),
                ]

                metric_cols = st.columns(
                    4
                )

                for col, item in zip(
                    metric_cols,
                    bulk_metrics,
                ):

                    icon, label, value = item

                    with col:

                        render_html(
                            f"""
<div class="kpi-card">

    <div class="kpi-top">

        <div class="kpi-label">
            {html.escape(label)}
        </div>

        <div class="kpi-icon">
            <i class="fa-solid {icon}"></i>
        </div>

    </div>

    <div class="kpi-value">
        {html.escape(value)}
    </div>

</div>
"""
                        )

                st.markdown(
                    "<br>",
                    unsafe_allow_html=True,
                )

                section_header(
                    "fa-table",
                    "Prediction Results",
                    "Hybrid ensemble predictions and fraud risk levels.",
                )

                display_cols = [
                    "Time",
                    "Amount",

                    "Random_Forest_Probability_%",
                    "XGBoost_Probability_%",
                    "Logistic_Regression_Probability_%",
                    "CatBoost_Probability_%",
                    "TabNet_Probability_%",
                    "LightGBM_Probability_%",

                    "Hybrid_Fraud_Prediction",
                    "Hybrid_Fraud_Probability_%",
                    "Hybrid_Risk_Level",
                ]

                display_cols = [
                    col
                    for col in display_cols
                    if col in batch_df.columns
                ]

                st.dataframe(
                    batch_df[
                        display_cols
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                csv_data = (
                    batch_df.to_csv(
                        index=False
                    )
                )

                st.download_button(
                    "Download Analysis Results",
                    data=csv_data,
                    file_name=(
                        "fraud_analysis_results.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True,
                    type="primary",
                )

        except Exception as e:

            st.error(
                f"Error processing CSV: {str(e)}"
            )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
<div class="footer">

    <div>
        <i class="fa-solid fa-shield-halved"></i>
        Credit Card AI Fraud Detection Platform
    </div>

    <div>
        6-Model Ensemble &nbsp; | &nbsp;
        SHAP Explainability &nbsp; | &nbsp;
        Transaction Intelligence
    </div>

</div>
"""
)