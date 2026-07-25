"""
===============================================================================
AI Predictive Maintenance Platform
Frontend - Streamlit
Author: Ali El Amrioui
===============================================================================

Modern frontend for the NASA C-MAPSS Predictive Maintenance project.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

###############################################################################
# PAGE CONFIGURATION
###############################################################################

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

###############################################################################
# CONFIGURATION & CONSTANTS
###############################################################################

API_HOST = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_HOST}/predict"
HEALTH_ENDPOINT = f"{API_HOST}/health"

# Application Thresholds (Defaults)
WINDOW_SIZE = 5
MAX_RUL = 125
INSPECT_THRESHOLD = 30
GROUND_THRESHOLD = 10

###############################################################################
# GLOBAL THEME
###############################################################################

st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #07111f;
    color: white;
}
section[data-testid="stSidebar"] {
    background: #0d1b2a;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
h1, h2, h3 {
    color: white;
}
.metric-card {
    background: #10253c;
    border-radius: 18px;
    padding: 22px;
    border: 1px solid rgba(255,255,255,.08);
    box-shadow: 0px 6px 18px rgba(0,0,0,.35);
}
.big-title {
    font-size: 42px;
    font-weight: 700;
    color: white;
}
.subtitle {
    color: #9db7d5;
}
.footer {
    color: #7c8ea3;
    text-align: center;
    margin-top: 50px;
}
</style>
""",
    unsafe_allow_html=True,
)

###############################################################################
# HELPERS
###############################################################################


@st.cache_data(ttl=5)
def check_api():
    """Ping FastAPI every 5 seconds."""
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=2)
        if r.status_code == 200:
            return True, r.json()
        return False, {}
    except Exception:
        return False, {}


def health_color(score: float):
    if score >= 70:
        return "green"
    if score >= 40:
        return "orange"
    return "red"


def status_badge(status):
    status_upper = status.upper()

    if "GROUND" in status_upper:
        return "🔴", "error"
    elif "REVIEW REQUIRED" in status_upper or "INSPECTION" in status_upper:
        return "🟠", "warning"
    else:
        return "🟢", "success"


###############################################################################
# SIDEBAR
###############################################################################

with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2906/2906231.png",
        width=90,
    )
    st.title("AI Maintenance")
    st.caption("Industrial Predictive Maintenance Platform")

    st.divider()

    api_ok, api_data = check_api()

    if api_ok:
        st.success("API Online")
    else:
        st.error("API Offline")

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "",
        [
            "🏠 Dashboard",
            "🔍 Engine Prediction",
            "📊 Fleet Analysis",
            "⚙️ Settings",
            "ℹ️ About",
        ],
    )

    st.divider()

    st.subheader("Runtime")
    st.write("**API**")
    st.code(API_HOST)
    st.write("**Time**")
    st.code(datetime.now().strftime("%H:%M:%S"))

###############################################################################
# DASHBOARD
###############################################################################

if page == "🏠 Dashboard":
    st.markdown(
        """
        <div class="big-title">AI Predictive Maintenance Platform</div>
        <div class="subtitle">NASA C-MAPSS Turbofan Engine Remaining Useful Life Prediction</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("API", "ONLINE" if api_ok else "OFFLINE")
    with c2:
        st.metric("Champion Model", "Loaded" if api_ok else "--")
    with c3:
        st.metric("Safety Model", "Loaded" if api_ok else "--")
    with c4:
        st.metric("Backend", "FastAPI")

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=82,
                title={"text": "Fleet Health"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "limegreen"},
                    "steps": [
                        {"range": [0, 30], "color": "darkred"},
                        {"range": [30, 60], "color": "orange"},
                        {"range": [60, 100], "color": "green"},
                    ],
                },
            )
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fleet = pd.DataFrame(
            {
                "Status": ["Healthy", "Inspection", "Grounded"],
                "Engines": [82, 11, 3],
            }
        )
        fig = px.bar(
            fleet,
            x="Status",
            y="Engines",
            color="Status",
            text="Engines",
        )
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Platform Overview")
    st.info(
        """
        This application predicts the Remaining Useful Life (RUL) of aircraft engines
        using an XGBoost-based predictive maintenance pipeline.

        The backend provides:
        • Champion model prediction
        • Conservative Safety RUL
        • Health Score
        • Maintenance recommendation
        • SHAP explainability
        • Out-of-distribution detection
        • Fleet analysis
        """
    )

###############################################################################
# ENGINE PREDICTION
###############################################################################

elif page == "🔍 Engine Prediction":
    st.title("🔍 Engine Prediction")
    st.caption("Predict Remaining Useful Life from an engine history.")

    st.divider()
    st.subheader("Engine Information")

    unit_id = st.number_input("Engine ID", min_value=1, value=1, step=1)
    history_length = st.slider("History Length", min_value=1, max_value=20, value=5)

    st.info(
        f"The API expects an **engine history**.\nCurrent history length: **{history_length} cycles**"
    )

    st.divider()
    st.subheader("Engine History")

    history = []
    for i in range(history_length):
        with st.expander(f"Cycle {i+1}", expanded=(i == history_length - 1)):
            cycle = st.number_input(
                "Cycle", key=f"cycle_{i}", value=i + 1, step=1
            )
            sensor_json = st.text_area(
                "Sensors (JSON)",
                key=f"sensors_{i}",
                height=200,
                value="""{
"s1":0, "s2":641.82, "s3":1589.70, "s4":1400.60, "s5":14.62,
"s6":21.61, "s7":554.36, "s8":2388.06, "s9":9046.19, "s10":1.30,
"s11":47.47, "s12":521.66, "s13":2388.02, "s14":8138.62, "s15":8.42,
"s16":0.03, "s17":392, "s18":2388, "s19":100, "s20":39.06, "s21":23.42,
"os1":0, "os2":0, "os3":100
}""",
            )

            try:
                sensors = json.loads(sensor_json)
            except Exception:
                st.error("Invalid JSON format.")
                sensors = {}

            history.append({"cycle": int(cycle), "sensors": sensors})

    st.divider()
    predict_button = st.button("🚀 Predict Engine Health", use_container_width=True)

    if predict_button:
        payload = {"unit_id": int(unit_id), "history": history}

        with st.spinner("Running AI diagnosis..."):
            try:
                start = time.perf_counter()
                response = requests.post(
                    PREDICT_ENDPOINT, json=payload, timeout=60
                )
                latency = (time.perf_counter() - start) * 1000
            except Exception as e:
                st.error(f"Connection error: {e}")
                st.stop()

        if response.status_code != 200:
            st.error(f"HTTP Error {response.status_code}")
            try:
                st.json(response.json())
            except Exception:
                st.write(response.text)
            st.stop()

        result = response.json()

        predicted_rul = result.get("predicted_RUL", 0)
        safety_rul = result.get("safety_RUL", 0)
        health_score = result.get("health_score", 0)
        status = result.get("status", "UNKNOWN")
        ood = result.get("out_of_distribution", False)
        violation_ratio = result.get("ood_violation_ratio", 0.0)
        violating_features = result.get("ood_violating_features", [])
        top_reasons = result.get("top_reasons", [])

        st.success("Prediction completed successfully.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted RUL", f"{predicted_rul:.2f}")
        c2.metric("Safety RUL", f"{safety_rul:.2f}")
        c3.metric("Health Score", f"{health_score:.1f} %")
        c4.metric("Latency", f"{latency:.0f} ms")

        st.divider()

        # Gauge
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=health_score,
                title={"text": "Engine Health Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "royalblue"},
                    "steps": [
                        {"range": [0, 20], "color": "#7f0000"},
                        {"range": [20, 50], "color": "#d97706"},
                        {"range": [50, 100], "color": "#16a34a"},
                    ],
                },
            )
        )
        gauge.update_layout(height=350)
        st.plotly_chart(gauge, use_container_width=True)

        st.subheader("Maintenance Recommendation")
        status_badge(status)

        st.subheader("Input Validation")
        if ood:
            st.error("Out-of-Distribution sample detected.")
            st.metric("Violation Ratio", f"{violation_ratio:.1%}")
            if violating_features:
                st.write("Outside training distribution:")
                st.code("\n".join(violating_features))
        else:
            st.success("Input is within training distribution.")

        if top_reasons:
            st.divider()
            st.subheader("Top Prediction Drivers")
            shap_df = pd.DataFrame(
                [
                    {
                        "Feature": x["feature"],
                        "Impact": x["shap_impact"],
                    }
                    for x in top_reasons
                ]
            )
            fig = px.bar(
                shap_df,
                x="Impact",
                y="Feature",
                orientation="h",
                text="Impact",
                color="Impact",
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(height=420, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Prediction JSON"):
            st.json(result)

###############################################################################
# FLEET ANALYSIS
###############################################################################

elif page == "📊 Fleet Analysis":
    st.title("📊 Fleet Analysis")
    st.caption("Run predictive maintenance on an entire fleet.")

    uploaded_file = st.file_uploader("Upload Fleet CSV", type=["csv"])

    if uploaded_file is not None:
        fleet = pd.read_csv(uploaded_file)
        st.success(f"{len(fleet)} rows loaded.")
        st.dataframe(fleet.head(), use_container_width=True)

        st.divider()

        if st.button("🚀 Analyze Fleet", use_container_width=True):
            progress = st.progress(0)
            predictions = []

            for i, row in fleet.iterrows():
                payload = {
                    "unit_id": int(row.get("unit", 1)),
                    "history": [
                        {
                            "cycle": int(row.get("cycle", 1)),
                            "sensors": {
                                k: float(v)
                                for k, v in row.items()
                                if k not in ["unit", "cycle"]
                            },
                        }
                    ],
                }

                try:
                    response = requests.post(
                        PREDICT_ENDPOINT, json=payload, timeout=60
                    )
                    if response.status_code == 200:
                        predictions.append(response.json())
                    else:
                        predictions.append(None)
                except Exception:
                    predictions.append(None)

                progress.progress((i + 1) / len(fleet))

            result_df = fleet.copy()
            result_df["Predicted_RUL"] = [
                p["predicted_RUL"] if p else np.nan for p in predictions
            ]
            result_df["Safety_RUL"] = [
                p["safety_RUL"] if p else np.nan for p in predictions
            ]
            result_df["Health_Score"] = [
                p["health_score"] if p else np.nan for p in predictions
            ]
            result_df["Status"] = [
                p["status"] if p else "FAILED" for p in predictions
            ]

            st.success("Fleet analysis completed.")
            st.dataframe(result_df, use_container_width=True)

            st.divider()

            c1, c2, c3 = st.columns(3)
            c1.metric("Fleet Size", len(result_df))
            c2.metric("Average Health", f"{result_df.Health_Score.mean():.1f} %")
            critical = (result_df.Status == "IMMEDIATE GROUNDING").sum()
            c3.metric("Critical Engines", int(critical))

            fig = px.histogram(
                result_df,
                x="Predicted_RUL",
                nbins=30,
                title="Predicted RUL Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = px.scatter(
                result_df,
                x="Predicted_RUL",
                y="Health_Score",
                color="Status",
                hover_data=["unit"] if "unit" in result_df.columns else None,
                title="Fleet Health Overview",
            )
            st.plotly_chart(fig, use_container_width=True)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Fleet Report",
                csv,
                file_name="fleet_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

###############################################################################
# SETTINGS
###############################################################################

elif page == "⚙️ Settings":
    st.title("⚙️ Application Settings")
    st.subheader("API Endpoint")
    st.code(API_HOST)

    st.divider()
    st.subheader("Connection Test")

    if st.button("Check API Status"):
        try:
            r = requests.get(HEALTH_ENDPOINT, timeout=5)
            if r.status_code == 200:
                st.success("API is healthy.")
                st.json(r.json())
            else:
                st.error(f"HTTP {r.status_code}")
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.subheader("Configuration")
    st.write(f"Window Size : **{WINDOW_SIZE}**")
    st.write(f"Maximum RUL : **{MAX_RUL}**")
    st.write(f"Inspection Threshold : **{INSPECT_THRESHOLD}**")
    st.write(f"Ground Threshold : **{GROUND_THRESHOLD}**")

###############################################################################
# ABOUT
###############################################################################

elif page == "ℹ️ About":
    st.title("ℹ️ AI Predictive Maintenance")
    st.markdown(
        """
    ### Aircraft Engine Remaining Useful Life Prediction

    This application demonstrates an end-to-end Predictive Maintenance platform
    built around the NASA C-MAPSS turbofan engine degradation dataset.

    ---
    ### Technology Stack
    - Python • FastAPI • Streamlit • XGBoost • SHAP • MLflow • Optuna • Docker • Kubernetes

    ---
    ### AI & MLOps Features
    - Remaining Useful Life prediction
    - Conservative Safety Model
    - Health Score computation
    - Maintenance recommendation
    - SHAP explainability & Out-of-Distribution detection
    - Fleet analytics
    """
    )

###############################################################################
# FOOTER
###############################################################################

st.divider()

st.markdown(
    """
    <center>
    <small>AI Predictive Maintenance Platform | NASA C-MAPSS | FastAPI • Streamlit • XGBoost • SHAP • MLflow</small>
    </center>
    """,
    unsafe_allow_html=True,
)