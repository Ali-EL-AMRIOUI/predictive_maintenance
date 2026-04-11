import streamlit as st
import requests
import pandas as pd
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="AI Predictive Maintenance Pro", 
    page_icon="✈️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOGIQUE DE CONNEXION DYNAMIQUE ---
# On utilise les variables d'environnement (standard industriel) avec fallback
API_HOST = os.getenv("API_URL", "http://localhost:8000")
API_URL = f"{API_HOST}/predict"

# Style CSS pour améliorer l'esthétique
st.markdown("""
    <style>
    .main { border-radius: 10px; }
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2906/2906231.png", width=100)
    st.title("Settings")
    st.info(f"📡 **API Endpoint:** {API_URL}")
    st.divider()
    st.markdown("### Status")
    try:
        # Check health of API
        health_check = requests.get(API_HOST, timeout=2)
        if health_check.status_code == 200:
            st.success("API Online")
        else:
            st.warning("API Unreachable")
    except:
        st.error("API Offline")

# --- MAIN INTERFACE ---
st.title("🛡️ Jet Engine Diagnostic Center")
st.markdown("---")

tab1, tab2 = st.tabs(["🎮 Manual Simulator", "📂 Fleet Analysis (CSV)"])

# --- TAB 1: MANUAL SIMULATOR ---
with tab1:
    st.subheader("Manual Sensor Input")
    with st.container():
        col_cyc, col_s2, col_s4, col_s11 = st.columns(4)
        
        with col_cyc:
            cycle = st.number_input("Current Cycle", value=50, min_value=1)
        with col_s2:
            s2 = st.number_input("Temp s2", value=641.8, format="%.2f")
        with col_s4:
            s4 = st.number_input("Press s4", value=1400.6, format="%.2f")
        with col_s11:
            s11 = st.number_input("Speed s11", value=47.4, format="%.2f")

    if st.button("🚀 Run Diagnostic", use_container_width=True):
        payload = {"sensor_data": {"cycle": cycle, "s2": s2, "s4": s4, "s11": s11}}
        with st.spinner("Analyzing sensors..."):
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    rul = result['predicted_RUL']
                    
                    # Logique de santé : on assume 200 cycles max pour l'exemple
                    health_pct = max(0, min(100, int((rul / 200) * 100)))
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Predicted RUL", f"{rul} Cycles")
                    c2.metric("Health Score", f"{health_pct}%")
                    
                    status = "✅ Stable" if health_pct > 50 else "⚠️ Warning" if health_pct > 20 else "🚨 Critical"
                    c3.metric("Status", status)
                    
                    color = "green" if health_pct > 50 else "orange" if health_pct > 20 else "red"
                    st.markdown(f"**Engine Health Timeline**")
                    st.progress(health_pct / 100)
                else:
                    st.error("Diagnostic failed. Check API logs.")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# --- TAB 2: BATCH CSV PREDICTION ---
with tab2:
    st.subheader("Bulk Fleet Diagnostic")
    uploaded_file = st.file_uploader("Upload fleet sensor data (CSV)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(5), use_container_width=True)
        
        if st.button("🔍 Analyze Entire Fleet", use_container_width=True):
            results = []
            progress_bar = st.progress(0)
            
            # Note: Pour une vraie "Wealth" tech, il faudrait une route API /predict-batch
            # Mais ici on garde ta logique de boucle pour la compatibilité
            for i, row in df.iterrows():
                payload = {"sensor_data": row[['cycle', 's2', 's4', 's11']].to_dict()}
                try:
                    r = requests.post(API_URL, json=payload)
                    results.append(r.json()['predicted_RUL'] if r.status_code == 200 else None)
                except:
                    results.append(None)
                progress_bar.progress((i + 1) / len(df))
            
            df['Predicted_RUL'] = results
            st.success("Analysis Complete!")
            st.area_chart(df['Predicted_RUL'])
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Report", csv, "fleet_report.csv", "text/csv")

st.markdown("---")
st.caption("AI Maintenance Pro | Secure | Scalable | No-Riba Tech Architecture")