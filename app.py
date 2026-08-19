import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    layout="wide"
)

@st.cache_resource
def load_assets():
    model_path = os.path.join('models', 'hybrid_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_assets()

st.title("💳 Credit Card Fraud Detection Dashboard")
st.subheader("Realtime Transaction Monitoring & Enterprise Assessment")

if model is None or scaler is None:
    st.error("⚠️ Model or scaler file not found! Please run `python train.py` first to generate the model files.")
    st.stop()

# Tab Layout
tab1, tab2, tab3 = st.tabs(["🏦 Banker / POS Mode", "📌 Technical Manual Check (V1-V28)", "📁 Bulk CSV Simulation"])

# TAB 1: BANKER / PRODUCTION MODE

with tab1:
    st.markdown("### 💳 Real-time Banker Transaction Entry")
    st.caption("Bankers or operators can input standard transaction details to evaluate real-time fraud risk.")
    
    col1, col2 = st.columns(2)
    with col1:
        card_no = st.text_input("Card Number", "4532 0155 8941 2345")
        amount_bank = st.number_input("Transaction Amount ($)", min_value=0.0, value=120.0, key="bank_amount")
    with col2:
        merchant = st.selectbox("Merchant Category", ["E-Commerce (Online Shopping)", "Retail Grocery", "Crypto Exchange", "ATM Withdrawal", "International Wire"])
        location = st.selectbox("Transaction Location", ["Domestic (Local)", "High-Risk Overseas Zone", "Unknown Proxy/VPN IP"])

    if st.button("Process & Assess Fraud Risk", type="primary"):
        # Base Features Initialization
        v_features = [0.0] * 28
        time_val_bank = 100.0
        
        # Risk Rule Based Adjustments with Stronger PCA Fraud Signals
        if merchant in ["Crypto Exchange", "International Wire"] or location != "Domestic (Local)":
            # Realistic PCA Fraud Pattern Injection (derived from high risk distributions)
            v_features[0] = -2.31   # V1
            v_features[1] = 1.95    # V2
            v_features[2] = -1.60   # V3
            v_features[3] = 3.99    # V4 (Strong Positive Signal for Fraud)
            v_features[10] = 3.20   # V11
            v_features[11] = -2.89  # V12
            v_features[13] = -4.28  # V14 (Strong Negative Signal for Fraud)
            v_features[16] = -1.58  # V17
            time_val_bank = 406.0

        input_data = np.array([[time_val_bank] + v_features + [amount_bank]])
        scaled_data = scaler.transform(input_data)
        
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        st.divider()
        res_col1, res_col2 = st.columns(2)
        
        masked_card = f"XXXX-XXXX-XXXX-{card_no[-4:]}" if len(card_no) >= 4 else "XXXX"
        
        with res_col1:
            if prediction == 1 or probability > 0.50:
                st.error(f"**ALERT: SUSPICIOUS TRANSACTION BLOCKED!**\n\n**Card:** {masked_card}")
            else:
                st.success(f"**TRANSACTION APPROVED & SAFE**\n\n**Card:** {masked_card}")
                
        with res_col2:
            st.metric(label="Fraud Risk Probability", value=f"{probability * 100:.2f}%")


# TAB 2: TECHNICAL MANUAL CHECK (V1-V28)

with tab2:
    st.markdown("### Technical Manual Input For Data Analysts")
    
    # --- Quick Sample Loader Buttons ---
    st.info("**Quick Test:** Click a button below to load preset test data:")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    # Session State Initialization
    if 'time_val' not in st.session_state:
        st.session_state.time_val = 100.0
    if 'amount_val' not in st.session_state:
        st.session_state.amount_val = 250.0
    for i in range(1, 29):
        if f'v_{i}' not in st.session_state:
            st.session_state[f'v_{i}'] = 0.0

    # Load Fraud Sample
    if btn_col1.button("Load Sample Fraud Data"):
        fraud_v = [-2.31, 1.95, -1.60, 3.99, -0.52, -1.42, -2.53, 1.39, -2.77, -2.77,
                   3.20, -2.89, -0.59, -4.28, -0.29, -0.71, -1.58, 0.45, 0.41, 0.12,
                   0.51, -0.03, -0.46, 0.32, 0.04, 0.17, 0.26, -0.14]
        for i, val in enumerate(fraud_v, start=1):
            st.session_state[f'v_{i}'] = val
        st.session_state.time_val = 406.0
        st.session_state.amount_val = 0.0
        st.rerun()

    # Load Normal Sample
    if btn_col2.button("Load Sample Normal Data"):
        for i in range(1, 29):
            st.session_state[f'v_{i}'] = 0.0
        st.session_state.time_val = 100.0
        st.session_state.amount_val = 50.0
        st.rerun()

    # Reset Input Fields
    if btn_col3.button("🔄 Reset All Fields"):
        for i in range(1, 29):
            st.session_state[f'v_{i}'] = 0.0
        st.session_state.time_val = 0.0
        st.session_state.amount_val = 0.0
        st.rerun()
        
    st.divider()

    # Inputs linked with Session State
    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input("Transaction Time (Seconds)", min_value=0.0, key="time_val")
    with col2:
        amount_val = st.number_input("Transaction Amount ($)", min_value=0.0, key="amount_val")

    st.markdown("#### PCA Features (V1 to V28)")
    v_cols = st.columns(4)
    v_values = []
    
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        with v_cols[col_idx]:
            val = st.number_input(f"V{i}", step=0.1, key=f"v_{i}")
            v_values.append(val)

    if st.button("🔍 Analyze Transaction", type="primary", key="btn_tech_analyze"):
        input_data = np.array([[time_val] + v_values + [amount_val]])
        scaled_data = scaler.transform(input_data)
        
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        st.divider()
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if prediction == 1:
                st.error("**ALERT: SUSPICIOUS / FRAUDULENT TRANSACTION DETECTED!**")
            else:
                st.success("**TRANSACTION IS LEGITIMATE**")
                
        with res_col2:
            st.metric(label="Fraud Risk Probability", value=f"{probability * 100:.2f}%")


# TAB 3: BULK CSV SIMULATION

with tab3:
    st.markdown("### Upload CSV File For Batch Stream Analysis")
    uploaded_file = st.file_uploader("Upload CSV containing transaction data", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        feature_cols = [f'V{i}' for i in range(1, 29)]
        required_cols = ['Time'] + feature_cols + ['Amount']
        
        if all(col in batch_df.columns for col in required_cols):
            X_batch = batch_df[required_cols]
            X_batch_scaled = scaler.transform(X_batch)
            
            predictions = model.predict(X_batch_scaled)
            probabilities = model.predict_proba(X_batch_scaled)[:, 1]

            batch_df['Fraud_Prediction'] = predictions
            batch_df['Fraud_Probability (%)'] = np.round(probabilities * 100, 2)

            st.write("### Analysis Results Summary")
            fraud_count = (predictions == 1).sum()
            legit_count = (predictions == 0).sum()

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Transactions", len(batch_df))
            m2.metric("Legitimate Transactions", legit_count)
            m3.metric("Flagged Fraud Transactions", fraud_count)

            st.dataframe(batch_df.style.highlight_between(subset=['Fraud_Prediction'], left=1, right=1, color='#ffcdd2'))
        else:
            st.error("Invalid CSV format. The file must contain 'Time', 'V1'-'V28', and 'Amount' columns.")