import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide", page_icon="💳")
st.title("💳 Credit Card Fraud Detection Simulator")

st.markdown("""
This application uses a trained **XGBoost** model to predict whether a credit card transaction is fraudulent.
Please input the transaction features below.
""")

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/xgboost_fraud_model.pkl')
        scaler_time = joblib.load('models/scaler_time.pkl')
        scaler_amount = joblib.load('models/scaler_amount.pkl')
        return model, scaler_time, scaler_amount
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the notebook has been run to generate the models.")
        return None, None, None

model, scaler_time, scaler_amount = load_models()

if model is not None:
    st.sidebar.header("Transaction Parameters")
    
    amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
    time = st.sidebar.number_input("Transaction Time (seconds since first transaction)", min_value=0.0, value=3600.0, step=100.0)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("PCA Features (V1 - V28)")
    
    # We will generate 28 inputs inside an expander for the UI
    with st.sidebar.expander("V1 - V28 Features (Advanced)"):
        v_features = []
        for i in range(1, 29):
            v = st.number_input(f"V{i}", value=0.0, step=0.1)
            v_features.append(v)
            
    if st.button("Predict Fraud Probability", type="primary"):
        # Scale inputs
        time_scaled = scaler_time.transform([[time]])[0][0]
        amount_scaled = scaler_amount.transform([[amount]])[0][0]
        
        # Combine into an array matching the training data feature order:
        # Amount_scaled, Time_scaled, V1, V2, ..., V28
        features = np.array([amount_scaled, time_scaled] + v_features).reshape(1, -1)
        
        # Predict
        prob = model.predict_proba(features)[0][1]
        
        st.markdown("---")
        st.subheader("Prediction Result")
        if prob > 0.5:
            st.error(f"🚨 **FRAUD ALERT!** 🚨\n\nProbability of Fraud: **{prob * 100:.2f}%**")
            st.progress(prob)
        else:
            st.success(f"✅ **LEGITIMATE TRANSACTION** ✅\n\nProbability of Fraud: **{prob * 100:.2f}%**")
            st.progress(prob)
