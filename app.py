import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide", page_icon="💳")
st.title("💳 Credit Card Fraud Detection Simulator")

st.markdown("""
This application is powered by a **FastAPI Microservice**. 
It uses both an **XGBoost** model (Supervised Learning) and a **PyTorch Autoencoder** (Unsupervised Learning) 
to predict whether a credit card transaction is fraudulent.
""")

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
        
API_URL = os.getenv("API_URL", "http://localhost:8000")

if st.button("Predict Fraud Probability", type="primary"):
    payload = {
        "Amount": amount,
        "Time": time,
        "V_features": v_features
    }
    
    try:
        with st.spinner('Querying FastAPI Backend...'):
            response = requests.post(f"{API_URL}/predict", json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            # XGBoost Results
            with col1:
                st.markdown("### XGBoost Model")
                prob = data.get('xgboost_probability', 0.0)
                if data.get('xgboost_prediction') == 'Fraud':
                    st.error(f"🚨 **FRAUD ALERT!** 🚨\n\nProbability: **{prob * 100:.2f}%**")
                else:
                    st.success(f"✅ **LEGITIMATE** ✅\n\nProbability: **{prob * 100:.2f}%**")
                st.progress(prob)
                
            # Autoencoder Results
            with col2:
                st.markdown("### PyTorch Autoencoder")
                mse = data.get('autoencoder_mse', 0.0)
                if data.get('autoencoder_prediction') == 'Anomaly':
                    st.warning(f"⚠️ **ANOMALY DETECTED!** ⚠️\n\nReconstruction Error: **{mse:.4f}**")
                elif 'autoencoder_prediction' in data:
                    st.info(f"🟢 **NORMAL PATTERN** 🟢\n\nReconstruction Error: **{mse:.4f}**")
                else:
                    st.write("Autoencoder model not available on backend.")
                    
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the backend API. Is the FastAPI server running on port 8000?")
    except Exception as e:
        st.error(f"An error occurred: {e}")
