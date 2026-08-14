from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import os
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Credit Card Fraud Detection API",
              description="MLOps deployed API for XGBoost and PyTorch Autoencoder",
              version="1.0.0")

# Define Autoencoder Architecture (must match training exactly)
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.Tanh(),
            nn.Linear(24, 14),
            nn.Tanh(),
            nn.Linear(14, 7),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(7, 14),
            nn.Tanh(),
            nn.Linear(14, 24),
            nn.Tanh(),
            nn.Linear(24, input_dim),
            nn.Identity()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Global variables for models
xgb_model = None
scaler_time = None
scaler_amount = None
autoencoder = None
INPUT_DIM = 30

@app.on_event("startup")
def load_models():
    global xgb_model, scaler_time, scaler_amount, autoencoder
    
    # In docker, working dir is /app, and models are in /app/models
    # Locally, if run from api/, models are in ../models
    # Let's dynamically find it based on current file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # If run in docker where api is mapped or copied differently, let's allow env var override
    models_dir = os.getenv("MODELS_DIR", os.path.join(base_dir, 'models'))
    
    try:
        xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_fraud_model.pkl'))
        scaler_time = joblib.load(os.path.join(models_dir, 'scaler_time.pkl'))
        scaler_amount = joblib.load(os.path.join(models_dir, 'scaler_amount.pkl'))
        
        autoencoder = Autoencoder(INPUT_DIM)
        ae_path = os.path.join(models_dir, 'autoencoder.pth')
        if os.path.exists(ae_path):
            autoencoder.load_state_dict(torch.load(ae_path, map_location=torch.device('cpu'), weights_only=True))
            autoencoder.eval()
        else:
            autoencoder = None
            print("Autoencoder model not found at", ae_path)
    except Exception as e:
        print(f"Error loading models: {e}")

class Transaction(BaseModel):
    Amount: float
    Time: float
    V_features: list[float]

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if len(transaction.V_features) != 28:
        raise HTTPException(status_code=400, detail="V_features must have exactly 28 items.")
    
    if scaler_time is None or scaler_amount is None:
        raise HTTPException(status_code=500, detail="Scalers not loaded.")

    # Scale inputs
    t_scaled = scaler_time.transform([[transaction.Time]])[0][0]
    a_scaled = scaler_amount.transform([[transaction.Amount]])[0][0]

    # Prepare feature array: Amount_scaled, Time_scaled, V1..V28
    features = np.array([a_scaled, t_scaled] + transaction.V_features).reshape(1, -1)
    
    response = {}
    
    # 1. XGBoost Prediction
    if xgb_model:
        prob = xgb_model.predict_proba(features)[0][1]
        response['xgboost_probability'] = float(prob)
        response['xgboost_prediction'] = "Fraud" if prob > 0.5 else "Legitimate"
    else:
        response['xgboost_prediction'] = "Model unavailable"

    # 2. Autoencoder Anomaly Score
    if autoencoder:
        with torch.no_grad():
            tensor_features = torch.FloatTensor(features)
            reconstructed = autoencoder(tensor_features)
            mse = torch.mean((tensor_features - reconstructed)**2).item()
            response['autoencoder_mse'] = float(mse)
            # Threshold around 1.5 - 2.0 based on normal data distribution
            response['autoencoder_prediction'] = "Anomaly" if mse > 2.0 else "Normal"
            
    return response

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": xgb_model is not None}
