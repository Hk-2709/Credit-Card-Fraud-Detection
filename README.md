# Credit Card Fraud Detection (MLOps & Deep Learning)

This project demonstrates a robust machine learning pipeline to classify credit card transactions as fraudulent or legitimate. It addresses extreme class imbalance and evaluates models based on Precision-Recall AUC (PR-AUC). 

It has been upgraded to a **Microservice Architecture** using **FastAPI** and **Docker**, and incorporates a **PyTorch Autoencoder** for unsupervised anomaly detection.

## Project Structure

* `data/`: Contains the ULB Credit Card Fraud dataset (must be downloaded from Kaggle).
* `notebooks/`: Contains the Jupyter notebooks for EDA, SMOTE, XGBoost Modeling, and the Deep Learning Autoencoder.
* `models/`: Stores the trained models (`xgboost_fraud_model.pkl`, `autoencoder.pth`) and scalers.
* `api/`: Contains the FastAPI backend application (`main.py`).
* `app.py`: A Streamlit application for real-time fraud scoring via the FastAPI backend.
* `docker-compose.yml`: Orchestration file for Docker deployment.
* `Dockerfile.api` & `Dockerfile.ui`: Docker build instructions for the services.

## Instructions to Run (Locally)

1. **Setup Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download Dataset**:
   Ensure you have configured your Kaggle API key (`~/.kaggle/kaggle.json`). Then run:
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data/
   unzip data/creditcardfraud.zip -d data/
   ```

3. **Train Models**:
   Open and execute the notebooks in the `notebooks/` directory to generate the `.pkl` and `.pth` models in `models/`.

4. **Run the Application (Microservices)**:
   You need two terminal windows.
   - **Terminal 1 (Backend)**: `uvicorn api.main:app --reload`
   - **Terminal 2 (Frontend)**: `streamlit run app.py`

## Instructions to Run (Docker)

If you have Docker Desktop installed, you can spin up the entire application (both the FastAPI backend and Streamlit frontend) with a single command:

```bash
docker-compose up --build
```
- Streamlit UI will be available at: `http://localhost:8501`
- FastAPI Docs will be available at: `http://localhost:8000/docs`
