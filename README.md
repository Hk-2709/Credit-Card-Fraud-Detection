# Credit Card Fraud Detection

This project demonstrates a robust machine learning pipeline to classify credit card transactions as fraudulent or legitimate. It addresses extreme class imbalance and evaluates models based on Precision-Recall AUC (PR-AUC) and cost-sensitive business metrics.

## Project Structure

* `data/`: Contains the ULB Credit Card Fraud dataset (must be downloaded from Kaggle).
* `notebooks/`: Contains the Jupyter notebook (`EDA_and_Modeling.ipynb`) used for exploratory data analysis, feature engineering, imbalance handling (SMOTE), and model training.
* `models/`: Stores the trained models (e.g., `xgboost_fraud_model.pkl`) and scalers.
* `app.py`: A Streamlit application for real-time fraud scoring.
* `Project_Synopsis.md`: The formalized synopsis of the project.

## Instructions to Run

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
   Open and execute the notebook in the `notebooks/` directory.
   ```bash
   jupyter notebook notebooks/EDA_and_Modeling.ipynb
   ```
   *Running this notebook will automatically save the trained XGBoost model and scalers to the `models/` directory.*

4. **Run the Streamlit App**:
   Once the models are saved, start the simulator app.
   ```bash
   streamlit run app.py
   ```
