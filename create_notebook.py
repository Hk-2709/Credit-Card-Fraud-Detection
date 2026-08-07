import nbformat as nbf
import os

os.makedirs('notebooks', exist_ok=True)
os.makedirs('models', exist_ok=True)

nb = nbf.v4.new_notebook()

title = nbf.v4.new_markdown_cell('# Credit Card Fraud Detection\n\nThis notebook handles Exploratory Data Analysis (EDA), Feature Engineering, Imbalance Handling, and Model Training.')

imports = nbf.v4.new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, f1_score
import joblib

import warnings
warnings.filterwarnings('ignore')''')

load_data = nbf.v4.new_markdown_cell('## 1. Load Data & EDA')
load_data_code = nbf.v4.new_code_cell('''# Load dataset
df = pd.read_csv('../data/creditcard.csv')
print(df.head())
print(f"Dataset shape: {df.shape}")

# Class imbalance
counts = df['Class'].value_counts()
print(counts)
print(f"Fraud Percentage: {counts[1] / len(df) * 100:.3f}%")

plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.show()''')

dist_viz = nbf.v4.new_code_cell('''# Time and Amount distribution
fig, ax = plt.subplots(1, 2, figsize=(12,4))
sns.histplot(df['Time'], bins=50, ax=ax[0], kde=True)
ax[0].set_title('Distribution of Transaction Time')
sns.histplot(df['Amount'], bins=50, ax=ax[1], kde=True)
ax[1].set_title('Distribution of Transaction Amount')
plt.show()''')

feat_eng = nbf.v4.new_markdown_cell('## 2. Feature Engineering & Scaling')
feat_eng_code = nbf.v4.new_code_cell('''# Scale Time and Amount
scaler_time = StandardScaler()
scaler_amount = StandardScaler()

df['Time_scaled'] = scaler_time.fit_transform(df['Time'].values.reshape(-1,1))
df['Amount_scaled'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1,1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Reorder columns
scaled_amount = df['Amount_scaled']
scaled_time = df['Time_scaled']
df.drop(['Amount_scaled', 'Time_scaled'], axis=1, inplace=True)
df.insert(0, 'Amount_scaled', scaled_amount)
df.insert(1, 'Time_scaled', scaled_time)

df.head()''')

split_data = nbf.v4.new_markdown_cell('## 3. Train-Test Split and Imbalance Handling')
split_data_code = nbf.v4.new_code_cell('''X = df.drop('Class', axis=1)
y = df['Class']

# Stratified split to maintain fraud ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}, Fraud cases: {sum(y_train)}")
print(f"Testing set: {X_test.shape}, Fraud cases: {sum(y_test)}")

# Apply SMOTE to training data ONLY
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print(f"SMOTE Training set: {X_train_sm.shape}, Fraud cases: {sum(y_train_sm)}")''')

modeling = nbf.v4.new_markdown_cell('## 4. Modeling & Evaluation')
modeling_code = nbf.v4.new_code_cell('''def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    
    print(f"--- {name} ---")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC: {pr_auc:.4f}\\n")
    return pr_auc

# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_sm, y_train_sm)
evaluate_model(lr, X_test, y_test, "Logistic Regression")

# 2. Random Forest (with class weight instead of SMOTE for comparison)
rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train) # Training on original imbalanced data with class weights
evaluate_model(rf, X_test, y_test, "Random Forest")

# 3. XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), random_state=42)
xgb.fit(X_train, y_train)
evaluate_model(xgb, X_test, y_test, "XGBoost")
''')

save_model = nbf.v4.new_markdown_cell('## 5. Save the Best Model')
save_model_code = nbf.v4.new_code_cell('''# We'll save XGBoost as it typically performs best for extreme imbalance
joblib.dump(xgb, '../models/xgboost_fraud_model.pkl')
joblib.dump(scaler_amount, '../models/scaler_amount.pkl')
joblib.dump(scaler_time, '../models/scaler_time.pkl')
print("Model and scalers saved to ../models/")''')

nb['cells'] = [title, imports, load_data, load_data_code, dist_viz, feat_eng, feat_eng_code, split_data, split_data_code, modeling, modeling_code, save_model, save_model_code]

with open('notebooks/EDA_and_Modeling.ipynb', 'w') as f:
    nbf.write(nb, f)
    
print("Notebook created at notebooks/EDA_and_Modeling.ipynb")
