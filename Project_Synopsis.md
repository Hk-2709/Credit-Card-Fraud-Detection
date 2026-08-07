# Project Synopsis: Credit Card Fraud Detection Using Machine Learning

## Title Page
**Title of the Project:** Credit Card Fraud Detection Using Machine Learning
**Team members name and Reg. No., their department:** [To be filled by student]
**Minor Specialization Name & Department offering:** Data Science Minor, [Department Name]
**Name of the Guide:** [To be filled by student]
**Date of Submission:** [To be filled by student]

---

## 1. Introduction
Credit card fraud represents one of the most economically damaging and technically challenging problems in modern financial systems. With global card fraud losses exceeding billions of dollars annually, and rising sharply due to the proliferation of e-commerce and contactless payments, the need for automated, intelligent fraud detection systems has never been more urgent. Fraudulent transactions represent a tiny fraction (often ~0.1–0.2%) of total volume. Traditional rule-based systems have proven inadequate in the face of evolving fraud tactics, generating excessive false positives that inconvenience legitimate users while failing to catch novel fraudulent patterns. Machine learning (ML) and deep learning (DL) have emerged as the dominant paradigms for addressing this challenge, offering the ability to learn complex non-linear patterns from large-scale transactional data and address extreme class imbalance.

## 2. Literature Review
A comprehensive review of recent literature (2019–2025) highlights a significant shift from classical machine learning algorithms to advanced ensemble methods, deep learning, and graph-based models for credit card fraud detection. The majority of studies heavily utilize the ULB European Credit Card dataset due to its standardized benchmark capabilities, though this introduces challenges in domain-specific feature engineering due to PCA anonymization. 

Class imbalance is the most universally cited challenge, with techniques like SMOTE remaining popular, while Generative Adversarial Networks (GANs) and cost-sensitive learning are increasingly explored to improve minority class representation without generating noise (Singh et al., 2022; Fiore et al., 2021). Methodologically, while deep learning architectures (e.g., Autoencoders, LSTMs) provide strong anomaly detection capabilities, tree-based ensemble models like XGBoost and Random Forest consistently demonstrate robust performance on tabular transaction data, often achieving PR-AUC scores above 0.95 (Alarfaj et al., 2022). 

Recent advancements also emphasize the need for Explainable AI (XAI) to meet regulatory requirements, using tools like SHAP to interpret model decisions (Yadav et al., 2023). However, significant gaps remain in the literature regarding real-time inference latency, concept drift over time (Pozzolo et al., 2020), and quantifiable cost-benefit analysis concerning false positives versus financial losses.

## 3. Problem Statement
Credit card fraud causes billions in annual global losses, with fraudulent transactions representing a tiny fraction (~0.1–0.2%) of total volume. Traditional rule-based systems struggle with evolving fraud patterns and produce high false positives, inconveniencing legitimate users. Furthermore, the extreme class imbalance (fraud vs. non-fraud) makes standard accuracy metrics misleading and demands specialized handling. This project aims to develop a robust machine learning pipeline to classify transactions as fraudulent or legitimate in near real-time, minimizing financial loss while strictly controlling false alarms.

## 4. Objectives
* Build and compare multiple classification models to accurately detect fraudulent credit card transactions.
* Address severe class imbalance using resampling (e.g., SMOTE) and algorithm-level techniques (e.g., class weights).
* Evaluate models using business-relevant metrics, specifically Precision-Recall AUC (PR-AUC), F1-score, and cost-sensitive evaluation.
* Deploy a simple interactive demo (using Streamlit) to simulate real-time fraud scoring.
* Demonstrate a complete data science workflow (data acquisition, cleaning, exploration, modeling, communication) using Python.

## 5. Methodology
1. **Data Preparation & EDA:** Load the ULB Credit Card Fraud dataset, handle missing values, and visualize distributions (e.g., Amount, Time patterns), correlation heatmaps, and imbalance severity.
2. **Feature Engineering:** Create derived features such as transaction velocity, amount bins, and hour-of-day from the Time variable, followed by feature scaling (StandardScaler).
3. **Imbalance Handling:** Apply oversampling techniques like SMOTE/ADASYN, RandomUnderSampler, and `class_weight='balanced'` in model parameters.
4. **Modeling:** Implement and rigorously compare multiple algorithms:
   * **Baseline:** Logistic Regression
   * **Tree-based:** Random Forest, XGBoost (preferred for extreme imbalance)
   * **Anomaly detection:** Isolation Forest
   * **Optional deep learning:** Simple Autoencoder for an unsupervised anomaly view.
   * *Hyperparameter tuning* via GridSearchCV or Optuna.
5. **Evaluation:** Focus strictly on PR-AUC (most informative for imbalance), Precision, Recall, and F1 at an optimal threshold. Implement a confusion matrix with cost analysis (e.g., false negative = high loss, false positive = customer friction).
6. **Deployment & Communication:** Build a Streamlit application to take transaction inputs (Amount, Time, V-features) and output a fraud probability alert. Generate a reproducible Jupyter report mimicking an R Markdown style with clear sections, visuals, and code cells.

## 6. References
* Alarfaj, F. K., et al. (2022). Credit Card Fraud Detection Using State-of-the-Art Machine Learning and Deep Learning Algorithms. *IEEE Access*.
* Fiore, U., et al. (2021). Using Generative Adversarial Networks for Improving Classification Effectiveness in Credit Card Fraud. *Information Sciences*.
* Pozzolo, A. D., et al. (2020). Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy. *IEEE Transactions on Neural Networks and Learning Systems*.
* Singh, A., et al. (2022). Credit Card Fraud Detection Under Extreme Imbalanced Data: A Comparative Study of Data-Level Algorithms. *Journal of Experimental & Theoretical Artificial Intelligence*.
* Yadav, S., et al. (2023). Credit Card Fraud Detection Using Ensemble Methods with Explainable AI (SHAP). *Procedia Computer Science*.
