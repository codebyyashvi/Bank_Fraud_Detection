# Bank Fraud Detection System

An end-to-end Machine Learning pipeline designed to detect fraudulent financial transactions. This project utilizes a highly imbalanced, synthetic dataset containing over 6.3 million transactions to train, tune, and evaluate multiple predictive models. The final tuned model is exported as a production-ready serialization artifact.

## 📌 Project Overview
Fraud detection is a critical application of machine learning in banking and finance. Since fraudulent activities are rare, the primary challenge is dealing with severe **class imbalance** (only ~0.13% of all transactions are fraudulent). 

This pipeline explores, preprocesses, and models the transaction data, utilizing **XGBoost** to achieve a high fraud capture rate (Recall) while minimizing false alarms (maximizing Precision) to optimize operational review costs.

---

## 📊 Dataset Description
The model is trained on `Fraud.csv` (which mimics the classic PaySim dataset):
*   **Total Transactions:** 6,362,620
*   **Fraudulent Transactions:** 8,213 (~0.129%)
*   **Non-Fraudulent Transactions:** 6,354,407 (~99.87%)

### Key Features
*   `step`: Maps a unit of time in the real world (1 step = 1 hour).
*   `type`: Type of transaction (`CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`).
*   `amount`: Amount of the transaction in local currency.
*   `nameOrig` / `nameDest`: Customer who started the transaction / recipient ID.
*   `oldbalanceOrig` / `newbalanceOrig`: Balance before and after the transaction for the sender.
*   `oldbalanceDest` / `newbalanceDest`: Balance before and after the transaction for the recipient.
*   `isFraud`: Ground truth label (1 = fraud, 0 = legitimate).

---

## 🛠️ Pipeline & Implementation Details

The implementation is detailed in [Analysis.ipynb](file:///c:/Users/yashv/Documents/ML_Project/Bank_Fraud_Detection/Analysis.ipynb) and consists of the following phases:

### 1. Exploratory Data Analysis (EDA)
*   **Fraud Type Mapping:** Identified that fraud exclusively occurs during `TRANSFER` and `CASH_OUT` transactions.
*   **Multicollinearity Detection:** Identified high correlation between origin/destination balances.
*   **Temporal Analysis:** Analyzed transactions by hour of the day to detect anomalous nighttime activity.
*   **Outlier Behavior:** Addressed heavy right-skewed transaction amounts.

### 2. Feature Engineering & Preprocessing
*   **Balance Differences:** Engineered balance-difference features (`errorBalanceOrig`, `errorBalanceDest`) to capture balance discrepancies, mitigating multicollinearity.
*   **Outlier Indicator:** Flagged extreme transaction values (top 1% percentile) with a log-transform and an `is_high_amount` indicator.
*   **Categorical Encoding:** Encoded transaction types via `LabelEncoder`.
*   **Feature Scaling:** Applied `RobustScaler` on numerical features to reduce the influence of outliers.

### 3. Model Training & Evaluation
Multiple machine learning classifiers were evaluated on a train/test split:
1.  **Logistic Regression** (baseline linear model)
2.  **Random Forest** (ensemble tree-based model)
3.  **XGBoost** (gradient boosting classifier)

*Evaluation metric of choice:* **Precision-Recall AUC (PR-AUC / Average Precision)** instead of ROC-AUC, due to the severe class imbalance.

---

## 🏆 Key Results & Performance Comparison

| Model | ROC-AUC | PR-AUC (Average Precision) |
| :--- | :---: | :---: |
| **Logistic Regression** | High | Low |
| **Random Forest** | High | Moderate |
| **XGBoost (Winner)** | **High** | **Highest (Best at catching rare classes)** |

### Threshold Optimization
By default, classifiers use a threshold of `0.5`. By plotting the **Precision-Recall Curve** for XGBoost, the decision threshold was optimized to **0.91** to meet operational requirements:
*   **Recall (Fraud Capture Rate):** **~90%** (catches 9 out of 10 frauds)
*   **Precision:** **~29% - 35%** (1 in 3 flagged alerts is true fraud, dramatically reducing false alarms compared to rule-based baselines).

---

## 💾 Model Serialization & Artifacts
The best-performing model (tuned XGBoost) along with scaling parameters is saved in the root directory:
*   **Artifact File:** [best_fraud_model_tuned.pkl](file:///c:/Users/yashv/Documents/ML_Project/Bank_Fraud_Detection/best_fraud_model_tuned.pkl)
*   **Format:** Serialized dictionary containing the trained model, label encoders, scaler, and optimized decision threshold.

---

## 🚀 Getting Started

### Prerequisites
Make sure Python 3.8+ and the following packages are installed:
```bash
pip install numpy pandas scikit-learn xgboost joblib matplotlib seaborn
```

### Running the Notebook
Open the Jupyter notebook to see the full modeling process:
```bash
jupyter notebook Analysis.ipynb
```

### Loading the Trained Model in Python
You can easily load the serialized model in your application using `joblib`:
```python
import joblib

# Load saved model artifact
artifact = joblib.load("best_fraud_model_tuned.pkl")
model = artifact["model"]
scaler = artifact.get("scaler")
threshold = artifact.get("threshold", 0.91)

# Make predictions on preprocessed features
# y_probs = model.predict_proba(X_test)[:, 1]
# y_preds = (y_probs >= threshold).astype(int)
```

---

## 📁 Repository Structure
*   [Analysis.ipynb](file:///c:/Users/yashv/Documents/ML_Project/Bank_Fraud_Detection/Analysis.ipynb) — Jupyter notebook containing EDA, Feature Engineering, Model Training, Hyperparameter Tuning, and Evaluation.
*   [best_fraud_model_tuned.pkl](file:///c:/Users/yashv/Documents/ML_Project/Bank_Fraud_Detection/best_fraud_model_tuned.pkl) — Serialization artifact of the final tuned model.
*   [README.md](file:///c:/Users/yashv/Documents/ML_Project/Bank_Fraud_Detection/README.md) — Documentation explaining the project.
*   `.gitignore` — Files to ignore in Git (such as `Fraud.csv`).
