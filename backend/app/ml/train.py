import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Define paths
PROCESSED_DATA_DIR = "/Users/NIRAJKUMAR/Desktop/Fraud_Detection/backend/app/ml/processed_data"
MODEL_PATH = "/Users/NIRAJKUMAR/Desktop/Fraud_Detection/backend/app/ml/fraud_model.joblib"

def train_and_compare():
    # Load processed data
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_train.csv"))["is_fraud"].values
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_test.csv"))["is_fraud"].values

    print("Training dataset shape:", X_train.shape)
    print("Fraud count in train set:", int(np.sum(y_train)), f"({np.sum(y_train)/len(y_train)*100:.2f}%)")

    # 1. Baseline Model (Imbalanced)
    print("\nTraining Baseline Random Forest model...")
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_baseline.fit(X_train, y_train)

    # 2. Resampled Model (SMOTE)
    print("Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("Resampled training dataset shape:", X_train_res.shape)
    print("Fraud count in resampled train set:", int(np.sum(y_train_res)), f"({np.sum(y_train_res)/len(y_train_res)*100:.2f}%)")

    print("Training SMOTE Random Forest model...")
    rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_smote.fit(X_train_res, y_train_res)

    # Evaluate Baseline
    y_pred_base = rf_baseline.predict(X_test)
    y_prob_base = rf_baseline.predict_proba(X_test)[:, 1]
    
    p_base = precision_score(y_test, y_pred_base)
    r_base = recall_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base)
    auc_base = roc_auc_score(y_test, y_prob_base)
    cm_base = confusion_matrix(y_test, y_pred_base)

    # Evaluate SMOTE
    y_pred_smote = rf_smote.predict(X_test)
    y_prob_smote = rf_smote.predict_proba(X_test)[:, 1]

    p_smote = precision_score(y_test, y_pred_smote)
    r_smote = recall_score(y_test, y_pred_smote)
    f1_smote = f1_score(y_test, y_pred_smote)
    auc_smote = roc_auc_score(y_test, y_prob_smote)
    cm_smote = confusion_matrix(y_test, y_pred_smote)

    # Print results comparison
    print("\n" + "="*50)
    print("             MODEL PERFORMANCE COMPARISON")
    print("="*50)
    print(f"{'Metric':<25} | {'Baseline RF':<12} | {'SMOTE RF':<12}")
    print("-"*50)
    print(f"{'Precision':<25} | {p_base:<12.4f} | {p_smote:<12.4f}")
    print(f"{'Recall (Sensitivity)':<25} | {r_base:<12.4f} | {r_smote:<12.4f}")
    print(f"{'F1-Score':<25} | {f1_base:<12.4f} | {f1_smote:<12.4f}")
    print(f"{'ROC-AUC':<25} | {auc_base:<12.4f} | {auc_smote:<12.4f}")
    print("-"*50)
    print("Confusion Matrix (Baseline):")
    print(cm_base)
    print("\nConfusion Matrix (SMOTE):")
    print(cm_smote)
    print("="*50)

    # Save the SMOTE RF model as our production fraud detection classifier
    joblib.dump(rf_smote, MODEL_PATH)
    print(f"Production model (SMOTE RF) saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_and_compare()
