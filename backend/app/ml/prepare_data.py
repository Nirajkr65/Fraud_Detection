import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

from pathlib import Path

# Define paths relative to script location
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "fraud_dataset.csv"
PREPROCESSOR_PATH = Path(__file__).resolve().parent / "preprocessor.joblib"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent / "processed_data"

def build_and_save_pipeline():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    
    # Separate features and target
    X = df.drop(columns=["transaction_id", "is_fraud"])
    y = df["is_fraud"]
    
    # Identify feature types
    numeric_features = ["amount", "transaction_hour", "device_trust_score", "velocity_last_24h", "cardholder_age"]
    categorical_features = ["merchant_category"]
    binary_features = ["foreign_transaction", "location_mismatch"]
    
    # Numerical preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Binary preprocessing pipeline
    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    
    # Column transformer to apply transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", binary_transformer, binary_features)
        ]
    )
    
    # Stratified Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit the preprocessor on the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the fitted preprocessor
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Preprocessor successfully saved to: {PREPROCESSOR_PATH}")
    
    # Save processed data split for subsequent training step
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DATA_DIR / "X_train.npy", X_train_processed)
    np.save(PROCESSED_DATA_DIR / "X_test.npy", X_test_processed)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)
    print(f"Processed train and test data splits saved to: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    build_and_save_pipeline()
