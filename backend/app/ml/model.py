import numpy as np
import pandas as pd
import joblib
import os

PREPROCESSOR_PATH = "/Users/NIRAJKUMAR/Desktop/Fraud_Detection/backend/app/ml/preprocessor.joblib"
MODEL_PATH = "/Users/NIRAJKUMAR/Desktop/Fraud_Detection/backend/app/ml/fraud_model.joblib"

class FraudModel:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.is_loaded = False
        self.load()

    def load(self):
        if os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH):
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
            self.model = joblib.load(MODEL_PATH)
            self.is_loaded = True
            print("Production ML model and preprocessor loaded successfully.")
        else:
            print("Model files not found. Running in fallback/mock mode.")

    def predict_single(self, data: dict):
        """
        Predicts fraud for a single transaction dictionary containing the raw features.
        """
        if not self.is_loaded:
            # Fallback mock decision rule
            amount = data.get("amount", 0.0)
            is_fraud = amount > 1000.0
            score = 0.85 if is_fraud else 0.05
            return int(is_fraud), float(score)

        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess using the ColumnTransformer
        processed_features = self.preprocessor.transform(df)
        
        # Predict class and probability
        pred = self.model.predict(processed_features)[0]
        prob = self.model.predict_proba(processed_features)[0][1]
        
        return int(pred), float(prob)

    def predict_batch(self, df: pd.DataFrame):
        """
        Predicts fraud for a batch DataFrame of transactions.
        """
        if not self.is_loaded:
            # Fallback mock decision rule
            preds = (df["amount"] > 1000.0).astype(int).values
            probs = preds * 0.8 + 0.05
            return preds, probs
            
        processed_features = self.preprocessor.transform(df)
        preds = self.model.predict(processed_features)
        probs = self.model.predict_proba(processed_features)[:, 1]
        return preds, probs

fraud_detector = FraudModel()
