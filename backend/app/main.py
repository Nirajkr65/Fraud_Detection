from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd
import io
import os
from datetime import datetime

from .database import engine, Base, get_db
from . import crud, schemas, auth, models
from .ml.model import fraud_detector

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Fraud Detection System API", version="1.2.0")

# Startup database initialization
@app.on_event("startup")
def startup_event():
    db = next(get_db())
    try:
        active_model = crud.get_active_model_info(db)
        if not active_model:
            # Register our trained SMOTE RF model details in the database
            crud.create_model_info(
                db,
                model_name="SMOTE Random Forest Classifier",
                version="1.0.0",
                filepath="/Users/NIRAJKUMAR/Desktop/Fraud_Detection/backend/app/ml/fraud_model.joblib",
                precision=1.0000,
                recall=0.8667,
                f1_score=0.9286,
                roc_auc=0.9998,
                status="active"
            )
            print("Default active model details registered in database successfully.")
    except Exception as e:
        print(f"Startup DB init failed: {e}")


# --- AUTH ENDPOINTS ---

@app.post("/signup", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    hashed_pw = auth.get_password_hash(user.password)
    new_user = crud.create_user(db, user, hashed_pw)
    return new_user

@app.post("/token", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, username=form_data.username)
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=schemas.UserResponse)
def read_users_me(current_user: models.User = Depends(auth.get_current_user)):
    return current_user


# --- PREDICTION ENDPOINTS ---

@app.post("/predict", response_model=schemas.TransactionResponse)
def predict_transaction(
    transaction: schemas.TransactionCreate, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    # Check if transaction ID already exists
    db_tx = crud.get_transaction(db, transaction.transaction_id)
    if db_tx:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Transaction ID already exists"
        )

    # Extract dictionary for the model
    features_dict = {
        "amount": transaction.amount,
        "transaction_hour": transaction.transaction_hour,
        "merchant_category": transaction.merchant_category,
        "foreign_transaction": transaction.foreign_transaction,
        "location_mismatch": transaction.location_mismatch,
        "device_trust_score": transaction.device_trust_score,
        "velocity_last_24h": transaction.velocity_last_24h,
        "cardholder_age": transaction.cardholder_age
    }

    # Run ML prediction
    is_fraud, score = fraud_detector.predict_single(features_dict)

    # Save transaction associated with the authenticated user
    new_tx = crud.create_transaction(
        db, 
        transaction, 
        is_fraud=bool(is_fraud), 
        score=score, 
        user_id=current_user.id
    )
    return new_tx

@app.post("/predict/batch", response_model=schemas.BatchPredictionResponse)
def predict_batch_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a CSV file."
        )
    
    try:
        # Read uploaded CSV content
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read CSV: {str(e)}"
        )

    # Check for required columns
    required_cols = [
        "transaction_id", "amount", "transaction_hour", "merchant_category",
        "foreign_transaction", "location_mismatch", "device_trust_score",
        "velocity_last_24h", "cardholder_age"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV missing columns: {', '.join(missing_cols)}"
        )

    # Make batch predictions using the ML model
    features_df = df[required_cols[1:]] # drop transaction_id column for ML inference
    preds, probs = fraud_detector.predict_batch(features_df)

    saved_transactions = []
    fraud_count = 0
    normal_count = 0

    for idx, row in df.iterrows():
        tx_id = str(row["transaction_id"])
        # Check if transaction_id exists to avoid duplicate constraint errors
        existing_tx = crud.get_transaction(db, tx_id)
        if existing_tx:
            continue

        is_f = bool(preds[idx])
        score = float(probs[idx])

        if is_f:
            fraud_count += 1
        else:
            normal_count += 1

        tx_create = schemas.TransactionCreate(
            transaction_id=tx_id,
            amount=float(row["amount"]),
            transaction_hour=int(row["transaction_hour"]),
            merchant_category=str(row["merchant_category"]),
            foreign_transaction=int(row["foreign_transaction"]),
            location_mismatch=int(row["location_mismatch"]),
            device_trust_score=int(row["device_trust_score"]),
            velocity_last_24h=int(row["velocity_last_24h"]),
            cardholder_age=int(row["cardholder_age"])
        )

        saved_tx = crud.create_transaction(
            db,
            tx_create,
            is_fraud=is_f,
            score=score,
            user_id=current_user.id
        )
        saved_transactions.append(saved_tx)

    total_processed = len(saved_transactions)
    fraud_ratio = fraud_count / total_processed if total_processed > 0 else 0.0

    # Save uploaded file details to database & disk
    upload_dir = "/Users/NIRAJKUMAR/Desktop/Fraud_Detection/backend/uploaded_files"
    os.makedirs(upload_dir, exist_ok=True)
    saved_file_path = os.path.join(upload_dir, f"{datetime.utcnow().timestamp()}_{file.filename}")
    try:
        with open(saved_file_path, "wb") as f:
            f.write(contents)
        
        crud.create_uploaded_file(
            db, 
            filename=file.filename, 
            filepath=saved_file_path, 
            record_count=total_processed, 
            fraud_count=fraud_count, 
            user_id=current_user.id
        )
    except Exception as e:
        print(f"Failed to save file log to DB/disk: {e}")

    return {
        "total_processed": total_processed,
        "fraud_detected": fraud_count,
        "normal_detected": normal_count,
        "fraud_ratio": fraud_ratio,
        "predictions": saved_transactions
    }


# --- PERFORMANCE METRICS ENDPOINT ---

@app.get("/metrics")
def get_model_metrics():
    # Return metrics calculated during our model evaluation stage
    return {
        "baseline_model": {
            "precision": 1.0,
            "recall": 0.8333,
            "f1_score": 0.9091,
            "roc_auc": 1.0,
            "confusion_matrix": {
                "tn": 1970,
                "fp": 0,
                "fn": 5,
                "tp": 25
            }
        },
        "smote_model": {
            "precision": 1.0,
            "recall": 0.8667,
            "f1_score": 0.9286,
            "roc_auc": 0.9998,
            "confusion_matrix": {
                "tn": 1970,
                "fp": 0,
                "fn": 4,
                "tp": 26
            }
        }
    }


# --- PREDICTION HISTORY ENDPOINT ---

@app.get("/transactions", response_model=List[schemas.TransactionResponse])
def get_user_transaction_history(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    return crud.get_transactions_by_user(db, user_id=current_user.id, skip=skip, limit=limit)


# --- UPLOADED FILES LOGS ---

@app.get("/uploaded-files", response_model=List[schemas.UploadedFileResponse])
def get_user_uploaded_files_history(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    return crud.get_uploaded_files_by_user(db, user_id=current_user.id, skip=skip, limit=limit)


# --- MODEL REGISTRY INFORMATION ---

@app.get("/models", response_model=List[schemas.ModelInfoResponse])
def get_model_registry(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    return crud.get_all_models_info(db)
