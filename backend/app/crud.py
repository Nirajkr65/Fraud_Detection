from sqlalchemy.orm import Session
from . import models, schemas

# User operations
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user: schemas.UserCreate, hashed_password: str):
    db_user = models.User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Transaction operations
def get_transaction(db: Session, transaction_id: str):
    return db.query(models.Transaction).filter(models.Transaction.transaction_id == transaction_id).first()

def get_transactions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Transaction).offset(skip).limit(limit).all()

def get_transactions_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.Transaction).filter(models.Transaction.user_id == user_id).offset(skip).limit(limit).all()

def create_transaction(db: Session, transaction: schemas.TransactionCreate, is_fraud: bool = False, score: float = 0.0, user_id: int = None):
    db_transaction = models.Transaction(
        transaction_id=transaction.transaction_id,
        amount=transaction.amount,
        transaction_hour=transaction.transaction_hour,
        merchant_category=transaction.merchant_category,
        foreign_transaction=transaction.foreign_transaction,
        location_mismatch=transaction.location_mismatch,
        device_trust_score=transaction.device_trust_score,
        velocity_last_24h=transaction.velocity_last_24h,
        cardholder_age=transaction.cardholder_age,
        is_fraud=is_fraud,
        prediction_score=score,
        user_id=user_id
    )
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)
    return db_transaction

# Uploaded Files operations
def create_uploaded_file(db: Session, filename: str, filepath: str, record_count: int, fraud_count: int, user_id: int):
    db_file = models.UploadedFile(
        filename=filename,
        filepath=filepath,
        record_count=record_count,
        fraud_count=fraud_count,
        user_id=user_id
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file

def get_uploaded_files_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.UploadedFile).filter(models.UploadedFile.user_id == user_id).offset(skip).limit(limit).all()

# Model Information operations
def create_model_info(db: Session, model_name: str, version: str, filepath: str, precision: float, recall: float, f1_score: float, roc_auc: float, status: str = "active"):
    # If adding an active model, set other models to archived
    if status == "active":
        db.query(models.ModelInfo).filter(models.ModelInfo.status == "active").update({"status": "archived"})
        db.commit()

    db_model = models.ModelInfo(
        model_name=model_name,
        version=version,
        filepath=filepath,
        status=status,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        roc_auc=roc_auc
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

def get_active_model_info(db: Session):
    return db.query(models.ModelInfo).filter(models.ModelInfo.status == "active").first()

def get_all_models_info(db: Session):
    return db.query(models.ModelInfo).order_by(models.ModelInfo.created_at.desc()).all()
