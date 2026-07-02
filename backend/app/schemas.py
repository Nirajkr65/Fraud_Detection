from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

# User Schemas
class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int

    class Config:
        from_attributes = True

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Transaction Schemas (Aligning with models.py and dataset columns)
class TransactionBase(BaseModel):
    transaction_id: str
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: int
    velocity_last_24h: int
    cardholder_age: int

class TransactionCreate(TransactionBase):
    pass

class TransactionResponse(TransactionBase):
    id: int
    is_fraud: Optional[bool] = None
    prediction_score: Optional[float] = None
    timestamp: datetime
    user_id: Optional[int] = None

    class Config:
        from_attributes = True

# Batch response
class BatchPredictionResponse(BaseModel):
    total_processed: int
    fraud_detected: int
    normal_detected: int
    fraud_ratio: float
    predictions: List[TransactionResponse]

# Uploaded File Schemas
class UploadedFileResponse(BaseModel):
    id: int
    filename: str
    filepath: str
    upload_timestamp: datetime
    record_count: int
    fraud_count: int
    user_id: int

    class Config:
        from_attributes = True

# Model Info Schemas
class ModelInfoResponse(BaseModel):
    id: int
    model_name: str
    version: str
    filepath: str
    status: str
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True
