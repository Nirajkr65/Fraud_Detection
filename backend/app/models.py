from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    transactions = relationship("Transaction", back_populates="user")
    uploaded_files = relationship("UploadedFile", back_populates="user")

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True)
    amount = Column(Float, nullable=False)
    transaction_hour = Column(Integer, nullable=False)
    merchant_category = Column(String, nullable=False)
    foreign_transaction = Column(Integer, nullable=False)
    location_mismatch = Column(Integer, nullable=False)
    device_trust_score = Column(Integer, nullable=False)
    velocity_last_24h = Column(Integer, nullable=False)
    cardholder_age = Column(Integer, nullable=False)
    is_fraud = Column(Boolean, nullable=True)
    prediction_score = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    user = relationship("User", back_populates="transactions")

class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    record_count = Column(Integer, default=0)
    fraud_count = Column(Integer, default=0)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    user = relationship("User", back_populates="uploaded_files")

class ModelInfo(Base):
    __tablename__ = "model_information"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    status = Column(String, default="active")  # active, archived
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
