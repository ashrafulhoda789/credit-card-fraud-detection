import os
import joblib
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. FastAPI App Config
app = FastAPI(title="Credit Card Fraud Detection Production API")

# 2. Database Setup (SQLite for production logging)
DATABASE_URL = "sqlite:///./transactions.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TransactionLog(Base):
    __tablename__ = "transaction_logs"
    id = Column(Integer, primary_key=True, index=True)
    card_number_hash = Column(String)
    amount = Column(Float)
    merchant_category = Column(String)
    location = Column(String)
    fraud_probability = Column(Float)
    is_fraud = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# 3. Load Models
MODEL_PATH = os.path.join('models', 'hybrid_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("Model or Scaler not found!")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# 4. Request Schema
class RawTransactionRequest(BaseModel):
    card_number: str
    amount: float
    merchant_category: str
    location: str

# 5. Core Processing & Prediction Endpoint
@app.post("/api/v1/predict")
def predict_transaction(tx: RawTransactionRequest):
    # Dummy Feature Extraction Pipeline (Raw Data -> PCA V1-V28 Map)
    # বাস্তবে এখানে ব্যাংকের নিজস্ব Preprocessing Engine থাকে
    v_features = [0.0] * 28  
    time_val = 100.0          

    input_data = np.array([[time_val] + v_features + [tx.amount]])
    scaled_data = scaler.transform(input_data)
    
    prediction = int(model.predict(scaled_data)[0])
    probability = float(model.predict_proba(scaled_data)[0][1])

    # Save Log to Database
    db = SessionLocal()
    masked_card = f"XXXX-XXXX-XXXX-{tx.card_number[-4:]}" if len(tx.card_number) >= 4 else "XXXX"
    log_entry = TransactionLog(
        card_number_hash=masked_card,
        amount=tx.amount,
        merchant_category=tx.merchant_category,
        location=tx.location,
        fraud_probability=round(probability * 100, 2),
        is_fraud=prediction
    )
    db.add(log_entry)
    db.commit()
    db.close()

    return {
        "status": "success",
        "card_number": masked_card,
        "is_fraud": bool(prediction),
        "fraud_risk_percentage": round(probability * 100, 2),
        "timestamp": datetime.utcnow().isoformat()
    }

# 6. Database Fetch Endpoint for Dashboard
@app.get("/api/v1/logs")
def get_transaction_logs():
    db = SessionLocal()
    logs = db.query(TransactionLog).order_by(TransactionLog.timestamp.desc()).all()
    db.close()
    return logs