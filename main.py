from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import asyncio
import os

# ========================
# FASTAPI APP CONFIGURATION
# ========================
app = FastAPI(title="Credit Card Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# REQUEST BODY
# ========================
class TransactionRequest(BaseModel):
    amount: float
    time: float

# ========================
# FRAUD DETECTION CLASS
# ========================
class FraudDetectionSystem:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_info = None
        self.feature_names = None
        self.load_model()

    def load_model(self):
        try:
            with open('model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('model_info.pkl', 'rb') as f:
                self.model_info = pickle.load(f)
            self.feature_names = self.model_info.get('feature_names') or \
                (list(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else 
                 ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Model load error: {e}")
            self.model_info = {'use_scaling': False}
            self.feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

    def create_transaction_data(self, amount, time):
        transaction = {'Time': time}
        for i in range(1, 29):
            transaction[f'V{i}'] = np.random.normal(0, 1)
        transaction['Amount'] = amount
        return transaction

    def predict_single_transaction(self, transaction_data):
        try:
            df = pd.DataFrame([transaction_data])
            df = df.reindex(columns=self.feature_names)
            X = self.scaler.transform(df) if self.model_info.get('use_scaling') else df
            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            return {
                'prediction': 'FRAUD' if pred == 1 else 'LEGITIMATE',
                'fraud_confidence': float(proba[1] * 100),
                'legitimate_confidence': float(proba[0] * 100)
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'prediction': 'LEGITIMATE',
                'fraud_confidence': 0.0,
                'legitimate_confidence': 100.0
            }

fraud_detector = FraudDetectionSystem()

# ========================
# MANUAL PREDICTION ENDPOINT
# ========================
@app.post("/predict")
async def predict_fraud(request: TransactionRequest):
    transaction_data = fraud_detector.create_transaction_data(
        amount=request.amount,
        time=request.time
    )
    result = fraud_detector.predict_single_transaction(transaction_data)
    return {
        "amount": request.amount,
        "time": request.time,
        **result
    }

# ========================
# TEST DATA ENDPOINT
# ========================
@app.get("/test-data")
async def get_test_data():
    return [
        {"name": "🚨 Midnight Fraud", "amount": 150000, "time": 3600},
        {"name": "🌙 Late Night High Amount", "amount": 200000, "time": 1800},
        {"name": "💰 Large Daytime", "amount": 500000, "time": 43200},
        {"name": "🏪 Normal Purchase", "amount": 15000, "time": 50400},
        {"name": "☕ Coffee Purchase", "amount": 2500, "time": 28800},
        {"name": "💳 Card Testing", "amount": 0.50, "time": 7200}
    ]

# ========================
# WEBSOCKET CONNECTION
# ========================
clients = set()
streaming_started = False

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    print(f"📡 Client connected: {len(clients)} total")

    global streaming_started
    if not streaming_started:
        streaming_started = True
        asyncio.create_task(simulation_loop())

    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        clients.remove(websocket)
        print(f"📴 Client disconnected: {len(clients)} total")

# ========================
# SIMULATION LOOP
# ========================
async def simulation_loop():
    csv_path = os.path.join(os.path.dirname(__file__), "raw_transactions.csv")
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        amount = float(row['amount'])
        
        # Convert timestamp → seconds since midnight
        try:
            dt = datetime.strptime(row['timestamp'], "%m/%d/%Y %H:%M")
            time_seconds = dt.hour * 3600 + dt.minute * 60
        except Exception:
            time_seconds = 0

        transaction_data = fraud_detector.create_transaction_data(amount, time_seconds)
        prediction = fraud_detector.predict_single_transaction(transaction_data)

        message = {
            "amount": amount,
            "time": time_seconds,
            **prediction
        }

        if clients:  # Only send if clients are connected
            for ws in clients:
                await ws.send_json(message)

        print(f"📤 Sent simulated transaction: {message}")
        await asyncio.sleep(2)  # pacing

# ========================
# ROOT ENDPOINT
# ========================
@app.get("/")
async def root():
    return {"message": "Credit Card Fraud Detection API with Simulation", "status": "running"}

# ========================
# MAIN ENTRY POINT
# ========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
