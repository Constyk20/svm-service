from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Chronic Kidney Disease Prediction",
    description="SVM Model - 94.1% Accuracy",
    version="1.0"
)

MODEL_PATH = "svm_model.pkl"
model = joblib.load(MODEL_PATH)

class Patient(BaseModel):
    age: float
    bp: float
    sg: float
    al: float
    su: float
    rbc: int
    pc: int
    pcc: int
    ba: int
    bgr: float
    bu: float
    sc: float
    sod: float
    pot: float  # ← FIXED: Added potassium
    hemo: float
    pcv: float
    wc: float
    rc: float
    htn: int
    dm: int
    cad: int
    appet: int
    pe: int
    ane: int

@app.get("/")
def home():
    return {"status": "SVM Service Running - 94.1% Accuracy!"}

@app.post("/predict")
def predict(patient: Patient):
    data = np.array([[
        patient.age, patient.bp, patient.sg, patient.al, patient.su,
        patient.rbc, patient.pc, patient.pcc, patient.ba, patient.bgr,
        patient.bu, patient.sc, patient.sod, patient.pot, patient.hemo,
        patient.pcv, patient.wc, patient.rc,
        patient.htn, patient.dm, patient.cad,
        patient.appet, patient.pe, patient.ane
    ]])

    try:
        probability = float(model.predict_proba(data)[0][1]) * 100
        prediction = int(model.predict(data)[0])
        
        return {
            "model": "SVM",
            "risk_score": round(probability, 2),
            "result": "CKD Detected" if prediction == 1 else "No CKD (Healthy)",
            "confidence": round(probability if prediction == 1 else 100 - probability, 2)
        }
    except Exception as e:
        return {"error": str(e), "message": "Prediction failed"}
