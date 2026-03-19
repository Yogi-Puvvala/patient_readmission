from src.predict import predict
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PatientInput(BaseModel):
    numeric: List[int]
    categorical: List[str]
    textual: List[str]
    sequential: List[str]

@app.get("/")
def sayWelcome():
    return {"message": "Welcome to patient readmission check"}

@app.post("/predict")
def predict_endpoint(data: PatientInput):
    payload = data.model_dump()
    result = predict(payload)
    return result

