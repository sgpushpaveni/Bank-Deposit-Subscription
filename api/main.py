

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running successfully!"}

# Load model and preprocessor (adjust paths as needed)
model = joblib.load("models/XGBoost.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")

# Define what input your API expects
class ClientFeatures(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: float
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@app.post("/predict")
def predict(features: ClientFeatures):
    # Convert to DataFrame
    input_df = pd.DataFrame([features.dict()])
    # Feature engineering (if needed)
    input_df['balance_x_duration'] = input_df['balance'] * input_df['duration']
    input_df['age_x_education'] = input_df['age'] * input_df['education'].map({'unknown':0, 'primary':1, 'secondary':2, 'tertiary':3})
    # Preprocess
    X_processed = preprocessor.transform(input_df)
    # Predict
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0, 1]
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }