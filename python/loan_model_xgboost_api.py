from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Define the data model for input
class CustomerData(BaseModel):
    years: float
    ownership: str
    income: float
    age: int
    grade: str
    amount: float
    interest: float

# Load your trained models
models = {"xgboost": joblib.load(r"D:\GenerativeAI LLMs\Day2\Result\xgboost_model.pkl")}

app = FastAPI()

@app.post("/predict_default/")
async def predict_default(customer_data: CustomerData):
    try:
        # Convert the input data to a DataFrame
        data = pd.DataFrame([customer_data.dict().values()], columns=customer_data.dict().keys())

        # Preprocess the data as per your model requirements
        # (e.g., scaling, encoding categorical variables, etc.)
        # Make sure this matches the preprocessing in your training script

        # Predict using each model
        predictions = {}
        for model_name, model in models.items():
            prediction = model.predict_proba(data)[0][1]  # Assuming the second class is default
            predictions[model_name] = float(prediction)  # Convert numpy.float32 to Python float

        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add other routes if needed