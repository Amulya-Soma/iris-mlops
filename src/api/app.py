from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import threading
import webbrowser
import os

# Paths
MODEL_DIR = "src/model/iris_model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# Load model
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

# Define request schema using Pydantic
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Mapping to match training column names
COLUMN_MAPPING = {
    "sepal_length": "sepal length (cm)",
    "sepal_width": "sepal width (cm)",
    "petal_length": "petal length (cm)",
    "petal_width": "petal width (cm)"
}

# Create FastAPI app
app = FastAPI(title="Iris Prediction API", description="Predict Iris species", version="1.0")

@app.post("/predict")
def predict(features: IrisFeatures):
    # Convert request to DataFrame
    input_df = pd.DataFrame([features.dict()])
    
    # Rename to match model training columns
    input_df.rename(columns=COLUMN_MAPPING, inplace=True)
    
    # Make prediction
    prediction = model.predict(input_df)
    
    return {"prediction": prediction.tolist()}

# Function to open browser automatically
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8003/docs")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=8003)