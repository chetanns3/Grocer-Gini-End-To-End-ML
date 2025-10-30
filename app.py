from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import pathlib

app = FastAPI(title="Supermarket Sales Predictor API")

# --- Load Model and Feature Names ---
MODEL_PATH = pathlib.Path("models/xgb_model.pkl")
FEATURES_PATH = pathlib.Path("models/feature_names.pkl")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

# --- Define Input Schema ---
class SalesInput(BaseModel):
    Item_Weight: float
    Item_Fat_Content: str
    Item_Visibility: float
    Item_Type: str
    Item_MRP: float
    Outlet_Establishment_Year: int
    Outlet_Size: str
    Outlet_Location_Type: str
    Outlet_Type: str

@app.get("/")
def home():
    return {"message": "Welcome to Supermarket Sales Predictor API 🚀"}

@app.post("/predict")
def predict_sales(data: SalesInput):
    df = pd.DataFrame([data.model_dump()])

    # 🔹 Encode categorical variables exactly as during training (dummy encoding)
    df_encoded = pd.get_dummies(df)

    # 🔹 Align columns with the trained model’s feature set
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)

    # 🔹 Predict
    prediction = model.predict(df_encoded)[0]
    return {"Predicted_Sales": round(float(prediction), 2)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
