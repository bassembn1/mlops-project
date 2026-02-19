from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# تحميل الموديل مرة واحدة عند تشغيل السيرفر
model = joblib.load("models/model.joblib")
scaler = joblib.load("models/scaler.joblib")


@app.get("/")
def home():
    return {"message": "ML Model API is running"}

@app.post("/predict")
def predict(age: float, salary: float, experience: float):

    data = np.array([[age, salary, experience]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    return {
        "prediction": int(prediction)
    }
