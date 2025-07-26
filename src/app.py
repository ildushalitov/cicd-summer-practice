from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import joblib
from src.data_loader import preprocess_data

app = FastAPI(title="Salary Predictor")

# Загрузка модели
model = joblib.load("models/model.joblib")


# 🎯 Модель входных данных
class VacancyFeatures(BaseModel):
    employees_number: int
    work_schedule: str
    employment: str
    length_of_employment: float
    region_name: str
    accept_teenagers: bool
    specialization: str
    response_count: int
    invitation_count: int
    month: int = Field(..., ge=1, le=12)
    day_of_week: int = Field(..., ge=0, le=6)
    industry: Optional[str] = None


@app.post("/predict")
def predict(data: VacancyFeatures):
    # Преобразуем JSON в DataFrame
    df_raw = pd.DataFrame([data.dict()])

    try:
        X, _ = preprocess_data(df_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предобработки: {e}")

    try:
        pred = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {e}")

    return {"Предсказанная зарплата": round(pred, 2)}


@app.get("/report")
def get_report():
    return FileResponse("report.html", media_type="text/html")
