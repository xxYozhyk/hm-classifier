# app.py
import pickle
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import uvicorn

app = FastAPI(
    title="H&M Product Classifier",
    description="Сервис для классификации товаров H&M по названию и описанию",
    version="1.0.0"
)

# Модель запроса
class ProductRequest(BaseModel):
    prod_name: str = Field(..., description="Название товара")
    detail_desc: str = Field(..., description="Описание товара")
    
    @validator('prod_name')
    def validate_prod_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Название товара не может быть пустым')
        return v.strip()
    
    @validator('detail_desc')
    def validate_detail_desc(cls, v):
        if not v or not v.strip():
            raise ValueError('Описание товара не может быть пустым')
        return v.strip()

# Модель ответа
class ProductResponse(BaseModel):
    predicted_group: str
    probabilities: dict
    model_used: str = "MLPClassifier"
    status: str = "success"

# Глобальные переменные для моделей
model = None
tfidf = None
label_encoder = None
model_metrics = None

def load_models():
    """Загрузка моделей и предобработчиков"""
    global model, tfidf, label_encoder, model_metrics
    
    try:
        with open('mlp_classifier.pickle', 'rb') as f:
            model = pickle.load(f)
        print("✅ MLPClassifier загружен")
        
        with open('tfidf_vectorizer.pickle', 'rb') as f:
            tfidf = pickle.load(f)
        print("✅ TF-IDF векторизатор загружен")
        
        with open('label_encoder.pickle', 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"✅ LabelEncoder загружен. Классы: {label_encoder.classes_.tolist()}")
        
        with open('model_metrics.json', 'r') as f:
            model_metrics = json.load(f)
        print("✅ Метрики модели загружены")
        
        return True
    except FileNotFoundError as e:
        print(f"❌ Ошибка загрузки: {e}")
        return False
    except Exception as e:
        print(f"❌ Непредвиденная ошибка: {e}")
        return False

# Загружаем модели при старте
load_models()

@app.get("/")
async def root():
    """Корневой эндпоинт с информацией"""
    return {
        "service": "H&M Product Classifier",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "POST /predict": "Predict product group",
            "GET /": "Service information"
        },
        "model_used": "MLPClassifier",
        "classes": label_encoder.classes_.tolist() if label_encoder else []
    }

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    if model is not None and tfidf is not None and label_encoder is not None:
        return {
            "status": "healthy",
            "model_loaded": True,
            "classes": label_encoder.classes_.tolist(),
            "model_metrics": model_metrics
        }
    else:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: models not loaded"
        )

@app.post("/predict", response_model=ProductResponse)
async def predict(request: ProductRequest):
    """
    Предсказание группы товара по названию и описанию
    """
    # Проверка, что модели загружены
    if model is None or tfidf is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: models not loaded"
        )
    
    # Объединяем название и описание
    text = f"{request.prod_name} {request.detail_desc}".strip()
    
    # Дополнительная проверка (на всякий случай)
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Combined text is empty. Please provide non-empty prod_name and detail_desc"
        )
    
    try:
        # Векторизация
        X = tfidf.transform([text])
        
        # Предсказание класса
        pred_class = model.predict(X)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]
        
        # Вероятности
        probabilities = model.predict_proba(X)[0]
        prob_dict = {
            label: float(prob) 
            for label, prob in zip(label_encoder.classes_, probabilities)
        }
        
        return ProductResponse(
            predicted_group=pred_label,
            probabilities=prob_dict,
            model_used="MLPClassifier",
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )