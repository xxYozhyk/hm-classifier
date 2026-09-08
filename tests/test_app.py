# tests/test_app.py
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Добавляем путь к приложению
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, load_models

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_models():
    """Загрузка моделей перед тестами"""
    success = load_models()
    assert success, "Models failed to load"
    yield

def test_health_endpoint(setup_models):
    """Тест health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True
    assert "classes" in data
    assert len(data["classes"]) > 0

def test_root_endpoint(setup_models):
    """Тест корневого эндпоинта"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "H&M Product Classifier"
    assert "model_used" in data

def test_predict_valid(setup_models):
    """Тест предсказания с корректными данными"""
    response = client.post(
        "/predict",
        json={
            "prod_name": "Strap top",
            "detail_desc": "Jersey top with narrow shoulder straps"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_group" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 5
    assert abs(sum(data["probabilities"].values()) - 1.0) < 0.01

@pytest.mark.parametrize("prod_name, detail_desc", [
    ("T-shirt", "Cotton t-shirt"),
    ("Jeans", "Denim jeans"),
    ("Socks", "Wool socks"),
    ("Dress", "Summer dress"),
])
def test_parametrized_predict(setup_models, prod_name, detail_desc):
    """Параметризованный тест с разными товарами"""
    response = client.post(
        "/predict",
        json={"prod_name": prod_name, "detail_desc": detail_desc}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_group"] in [
        "Accessories", "Jersey Basic", "Jersey Fancy", 
        "Knitwear", "Under-, Nightwear"
    ]
    assert len(data["probabilities"]) == 5

def test_empty_text(setup_models):
    """Тест с пустыми полями"""
    # Пустое название
    response = client.post(
        "/predict",
        json={"prod_name": "", "detail_desc": "Some description"}
    )
    assert response.status_code == 400 or response.status_code == 422
    
    # Пустое описание
    response = client.post(
        "/predict",
        json={"prod_name": "Product", "detail_desc": ""}
    )
    assert response.status_code == 400 or response.status_code == 422
    
    # Оба пустые
    response = client.post(
        "/predict",
        json={"prod_name": "", "detail_desc": ""}
    )
    assert response.status_code == 400 or response.status_code == 422

def test_invalid_input(setup_models):
    """Тест с некорректным вводом"""
    # Отсутствие полей
    response = client.post("/predict", json={})
    assert response.status_code == 422