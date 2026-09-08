import requests
import json

BASE_URL = "http://localhost:8000"

print("="*60)
print("ТЕСТИРОВАНИЕ H&M CLASSIFIER API")
print("="*60)
print()

# 1. Тест GET /
print("1. Тест GET /")
try:
    response = requests.get(f"{BASE_URL}/")
    print(f"✅ Статус: {response.status_code} OK")
    data = response.json()
    print(f"   Сервис: {data['service']}")
    print(f"   Версия: {data['version']}")
    print(f"   Модель: {data['model_used']}")
except Exception as e:
    print(f"❌ Ошибка: {e}")
print()

# 2. Тест GET /health
print("2. Тест GET /health")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"✅ Статус: {response.status_code} OK")
    data = response.json()
    print(f"   Статус: {data['status']}")
    print(f"   Модель загружена: {data['model_loaded']}")
    print(f"   Классы: {', '.join(data['classes'])}")
except Exception as e:
    print(f"❌ Ошибка: {e}")
print()

# 3. Тест POST /predict - корректные запросы
print("3. Тест POST /predict (корректные запросы)")
test_cases = [
    ("Strap top", "Jersey top with narrow shoulder straps"),
    ("Socks", "Fine-knit socks in cotton blend"),
    ("Jeans", "5-pocket jeans in washed denim"),
    ("Dress", "Short dress in soft organic cotton jersey"),
]

for prod_name, detail_desc in test_cases:
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"prod_name": prod_name, "detail_desc": detail_desc}
        )
        if response.status_code == 200:
            data = response.json()
            confidence = data['probabilities'][data['predicted_group']] * 100
            print(f"✅ {prod_name}: {data['predicted_group']} (уверенность: {confidence:.2f}%)")
        else:
            print(f"❌ {prod_name}: Ошибка {response.status_code}")
            print(f"   Детали: {response.json()}")
    except Exception as e:
        print(f"❌ {prod_name}: {e}")
print()

# 4. Тест POST /predict - некорректные запросы
print("4. Тест POST /predict (некорректные запросы)")

invalid_cases = [
    {"prod_name": "", "detail_desc": "Description without name"},
    {"prod_name": "Name without description", "detail_desc": ""},
    {"prod_name": "", "detail_desc": ""}
]

for case in invalid_cases:
    try:
        response = requests.post(f"{BASE_URL}/predict", json=case)
        if response.status_code == 400:
            print(f"✅ Ожидаемая ошибка 400: {case}")
        else:
            print(f"⚠️ Статус {response.status_code}: {case}")
            print(f"   Ответ: {response.json()}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

print()
print("="*60)
print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
print("="*60)