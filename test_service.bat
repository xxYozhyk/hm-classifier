@echo off
chcp 65001 > nul
echo ========================================
echo ТЕСТИРОВАНИЕ H&M CLASSIFIER SERVICE
echo ========================================
echo.

echo 1. Проверка Health...
echo.
curl -s -X GET http://localhost:8000/health
echo.
echo.

echo 2. Проверка корневого эндпоинта...
echo.
curl -s -X GET http://localhost:8000/
echo.
echo.

echo 3. Тест предсказания: Strap top...
echo.
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"prod_name\":\"Strap top\",\"detail_desc\":\"Jersey top with narrow shoulder straps\"}"
echo.
echo.

echo 4. Тест предсказания: Socks...
echo.
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"prod_name\":\"Socks\",\"detail_desc\":\"Fine-knit socks in cotton blend\"}"
echo.
echo.

echo 5. Тест предсказания: Jeans...
echo.
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"prod_name\":\"Jeans\",\"detail_desc\":\"5-pocket jeans in washed denim\"}"
echo.
echo.

echo 6. Тест предсказания: Dress...
echo.
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"prod_name\":\"Dress\",\"detail_desc\":\"Short dress in soft organic cotton jersey\"}"
echo.
echo.

echo 7. Тест с пустыми данными (должна быть ошибка)...
echo.
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"prod_name\":\"\",\"detail_desc\":\"\"}"
echo.
echo.

echo ========================================
echo ТЕСТИРОВАНИЕ ЗАВЕРШЕНО
echo ========================================
pause