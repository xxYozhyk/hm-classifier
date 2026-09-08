FROM python:3.9-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Проверяем версии
RUN python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
RUN python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
RUN python -c "import pandas; print(f'Pandas: {pandas.__version__}')"

# Копируем данные и скрипт обучения
COPY articles.csv .
COPY train_in_container.py .

# Обучаем модели
RUN python train_in_container.py

# Проверяем, что модели создались
RUN ls -la /app/*.pickle

# Копируем приложение
COPY app.py .

# Создаем пользователя
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]