FROM python:3.9-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Проверяем установку
RUN python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
RUN python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
RUN python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"

# Копируем модель и файлы для обучения
COPY articles.csv .
COPY train_in_container.py .
RUN python train_in_container.py

# Копируем приложение (ИСПРАВЛЕНО!)
COPY main.py .

# Копируем остальные файлы
COPY lstm_model.pth .
COPY tfidf_vectorizer.pickle .
COPY label_encoder.pickle .
COPY mlp_classifier.pickle .
COPY model_metrics.json .

# Создаем пользователя
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Запускаем приложение
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]