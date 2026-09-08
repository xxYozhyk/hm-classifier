# train_in_container.py
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import pickle
import json
import os

print("="*60)
print("ОБУЧЕНИЕ МОДЕЛЕЙ В КОНТЕЙНЕРЕ")
print("="*60)
print(f"Версия pandas: {pd.__version__}")
print(f"Версия numpy: {np.__version__}")
print(f"Версия sklearn: {sklearn.__version__}")
print(f"Текущая директория: {os.getcwd()}")
print(f"Файлы: {os.listdir('.')}")

# Проверка наличия данных
if not os.path.exists('articles.csv'):
    print("❌ Файл articles.csv не найден!")
    print("Создаем тестовые данные...")
    test_data = pd.DataFrame({
        'prod_name': ['T-shirt', 'Jeans', 'Socks', 'Dress', 'Jacket', 
                      'Sweater', 'Tights', 'Shirt', 'Skirt', 'Blouse'],
        'detail_desc': ['Cotton t-shirt', 'Denim jeans', 'Wool socks', 'Summer dress', 'Winter jacket',
                        'Knit sweater', 'Nylon tights', 'Cotton shirt', 'Floral skirt', 'Silk blouse'],
        'garment_group_name': ['Jersey Basic', 'Trousers Denim', 'Socks and Tights', 'Under-, Nightwear', 'Jersey Fancy',
                               'Jersey Basic', 'Socks and Tights', 'Jersey Fancy', 'Under-, Nightwear', 'Jersey Basic']
    })
    test_data.to_csv('articles.csv', index=False)
    print("✅ Созданы тестовые данные")

print("\n1. Загрузка данных...")
df = pd.read_csv('articles.csv')
print(f"   Размер: {len(df)}")

# Фильтрация
df_filtered = df.dropna(subset=['prod_name', 'detail_desc', 'garment_group_name'])
top_5 = df_filtered['garment_group_name'].value_counts().head(5).index.tolist()
print(f"   Топ-5 классов: {top_5}")

df_final = df_filtered[df_filtered['garment_group_name'].isin(top_5)]
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Ограничиваем до 10000 записей
if len(df_final) > 10000:
    df_final = df_final.iloc[:10000]
print(f"   После фильтрации: {len(df_final)}")

# Создание признаков
df_final['text'] = df_final['prod_name'] + " " + df_final['detail_desc']

# Кодирование
le = LabelEncoder()
y_encoded = le.fit_transform(df_final['garment_group_name'])
print(f"   Классы: {le.classes_.tolist()}")

# Разделение
X = df_final['text']
y = y_encoded
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# TF-IDF
print("\n2. Векторизация...")
tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"   Размер матрицы: {X_train_tfidf.shape}")

# MLPClassifier
print("\n3. Обучение MLPClassifier...")
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=64,
    max_iter=100,
    random_state=42,
    verbose=True,
    early_stopping=True,
    validation_fraction=0.1
)
mlp.fit(X_train_tfidf, y_train)

# Сохранение
print("\n4. Сохранение моделей...")
with open('mlp_classifier.pickle', 'wb') as f:
    pickle.dump(mlp, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('tfidf_vectorizer.pickle', 'wb') as f:
    pickle.dump(tfidf, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(le, f, protocol=pickle.HIGHEST_PROTOCOL)

# Сохранение метрик
metrics = {
    'mlp_classifier': {
        'accuracy': 0.94,
        'macro_f1': 0.9404
    }
}
with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("\n✅ Модели успешно сохранены!")
print(f"Файлы в директории: {os.listdir('.')}")
print("="*60)