import os
import joblib

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.data_loader import load_sample_data

GOOGLE_DRIVE_FILE_ID = '1fekiVOa1A0-n8ZTW68X7kgqfgQ7VjYsR'
CSV_URL = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'

MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')
os.makedirs(MODEL_DIR, exist_ok=True)

# Загрузка данных
X_train, X_test, y_train, y_test = load_sample_data(CSV_URL)

cat_features = X_train.select_dtypes(include="category").columns.tolist()

# Обучение модели
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    verbose=100,
    random_state=42
)

model.fit(X_train, y_train, cat_features=cat_features)

# Оценка качества
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'✅ MAE: {mae:.2f}')
print(f'✅ R²: {r2:.4f}')

# Сохранение модели
joblib.dump(model, MODEL_PATH)
print(f'✅ Model saved to {MODEL_PATH}')

# Вывод признаков
print('🔍 Использованные признаки:')
print(model.feature_names_)
