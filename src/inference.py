import os
import joblib
from datetime import datetime
from src.data_loader import get_sample_features

# Путь к модели
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 'models', 'model.joblib'
)

CSV_URL = 'https://drive.google.com/uc?id=1fekiVOa1A0-n8ZTW68X7kgqfgQ7VjYsR'

# Пути к выходным файлам
PRED_PATH = os.path.join(os.path.dirname(__file__), '..', 'predictions.csv')
REPORT_PATH = os.path.join(os.path.dirname(__file__), '..', 'report.html')

# Проверка наличия модели
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Модель не найдена: {MODEL_PATH}")

# Загрузка модели
model = joblib.load(MODEL_PATH)

# Получение 5 примеров
X = get_sample_features(CSV_URL)

# Предсказание
preds = model.predict(X)
X_result = X.copy()
X_result['Предсказанная зарплата'] = preds

# Сохранение предсказаний
X_result.to_csv(PRED_PATH, index=False, encoding='utf-8-sig')
print(f"✅ Предсказания сохранены в {PRED_PATH}")

# HTML отчёт
html = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Отчёт о предсказаниях зарплаты</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
    </style>
</head>
<body>
    <h1>📊 Отчёт о предсказаниях зарплаты</h1>
    <p><strong>Дата генерации:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Количество примеров:</strong> {len(X_result)}</p>
    {X_result.to_html(index=False, border=0)}
</body>
</html>
"""

# Сохранение отчёта
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"📄 HTML-отчёт сохранён в {REPORT_PATH}")
