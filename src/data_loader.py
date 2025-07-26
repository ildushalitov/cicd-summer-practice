import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str):
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Средняя зарплата как целевая переменная
    df["salary"] = df[["compensation_from", "compensation_to"]].mean(axis=1)

    # Удаление строк без зарплаты и с аномальными значениями
    df = df.dropna(subset=["salary"])
    df = df[(df["salary"] >= 10000) & (df["salary"] <= 300000)]

    # Преобразуем дату в числовые признаки
    df["creation_date"] = pd.to_datetime(df["creation_date"], errors='coerce')
    df["month"] = df["creation_date"].dt.month
    df["day_of_week"] = df["creation_date"].dt.dayofweek

    # Топ-10 отраслей (по industry_id_list) как бинарные признаки
    df["industry_id_list"] = df["industry_id_list"].astype(str)
    top_industries = df["industry_id_list"].value_counts().head(10).index
    for ind in top_industries:
        df[f"industry_{ind}"] = df["industry_id_list"].apply(lambda x: ind in x)

    # Признаки и целевая переменная
    features = [
                   "employees_number", "work_schedule", "employment", "length_of_employment",
                   "region_name", "accept_teenagers", "specialization",
                   "response_count", "invitation_count", "month", "day_of_week"
               ] + [f"industry_{ind}" for ind in top_industries]

    # Удаление строк с пропущенными признаками
    df = df.dropna(subset=features)

    # ✅ Преобразуем все строковые признаки в категориальные
    for col in df[features].select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")

    X = df[features]
    y = df["salary"]

    return X, y


def load_and_preprocess(path: str):
    df = load_data(path)
    return preprocess_data(df)


def load_sample_data(path: str, test_size=0.2, random_state=42):
    X, y = load_and_preprocess(path)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_sample_features(path: str, n: int = 5):
    X, _ = load_and_preprocess(path)
    return X.head(n)
