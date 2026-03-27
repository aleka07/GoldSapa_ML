"""
Model V2 — CatBoost с расширенными признаками.

Новые признаки поверх baseline:
  - Температура (Open-Meteo Archive)
  - Лаги продаж (7, 14, 30 дней)
  - Скользящие средние (MA7, MA14, MA30)
  - Средняя цена товара

Запуск:
    python -m src.model_v2
"""
import logging
import os

import holidays
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.config import DAILY_SALES_PATH, DATA_DIR, LATITUDE, LONGITUDE

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(DATA_DIR, "model")
KZ_HOLIDAYS = holidays.Kazakhstan(years=range(2023, 2028))
VAL_CUTOFF = "2026-01-01"

CAT_FEATURES = ["Номенклатура_Key", "Склад_Key"]
TARGET = "Количество"


# ─── Feature Engineering ────────────────────────────────────

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["Date"])
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = dt.apply(lambda x: 1 if x in KZ_HOLIDAYS else 0)
    return df


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить температуру из Open-Meteo Archive."""
    import requests

    df = df.copy()
    date_min = df["Date"].min().strftime("%Y-%m-%d")
    date_max = df["Date"].max().strftime("%Y-%m-%d")

    logger.info("Запрос погоды %s → %s …", date_min, date_max)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": date_min,
        "end_date": date_max,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "Asia/Almaty",
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    weather = pd.DataFrame({
        "Date": pd.to_datetime(data["daily"]["time"]),
        "temperature": data["daily"]["temperature_2m_mean"],
        "precipitation": data["daily"]["precipitation_sum"],
    })
    logger.info("Получено %d дней погоды.", len(weather))

    df = df.merge(weather, on="Date", how="left")
    df["temperature"] = df["temperature"].fillna(df["temperature"].median())
    df["precipitation"] = df["precipitation"].fillna(0)
    return df


def add_price(df: pd.DataFrame) -> pd.DataFrame:
    """Средняя цена за единицу."""
    df = df.copy()
    df["avg_price"] = np.where(
        df["Количество"] > 0,
        df["Сумма"] / df["Количество"],
        0,
    )
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Лаги и скользящие средние по (Номенклатура_Key, Склад_Key)."""
    df = df.copy()
    df = df.sort_values(["Номенклатура_Key", "Склад_Key", "Date"])

    group = df.groupby(["Номенклатура_Key", "Склад_Key"])["Количество"]

    # Лаги
    for lag in [7, 14, 30]:
        df[f"lag_{lag}"] = group.shift(lag)

    # Скользящие средние
    for window in [7, 14, 30]:
        df[f"ma_{window}"] = group.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # Скользящее стд (для понимания волатильности)
    df["std_7"] = group.transform(
        lambda x: x.shift(1).rolling(7, min_periods=2).std()
    )

    # Заполняем NaN нулями (первые дни без истории)
    lag_cols = [c for c in df.columns if c.startswith(("lag_", "ma_", "std_"))]
    df[lag_cols] = df[lag_cols].fillna(0)

    logger.info("Лаги и MA добавлены: %s", lag_cols)
    return df


# ─── Метрики ────────────────────────────────────────────────

def mape(y_true, y_pred):
    mask = y_true > 0
    return (abs(y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100


# ─── Обучение ───────────────────────────────────────────────

def train_v2():
    # Загрузка
    df = pd.read_parquet(DAILY_SALES_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # Feature engineering
    logger.info("═══ Feature Engineering ═══")
    df = add_calendar(df)
    df = add_weather(df)
    df = add_price(df)
    df = add_lags(df)

    # Список признаков
    num_features = [
        "day_of_week", "month", "day_of_year", "week_of_year",
        "is_weekend", "is_holiday",
        "temperature", "precipitation",
        "avg_price",
        "lag_7", "lag_14", "lag_30",
        "ma_7", "ma_14", "ma_30",
        "std_7",
    ]
    all_features = CAT_FEATURES + num_features

    # Split
    train = df[df["Date"] < VAL_CUTOFF].copy()
    val = df[df["Date"] >= VAL_CUTOFF].copy()
    logger.info("Train: %d строк, Val: %d строк", len(train), len(val))

    # CatBoost
    cat_idx = [all_features.index(f) for f in CAT_FEATURES]

    train_pool = Pool(train[all_features], train[TARGET], cat_features=cat_idx)
    val_pool = Pool(val[all_features], val[TARGET], cat_features=cat_idx)

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric="MAE",
        random_seed=42,
        verbose=100,
    )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

    # Предсказания
    val_pred = model.predict(val[all_features])

    # Метрики
    mae = mean_absolute_error(val[TARGET], val_pred)
    rmse = root_mean_squared_error(val[TARGET], val_pred)
    r2 = r2_score(val[TARGET], val_pred)
    mape_val = mape(val[TARGET].values, val_pred)

    print(f"\n{'='*50}")
    print(f"  MODEL V2 РЕЗУЛЬТАТЫ (Val: 2026)")
    print(f"{'='*50}")
    print(f"  MAE:   {mae:.2f} шт")
    print(f"  RMSE:  {rmse:.2f} шт")
    print(f"  MAPE:  {mape_val:.1f}%")
    print(f"  R²:    {r2:.4f}")
    print(f"{'='*50}")

    # Сравнение с baseline
    baseline_path = os.path.join(MODEL_DIR, "metrics.csv")
    if os.path.exists(baseline_path):
        bl = pd.read_csv(baseline_path)
        bl_mae = bl.iloc[0]["MAE"]
        delta = mae - bl_mae
        pct = delta / bl_mae * 100
        print(f"\n  vs Baseline:")
        print(f"  MAE:  {bl_mae:.2f} → {mae:.2f}  ({delta:+.2f}, {pct:+.1f}%)")
        print(f"  {'✅ УЛУЧШЕНИЕ' if delta < 0 else '❌ УХУДШЕНИЕ'}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": all_features,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    print(f"\n  Feature Importance:")
    for _, row in importance.iterrows():
        bar = "█" * int(row["importance"] / 2)
        print(f"  {row['feature']:20s} {row['importance']:6.1f}  {bar}")

    # Сохранение
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(MODEL_DIR, "v2.cbm"))

    # Обновляем metrics.csv
    metrics = pd.DataFrame([{
        "model": "v2_full_features",
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape_val, 1),
        "R2": round(r2, 4),
        "features": ", ".join(all_features),
    }])
    if os.path.exists(baseline_path):
        old = pd.read_csv(baseline_path)
        metrics = pd.concat([old, metrics], ignore_index=True)
    metrics.to_csv(baseline_path, index=False)

    # Val predictions
    val_result = val[["Date", "Номенклатура", "Склад_Key", TARGET]].copy()
    val_result["Predicted"] = val_pred.round(1)
    val_result["Error"] = (val_result[TARGET] - val_result["Predicted"]).round(1)
    val_result.to_csv(os.path.join(MODEL_DIR, "val_predictions_v2.csv"), index=False)

    logger.info("Модель → data/model/v2.cbm")
    logger.info("Метрики → data/model/metrics.csv")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    train_v2()
