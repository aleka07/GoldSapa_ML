"""
Model V3 — улучшенная модель на основе research.

Улучшения поверх V2:
  1. Лог-трансформация таргета (log1p/expm1)
  2. EWMA (экспоненциальное скользящее среднее)
  3. Категория товара из Parent_Key
  4. Тренд (MA7/MA30)
  5. Циклическое кодирование дня недели и месяца

Валидация: 2026 год.

Запуск:
    python -m src.model_v3
"""
import logging
import os
import math

import holidays
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.config import DAILY_SALES_PATH, DATA_DIR, LATITUDE, LONGITUDE, NOMENCLATURE_PATH

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(DATA_DIR, "model")
KZ_HOLIDAYS = holidays.Kazakhstan(years=range(2023, 2028))
VAL_CUTOFF = "2026-01-01"
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

    # Циклическое кодирование
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    import requests

    df = df.copy()
    date_min = df["Date"].min().strftime("%Y-%m-%d")
    date_max = df["Date"].max().strftime("%Y-%m-%d")

    logger.info("Запрос погоды %s → %s …", date_min, date_max)
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "start_date": date_min, "end_date": date_max,
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "Asia/Almaty",
        },
        timeout=60,
    )
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
    df = df.copy()
    df["avg_price"] = np.where(df["Количество"] > 0, df["Сумма"] / df["Количество"], 0)
    return df


def add_category(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить категорию из Parent_Key справочника."""
    df = df.copy()
    if not os.path.exists(NOMENCLATURE_PATH):
        df["Категория"] = "unknown"
        return df

    nom = pd.read_parquet(NOMENCLATURE_PATH)

    # Строим маппинг: Ref_Key → Parent_Key
    parent_map = nom.set_index("Ref_Key")["Parent_Key"].to_dict()
    name_map = nom.set_index("Ref_Key")["Description"].to_dict()

    # Для каждого товара — берём имя его Parent_Key
    df["Категория"] = (
        df["Номенклатура_Key"]
        .map(parent_map)
        .map(name_map)
        .fillna("Прочее")
    )

    cats = df["Категория"].nunique()
    logger.info("Категории из справочника: %d уникальных", cats)
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Лаги, MA, EWMA, тренд."""
    df = df.copy()
    df = df.sort_values(["Номенклатура_Key", "Склад_Key", "Date"])

    group = df.groupby(["Номенклатура_Key", "Склад_Key"])["Количество"]

    # Лаги
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = group.shift(lag)

    # Обычные MA (shift(1) чтобы не было data leakage)
    for w in [7, 14, 30]:
        df[f"ma_{w}"] = group.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )

    # EWMA (экспоненциальное скользящее)
    for span in [7, 14]:
        df[f"ewma_{span}"] = group.transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=1).mean()
        )

    # Волатильность
    df["std_7"] = group.transform(
        lambda x: x.shift(1).rolling(7, min_periods=2).std()
    )

    # Тренд: MA7 / MA30 (>1 = растёт, <1 = падает)
    df["trend_7_30"] = np.where(
        df["ma_30"] > 0,
        df["ma_7"] / df["ma_30"],
        1.0,
    )

    # Заполняем NaN
    lag_cols = [c for c in df.columns if c.startswith(("lag_", "ma_", "ewma_", "std_", "trend_"))]
    df[lag_cols] = df[lag_cols].fillna(0)

    logger.info("Лаги/MA/EWMA/тренд добавлены: %d признаков", len(lag_cols))
    return df


# ─── Метрики ────────────────────────────────────────────────

def mape(y_true, y_pred):
    mask = y_true > 0
    return (abs(y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100


# ─── Обучение ───────────────────────────────────────────────

def train_v3():
    df = pd.read_parquet(DAILY_SALES_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # Feature engineering
    logger.info("═══ V3 Feature Engineering ═══")
    df = add_calendar(df)
    df = add_weather(df)
    df = add_price(df)
    df = add_category(df)
    df = add_lags(df)

    # Log-трансформация таргета
    df["log_target"] = np.log1p(df[TARGET])

    # Категориальные
    cat_features = ["Номенклатура_Key", "Склад_Key", "Категория"]

    # Числовые
    num_features = [
        # Календарь
        "day_of_week", "month", "day_of_year", "week_of_year",
        "is_weekend", "is_holiday",
        # Циклические
        "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
        # Погода
        "temperature", "precipitation",
        # Цена
        "avg_price",
        # Лаги
        "lag_1", "lag_7", "lag_14", "lag_30",
        # MA
        "ma_7", "ma_14", "ma_30",
        # EWMA
        "ewma_7", "ewma_14",
        # Волатильность и тренд
        "std_7", "trend_7_30",
    ]

    all_features = cat_features + num_features

    # Split
    train = df[df["Date"] < VAL_CUTOFF].copy()
    val = df[df["Date"] >= VAL_CUTOFF].copy()
    logger.info("Train: %d строк, Val: %d строк", len(train), len(val))

    cat_idx = [all_features.index(f) for f in cat_features]

    # ─── Модель с log-таргетом ──────────────────────────────
    train_pool = Pool(train[all_features], train["log_target"], cat_features=cat_idx)
    val_pool_log = Pool(val[all_features], val["log_target"], cat_features=cat_idx)

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        eval_metric="MAE",
        random_seed=42,
        verbose=200,
    )

    model.fit(train_pool, eval_set=val_pool_log, early_stopping_rounds=100)

    # Предсказания (обратная трансформация)
    val_pred_log = model.predict(val[all_features])
    val_pred = np.expm1(val_pred_log)
    val_pred = np.maximum(val_pred, 0)  # не может быть < 0

    # Метрики
    y_true = val[TARGET].values
    mae = mean_absolute_error(y_true, val_pred)
    rmse = root_mean_squared_error(y_true, val_pred)
    r2 = r2_score(y_true, val_pred)
    mape_val = mape(y_true, val_pred)

    print(f"\n{'='*55}")
    print(f"  MODEL V3 РЕЗУЛЬТАТЫ (Val: 2026)")
    print(f"{'='*55}")
    print(f"  MAE:   {mae:.2f} шт")
    print(f"  RMSE:  {rmse:.2f} шт")
    print(f"  MAPE:  {mape_val:.1f}%")
    print(f"  R²:    {r2:.4f}")
    print(f"  Признаков: {len(all_features)}")
    print(f"{'='*55}")

    # Сравнение с предыдущими
    metrics_path = os.path.join(MODEL_DIR, "metrics.csv")
    if os.path.exists(metrics_path):
        prev = pd.read_csv(metrics_path)
        print(f"\n  ─── Сравнение ───")
        for _, row in prev.iterrows():
            delta = mae - row["MAE"]
            print(f"  vs {row['model']:25s}  MAE {row['MAE']:.2f} → {mae:.2f}  ({delta:+.2f})")

    # Feature importance (топ-15)
    importance = pd.DataFrame({
        "feature": all_features,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    print(f"\n  Feature Importance (топ-15):")
    for _, row in importance.head(15).iterrows():
        bar = "█" * int(row["importance"] / 2)
        print(f"  {row['feature']:20s} {row['importance']:6.1f}  {bar}")

    # Сохранение
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(MODEL_DIR, "v3.cbm"))

    # Метрики
    new_row = pd.DataFrame([{
        "model": "v3_log_ewma_cat",
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape_val, 1),
        "R2": round(r2, 4),
        "features": f"{len(all_features)} features",
    }])
    if os.path.exists(metrics_path):
        old = pd.read_csv(metrics_path)
        new_row = pd.concat([old, new_row], ignore_index=True)
    new_row.to_csv(metrics_path, index=False)

    # Val predictions
    val_result = val[["Date", "Номенклатура", "Склад_Key", TARGET]].copy()
    val_result["Predicted"] = np.round(val_pred, 1)
    val_result["Error"] = (val_result[TARGET] - val_result["Predicted"]).round(1)
    val_result.to_csv(os.path.join(MODEL_DIR, "val_predictions_v3.csv"), index=False)

    logger.info("Модель → data/model/v3.cbm")
    logger.info("Метрики → data/model/metrics.csv")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    train_v3()
