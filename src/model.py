"""
Baseline модель CatBoost — прогноз дневных продаж.

Train: 2024–2025, Val: 2026.
Признаки: календарные + Номенклатура + Склад (как категориальные).

Запуск:
    python -m src.model
"""
import logging
import os

import holidays
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.config import DAILY_SALES_PATH, DATA_DIR

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(DATA_DIR, "model")
KZ_HOLIDAYS = holidays.Kazakhstan(years=range(2023, 2028))

VAL_CUTOFF = "2026-01-01"

# ─── Признаки ───────────────────────────────────────────────

CAT_FEATURES = ["Номенклатура_Key", "Склад_Key"]

CALENDAR_FEATURES = [
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "is_holiday",
]

ALL_FEATURES = CAT_FEATURES + CALENDAR_FEATURES
TARGET = "Количество"


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить календарные признаки."""
    df = df.copy()
    dt = pd.to_datetime(df["Date"])
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = dt.apply(lambda x: 1 if x in KZ_HOLIDAYS else 0)
    return df


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (игнорируем нули)."""
    mask = y_true > 0
    return (abs(y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100


def train_baseline():
    """Обучить baseline и оценить на валидации."""
    # Загрузка
    df = pd.read_parquet(DAILY_SALES_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = add_calendar(df)

    # Split
    train = df[df["Date"] < VAL_CUTOFF].copy()
    val = df[df["Date"] >= VAL_CUTOFF].copy()
    logger.info("Train: %d строк (%s — %s)", len(train),
                train["Date"].min().date(), train["Date"].max().date())
    logger.info("Val:   %d строк (%s — %s)", len(val),
                val["Date"].min().date(), val["Date"].max().date())

    # CatBoost pools
    cat_idx = [ALL_FEATURES.index(f) for f in CAT_FEATURES]

    train_pool = Pool(
        train[ALL_FEATURES], train[TARGET],
        cat_features=cat_idx,
    )
    val_pool = Pool(
        val[ALL_FEATURES], val[TARGET],
        cat_features=cat_idx,
    )

    # Модель
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        eval_metric="MAE",
        random_seed=42,
        verbose=100,
    )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

    # Предсказания
    val_pred = model.predict(val[ALL_FEATURES])

    # Метрики
    mae = mean_absolute_error(val[TARGET], val_pred)
    rmse = root_mean_squared_error(val[TARGET], val_pred)
    r2 = r2_score(val[TARGET], val_pred)
    mape_val = mape(val[TARGET].values, val_pred)

    print(f"\n{'='*50}")
    print(f"  BASELINE РЕЗУЛЬТАТЫ (Val: 2026)")
    print(f"{'='*50}")
    print(f"  MAE:   {mae:.2f} шт")
    print(f"  RMSE:  {rmse:.2f} шт")
    print(f"  MAPE:  {mape_val:.1f}%")
    print(f"  R²:    {r2:.4f}")
    print(f"{'='*50}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    print(f"\n  Feature Importance:")
    for _, row in importance.iterrows():
        bar = "█" * int(row["importance"] / 2)
        print(f"  {row['feature']:20s} {row['importance']:6.1f}  {bar}")

    # Сохранение модели
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "baseline.cbm")
    model.save_model(model_path)
    logger.info("Модель → %s", model_path)

    # Сохраняем метрики
    metrics = pd.DataFrame([{
        "model": "baseline_calendar",
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape_val, 1),
        "R2": round(r2, 4),
        "features": ", ".join(ALL_FEATURES),
    }])
    metrics_path = os.path.join(MODEL_DIR, "metrics.csv")
    if os.path.exists(metrics_path):
        old = pd.read_csv(metrics_path)
        metrics = pd.concat([old, metrics], ignore_index=True)
    metrics.to_csv(metrics_path, index=False)
    logger.info("Метрики → %s", metrics_path)

    # Сохраняем val predictions для анализа
    val_result = val[["Date", "Номенклатура", "Склад_Key", TARGET]].copy()
    val_result["Predicted"] = val_pred.round(1)
    val_result["Error"] = (val_result[TARGET] - val_result["Predicted"]).round(1)
    val_result.to_csv(os.path.join(MODEL_DIR, "val_predictions.csv"), index=False)

    return model, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    train_baseline()
