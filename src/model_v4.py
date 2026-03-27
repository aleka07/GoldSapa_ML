"""
Model V4 — Расширенный Feature Engineering + CatBoost vs LightGBM.

Новые фичи поверх V2:
  - Сезон (зима/весна/лето/осень)
  - Начало/конец месяца
  - Дни до/после праздника, предпраздничный день
  - YoY: продажи тот же день/неделя прошлого года
  - Тренды: momentum, trend_7_14, trend_7_30
  - EWMA (7, 14)
  - Погода+: temp_ma_3, temp_delta, is_rain
  - Циклическое кодирование (sin/cos)

Запуск:
    python -m src.model_v4
"""
import logging
import os
from datetime import timedelta

import holidays
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.config import DAILY_SALES_PATH, DATA_DIR, LATITUDE, LONGITUDE

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(DATA_DIR, "model")
KZ_HOLIDAYS = holidays.Kazakhstan(years=range(2023, 2028))
VAL_CUTOFF = "2026-01-01"
TARGET = "Количество"

CAT_FEATURES = ["Номенклатура_Key", "Склад_Key"]


# ═══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Календарные + расширенные признаки."""
    df = df.copy()
    dt = pd.to_datetime(df["Date"])

    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = dt.apply(lambda x: 1 if x in KZ_HOLIDAYS else 0)

    # ─── Сезон ──────────────────────────────────────────────
    # 0=зима (12,1,2), 1=весна (3,4,5), 2=лето (6,7,8), 3=осень (9,10,11)
    df["season"] = df["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )

    # ─── Начало / конец месяца ──────────────────────────────
    df["is_month_start"] = (dt.dt.day <= 3).astype(int)
    df["is_month_end"] = (dt.dt.day >= 28).astype(int)

    # ─── Дни до / после праздника ───────────────────────────
    holiday_dates = sorted(KZ_HOLIDAYS.keys())

    def days_to_next_holiday(date):
        for h in holiday_dates:
            delta = (h - date.date()).days
            if delta >= 0:
                return min(delta, 30)
        return 30

    def days_since_last_holiday(date):
        for h in reversed(holiday_dates):
            delta = (date.date() - h).days
            if delta >= 0:
                return min(delta, 30)
        return 30

    df["days_to_holiday"] = dt.apply(days_to_next_holiday)
    df["days_after_holiday"] = dt.apply(days_since_last_holiday)
    df["is_pre_holiday"] = (df["days_to_holiday"] == 1).astype(int)

    # ─── Циклическое кодирование ────────────────────────────
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    logger.info("Календарные признаки добавлены (сезон, праздники±, цикл).")
    return df


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Температура + осадки + расширенные погодные фичи."""
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

    # Погодные производные
    weather = weather.sort_values("Date")
    weather["temp_ma_3"] = weather["temperature"].rolling(3, min_periods=1).mean()
    weather["temp_delta"] = weather["temperature"].diff().fillna(0)
    weather["is_rain"] = (weather["precipitation"] > 1.0).astype(int)

    logger.info("Получено %d дней погоды.", len(weather))

    df = df.merge(weather, on="Date", how="left")
    df["temperature"] = df["temperature"].fillna(df["temperature"].median())
    df["precipitation"] = df["precipitation"].fillna(0)
    df["temp_ma_3"] = df["temp_ma_3"].fillna(df["temperature"])
    df["temp_delta"] = df["temp_delta"].fillna(0)
    df["is_rain"] = df["is_rain"].fillna(0).astype(int)

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
    """Лаги, MA, EWMA, волатильность, тренды, momentum."""
    df = df.copy()
    df = df.sort_values(["Номенклатура_Key", "Склад_Key", "Date"])

    group = df.groupby(["Номенклатура_Key", "Склад_Key"])["Количество"]

    # Лаги
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = group.shift(lag)

    # Скользящие средние (shift(1) — без data leakage)
    for w in [7, 14, 30]:
        df[f"ma_{w}"] = group.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )

    # EWMA
    for span in [7, 14]:
        df[f"ewma_{span}"] = group.transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=1).mean()
        )

    # Волатильность
    df["std_7"] = group.transform(
        lambda x: x.shift(1).rolling(7, min_periods=2).std()
    )

    # ─── Тренды ─────────────────────────────────────────────
    df["trend_7_30"] = np.where(df["ma_30"] > 0, df["ma_7"] / df["ma_30"], 1.0)
    df["trend_7_14"] = np.where(df["ma_14"] > 0, df["ma_7"] / df["ma_14"], 1.0)

    # Momentum: ускорение/замедление
    df["momentum"] = np.where(
        df["lag_7"] > 0,
        (df["lag_1"] - df["lag_7"]) / df["lag_7"],
        0,
    )

    # Заполняем NaN
    lag_cols = [c for c in df.columns
                if c.startswith(("lag_", "ma_", "ewma_", "std_", "trend_", "momentum"))]
    df[lag_cols] = df[lag_cols].fillna(0)

    logger.info("Лаги/MA/EWMA/тренды добавлены: %d признаков", len(lag_cols))
    return df


def add_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """Year-over-Year: сравнение с прошлым годом."""
    df = df.copy()
    df = df.sort_values(["Номенклатура_Key", "Склад_Key", "Date"])

    # Продажи ровно 365 дней назад (тот же день прошлого года)
    group = df.groupby(["Номенклатура_Key", "Склад_Key"])["Количество"]
    df["qty_same_day_ly"] = group.shift(365)

    # Среднее за ту же неделю прошлого года (±3 дня ≈ shift 362..368)
    shifts = [group.shift(d) for d in range(362, 369)]
    df["qty_same_week_ly"] = pd.concat(shifts, axis=1).mean(axis=1)

    # YoY ratio: MA_7 текущий / MA_7 год назад
    ma7_ly = group.transform(
        lambda x: x.shift(365).rolling(7, min_periods=1).mean()
    )
    df["yoy_ratio"] = np.where(ma7_ly > 0, df.get("ma_7", 0) / ma7_ly, 1.0)

    # Заполнение (первый год — нет YoY данных)
    df["qty_same_day_ly"] = df["qty_same_day_ly"].fillna(0)
    df["qty_same_week_ly"] = df["qty_same_week_ly"].fillna(0)
    df["yoy_ratio"] = df["yoy_ratio"].fillna(1.0)
    df["yoy_ratio"] = df["yoy_ratio"].clip(0, 10)  # clamp выбросы

    logger.info("YoY признаки добавлены.")
    return df


# ═══════════════════════════════════════════════════════════════
#  МЕТРИКИ
# ═══════════════════════════════════════════════════════════════

def mape(y_true, y_pred):
    mask = y_true > 0
    return (abs(y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100


def eval_metrics(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(root_mean_squared_error(y_true, y_pred), 2),
        "MAPE": round(mape(y_true, y_pred), 1),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


# ═══════════════════════════════════════════════════════════════
#  ОБУЧЕНИЕ
# ═══════════════════════════════════════════════════════════════

NUM_FEATURES = [
    # Календарь
    "day_of_week", "month", "day_of_year", "week_of_year",
    "is_weekend", "is_holiday",
    "season", "is_month_start", "is_month_end",
    "days_to_holiday", "days_after_holiday", "is_pre_holiday",
    # Циклические
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    # Погода
    "temperature", "precipitation", "temp_ma_3", "temp_delta", "is_rain",
    # Цена
    "avg_price",
    # Лаги
    "lag_1", "lag_7", "lag_14", "lag_30",
    # MA
    "ma_7", "ma_14", "ma_30",
    # EWMA
    "ewma_7", "ewma_14",
    # Волатильность / тренды
    "std_7", "trend_7_30", "trend_7_14", "momentum",
    # YoY
    "qty_same_day_ly", "qty_same_week_ly", "yoy_ratio",
]

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

# Не-производственные позиции (не идут в печь)
EXCLUDE_KEYWORDS = ["Бумага", "Пакет", "Заезд", "Пленка", "Скотч", "Коробка"]

# Минимальные пороги
MIN_DAYS_SOLD = 60      # минимум 60 дней продаж
MIN_AVG_PER_DAY = 10    # минимум 10 шт/день в среднем


def filter_products(df: pd.DataFrame) -> pd.DataFrame:
    """Оставить только активные производственные товары."""
    df = df.copy()

    before = df["Номенклатура"].nunique()

    # 1. Убрать не-производственные позиции
    mask = ~df["Номенклатура"].str.contains(
        "|".join(EXCLUDE_KEYWORDS), case=False, na=False
    )
    df = df[mask]

    # 2. Статистика по товарам
    stats = df.groupby("Номенклатура").agg(
        days_sold=("Date", "nunique"),
        avg_qty=("Количество", "mean"),
    )

    # 3. Фильтр: достаточно истории и объёма
    active = stats[
        (stats["days_sold"] >= MIN_DAYS_SOLD) &
        (stats["avg_qty"] >= MIN_AVG_PER_DAY)
    ].index

    df = df[df["Номенклатура"].isin(active)]
    after = df["Номенклатура"].nunique()

    logger.info("Фильтр товаров: %d → %d (убрано %d малоактивных/нерелевантных)",
                before, after, before - after)
    return df


def prepare_data():
    """Полный FE пайплайн → (train, val, all_features)."""
    df = pd.read_parquet(DAILY_SALES_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    logger.info("═══ V4 Feature Engineering ═══")
    logger.info("Исходных строк: %d", len(df))

    # Фильтрация товаров
    df = filter_products(df)
    logger.info("После фильтрации: %d строк, %d товаров",
                len(df), df["Номенклатура"].nunique())

    df = add_calendar(df)
    df = add_weather(df)
    df = add_price(df)
    df = add_lags(df)
    df = add_yoy(df)

    # Split
    train = df[df["Date"] < VAL_CUTOFF].copy()
    val = df[df["Date"] >= VAL_CUTOFF].copy()
    logger.info("Train: %d строк, Val: %d строк", len(train), len(val))
    logger.info("Признаков: %d (%d кат + %d числ)",
                len(ALL_FEATURES), len(CAT_FEATURES), len(NUM_FEATURES))

    return train, val, df


def train_catboost(train, val):
    """Обучить CatBoost V4."""
    cat_idx = [ALL_FEATURES.index(f) for f in CAT_FEATURES]
    train_pool = Pool(train[ALL_FEATURES], train[TARGET], cat_features=cat_idx)
    val_pool = Pool(val[ALL_FEATURES], val[TARGET], cat_features=cat_idx)

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        eval_metric="MAE",
        random_seed=42,
        verbose=200,
    )
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)

    pred = model.predict(val[ALL_FEATURES])
    pred = np.maximum(pred, 0)

    metrics = eval_metrics(val[TARGET].values, pred)
    return model, pred, metrics


def train_lightgbm(train, val):
    """Обучить LightGBM V4."""
    import lightgbm as lgb

    cat_cols = CAT_FEATURES

    # LightGBM требует категории как category dtype
    for col in cat_cols:
        train[col] = train[col].astype("category")
        val[col] = val[col].astype("category")

    train_ds = lgb.Dataset(
        train[ALL_FEATURES], train[TARGET],
        categorical_feature=cat_cols,
    )
    val_ds = lgb.Dataset(
        val[ALL_FEATURES], val[TARGET],
        categorical_feature=cat_cols,
        reference=train_ds,
    )

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [
        lgb.log_evaluation(200),
        lgb.early_stopping(100),
    ]

    model = lgb.train(
        params, train_ds,
        num_boost_round=1500,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=callbacks,
    )

    pred = model.predict(val[ALL_FEATURES])
    pred = np.maximum(pred, 0)

    metrics = eval_metrics(val[TARGET].values, pred)
    return model, pred, metrics


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def train_v4():
    train, val, full_df = prepare_data()

    # ─── CatBoost ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  🚀 Training CatBoost V4 ...")
    print("=" * 55)
    cb_model, cb_pred, cb_metrics = train_catboost(train, val)

    print(f"\n  CatBoost V4:")
    for k, v in cb_metrics.items():
        print(f"    {k}: {v}")

    # ─── LightGBM ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  🚀 Training LightGBM V4 ...")
    print("=" * 55)
    try:
        lgb_model, lgb_pred, lgb_metrics = train_lightgbm(train.copy(), val.copy())
        has_lgb = True
        print(f"\n  LightGBM V4:")
        for k, v in lgb_metrics.items():
            print(f"    {k}: {v}")
    except ImportError:
        print("  ⚠️  lightgbm не установлен, пропускаем")
        lgb_model, lgb_pred, lgb_metrics = None, None, None
        has_lgb = False

    # ─── Ensemble ──────────────────────────────────────────
    if has_lgb:
        ens_pred = 0.5 * cb_pred + 0.5 * lgb_pred
        ens_pred = np.maximum(ens_pred, 0)
        ens_metrics = eval_metrics(val[TARGET].values, ens_pred)

        print(f"\n  Ensemble (50/50):")
        for k, v in ens_metrics.items():
            print(f"    {k}: {v}")
    else:
        ens_metrics = None
        ens_pred = cb_pred

    # ─── Сравнительная таблица ─────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {'Модель':25s} {'MAE':>8s} {'RMSE':>8s} {'MAPE':>8s} {'R²':>8s}")
    print(f"{'─'*65}")

    # Загрузить прошлые метрики
    metrics_path = os.path.join(MODEL_DIR, "metrics.csv")
    if os.path.exists(metrics_path):
        prev = pd.read_csv(metrics_path)
        for _, row in prev.iterrows():
            print(f"  {row['model']:25s} {row['MAE']:8.2f} {row['RMSE']:8.2f} "
                  f"{row['MAPE']:8.1f} {row['R2']:8.4f}")
        print(f"{'─'*65}")

    print(f"  {'v4_catboost':25s} {cb_metrics['MAE']:8.2f} {cb_metrics['RMSE']:8.2f} "
          f"{cb_metrics['MAPE']:8.1f} {cb_metrics['R2']:8.4f}")
    if has_lgb:
        print(f"  {'v4_lightgbm':25s} {lgb_metrics['MAE']:8.2f} {lgb_metrics['RMSE']:8.2f} "
              f"{lgb_metrics['MAPE']:8.1f} {lgb_metrics['R2']:8.4f}")
        print(f"  {'v4_ensemble':25s} {ens_metrics['MAE']:8.2f} {ens_metrics['RMSE']:8.2f} "
              f"{ens_metrics['MAPE']:8.1f} {ens_metrics['R2']:8.4f}")
    print(f"{'='*65}")

    # ─── Определяем лучшую модель ──────────────────────────
    best_name = "v4_catboost"
    best_metrics = cb_metrics
    best_pred = cb_pred

    if has_lgb:
        if lgb_metrics["MAE"] < best_metrics["MAE"]:
            best_name = "v4_lightgbm"
            best_metrics = lgb_metrics
            best_pred = lgb_pred
        if ens_metrics["MAE"] < best_metrics["MAE"]:
            best_name = "v4_ensemble"
            best_metrics = ens_metrics
            best_pred = ens_pred

    print(f"\n  ✅ Лучшая: {best_name} (MAE={best_metrics['MAE']:.2f})")

    # ─── Feature Importance (CatBoost) ─────────────────────
    importance = pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance": cb_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    print(f"\n  Feature Importance (CatBoost, топ-15):")
    for _, row in importance.head(15).iterrows():
        bar = "█" * int(row["importance"] / 2)
        print(f"  {row['feature']:22s} {row['importance']:6.1f}  {bar}")

    # ─── Сохранение ────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    cb_model.save_model(os.path.join(MODEL_DIR, "v4_catboost.cbm"))
    if has_lgb:
        lgb_model.save_model(os.path.join(MODEL_DIR, "v4_lightgbm.txt"))

    importance.to_csv(os.path.join(MODEL_DIR, "importance_v4.csv"), index=False)

    # Val predictions
    val_result = val[["Date", "Номенклатура", "Склад_Key", TARGET]].copy()
    val_result["Predicted"] = np.round(best_pred, 1)
    val_result["Error"] = (val_result[TARGET] - val_result["Predicted"]).round(1)
    val_result.to_csv(os.path.join(MODEL_DIR, "val_predictions_v4.csv"), index=False)

    # Обновляем общий metrics.csv
    new_rows = [{"model": "v4_catboost", **cb_metrics,
                 "features": f"{len(ALL_FEATURES)} features"}]
    if has_lgb:
        new_rows.append({"model": "v4_lightgbm", **lgb_metrics,
                         "features": f"{len(ALL_FEATURES)} features"})
        new_rows.append({"model": "v4_ensemble", **ens_metrics,
                         "features": f"{len(ALL_FEATURES)} features"})
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(metrics_path):
        old = pd.read_csv(metrics_path)
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(metrics_path, index=False)

    logger.info("Модели сохранены в %s", MODEL_DIR)
    return cb_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    train_v4()
