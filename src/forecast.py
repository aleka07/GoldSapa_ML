"""
Прогноз на завтра — загрузка мощностей для технолога.

Загружает свежие продажи из 1С, прогноз погоды,
и выдаёт таблицу: сколько штук каждого товара произвести.

Запуск:
    python -m src.forecast
    python -m src.forecast --date 2026-03-27    # конкретная дата
"""
import logging
import os
from datetime import datetime, timedelta

import holidays
import numpy as np
import pandas as pd
import requests
from catboost import CatBoostRegressor
from requests.auth import HTTPBasicAuth

from src.config import (
    DATA_DIR,
    DAILY_SALES_PATH,
    LATITUDE,
    LONGITUDE,
    NOMENCLATURE_PATH,
    ODATA_BASE_URL,
    ODATA_BATCH_SIZE,
    ODATA_LOGIN,
    ODATA_PASSWORD,
)

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(DATA_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "v2.cbm")
KZ_HOLIDAYS = holidays.Kazakhstan(years=range(2023, 2028))

DAYS_RU = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]

# V2 feature list (должен совпадать с model_v2.py)
CAT_FEATURES = ["Номенклатура_Key", "Склад_Key"]
NUM_FEATURES = [
    "day_of_week", "month", "day_of_year", "week_of_year",
    "is_weekend", "is_holiday",
    "temperature", "precipitation",
    "avg_price",
    "lag_7", "lag_14", "lag_30",
    "ma_7", "ma_14", "ma_30",
    "std_7",
]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


# ─── Загрузка свежих данных ─────────────────────────────────

def fetch_recent_sales(days_back: int = 45) -> pd.DataFrame:
    """Скачать последние N дней продаж из 1С для расчёта лагов."""
    auth = HTTPBasicAuth(ODATA_LOGIN, ODATA_PASSWORD)
    endpoint = "/AccumulationRegister_РеализацияТМЗ_RecordType"

    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00")
    date_to = datetime.now().strftime("%Y-%m-%dT23:59:59")

    odata_filter = (
        f"Period ge datetime'{date_from}' and Period le datetime'{date_to}'"
        f" and Recorder_Type eq 'StandardODATA.Document_РеализацияТоваровУслуг'"
    )

    url = (
        f"{ODATA_BASE_URL}{endpoint}"
        f"?$format=json"
        f"&$select=Period,Номенклатура_Key,Склад_Key,Количество,Сумма"
        f"&$filter={odata_filter}"
        f"&$orderby=Period asc"
    )

    logger.info("Загрузка продаж за последние %d дней …", days_back)
    resp = requests.get(url, auth=auth, timeout=120)
    resp.raise_for_status()
    records = resp.json().get("value", [])
    logger.info("Получено %d записей", len(records))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["Period"] = pd.to_datetime(df["Period"])
    df["Date"] = df["Period"].dt.normalize()

    # Агрегация по дням
    daily = (
        df.groupby(["Date", "Номенклатура_Key", "Склад_Key"], as_index=False)
        .agg(Количество=("Количество", "sum"), Сумма=("Сумма", "sum"))
    )
    daily = daily[daily["Количество"] > 0]
    return daily


# ─── Прогноз погоды ─────────────────────────────────────────

def get_weather_forecast(target_date: datetime) -> dict:
    """Получить прогноз погоды на конкретную дату."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "Asia/Almaty",
        "forecast_days": 3,
    }

    logger.info("Запрос прогноза погоды …")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    date_str = target_date.strftime("%Y-%m-%d")
    temps = dict(zip(data["daily"]["time"], data["daily"]["temperature_2m_mean"]))
    precips = dict(zip(data["daily"]["time"], data["daily"]["precipitation_sum"]))

    temp = temps.get(date_str, 10.0)
    precip = precips.get(date_str, 0.0)
    logger.info("Погода на %s: %.1f°C, осадки %.1fмм", date_str, temp, precip)
    return {"temperature": temp, "precipitation": precip}


# ─── Построение признаков ───────────────────────────────────

def build_forecast_features(
    target_date: datetime,
    recent_sales: pd.DataFrame,
    weather: dict,
) -> pd.DataFrame:
    """Собрать feature-матрицу для прогноза на target_date."""

    # Уникальные комбинации товар × склад (из недавних продаж)
    combos = (
        recent_sales.groupby(["Номенклатура_Key", "Склад_Key"])
        .agg(Количество=("Количество", "sum"))
        .reset_index()
    )
    # Только активные товары (продавались хотя бы что-то)
    combos = combos[combos["Количество"] > 0][["Номенклатура_Key", "Склад_Key"]]

    # Создаём строки для каждого товара × склад
    forecast_df = combos.copy()
    forecast_df["Date"] = pd.Timestamp(target_date)

    # Календарные
    forecast_df["day_of_week"] = target_date.weekday()
    forecast_df["month"] = target_date.month
    forecast_df["day_of_year"] = target_date.timetuple().tm_yday
    forecast_df["week_of_year"] = target_date.isocalendar()[1]
    forecast_df["is_weekend"] = int(target_date.weekday() in [5, 6])
    forecast_df["is_holiday"] = int(target_date in KZ_HOLIDAYS)

    # Погода
    forecast_df["temperature"] = weather["temperature"]
    forecast_df["precipitation"] = weather["precipitation"]

    # Средняя цена (из недавних данных)
    price_map = recent_sales.groupby(["Номенклатура_Key", "Склад_Key"]).apply(
        lambda g: g["Сумма"].sum() / max(g["Количество"].sum(), 1),
        include_groups=False,
    ).reset_index(name="avg_price")
    forecast_df = forecast_df.merge(price_map, on=["Номенклатура_Key", "Склад_Key"], how="left")
    forecast_df["avg_price"] = forecast_df["avg_price"].fillna(0)

    # Лаги и MA — считаем из recent_sales
    recent = recent_sales.sort_values(["Номенклатура_Key", "Склад_Key", "Date"])

    for nk, sk in zip(forecast_df["Номенклатура_Key"], forecast_df["Склад_Key"]):
        mask = (recent["Номенклатура_Key"] == nk) & (recent["Склад_Key"] == sk)
        product_history = recent[mask].sort_values("Date")

        idx = forecast_df[
            (forecast_df["Номенклатура_Key"] == nk) & (forecast_df["Склад_Key"] == sk)
        ].index[0]

        qty = product_history["Количество"].values

        # Лаги (относительно target_date)
        forecast_df.loc[idx, "lag_7"] = qty[-7] if len(qty) >= 7 else 0
        forecast_df.loc[idx, "lag_14"] = qty[-14] if len(qty) >= 14 else 0
        forecast_df.loc[idx, "lag_30"] = qty[-30] if len(qty) >= 30 else 0

        # MA
        forecast_df.loc[idx, "ma_7"] = qty[-7:].mean() if len(qty) >= 7 else qty.mean() if len(qty) > 0 else 0
        forecast_df.loc[idx, "ma_14"] = qty[-14:].mean() if len(qty) >= 14 else qty.mean() if len(qty) > 0 else 0
        forecast_df.loc[idx, "ma_30"] = qty[-30:].mean() if len(qty) >= 30 else qty.mean() if len(qty) > 0 else 0

        # Std
        forecast_df.loc[idx, "std_7"] = qty[-7:].std() if len(qty) >= 7 else 0

    # Заполняем NaN
    for col in NUM_FEATURES:
        forecast_df[col] = forecast_df[col].fillna(0)

    return forecast_df


# ─── Прогноз ────────────────────────────────────────────────

def run_forecast(target_date: datetime = None):
    """Запустить прогноз на target_date (по умолчанию — завтра)."""
    if target_date is None:
        target_date = (datetime.now() + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    date_str = target_date.strftime("%Y-%m-%d")
    dow_ru = DAYS_RU[target_date.weekday()]
    is_hol = "Да" if target_date in KZ_HOLIDAYS else "Нет"

    print(f"\n{'='*60}")
    print(f"  🍞 ПРОГНОЗ ЗАГРУЗКИ МОЩНОСТЕЙ — {date_str} ({dow_ru})")
    print(f"{'='*60}")

    # 1. Загрузить модель
    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ Модель не найдена: {MODEL_PATH}")
        print(f"     Сначала обучите: python -m src.model_v2")
        return

    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    logger.info("Модель загружена: %s", MODEL_PATH)

    # 2. Свежие продажи из 1С
    recent = fetch_recent_sales(days_back=45)
    if recent.empty:
        print("  ❌ Нет данных о продажах")
        return

    # 3. Погода
    weather = get_weather_forecast(target_date)

    # 4. Построить признаки
    forecast_df = build_forecast_features(target_date, recent, weather)
    logger.info("Подготовлено %d позиций для прогноза", len(forecast_df))

    # 5. Предсказание
    predictions = model.predict(forecast_df[ALL_FEATURES])
    predictions = np.maximum(predictions, 0).round().astype(int)
    forecast_df["Прогноз_шт"] = predictions

    # 6. Подтянуть имена товаров
    if os.path.exists(NOMENCLATURE_PATH):
        nom = pd.read_parquet(NOMENCLATURE_PATH)
        nom = nom.rename(columns={"Ref_Key": "Номенклатура_Key", "Description": "Номенклатура"})
        forecast_df = forecast_df.merge(
            nom[["Номенклатура_Key", "Номенклатура"]], on="Номенклатура_Key", how="left"
        )
        forecast_df["Номенклатура"] = forecast_df["Номенклатура"].fillna("Неизвестно")
    else:
        forecast_df["Номенклатура"] = forecast_df["Номенклатура_Key"]

    # 7. Агрегация по товару (суммируем по складам)
    result = (
        forecast_df.groupby("Номенклатура", as_index=False)
        .agg(Прогноз_шт=("Прогноз_шт", "sum"))
        .sort_values("Прогноз_шт", ascending=False)
    )
    result = result[result["Прогноз_шт"] > 0]

    # 8. Вывод
    print(f"  📅 Дата:        {date_str} ({dow_ru})")
    print(f"  🌡️  Температура: {weather['temperature']:.1f}°C")
    print(f"  🌧️  Осадки:      {weather['precipitation']:.1f}мм")
    print(f"  🎉 Праздник:    {is_hol}")
    print(f"  📦 Позиций:     {len(result)}")
    print(f"  📊 Всего штук:  {result['Прогноз_шт'].sum()}")
    print(f"\n{'─'*45}")
    print(f"  {'Номенклатура':35s} {'Шт':>6s}")
    print(f"{'─'*45}")
    for _, row in result.iterrows():
        name = row["Номенклатура"][:35]
        print(f"  {name:35s} {row['Прогноз_шт']:6d}")
    print(f"{'─'*45}")
    print(f"  {'ИТОГО':35s} {result['Прогноз_шт'].sum():6d}")
    print()

    # Сохранение в CSV
    out_path = os.path.join(DATA_DIR, f"forecast_{date_str}.csv")
    result.to_csv(out_path, index=False)
    logger.info("Прогноз сохранён → %s", out_path)

    return result


# ─── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    parser = argparse.ArgumentParser(description="Прогноз загрузки мощностей на завтра")
    parser.add_argument("--date", default=None, help="Дата прогноза (YYYY-MM-DD), по умолчанию завтра")
    args = parser.parse_args()

    if args.date:
        target = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        target = None

    run_forecast(target)
