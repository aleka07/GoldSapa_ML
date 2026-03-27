"""
Шаг 3 — Добавление внешних факторов (фичи).

Календарные признаки + праздники КЗ + архивная температура Open-Meteo.
Результат → data/features.parquet.
"""
import logging

import holidays
import pandas as pd
import requests

from src.config import (
    DAILY_SALES_PATH,
    FEATURES_PATH,
    LATITUDE,
    LONGITUDE,
)

logger = logging.getLogger(__name__)

# Праздники Казахстана — кэшируем один раз
KZ_HOLIDAYS = holidays.Kazakhstan(years=range(2023, 2028))


# ── Календарные признаки ────────────────────────────────────
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """day_of_week, month, day_of_year, is_weekend, is_holiday."""
    df = df.copy()
    dt = pd.to_datetime(df["Date"])

    df["day_of_week"] = dt.dt.dayofweek          # 0=Пн … 6=Вс
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = dt.apply(lambda x: 1 if x in KZ_HOLIDAYS else 0)

    logger.info("Календарные признаки добавлены.")
    return df


# ── Погода (Open-Meteo Archive API) ─────────────────────────
def get_weather(date_min: str, date_max: str) -> pd.DataFrame:
    """
    Средняя дневная температура из Open-Meteo Archive API.

    Parameters
    ----------
    date_min, date_max : str  — 'YYYY-MM-DD'

    Returns
    -------
    DataFrame с колонками Date, temperature
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": date_min,
        "end_date": date_max,
        "daily": "temperature_2m_mean",
        "timezone": "Asia/Almaty",
    }

    logger.info("Запрос погоды %s → %s …", date_min, date_max)
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    weather = pd.DataFrame({
        "Date": pd.to_datetime(data["daily"]["time"]),
        "temperature": data["daily"]["temperature_2m_mean"],
    })
    logger.info("Получено %d дней погоды.", len(weather))
    return weather


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить температуру через merge по Date."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    date_min = df["Date"].min().strftime("%Y-%m-%d")
    date_max = df["Date"].max().strftime("%Y-%m-%d")

    weather = get_weather(date_min, date_max)
    df = df.merge(weather, on="Date", how="left")

    missing = df["temperature"].isna().sum()
    if missing:
        logger.warning("Пропущена температура для %d строк — заполняю медианой.", missing)
        df["temperature"] = df["temperature"].fillna(df["temperature"].median())

    return df


# ── Полный пайплайн фичей ──────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Полный пайплайн: календарь → погода."""
    df = add_calendar_features(df)
    df = add_weather(df)
    return df


def save_features(df: pd.DataFrame) -> None:
    df.to_parquet(FEATURES_PATH, index=False)
    logger.info("Сохранено → %s  (%d строк, %d колонок)", FEATURES_PATH, *df.shape)


# ─── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    daily = pd.read_parquet(DAILY_SALES_PATH)
    logger.info("Загружено %d строк из %s", len(daily), DAILY_SALES_PATH)

    featured = build_features(daily)
    save_features(featured)
    print(featured.head(10))
    print(f"\nФинальные колонки: {list(featured.columns)}")
