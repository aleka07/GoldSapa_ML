"""
Конфигурация проекта GoldSapa ML.
"""
import os

# ─── 1С OData ───────────────────────────────────────────────
ODATA_BASE_URL = os.getenv(
    "ODATA_BASE_URL",
    "https://goldsapa.24c.kz/goldsapa/odata/standard.odata"
)
ODATA_LOGIN = os.getenv("ODATA_LOGIN", "user1")
ODATA_PASSWORD = os.getenv("ODATA_PASSWORD", "0o9i8u7y")

# ─── Open-Meteo (координаты Алматы) ────────────────────────
LATITUDE = 43.25
LONGITUDE = 76.95

# ─── Пути к данным ──────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_SALES_PATH = os.path.join(DATA_DIR, "raw_sales.parquet")
NOMENCLATURE_PATH = os.path.join(DATA_DIR, "nomenclature.parquet")
DAILY_SALES_PATH = os.path.join(DATA_DIR, "daily_sales.parquet")
FEATURES_PATH = os.path.join(DATA_DIR, "features.parquet")

# ─── Параметры выгрузки ────────────────────────────────────
ODATA_BATCH_SIZE = 5000
