"""
Шаг 1 — Выгрузка данных из 1С OData.

Продажи: грузим по неделям через $filter по Period.
Справочник: Catalog_Номенклатура целиком.

Результат:
    data/raw_sales.parquet
    data/nomenclature.parquet
"""
import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

from src.config import (
    DATA_DIR,
    ODATA_BASE_URL,
    ODATA_LOGIN,
    ODATA_PASSWORD,
    ODATA_BATCH_SIZE,
    RAW_SALES_PATH,
    NOMENCLATURE_PATH,
)

logger = logging.getLogger(__name__)

WEEKS_DIR = os.path.join(DATA_DIR, "weeks")


# ─── Утилиты ────────────────────────────────────────────────

def _build_url(endpoint: str, params: dict) -> str:
    """Собрать OData URL вручную (чтобы $ не кодировался в %24)."""
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{ODATA_BASE_URL}{endpoint}?{qs}"


def _odata_get(endpoint: str, params: dict, auth) -> list:
    """GET-запрос к OData с 3 ретраями."""
    url = _build_url(endpoint, params)
    for attempt in range(3):
        try:
            resp = requests.get(url, auth=auth, timeout=120)
            resp.raise_for_status()
            return resp.json().get("value", [])
        except (requests.RequestException, requests.HTTPError) as e:
            wait = 5 * (attempt + 1)
            logger.warning("Ошибка: %s → повтор через %dс", e, wait)
            time.sleep(wait)
    logger.error("3 неудачных попытки. Пропускаем.")
    return []


def _week_ranges(date_from: str, date_to: str):
    """Генератор (start, end) по 7 дней."""
    start = datetime.strptime(date_from, "%Y-%m-%d")
    end = datetime.strptime(date_to, "%Y-%m-%d")
    while start <= end:
        week_end = min(start + timedelta(days=6), end)
        yield start, week_end
        start = week_end + timedelta(days=1)


# ─── Продажи ────────────────────────────────────────────────

def get_sales(date_from: str, date_to: str) -> pd.DataFrame:
    """Выгрузить продажи по неделям через $filter."""
    os.makedirs(WEEKS_DIR, exist_ok=True)
    endpoint = "/AccumulationRegister_РеализацияТМЗ_RecordType"
    auth = HTTPBasicAuth(ODATA_LOGIN, ODATA_PASSWORD)

    for w_start, w_end in _week_ranges(date_from, date_to):
        chunk_name = f"week_{w_start.strftime('%Y-%m-%d')}.parquet"
        chunk_path = os.path.join(WEEKS_DIR, chunk_name)

        # Resume: если чанк есть — пропускаем
        if os.path.exists(chunk_path):
            logger.info("✓ %s — уже скачан, пропускаем", chunk_name)
            continue

        s = w_start.strftime("%Y-%m-%dT00:00:00")
        e = w_end.strftime("%Y-%m-%dT23:59:59")

        odata_filter = (
            f"Period ge datetime'{s}' and Period le datetime'{e}'"
            f" and Recorder_Type eq 'StandardODATA.Document_РеализацияТоваровУслуг'"
        )

        logger.info("Загрузка %s … %s", w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d"))

        # Пагинация внутри недели
        all_records = []
        skip = 0

        while True:
            params = {
                "$format": "json",
                "$select": "Period,Номенклатура_Key,Склад_Key,Количество,Сумма",
                "$filter": odata_filter,
                "$orderby": "Period asc",
                "$top": ODATA_BATCH_SIZE,
                "$skip": skip,
            }

            batch = _odata_get(endpoint, params, auth)
            if not batch:
                break

            all_records.extend(batch)
            logger.info("  +%d записей (всего %d)", len(batch), len(all_records))

            if len(batch) < ODATA_BATCH_SIZE:
                break

            skip += ODATA_BATCH_SIZE
            time.sleep(0.2)

        if all_records:
            pd.DataFrame(all_records).to_parquet(chunk_path, index=False)
            logger.info("  → %s (%d записей)", chunk_name, len(all_records))
        else:
            # Сохраняем пустой чанк чтобы не грузить повторно
            pd.DataFrame().to_parquet(chunk_path, index=False)
            logger.info("  → %s (пусто)", chunk_name)

        time.sleep(0.3)

    # Собираем всё
    return merge_weeks()


def merge_weeks() -> pd.DataFrame:
    """Объединить все недельные чанки."""
    if not os.path.isdir(WEEKS_DIR):
        return pd.DataFrame()
    files = sorted(f for f in os.listdir(WEEKS_DIR) if f.endswith(".parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(os.path.join(WEEKS_DIR, f)) for f in files]
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["Period"] = pd.to_datetime(df["Period"])
    logger.info("Собрано %d недель → %d строк", len(files), len(df))
    return df


# ─── Справочник номенклатуры ────────────────────────────────

def get_nomenclature() -> pd.DataFrame:
    """Скачать справочник Catalog_Номенклатура."""
    endpoint = "/Catalog_Номенклатура"
    auth = HTTPBasicAuth(ODATA_LOGIN, ODATA_PASSWORD)

    all_records = []
    skip = 0

    logger.info("Загрузка справочника номенклатуры …")

    while True:
        params = {
            "$format": "json",
            "$select": "Ref_Key,Description,Parent_Key",
            "$top": ODATA_BATCH_SIZE,
            "$skip": skip,
        }

        batch = _odata_get(endpoint, params, auth)
        if not batch:
            break

        all_records.extend(batch)
        logger.info("  +%d (всего %d)", len(batch), len(all_records))

        if len(batch) < ODATA_BATCH_SIZE:
            break

        skip += ODATA_BATCH_SIZE
        time.sleep(0.2)

    df = pd.DataFrame(all_records)
    logger.info("Справочник: %d позиций", len(df))
    return df


def save_nomenclature(df: pd.DataFrame) -> None:
    """Сохранить справочник в parquet."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(NOMENCLATURE_PATH, index=False)
    logger.info("Сохранено → %s  (%d строк)", NOMENCLATURE_PATH, len(df))


def save_raw(df: pd.DataFrame, date_from: str = None, date_to: str = None) -> None:
    """Сохранить финальный parquet продаж."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(RAW_SALES_PATH, index=False)
    logger.info("Сохранено → %s  (%d строк)", RAW_SALES_PATH, len(df))


# ─── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    parser = argparse.ArgumentParser(description="Выгрузка продаж из 1С OData")
    parser.add_argument("--date-from", default="2024-01-01")
    parser.add_argument("--date-to", default="2025-12-31")
    parser.add_argument("--merge-only", action="store_true", help="Только собрать недельные чанки")
    parser.add_argument("--nomenclature", action="store_true", help="Скачать справочник номенклатуры")
    args = parser.parse_args()

    if args.nomenclature:
        nom = get_nomenclature()
        save_nomenclature(nom)
    elif args.merge_only:
        df = merge_weeks()
        save_raw(df, args.date_from, args.date_to)
    else:
        df = get_sales(args.date_from, args.date_to)
        save_raw(df, args.date_from, args.date_to)
