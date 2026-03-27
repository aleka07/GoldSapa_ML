"""
Шаг 2 — Агрегация продаж по дням + подтягивание справочника.

Считывает raw_sales.parquet + nomenclature.parquet
→ джоинит имена номенклатуры
→ группирует по (Date, Номенклатура, Склад_Key)
→ убирает возвраты
→ сохраняет daily_sales.parquet.
"""
import logging
import os

import pandas as pd

from src.config import DAILY_SALES_PATH, RAW_SALES_PATH, NOMENCLATURE_PATH

logger = logging.getLogger(__name__)


def _load_nomenclature() -> pd.DataFrame:
    """Загрузить справочник номенклатуры."""
    if not os.path.exists(NOMENCLATURE_PATH):
        logger.warning("Справочник %s не найден — имена не подтянутся", NOMENCLATURE_PATH)
        return pd.DataFrame()
    nom = pd.read_parquet(NOMENCLATURE_PATH)
    # Убираем папки (у них Description = 'Готовая продукция' и т.п.)
    # Оставляем все — пусть джоинится что найдётся
    nom = nom.rename(columns={"Ref_Key": "Номенклатура_Key", "Description": "Номенклатура"})
    return nom[["Номенклатура_Key", "Номенклатура", "Parent_Key"]]


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегировать продажи до дневного уровня.

    - Подтягивает имена из справочника номенклатуры.
    - Суммирует Количество и Сумма по (Date, Номенклатура, Склад_Key).
    - Удаляет строки с Количество <= 0 (возвраты).
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Period"]).dt.date

    # Подтягиваем имена
    nom = _load_nomenclature()
    if not nom.empty:
        df = df.merge(nom, on="Номенклатура_Key", how="left")
        unnamed = df["Номенклатура"].isna().sum()
        if unnamed:
            logger.warning("%d записей без имени в справочнике", unnamed)
        df["Номенклатура"] = df["Номенклатура"].fillna("Неизвестно")
    else:
        df["Номенклатура"] = df["Номенклатура_Key"]

    # Агрегация
    daily = (
        df
        .groupby(["Date", "Номенклатура_Key", "Номенклатура", "Склад_Key"], as_index=False)
        .agg(
            Количество=("Количество", "sum"),
            Сумма=("Сумма", "sum"),
        )
    )

    # Убираем возвраты (отрицательные) и нулевые
    before = len(daily)
    daily = daily[daily["Количество"] > 0].copy()
    logger.info(
        "Агрегация: %d → %d строк (убрано %d возвратов/нулей)",
        before, len(daily), before - len(daily),
    )

    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values(["Date", "Номенклатура"]).reset_index(drop=True)
    return daily


def save_daily(daily: pd.DataFrame) -> None:
    """Сохранить агрегированные данные."""
    daily.to_parquet(DAILY_SALES_PATH, index=False)
    logger.info("Сохранено → %s  (%d строк)", DAILY_SALES_PATH, len(daily))


# ─── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    raw = pd.read_parquet(RAW_SALES_PATH)
    logger.info("Загружено %d строк из %s", len(raw), RAW_SALES_PATH)

    daily = aggregate_daily(raw)
    save_daily(daily)

    # Показать пример
    print("\n=== Топ-20 по сумме за всё время ===")
    top = (
        daily.groupby("Номенклатура", as_index=False)
        .agg(Количество=("Количество", "sum"), Сумма=("Сумма", "sum"))
        .sort_values("Сумма", ascending=False)
        .head(20)
    )
    print(top.to_string(index=False))
    print(f"\nВсего строк: {len(daily)}")
    print(f"Период: {daily['Date'].min()} — {daily['Date'].max()}")
    print(f"Уникальных товаров: {daily['Номенклатура'].nunique()}")
