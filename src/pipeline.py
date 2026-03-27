"""
GoldSapa ML — Пайплайн данных (Шаги 1→2→3).

Запуск:
    source .venv/bin/activate
    python -m src.pipeline --date-from 2024-01-01 --date-to 2025-12-31

Для каждого шага можно запускать отдельно:
    python -m src.extract   --date-from 2024-01-01 --date-to 2025-12-31
    python -m src.aggregate
    python -m src.features
"""
import argparse
import logging

from src.extract import get_sales, save_raw, get_nomenclature, save_nomenclature
from src.aggregate import aggregate_daily, save_daily
from src.features import build_features, save_features

logger = logging.getLogger(__name__)


def run(date_from: str, date_to: str) -> None:
    # Шаг 1: Выгрузка из 1С
    logger.info("═══ Шаг 1: Выгрузка из 1С ═══")
    raw = get_sales(date_from, date_to)
    save_raw(raw, date_from, date_to)

    # Справочник номенклатуры
    nom = get_nomenclature()
    save_nomenclature(nom)

    # Шаг 2: Агрегация по дням
    logger.info("═══ Шаг 2: Агрегация по дням ═══")
    daily = aggregate_daily(raw)
    save_daily(daily)

    # Шаг 3: Внешние факторы
    logger.info("═══ Шаг 3: Внешние факторы ═══")
    featured = build_features(daily)
    save_features(featured)

    logger.info("═══ Готово! ═══")
    print(featured.head(10))
    print(f"\nКолонки: {list(featured.columns)}")
    print(f"Строк: {len(featured)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    parser = argparse.ArgumentParser(description="GoldSapa ML — полный пайплайн данных")
    parser.add_argument("--date-from", default="2024-01-01")
    parser.add_argument("--date-to", default="2025-12-31")
    args = parser.parse_args()

    run(args.date_from, args.date_to)
