"""
Сравнение прогноза vs факт на данных 2026 (валидация).

Показывает точность модели по каждому товару и по дням.

Запуск:
    python -m src.validate
"""
import logging
import os

import numpy as np
import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(DATA_DIR, "model")


def run_validation():
    # Загрузка predictions
    pred_path = os.path.join(MODEL_DIR, "val_predictions_v2.csv")
    if not os.path.exists(pred_path):
        print("❌ Нет файла val_predictions_v2.csv. Сначала запустите: python -m src.model_v2")
        return

    df = pd.read_csv(pred_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Predicted"] = df["Predicted"].clip(lower=0)

    # ─── Общая статистика ───────────────────────────────────
    mae = abs(df["Количество"] - df["Predicted"]).mean()
    total_fact = df["Количество"].sum()
    total_pred = df["Predicted"].sum()
    diff_pct = (total_pred - total_fact) / total_fact * 100

    print(f"\n{'='*65}")
    print(f"  📊 ВАЛИДАЦИЯ МОДЕЛИ V2 — Прогноз vs Факт (2026)")
    print(f"{'='*65}")
    print(f"  Период:       {df['Date'].min().date()} — {df['Date'].max().date()}")
    print(f"  Строк:        {len(df)}")
    print(f"  MAE:          {mae:.1f} шт (средняя ошибка на позицию)")
    print(f"  Факт всего:   {total_fact:,.0f} шт")
    print(f"  Прогноз всего:{total_pred:,.0f} шт")
    print(f"  Разница:      {diff_pct:+.1f}%")
    print(f"{'='*65}")

    # ─── По товарам: топ-20 ─────────────────────────────────
    by_product = (
        df.groupby("Номенклатура")
        .agg(
            Факт=("Количество", "sum"),
            Прогноз=("Predicted", "sum"),
            Дней=("Date", "nunique"),
            MAE=("Error", lambda x: abs(x).mean()),
        )
        .reset_index()
    )
    by_product["Разница_%"] = (
        (by_product["Прогноз"] - by_product["Факт"]) / by_product["Факт"].clip(lower=1) * 100
    ).round(1)
    by_product["Прогноз"] = by_product["Прогноз"].round(0).astype(int)
    by_product["Факт"] = by_product["Факт"].astype(int)
    by_product["MAE"] = by_product["MAE"].round(1)
    by_product = by_product.sort_values("Факт", ascending=False)

    print(f"\n  {'Номенклатура':35s} {'Факт':>7s} {'Прогноз':>8s} {'Δ%':>6s} {'MAE':>6s}")
    print(f"  {'─'*65}")
    for _, row in by_product.head(25).iterrows():
        name = row["Номенклатура"][:35]
        delta = row["Разница_%"]
        marker = "✅" if abs(delta) < 15 else ("⚠️" if abs(delta) < 30 else "❌")
        print(f"  {name:35s} {row['Факт']:7d} {row['Прогноз']:8d} {delta:+6.1f} {row['MAE']:6.1f} {marker}")

    # ─── По дням ────────────────────────────────────────────
    by_day = (
        df.groupby("Date")
        .agg(Факт=("Количество", "sum"), Прогноз=("Predicted", "sum"))
        .reset_index()
    )
    by_day["Разница_%"] = ((by_day["Прогноз"] - by_day["Факт"]) / by_day["Факт"] * 100).round(1)
    by_day["Факт"] = by_day["Факт"].astype(int)
    by_day["Прогноз"] = by_day["Прогноз"].round(0).astype(int)

    daily_mae = abs(by_day["Факт"] - by_day["Прогноз"]).mean()
    daily_mape = abs(by_day["Разница_%"]).mean()

    print(f"\n  ─── По дням (агрегат) ───")
    print(f"  MAE дневной итого: {daily_mae:.0f} шт")
    print(f"  MAPE дневной:      {daily_mape:.1f}%")
    print(f"\n  {'Дата':12s} {'Факт':>7s} {'Прогноз':>8s} {'Δ%':>7s}")
    print(f"  {'─'*40}")
    for _, row in by_day.head(20).iterrows():
        marker = "✅" if abs(row["Разница_%"]) < 10 else ("⚠️" if abs(row["Разница_%"]) < 20 else "❌")
        print(f"  {str(row['Date'].date()):12s} {row['Факт']:7d} {row['Прогноз']:8d} {row['Разница_%']:+7.1f}% {marker}")

    if len(by_day) > 20:
        print(f"  ... ещё {len(by_day) - 20} дней")

    # ─── Сохранение ─────────────────────────────────────────
    by_product.to_csv(os.path.join(MODEL_DIR, "validation_by_product.csv"), index=False)
    by_day.to_csv(os.path.join(MODEL_DIR, "validation_by_day.csv"), index=False)

    print(f"\n  📁 Детали сохранены:")
    print(f"     data/model/validation_by_product.csv")
    print(f"     data/model/validation_by_day.csv")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    run_validation()
