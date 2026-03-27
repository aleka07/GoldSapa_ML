"""
Аналитика продаж GoldSapa.

Генерирует графики и сводный отчёт в data/analytics/.
"""
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd

from src.config import DAILY_SALES_PATH, DATA_DIR

logger = logging.getLogger(__name__)

OUT_DIR = os.path.join(DATA_DIR, "analytics")
os.makedirs(OUT_DIR, exist_ok=True)

DAYS_RU = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
MONTHS_RU = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
             "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]


def _setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.figsize"] = (14, 7)
    plt.rcParams["figure.dpi"] = 120


def run_analytics():
    _setup_style()
    df = pd.read_parquet(DAILY_SALES_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # ─── 1. Общие продажи по дням (Сумма) ───────────────────
    daily_total = df.groupby("Date").agg(
        Количество=("Количество", "sum"),
        Сумма=("Сумма", "sum"),
    ).reset_index().sort_values("Date")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    ax1.plot(daily_total["Date"], daily_total["Сумма"], linewidth=0.7, alpha=0.5, color="steelblue")
    # Скользящее среднее 7 дней
    daily_total["Сумма_MA7"] = daily_total["Сумма"].rolling(7).mean()
    ax1.plot(daily_total["Date"], daily_total["Сумма_MA7"], linewidth=2, color="navy", label="Скользящее 7 дн.")
    ax1.set_title("Ежедневная выручка (тг)", fontsize=14, fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax1.legend()

    ax2.plot(daily_total["Date"], daily_total["Количество"], linewidth=0.7, alpha=0.5, color="coral")
    daily_total["Кол_MA7"] = daily_total["Количество"].rolling(7).mean()
    ax2.plot(daily_total["Date"], daily_total["Кол_MA7"], linewidth=2, color="darkred", label="Скользящее 7 дн.")
    ax2.set_title("Ежедневное количество (шт)", fontsize=14, fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_daily_trend.png"))
    plt.close()
    logger.info("✓ 01_daily_trend.png")

    # ─── 2. Топ-15 товаров по выручке ───────────────────────
    top15 = (
        df.groupby("Номенклатура")
        .agg(Сумма=("Сумма", "sum"), Количество=("Количество", "sum"))
        .sort_values("Сумма", ascending=True)
        .tail(15)
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(top15.index, top15["Сумма"], color=sns.color_palette("viridis", 15))
    ax.set_title("Топ-15 товаров по выручке (тг)", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    for bar, val in zip(bars, top15["Сумма"]):
        ax.text(val, bar.get_y() + bar.get_height()/2, f" {val/1e6:.1f}M",
                va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_top15_revenue.png"))
    plt.close()
    logger.info("✓ 02_top15_revenue.png")

    # ─── 3. Топ-15 товаров по количеству ────────────────────
    top15q = (
        df.groupby("Номенклатура")
        .agg(Количество=("Количество", "sum"))
        .sort_values("Количество", ascending=True)
        .tail(15)
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(top15q.index, top15q["Количество"], color=sns.color_palette("magma", 15))
    ax.set_title("Топ-15 товаров по количеству (шт)", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_top15_quantity.png"))
    plt.close()
    logger.info("✓ 03_top15_quantity.png")

    # ─── 4. Продажи по дням недели ──────────────────────────
    df["dow"] = df["Date"].dt.dayofweek
    dow_stats = df.groupby("dow").agg(
        Сумма=("Сумма", "sum"), Количество=("Количество", "sum")
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.bar([DAYS_RU[i] for i in dow_stats["dow"]], dow_stats["Сумма"], color="steelblue")
    ax1.set_title("Выручка по дням недели", fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

    ax2.bar([DAYS_RU[i] for i in dow_stats["dow"]], dow_stats["Количество"], color="coral")
    ax2.set_title("Количество по дням недели", fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_weekday.png"))
    plt.close()
    logger.info("✓ 04_weekday.png")

    # ─── 5. Помесячная динамика ─────────────────────────────
    df["YearMonth"] = df["Date"].dt.to_period("M")
    monthly = df.groupby("YearMonth").agg(
        Сумма=("Сумма", "sum"), Количество=("Количество", "sum")
    ).reset_index()
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(range(len(monthly)), monthly["Сумма"], color="teal", alpha=0.8)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["YearMonth"], rotation=45, ha="right", fontsize=9)
    ax.set_title("Помесячная выручка (тг)", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_monthly.png"))
    plt.close()
    logger.info("✓ 05_monthly.png")

    # ─── 6. Тепловая карта: товары × месяц ──────────────────
    top10_names = (
        df.groupby("Номенклатура")["Сумма"].sum()
        .nlargest(10).index.tolist()
    )
    heat_data = (
        df[df["Номенклатура"].isin(top10_names)]
        .groupby([df["Date"].dt.to_period("M").astype(str), "Номенклатура"])["Количество"]
        .sum()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(heat_data.T, cmap="YlOrRd", linewidths=0.3, ax=ax, fmt=".0f")
    ax.set_title("Топ-10 товаров: количество по месяцам", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_heatmap_top10.png"))
    plt.close()
    logger.info("✓ 06_heatmap_top10.png")

    # ─── 7. Сводный отчёт в CSV ────────────────────────────
    summary = (
        df.groupby("Номенклатура")
        .agg(
            Дней_продаж=("Date", "nunique"),
            Количество_всего=("Количество", "sum"),
            Сумма_всего=("Сумма", "sum"),
            Среднее_в_день_шт=("Количество", "mean"),
            Среднее_в_день_тг=("Сумма", "mean"),
        )
        .sort_values("Сумма_всего", ascending=False)
        .reset_index()
    )
    summary["Средняя_цена"] = (summary["Сумма_всего"] / summary["Количество_всего"]).round(1)
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    logger.info("✓ summary.csv")

    # ─── Итого ──────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  АНАЛИТИКА GOLDSAPA")
    print(f"{'='*50}")
    print(f"  Период:          {df['Date'].min().date()} — {df['Date'].max().date()}")
    print(f"  Всего дней:      {df['Date'].nunique()}")
    print(f"  Товаров:         {df['Номенклатура'].nunique()}")
    print(f"  Общая выручка:   {df['Сумма'].sum()/1e6:.1f}M тг")
    print(f"  Общее кол-во:    {df['Количество'].sum()/1e3:.0f}K шт")
    print(f"  Средняя / день:  {daily_total['Сумма'].mean()/1e3:.0f}K тг")
    print(f"{'='*50}")
    print(f"\n  📊  Графики сохранены в: {OUT_DIR}/")
    print(f"  📋  Сводка по товарам:   {OUT_DIR}/summary.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    run_analytics()
