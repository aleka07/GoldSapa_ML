import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", app_title="GoldSapa ML — Аналитика продаж")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🍞 GoldSapa ML — Аналитика продаж

    Интерактивный дашборд для анализа продаж хлебозавода Gold Sapa.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import marimo as mo

    return go, mo, pd, px


@app.cell
def _(pd):
    # Загрузка данных
    daily = pd.read_parquet("data/daily_sales.parquet")
    daily["Date"] = pd.to_datetime(daily["Date"])
    nom = pd.read_parquet("data/nomenclature.parquet")
    return (daily,)


@app.cell(hide_code=True)
def _(daily, mo):
    # Сводка по данным
    period_start = daily["Date"].min().strftime("%d.%m.%Y")
    period_end = daily["Date"].max().strftime("%d.%m.%Y")
    total_revenue = daily["Сумма"].sum()
    total_qty = daily["Количество"].sum()
    n_products = daily["Номенклатура"].nunique()

    mo.md(f"""
    ## 📊 Обзор данных

    | Метрика | Значение |
    |---------|----------|
    | Период | {period_start} — {period_end} |
    | Товаров | {n_products} |
    | Общая выручка | {total_revenue / 1e6:.1f} млн тг |
    | Общее количество | {total_qty / 1e3:.0f} тыс. шт |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Слайдер для выбора кол-ва топ-товаров
    top_n = mo.ui.slider(5, 30, value=15, step=1, label="Топ-N товаров")
    mo.md(f"## 🏆 Топ товаров по выручке\n\n{top_n}")
    return (top_n,)


@app.cell
def _(daily, px, top_n):
    # Топ-N товаров по выручке
    top_products = (
        daily.groupby("Номенклатура", as_index=False)
        .agg({"Сумма": "sum", "Количество": "sum"})
        .nlargest(top_n.value, "Сумма")
        .sort_values("Сумма")
    )

    fig_top = px.bar(
        top_products,
        x="Сумма",
        y="Номенклатура",
        orientation="h",
        title=f"Топ-{top_n.value} товаров по выручке",
        labels={"Сумма": "Выручка (тг)", "Номенклатура": ""},
        color="Сумма",
        color_continuous_scale="Viridis",
    )
    fig_top.update_layout(height=max(400, top_n.value * 30), showlegend=False)
    fig_top
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 📈 Тренд продаж по дням
    """)
    return


@app.cell
def _(daily, go):
    # Тренд выручки с MA-7
    trend = (
        daily.groupby("Date", as_index=False)
        .agg({"Сумма": "sum", "Количество": "sum"})
        .sort_values("Date")
    )
    trend["MA_7"] = trend["Сумма"].rolling(7).mean()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend["Date"], y=trend["Сумма"],
        mode="lines", name="Выручка", opacity=0.3,
        line=dict(color="steelblue"),
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend["Date"], y=trend["MA_7"],
        mode="lines", name="MA-7",
        line=dict(color="navy", width=2),
    ))
    fig_trend.update_layout(
        title="Ежедневная выручка (тг)",
        yaxis_title="тг",
        height=450,
    )
    fig_trend
    return


@app.cell(hide_code=True)
def _(daily, mo):
    # Dropdown для выбора товара
    products = sorted(daily["Номенклатура"].unique().tolist())
    product_select = mo.ui.dropdown(
        options=products,
        value=products[0],
        label="Выберите товар",
    )
    mo.md(f"## 🔍 Детализация по товару\n\n{product_select}")
    return (product_select,)


@app.cell
def _(daily, go, product_select):
    # График продаж выбранного товара
    prod_data = (
        daily[daily["Номенклатура"] == product_select.value]
        .groupby("Date", as_index=False)
        .agg({"Количество": "sum"})
        .sort_values("Date")
    )
    prod_data["MA_7"] = prod_data["Количество"].rolling(7).mean()

    fig_prod = go.Figure()
    fig_prod.add_trace(go.Scatter(
        x=prod_data["Date"], y=prod_data["Количество"],
        mode="lines", name="Продажи (шт)", opacity=0.4,
        line=dict(color="coral"),
    ))
    fig_prod.add_trace(go.Scatter(
        x=prod_data["Date"], y=prod_data["MA_7"],
        mode="lines", name="MA-7",
        line=dict(color="darkred", width=2),
    ))
    fig_prod.update_layout(
        title=f"Продажи: {product_select.value}",
        yaxis_title="шт",
        height=400,
    )
    fig_prod
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 📅 Продажи по дням недели
    """)
    return


@app.cell
def _(daily, px):
    DAYS_RU = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    dow = daily.copy()
    dow["dow"] = dow["Date"].dt.dayofweek
    dow_stats = dow.groupby("dow", as_index=False).agg({"Количество": "sum"})
    dow_stats["День"] = dow_stats["dow"].map(lambda i: DAYS_RU[i])

    fig_dow = px.bar(
        dow_stats,
        x="День",
        y="Количество",
        title="Количество продаж по дням недели",
        color="Количество",
        color_continuous_scale="Sunset",
    )
    fig_dow.update_layout(height=400, showlegend=False)
    fig_dow
    return


# ═══════════════════════════════════════════════════════════════
#  MODEL V4 — Feature Engineering + Training
# ═══════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 🤖 Model V4 — Feature Engineering + Обучение

    Расширенный набор признаков:
    - **Календарь+**: сезон, начало/конец месяца, дни до/после праздника
    - **YoY**: сравнение с тем же днём прошлого года
    - **Тренды**: momentum, trend_7_14, trend_7_30, EWMA
    - **Погода+**: temp_ma_3, temp_delta, is_rain
    - **Модели**: CatBoost, LightGBM, Ensemble
    """)
    return


@app.cell
def _():
    import numpy as np
    import holidays
    from catboost import CatBoostRegressor, Pool
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    return CatBoostRegressor, Pool, lgb, np, holidays, mean_absolute_error, root_mean_squared_error, r2_score


@app.cell
def _(daily, pd, np, holidays):
    # ─── Feature Engineering V4 ─────────────────────────────
    KZ_HOLIDAYS = holidays.Kazakhstan(years=range(2023, 2028))
    holiday_dates = sorted(KZ_HOLIDAYS.keys())

    fe = daily.copy()
    dt = pd.to_datetime(fe["Date"])

    # Календарь
    fe["day_of_week"] = dt.dt.dayofweek
    fe["month"] = dt.dt.month
    fe["day_of_year"] = dt.dt.dayofyear
    fe["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    fe["is_weekend"] = fe["day_of_week"].isin([5, 6]).astype(int)
    fe["is_holiday"] = dt.apply(lambda x: 1 if x in KZ_HOLIDAYS else 0)

    # Сезон
    fe["season"] = fe["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )

    # Начало / конец месяца
    fe["is_month_start"] = (dt.dt.day <= 3).astype(int)
    fe["is_month_end"] = (dt.dt.day >= 28).astype(int)

    # Дни до / после праздника
    def _days_to_hol(date):
        for h in holiday_dates:
            d = (h - date.date()).days
            if d >= 0:
                return min(d, 30)
        return 30

    def _days_after_hol(date):
        for h in reversed(holiday_dates):
            d = (date.date() - h).days
            if d >= 0:
                return min(d, 30)
        return 30

    fe["days_to_holiday"] = dt.apply(_days_to_hol)
    fe["days_after_holiday"] = dt.apply(_days_after_hol)
    fe["is_pre_holiday"] = (fe["days_to_holiday"] == 1).astype(int)

    # Циклическое кодирование
    fe["dow_sin"] = np.sin(2 * np.pi * fe["day_of_week"] / 7)
    fe["dow_cos"] = np.cos(2 * np.pi * fe["day_of_week"] / 7)
    fe["month_sin"] = np.sin(2 * np.pi * fe["month"] / 12)
    fe["month_cos"] = np.cos(2 * np.pi * fe["month"] / 12)

    # Цена
    fe["avg_price"] = np.where(fe["Количество"] > 0, fe["Сумма"] / fe["Количество"], 0)

    # ─── Погода (Open-Meteo Archive) ────────────────────────
    import requests
    date_min = fe["Date"].min().strftime("%Y-%m-%d")
    date_max = fe["Date"].max().strftime("%Y-%m-%d")
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": 43.25, "longitude": 76.95,
            "start_date": date_min, "end_date": date_max,
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "Asia/Almaty",
        }, timeout=60,
    )
    resp.raise_for_status()
    wdata = resp.json()
    weather = pd.DataFrame({
        "Date": pd.to_datetime(wdata["daily"]["time"]),
        "temperature": wdata["daily"]["temperature_2m_mean"],
        "precipitation": wdata["daily"]["precipitation_sum"],
    }).sort_values("Date")
    weather["temp_ma_3"] = weather["temperature"].rolling(3, min_periods=1).mean()
    weather["temp_delta"] = weather["temperature"].diff().fillna(0)
    weather["is_rain"] = (weather["precipitation"] > 1.0).astype(int)

    fe = fe.merge(weather, on="Date", how="left")
    fe["temperature"] = fe["temperature"].fillna(fe["temperature"].median())
    fe["precipitation"] = fe["precipitation"].fillna(0)
    fe["temp_ma_3"] = fe["temp_ma_3"].fillna(fe["temperature"])
    fe["temp_delta"] = fe["temp_delta"].fillna(0)
    fe["is_rain"] = fe["is_rain"].fillna(0).astype(int)

    # ─── Лаги, MA, EWMA, тренды ────────────────────────────
    fe = fe.sort_values(["Номенклатура_Key", "Склад_Key", "Date"])
    grp = fe.groupby(["Номенклатура_Key", "Склад_Key"])["Количество"]

    for lag in [1, 7, 14, 30]:
        fe[f"lag_{lag}"] = grp.shift(lag)

    for w in [7, 14, 30]:
        fe[f"ma_{w}"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

    for span in [7, 14]:
        fe[f"ewma_{span}"] = grp.transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())

    fe["std_7"] = grp.transform(lambda x: x.shift(1).rolling(7, min_periods=2).std())

    fe["trend_7_30"] = np.where(fe["ma_30"] > 0, fe["ma_7"] / fe["ma_30"], 1.0)
    fe["trend_7_14"] = np.where(fe["ma_14"] > 0, fe["ma_7"] / fe["ma_14"], 1.0)
    fe["momentum"] = np.where(fe["lag_7"] > 0, (fe["lag_1"] - fe["lag_7"]) / fe["lag_7"], 0)

    # ─── YoY ────────────────────────────────────────────────
    fe["qty_same_day_ly"] = grp.shift(365)
    shifts = [grp.shift(d) for d in range(362, 369)]
    fe["qty_same_week_ly"] = pd.concat(shifts, axis=1).mean(axis=1)
    ma7_ly = grp.transform(lambda x: x.shift(365).rolling(7, min_periods=1).mean())
    fe["yoy_ratio"] = np.where(ma7_ly > 0, fe["ma_7"] / ma7_ly, 1.0)

    # Fill NaN
    fill_cols = [c for c in fe.columns if c.startswith(("lag_","ma_","ewma_","std_","trend_","momentum","qty_same","yoy_"))]
    fe[fill_cols] = fe[fill_cols].fillna(0)
    fe["yoy_ratio"] = fe["yoy_ratio"].clip(0, 10)

    return (fe,)


@app.cell
def _(fe, mo):
    mo.md(f"""
    ### ✅ Feature Engineering завершён

    - **Строк**: {len(fe):,}
    - **Признаков**: {len(fe.columns)}
    - **Период**: {fe['Date'].min().strftime('%d.%m.%Y')} — {fe['Date'].max().strftime('%d.%m.%Y')}
    """)
    return


@app.cell
def _(fe, np, pd, CatBoostRegressor, Pool, lgb, mean_absolute_error, root_mean_squared_error, r2_score):
    # ─── Обучение моделей ───────────────────────────────────
    CAT_FEATURES = ["Номенклатура_Key", "Склад_Key"]
    NUM_FEATURES = [
        "day_of_week", "month", "day_of_year", "week_of_year",
        "is_weekend", "is_holiday", "season", "is_month_start", "is_month_end",
        "days_to_holiday", "days_after_holiday", "is_pre_holiday",
        "dow_sin", "dow_cos", "month_sin", "month_cos",
        "temperature", "precipitation", "temp_ma_3", "temp_delta", "is_rain",
        "avg_price",
        "lag_1", "lag_7", "lag_14", "lag_30",
        "ma_7", "ma_14", "ma_30", "ewma_7", "ewma_14",
        "std_7", "trend_7_30", "trend_7_14", "momentum",
        "qty_same_day_ly", "qty_same_week_ly", "yoy_ratio",
    ]
    ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
    TARGET = "Количество"
    VAL_CUTOFF = "2026-01-01"

    train_df = fe[fe["Date"] < VAL_CUTOFF].copy()
    val_df = fe[fe["Date"] >= VAL_CUTOFF].copy()

    def _mape(y_true, y_pred):
        mask = y_true > 0
        return (abs(y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100

    def _eval(y_true, y_pred):
        return {
            "MAE": round(mean_absolute_error(y_true, y_pred), 2),
            "RMSE": round(root_mean_squared_error(y_true, y_pred), 2),
            "MAPE": round(_mape(y_true, y_pred), 1),
            "R²": round(r2_score(y_true, y_pred), 4),
        }

    # ─── CatBoost ───────────────────────────────────────────
    cat_idx = [ALL_FEATURES.index(f) for f in CAT_FEATURES]
    train_pool = Pool(train_df[ALL_FEATURES], train_df[TARGET], cat_features=cat_idx)
    val_pool = Pool(val_df[ALL_FEATURES], val_df[TARGET], cat_features=cat_idx)

    cb_model = CatBoostRegressor(
        iterations=1500, learning_rate=0.05, depth=6,
        l2_leaf_reg=3, eval_metric="MAE", random_seed=42, verbose=0,
    )
    cb_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
    cb_pred = np.maximum(cb_model.predict(val_df[ALL_FEATURES]), 0)
    cb_metrics = _eval(val_df[TARGET].values, cb_pred)

    # ─── LightGBM ──────────────────────────────────────────
    train_lgb = train_df.copy()
    val_lgb = val_df.copy()
    for col in CAT_FEATURES:
        train_lgb[col] = train_lgb[col].astype("category")
        val_lgb[col] = val_lgb[col].astype("category")

    train_ds = lgb.Dataset(train_lgb[ALL_FEATURES], train_lgb[TARGET], categorical_feature=CAT_FEATURES)
    val_ds = lgb.Dataset(val_lgb[ALL_FEATURES], val_lgb[TARGET], categorical_feature=CAT_FEATURES, reference=train_ds)

    lgb_model = lgb.train(
        {"objective": "regression", "metric": "mae", "learning_rate": 0.05,
         "num_leaves": 63, "min_child_samples": 20,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
         "verbose": -1, "seed": 42},
        train_ds, num_boost_round=1500,
        valid_sets=[val_ds], valid_names=["val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    lgb_pred = np.maximum(lgb_model.predict(val_df[ALL_FEATURES]), 0)
    lgb_metrics = _eval(val_df[TARGET].values, lgb_pred)

    # ─── Ensemble ──────────────────────────────────────────
    ens_pred = np.maximum(0.5 * cb_pred + 0.5 * lgb_pred, 0)
    ens_metrics = _eval(val_df[TARGET].values, ens_pred)

    # Feature importance
    fi = pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance": cb_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    # Собираем результаты
    results = pd.DataFrame([
        {"Модель": "Baseline (calendar)", "MAE": 82.77, "RMSE": 359.08, "MAPE": 165.3, "R²": 0.5540},
        {"Модель": "V2 (lags+weather)", "MAE": 55.90, "RMSE": 266.69, "MAPE": 109.1, "R²": 0.7540},
        {"Модель": "V3 (log+ewma)", "MAE": 60.54, "RMSE": 326.30, "MAPE": 54.1, "R²": 0.6317},
        {"Модель": "V4 CatBoost ⚡", **cb_metrics},
        {"Модель": "V4 LightGBM ⚡", **lgb_metrics},
        {"Модель": "V4 Ensemble ⚡", **ens_metrics},
    ])

    return ALL_FEATURES, cb_model, cb_pred, cb_metrics, lgb_model, lgb_pred, lgb_metrics, ens_pred, ens_metrics, fi, results, val_df, TARGET


@app.cell(hide_code=True)
def _(mo, results):
    # Таблица сравнения моделей
    mo.md("### 📊 Сравнение всех моделей")
    return


@app.cell
def _(mo, results):
    mo.ui.table(results, selection=None)
    return


@app.cell(hide_code=True)
def _(fi, px, mo):
    # Feature Importance
    mo.md("### 🏆 Feature Importance (CatBoost V4)")
    return


@app.cell
def _(fi, px):
    top_fi = fi.head(20).sort_values("importance")
    fig_fi = px.bar(
        top_fi, x="importance", y="feature", orientation="h",
        title="Топ-20 признаков по важности (CatBoost V4)",
        color="importance", color_continuous_scale="Viridis",
    )
    fig_fi.update_layout(height=600, showlegend=False)
    fig_fi
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("### 🎯 Факт vs Прогноз (валидация 2026)")
    return


@app.cell
def _(val_df, ens_pred, px, np, TARGET):
    import pandas as _pd
    scatter_df = _pd.DataFrame({
        "Факт": val_df[TARGET].values,
        "Прогноз": np.round(ens_pred, 1),
    })
    fig_scatter = px.scatter(
        scatter_df, x="Факт", y="Прогноз",
        title="Факт vs Прогноз (Ensemble V4)",
        opacity=0.3,
    )
    # Диагональ
    max_val = max(scatter_df["Факт"].max(), scatter_df["Прогноз"].max())
    fig_scatter.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="red", dash="dash"),
    )
    fig_scatter.update_layout(height=500, width=600)
    fig_scatter
    return


if __name__ == "__main__":
    app.run()
