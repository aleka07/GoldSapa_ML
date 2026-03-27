**Общая архитектура**

1С OData Python/Pandas ML Model Результат

───────── ──────────── ──────── ─────────

РеализацияТМЗ → Очистка+Агрегация → CatBoost/Prophet → Прогноз

Номенклатура Внешние факторы Обучение на завтра

(погода, календарь) Предсказание

**Шаг 1 - Выгрузка данных из 1С**

import requests

import pandas as pd

from requests.auth import HTTPBasicAuth

def get_sales(date_from, date_to):

base_url = "<https://goldsapa.24c.kz/goldsapa/odata/standard.odata>"

endpoint = "/AccumulationRegister_РеализацияТМЗ/RecordType"

params = {

"\$format": "json",

"\$select": "Period,Номенклатура_Key,Склад_Key,Количество",

"\$filter": f"Period ge datetime'{date_from}T00:00:00' and Period le datetime'{date_to}T23:59:59'",

"\$orderby": "Period asc",

"\$top": 1000,

"\$skip": 0

}

auth = HTTPBasicAuth("логин", "пароль")

all_records = \[\]

while True:

r = requests.get(base_url + endpoint, params=params, auth=auth)

batch = r.json().get("value", \[\])

all_records.extend(batch)

if len(batch) < 1000:

break

params\["\$skip"\] += 1000

return pd.DataFrame(all_records)

df = get_sales("2024-01-01", "2025-12-31")

**Шаг 2 - Агрегация по дням**

\# Дата без времени

df\["Date"\] = pd.to_datetime(df\["Period"\]).dt.date

\# Суммируем по дню + номенклатура + склад

daily = df.groupby(\["Date", "Номенклатура_Key", "Склад_Key"\]).agg(

Количество=("Количество", "sum")

).reset_index()

\# Убираем возвраты (отрицательные значения)

daily = daily\[daily\["Количество"\] > 0\]

**Шаг 3 - Добавляем внешние факторы**

import openmeteo_requests # pip install openmeteo-requests

\# --- День недели, месяц, праздники ---

daily\["Date"\] = pd.to_datetime(daily\["Date"\])

daily\["day_of_week"\] = daily\["Date"\].dt.dayofweek # 0=пн, 6=вс

daily\["month"\] = daily\["Date"\].dt.month

daily\["day_of_year"\] = daily\["Date"\].dt.dayofyear

daily\["is_weekend"\] = daily\["day_of_week"\].isin(\[5, 6\]).astype(int)

\# --- Праздники Казахстана ---

import holidays

kz_holidays = holidays.Kazakhstan()

daily\["is_holiday"\] = daily\["Date"\].apply(

lambda x: 1 if x in kz_holidays else 0

)

\# --- Температура (Open-Meteo, бесплатно) ---

import openmeteo_requests

def get_weather(dates):

\# Алматы: lat=43.25, lon=76.95

url = "<https://archive-api.open-meteo.com/v1/archive>"

params = {

"latitude": 43.25,

"longitude": 76.95,

"start_date": str(dates.min().date()),

"end_date": str(dates.max().date()),

"daily": "temperature_2m_mean"

}

r = requests.get(url, params=params).json()

weather = pd.DataFrame({

"Date": pd.to_datetime(r\["daily"\]\["time"\]),

"temperature": r\["daily"\]\["temperature_2m_mean"\]

})

return weather

weather = get_weather(daily\["Date"\])

daily = daily.merge(weather, on="Date", how="left")

**Шаг 4 - Обучение модели**

from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split

\# Признаки для модели

features = \[

"day_of_week",

"month",

"day_of_year",

"is_weekend",

"is_holiday",

"temperature",

"Номенклатура_Key", # CatBoost сам обработает как категорию

"Склад_Key"

\]

target = "Количество"

X = daily\[features\]

y = daily\[target\]

X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size=0.2, shuffle=False # shuffle=False - важно для временных рядов!

)

model = CatBoostRegressor(

iterations=500,

learning_rate=0.05,

depth=6,

cat_features=\["Номенклатура_Key", "Склад_Key"\], # категориальные признаки

eval_metric="RMSE",

verbose=50

)

model.fit(X_train, y_train, eval_set=(X_test, y_test))

**Шаг 5 - Прогноз на завтра**

from datetime import date, timedelta

tomorrow = date.today() + timedelta(days=1)

\# Получаем прогноз погоды на завтра

def get_forecast_temp(date):

url = "<https://api.open-meteo.com/v1/forecast>"

params = {

"latitude": 43.25,

"longitude": 76.95,

"daily": "temperature_2m_mean",

"forecast_days": 2

}

r = requests.get(url, params=params).json()

temps = dict(zip(r\["daily"\]\["time"\], r\["daily"\]\["temperature_2m_mean"\]))

return temps.get(str(date), 10)

temp_tomorrow = get_forecast_temp(tomorrow)

\# Берём все уникальные номенклатуры

nomenclatures = daily\["Номенклатура_Key"\].unique()

skl = daily\["Склад_Key"\].unique()\[0\] # или все склады

predict_df = pd.DataFrame({

"Номенклатура_Key": nomenclatures,

"Склад_Key": skl,

"day_of_week": tomorrow.weekday(),

"month": tomorrow.month,

"day_of_year": tomorrow.timetuple().tm_yday,

"is_weekend": int(tomorrow.weekday() in \[5, 6\]),

"is_holiday": int(tomorrow in kz_holidays),

"temperature": temp_tomorrow

})

predict*df\["прогноз*количество"\] = model.predict(predict_df\[features\])

predict*df\["прогноз*количество"\] = predict*df\["прогноз*количество"\].round().astype(int)

\# Оставляем только позиции с ненулевым прогнозом

result = predict*df\[predict_df\["прогноз*количество"\] > 0\].sort_values(

"прогноз_количество", ascending=False

)

print(result\[\["Номенклатура*Key", "прогноз*количество"\]\])

**Итоговый вывод технологу**

| **Номенклатура** | **Прогноз на завтра (шт)** |
| ---------------- | -------------------------- |
| Сэндвич Панини   | 95                         |
| Круассан масло   | 140                        |
| Пирог яблоко     | 60                         |

**Что нужно для старта**

- История продаж минимум **6 месяцев** (лучше год+)
- Python с библиотеками: pandas, catboost, requests, holidays
- Доступ к OData (уже есть ✅)
- Около **2-3 дней** на настройку и первые результаты

Хочешь начнём с конкретного шага - например напишем полный рабочий скрипт?