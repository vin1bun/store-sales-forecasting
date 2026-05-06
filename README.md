# 📈 Store Sales Forecasting

### 🔗 [Live App](https://store-sales-forecasting-hosig7yav6sekcchgkrkkk.streamlit.app/) | [GitHub](https://github.com/vin1bun) | [LinkedIn](https://linkedin.com/in/vineetprakash03)

> Predict daily retail sales across 54 stores and 33 product families using Prophet, XGBoost and ARIMA — with a live interactive dashboard built on real Ecuador grocery chain data.

---

## 🧩 Business Problem

Retail chains lose crores every year from two mistakes — overstocking (products expire) and stockouts (shelves go empty). This project solves that by forecasting daily sales per store per product family — giving procurement teams a data-driven answer to *"How much should we order next week?"*

---

## 📦 Dataset

| File | Description |
|---|---|
| train.csv | 3 million rows of daily sales per store per family |
| stores.csv | Store info — city, state, type, cluster |
| oil.csv | Daily Ecuador oil prices |
| holidays_events.csv | National, regional and local holidays |
| transactions.csv | Daily transaction counts per store |

- **Source** → Kaggle Store Sales Time Series Forecasting Competition
- **Period** → January 2013 to August 2017
- **Scale** → 54 stores × 33 product families × 1,684 days

---

## ⚙️ What Makes This Project Different

- 🔗 **Merged 6 real-world data files** using relational join keys — not a single clean CSV
- 🛢️ **Oil price as economic signal** — Ecuador's economy depends on oil exports, directly affecting consumer spending
- 📅 **3 types of seasonality validated from data** — weekly, monthly and yearly — before any feature engineering
- 🔄 **Group-aware lag features** — created within store-family groups using `groupby().shift()` to prevent group leakage
- ⏩ **Look-ahead bias prevention** — used `shift(1)` inside rolling calculations so model never sees today's sales
- 🎯 **3 models compared with business reasoning** — not just accuracy numbers

---

## 🧠 Feature Engineering

| Feature Type | Features Created |
|---|---|
| Lag Features | lag_7, lag_14, lag_28 |
| Rolling Averages | rolling_mean_7, rolling_mean_14, rolling_mean_28 |
| Date Features | day, week, month, quarter, year, dayofweek, is_weekend |
| External Features | oil price, is_holiday, onpromotion, transactions |
| Encoded Features | family_encoded, type_encoded |

---

## 🤖 Models Compared

| Model | MAPE | Level | Best For |
|---|---|---|---|
| **Prophet** | **9.98%** 🏆 | Daily Total | Business planning and trend forecasting |
| ARIMA | 11.21% | Daily Total | Statistical baseline |
| XGBoost | 47.86%* | Store + Family | Granular inventory decisions |

*XGBoost MAPE is inflated by 44% zero-sales rows in sparse retail data. RMSE of 196 is the reliable metric at store-family level.

---

## 🚀 Live App Features

| Page | What It Does |
|---|---|
| 📊 Sales Dashboard | KPI cards, overall trend, top stores and families |
| 🔮 Forecast | Store + family selector with Prophet forecast and confidence intervals |
| 📈 Model Comparison | All 3 models compared with metrics and business reasoning |
| 🎮 What-If Simulator | Toggle promotions and holidays — see instant sales impact |

---

## 💰 Business Impact

> A 1% MAPE improvement on a ₹10 Crore inventory = ₹10 Lakh saved.
> Prophet improved over ARIMA baseline from 11.21% to 9.98% — approximately ₹12 Lakh in potential savings per ₹10 Crore inventory.

---

## 🛠️ Tech Stack

`Python` `XGBoost` `Prophet` `ARIMA` `Pandas` `NumPy` `SHAP` `Plotly` `Streamlit` `Google Colab` `Scikit-learn`

---

## 📁 Project Structure

```
store-sales-forecasting/
│
├── app.py                  ← Streamlit app
├── requirements.txt        ← Dependencies
├── xgb_model.pkl          ← Trained XGBoost model
├── prophet_model.pkl      ← Trained Prophet model
└── master.csv             ← Cleaned and merged dataset
```

---

## 👨‍💻 Author

**Vineet Prakash** — Aspiring Data Scientist

[GitHub](https://github.com/vin1bun) | [LinkedIn](https://linkedin.com/in/vineetprakash03) | [Live App](https://store-sales-forecasting-hosig7yav6sekcchgkrkkk.streamlit.app/)



*Project 3 of 5 in my Data Science Portfolio — targeting roles at Amazon India, Flipkart, MoEngage and Google India*

---

Paste this into your GitHub README file and your project is 100% complete and portfolio ready. 🚀
