
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

# ─── PAGE CONFIG ──────────────────────────────────────
st.set_page_config(
    page_title = "Store Sales Forecasting",
    page_icon  = "📈",
    layout     = "wide"
)

# ─── LOAD DATA AND MODELS ─────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("master.csv", parse_dates=["date"])
    return df

@st.cache_resource
def load_models():
    with open("xgb_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open("prophet_model.pkl", "rb") as f:
        prophet = pickle.load(f)
    return xgb, prophet

df                       = load_data()
xgb_model, prophet_model = load_models()

# ─── SIDEBAR ──────────────────────────────────────────
st.sidebar.title("📈 Store Sales Forecasting")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Sales Dashboard",
    "🔮 Forecast",
    "📈 Model Comparison",
    "🎮 What-If Simulator"
])
st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Created by")
st.sidebar.markdown("## Vineet Prakash")
st.sidebar.markdown("Aspiring Data Scientist")
st.sidebar.markdown("[GitHub](https://github.com/vin1bun) | [LinkedIn](#)")

# ══════════════════════════════════════════════════════
# PAGE 1 — SALES DASHBOARD
# ══════════════════════════════════════════════════════
if page == "📊 Sales Dashboard":
    st.title("📊 Sales Dashboard")
    st.markdown("Overview of store sales performance across all stores and product families.")
    st.markdown("**Created by Vineet Prakash** | Store Sales Forecasting Project")

    # KPI Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales",      f"{df['sales'].sum():,.0f}")
    col2.metric("Total Stores",     f"{df['store_nbr'].nunique()}")
    col3.metric("Product Families", f"{df['family'].nunique()}")

    # Overall Sales Trend
    st.subheader("Overall Sales Trend")
    daily = df.groupby("date")["sales"].sum().reset_index()
    fig   = px.line(daily, x="date", y="sales", title="Daily Total Sales")
    st.plotly_chart(fig, use_container_width=True)

    # Top Stores and Families side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Stores")
        top_stores = df.groupby("store_nbr")["sales"].sum().nlargest(10).reset_index()
        fig = px.bar(top_stores, x="store_nbr", y="sales", color="sales")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Product Families")
        top_families = df.groupby("family")["sales"].sum().nlargest(10).reset_index()
        fig = px.bar(top_families, x="sales", y="family", orientation="h", color="sales")
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("*Built by Vineet Prakash | Sales Forecasting using Prophet, XGBoost & ARIMA*")

# ══════════════════════════════════════════════════════
# PAGE 2 — FORECAST
# ══════════════════════════════════════════════════════
elif page == "🔮 Forecast":
    st.title("🔮 Sales Forecast")
    st.markdown("**Built by Vineet Prakash** | Powered by Prophet")

    col1, col2 = st.columns(2)
    with col1:
        store = st.selectbox("Select Store", sorted(df["store_nbr"].unique()))
    with col2:
        family = st.selectbox("Select Product Family", sorted(df["family"].unique()))

    filtered      = df[(df["store_nbr"] == store) & (df["family"] == family)]
    prophet_input = filtered.groupby("date")["sales"].sum().reset_index()
    prophet_input.columns = ["ds", "y"]

    future   = prophet_model.make_future_dataframe(periods=15)
    forecast = prophet_model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                             name="Forecast", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"],
                             fill=None, line=dict(color="lightblue"), name="Upper Bound"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"],
                             fill="tonexty", line=dict(color="lightblue"), name="Lower Bound"))
    fig.add_trace(go.Scatter(x=prophet_input["ds"], y=prophet_input["y"],
                             name="Actual", mode="markers", marker=dict(color="black")))
    fig.update_layout(title=f"Forecast — Store {store} | {family} | Vineet Prakash")
    st.plotly_chart(fig, use_container_width=True)

    last_forecast = forecast.tail(15)["yhat"].mean()
    avg_sales     = prophet_input["y"].mean()
    if last_forecast > avg_sales * 1.2:
        st.error(f"🚨 Restocking Alert — Forecasted spike of {((last_forecast/avg_sales)-1)*100:.1f}% above average. Order extra stock!")
    else:
        st.success("✅ Sales within normal range — no restocking alert")

    st.markdown("---")
    st.markdown("*Created by Vineet Prakash*")

# ══════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")
    st.markdown("**Vineet Prakash** | Comparing Prophet vs XGBoost vs ARIMA")

    st.subheader("Model Performance Metrics")
    metrics = pd.DataFrame({
        "Model"   : ["XGBoost", "Prophet", "ARIMA"],
        "MAPE"    : ["47.86%",  "9.98%",   "11.21%"],
        "RMSE"    : ["196.25",  "117113",  "113213"],
        "MAE"     : ["60.59",   "86134",   "92013"],
        "Level"   : ["Store+Family", "Daily Total", "Daily Total"]
    })
    st.dataframe(metrics, use_container_width=True)
    st.info("ℹ️ XGBoost operates at store-family level. Prophet and ARIMA operate on aggregated daily sales. Direct MAPE comparison is not apples-to-apples.")

    st.subheader("When to Use Each Model")
    col1, col2, col3 = st.columns(3)
    col1.success("**XGBoost**

Best for granular store level inventory decisions")
    col2.success("**Prophet**

Best for overall business planning and trend forecasting")
    col3.success("**ARIMA**

Best as statistical baseline for stable trends")

    st.markdown("---")
    st.markdown("*Built by Vineet Prakash*")

# ══════════════════════════════════════════════════════
# PAGE 4 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════
elif page == "🎮 What-If Simulator":
    st.title("🎮 What-If Sales Simulator")
    st.markdown("**Vineet Prakash** | Toggle conditions and see forecast impact")

    col1, col2 = st.columns(2)
    with col1:
        promotion  = st.toggle("🎯 Promotion Active", value=False)
        holiday    = st.toggle("🎉 Holiday",          value=False)
    with col2:
        base_sales = st.slider("Base Sales Level", 500, 5000, 1000, step=100)

    simulated = base_sales
    if promotion : simulated *= 1.25
    if holiday   : simulated *= 0.85

    st.subheader("Simulated Sales Impact")
    col1, col2, col3 = st.columns(3)
    col1.metric("Base Sales",      f"{base_sales:,}")
    col2.metric("Simulated Sales", f"{simulated:,.0f}")
    col3.metric("Impact",          f"{((simulated/base_sales)-1)*100:+.1f}%")

    fig = px.bar(
        x      = ["Base Sales", "Simulated Sales"],
        y      = [base_sales, simulated],
        color  = ["Base", "Simulated"],
        title  = "Sales Impact Simulation — Vineet Prakash",
        labels = {"x": "Scenario", "y": "Sales"}
    )
    st.plotly_chart(fig, use_container_width=True)

    if promotion and holiday:
        st.warning("⚠️ Promotion during holiday — mixed effect. Net impact: +6.25%")
    elif promotion:
        st.success("✅ Promotion active — expected +25% sales lift")
    elif holiday:
        st.info("ℹ️ Holiday detected — expected -15% reduction in sales")

    st.markdown("---")
    st.markdown("*Created by Vineet Prakash | Store Sales Forecasting Project*")
