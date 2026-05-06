import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Store Sales Forecasting", page_icon="📈", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("master.csv", parse_dates=["date"])

@st.cache_resource
def load_models():
    with open("xgb_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open("prophet_model.pkl", "rb") as f:
        prophet = pickle.load(f)
    return xgb, prophet

df = load_data()
xgb_model, prophet_model = load_models()

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
st.sidebar.markdown("Data Scientist")

if page == "📊 Sales Dashboard":
    st.title("📊 Sales Dashboard")
    st.markdown("**Created by Vineet Prakash** | Store Sales Forecasting Project")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"{df['sales'].sum():,.0f}")
    col2.metric("Total Stores", f"{df['store_nbr'].nunique()}")
    col3.metric("Product Families", f"{df['family'].nunique()}")

    st.subheader("Overall Sales Trend")
    daily = df.groupby("date")["sales"].sum().reset_index()
    fig = px.line(daily, x="date", y="sales", title="Daily Total Sales")
    st.plotly_chart(fig, use_container_width=True)

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

    st.markdown("---")
    st.markdown("*Built by Vineet Prakash | Sales Forecasting using Prophet, XGBoost and ARIMA*")

elif page == "🔮 Forecast":
    st.title("🔮 Sales Forecast")
    st.markdown("**Built by Vineet Prakash** | Powered by Prophet")

    col1, col2 = st.columns(2)
    with col1:
        store = st.selectbox("Select Store", sorted(df["store_nbr"].unique()))
    with col2:
        family = st.selectbox("Select Product Family", sorted(df["family"].unique()))

    filtered = df[(df["store_nbr"] == store) & (df["family"] == family)]
    prophet_input = filtered.groupby("date")["sales"].sum().reset_index()
    prophet_input.columns = ["ds", "y"]

    future = prophet_model.make_future_dataframe(periods=15)
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
    fig.update_layout(title=f"Forecast — Store {store} | {family}")
    st.plotly_chart(fig, use_container_width=True)

    last_forecast = forecast.tail(15)["yhat"].mean()
    avg_sales = prophet_input["y"].mean()
    if last_forecast > avg_sales * 1.2:
        st.error(f"Restocking Alert — Forecasted spike of {((last_forecast/avg_sales)-1)*100:.1f}% above average!")
    else:
        st.success("Sales within normal range — no restocking alert")

    st.markdown("---")
    st.markdown("*Created by Vineet Prakash*")

elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")
    st.markdown("**Vineet Prakash** | Comparing Prophet vs XGBoost vs ARIMA")

    metrics = pd.DataFrame({
        "Model"  : ["XGBoost", "Prophet", "ARIMA"],
        "MAPE"   : ["47.86%", "9.98%", "11.21%"],
        "RMSE"   : ["196.25", "117113", "113213"],
        "MAE"    : ["60.59", "86134", "92013"],
        "Level"  : ["Store+Family", "Daily Total", "Daily Total"]
    })
    st.dataframe(metrics, use_container_width=True)
    st.info("XGBoost operates at store-family level. Prophet and ARIMA on aggregated daily sales.")

    st.subheader("When to Use Each Model")
    col1, col2, col3 = st.columns(3)
    col1.success("XGBoost — Best for granular store level inventory decisions")
    col2.success("Prophet — Best for overall business planning and trend forecasting")
    col3.success("ARIMA — Best as statistical baseline for stable trends")
    st.markdown("---")
    st.markdown("*Built by Vineet Prakash*")

elif page == "🎮 What-If Simulator":
    st.title("🎮 What-If Sales Simulator")
    st.markdown("**Vineet Prakash** | Toggle conditions and see forecast impact")

    col1, col2 = st.columns(2)
    with col1:
        promotion = st.toggle("Promotion Active", value=False)
        holiday = st.toggle("Holiday", value=False)
    with col2:
        base_sales = st.slider("Base Sales Level", 500, 5000, 1000, step=100)

    simulated = base_sales
    if promotion:
        simulated *= 1.25
    if holiday:
        simulated *= 0.85

    col1, col2, col3 = st.columns(3)
    col1.metric("Base Sales", f"{base_sales:,}")
    col2.metric("Simulated Sales", f"{simulated:,.0f}")
    col3.metric("Impact", f"{((simulated/base_sales)-1)*100:+.1f}%")

    fig = px.bar(
        x=["Base Sales", "Simulated Sales"],
        y=[base_sales, simulated],
        color=["Base", "Simulated"],
        title="Sales Impact Simulation — Vineet Prakash",
        labels={"x": "Scenario", "y": "Sales"}
    )
    st.plotly_chart(fig, use_container_width=True)

    if promotion and holiday:
        st.warning("Mixed effect — Promotion +25% but Holiday -15%. Net: +6.25%")
    elif promotion:
        st.success("Promotion active — expected +25% sales lift")
    elif holiday:
        st.info("Holiday detected — expected -15% reduction in sales")

    st.markdown("---")
    st.markdown("*Created by Vineet Prakash | Store Sales Forecasting Project*")
