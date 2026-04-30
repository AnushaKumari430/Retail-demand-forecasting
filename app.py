import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Retail Forecast", layout="wide")

st.title("📊 Retail Demand Forecasting Dashboard")
st.caption("Time-series forecasting using Prophet | Retail inventory planning")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Sidebar
st.sidebar.header("⚙ Controls")
days = st.sidebar.slider("Forecast Days", 7, 90, 30)

# Forecast
future = model.make_future_dataframe(periods=days)
forecast = model.predict(future)

historical = forecast[:-days]
future_data = forecast[-days:]

# --- METRICS ---
st.markdown("### 📌 Key Metrics")

growth = ((future_data['yhat'].mean() - historical['yhat'].iloc[-1]) / historical['yhat'].iloc[-1]) * 100

col1, col2, col3 = st.columns(3)

col1.metric("Last Observed", f"{historical['yhat'].iloc[-1]:.0f}")
col2.metric("Avg Forecast", f"{future_data['yhat'].mean():.0f}", f"{growth:.2f}%")
col3.metric("Peak Forecast", f"{future_data['yhat'].max():.0f}")

st.markdown("---")

# --- INTERACTIVE CHART ---
st.subheader("📈 Sales Forecast")

fig = go.Figure()

# Historical data
fig.add_trace(go.Scatter(
    x=historical['ds'],
    y=historical['yhat'],
    mode='lines',
    name='Historical',
    line=dict(color='blue')
))

# Forecast data
fig.add_trace(go.Scatter(
    x=future_data['ds'],
    y=future_data['yhat'],
    mode='lines',
    name='Forecast',
    line=dict(color='green')
))

# Confidence interval
fig.add_trace(go.Scatter(
    x=future_data['ds'],
    y=future_data['yhat_upper'],
    mode='lines',
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=future_data['ds'],
    y=future_data['yhat_lower'],
    mode='lines',
    fill='tonexty',
    name='Confidence Range',
    fillcolor='rgba(0,200,0,0.2)',
    line=dict(width=0)
))

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- COMPONENTS ---
st.subheader("📊 Trend & Seasonality")

fig2 = model.plot_components(forecast)
st.pyplot(fig2)

st.markdown("---")

# --- INSIGHTS ---
st.subheader("📌 Business Insights")

direction = "increase 📈" if growth > 0 else "decrease 📉"

st.write(f"""
### Key Observations:
- Expected **{direction} of {abs(growth):.2f}%**
- Average demand: **{int(future_data['yhat'].mean())} units/day**
- Peak demand: **{int(future_data['yhat'].max())} units**
- Variability suggests seasonal patterns

### Business Recommendations:
- Adjust inventory for predicted peaks  
- Maintain safety stock for fluctuations  
- Optimize supply planning based on forecast  
""")