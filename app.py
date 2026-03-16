import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -----------------------------
# PAGE CONFIG                 -
# -----------------------------
st.set_page_config(page_title="AI Trading Assistant", layout="wide")

st.title("📈 AI Trading Assistant")


# -----------------------------
# STOCK INPUT                 -
# -----------------------------
ticker = st.text_input("Enter Stock Symbol", "AAPL")


# -----------------------------
# LOAD DATA                   -
# -----------------------------
data = yf.download(ticker, period="1y", interval="1d")

if data.empty:
    st.error("No data found")
    st.stop()

data = data.reset_index()

# -------- FIX MULTIINDEX PROBLEM ----------
data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

# -------- FORCE ALL PRICE COLUMNS TO 1D --------
for col in ["Open", "High", "Low", "Close", "Volume"]:
    data[col] = pd.Series(np.ravel(data[col]))


# -----------------------------
# TECHNICAL INDICATORS        -
# -----------------------------
data["RSI"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
data["SMA20"] = data["Close"].rolling(20).mean()

data = data.dropna().reset_index(drop=True)


# -----------------------------
# AI PREDICTION               -
# -----------------------------
prices = data["Close"].values.reshape(-1,1)

if len(prices) < 60:
    st.error("Not enough data for prediction")
    st.stop()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices)

model = load_model("model/stock_model.h5", compile=False)

last_60 = scaled[-60:]
X = last_60.reshape(1,60,1)

prediction = model.predict(X, verbose=0)

pred_price = scaler.inverse_transform(prediction)[0][0]
current_price = prices[-1][0]


# -----------------------------
# SIGNAL LOGIC                -
# -----------------------------
rsi = data["RSI"].iloc[-1]

reasons = []

if pred_price > current_price:
    signal = "BUY"
    reasons.append("AI predicts price increase")
elif pred_price < current_price:
    signal = "SELL"
    reasons.append("AI predicts price decrease")
else:
    signal = "HOLD"

if rsi > 70:
    reasons.append("RSI indicates overbought market")
elif rsi < 30:
    reasons.append("RSI indicates oversold market")

if current_price > data["SMA20"].iloc[-1]:
    reasons.append("Price above 20-day trend")
else:
    reasons.append("Price below 20-day trend")


# -----------------------------
# METRICS                     -
# -----------------------------
col1, col2 = st.columns(2)

col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Predicted Price", f"${pred_price:.2f}")


# -----------------------------
# SIGNAL DISPLAY              -
# -----------------------------
if signal == "BUY":
    st.success("🟢 BUY SIGNAL")
elif signal == "SELL":
    st.error("🔴 SELL SIGNAL")
else:
    st.warning("🟡 HOLD SIGNAL")


# -----------------------------
# EXPLANATION                 -
# -----------------------------
st.subheader("📌 Why this signal?")

for r in reasons:
    st.write("•", r)


# -----------------------------
# CANDLESTICK CHART           -
# -----------------------------
st.subheader("📊 Stock Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data["Date"],
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Price"
))

future_date = data["Date"].iloc[-1] + pd.Timedelta(days=1)

fig.add_trace(go.Scatter(
    x=[future_date],
    y=[pred_price],
    mode="markers",
    marker=dict(color="red", size=12),
    name="AI Prediction"
))

fig.update_layout(
    template="plotly_dark",
    height=600,
    title=f"{ticker} Price Chart",
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(fig, width="stretch")


# -----------------------------
# VOLUME CHART                -
# -----------------------------
st.subheader("📊 Volume")

vol_fig = go.Figure()

vol_fig.add_trace(go.Bar(
    x=data["Date"],
    y=data["Volume"],
    name="Volume"
))

vol_fig.update_layout(
    template="plotly_dark",
    height=300
)

st.plotly_chart(vol_fig, width="stretch")


# -----------------------------
# PRICE COMPARISON            -
# -----------------------------
st.subheader("🤖 Current vs Predicted")

comparison_df = pd.DataFrame({
    "Type": ["Current Price", "Predicted Price"],
    "Price": [current_price, pred_price]
})

st.bar_chart(comparison_df.set_index("Type"))