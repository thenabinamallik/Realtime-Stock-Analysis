import pandas as pd
import numpy as np

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Load dataset                -
# -----------------------------
data = pd.read_csv("dataset/AAPL_Stock_Price_Dataset.csv")

# Ensure sorted by date
data = data.sort_values("Date")

# Extract closing prices
prices = data["Close"].values.reshape(-1, 1)

# -----------------------------
# Scale prices                -
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(prices)

# -----------------------------
# Load trained model          -
# -----------------------------
model = load_model("model/stock_model.h5", compile=False)

# -----------------------------
# Prepare prediction input    -
# -----------------------------
last_60 = scaled[-60:]
X = last_60.reshape(1, 60, 1)

# -----------------------------
# Predict next price          -
# -----------------------------
pred = model.predict(X, verbose=0)

pred_price = scaler.inverse_transform(pred)[0][0]
current_price = prices[-1][0]

# -----------------------------
# Trading signal logic        -
# -----------------------------
if pred_price > current_price * 1.01:
    signal = "BUY"
elif pred_price < current_price * 0.99:
    signal = "SELL"
else:
    signal = "HOLD"

# -----------------------------
# Output                      -
# -----------------------------
print("\n📊 AI Stock Prediction\n")

print(f"Current Price: ${current_price:.2f}")
print(f"Predicted Price: ${pred_price:.2f}")
print(f"Trading Signal: {signal}")