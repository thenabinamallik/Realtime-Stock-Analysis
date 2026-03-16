import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------
# Load dataset                -
# -----------------------------
data = pd.read_csv("dataset/AAPL_Stock_Price_Dataset.csv")

# Ensure sorted by date
data = data.sort_values("Date")

# Use closing price
prices = data["Close"].values.reshape(-1, 1)

# -----------------------------
# Scale data                  -
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(prices)

# -----------------------------
# Create training sequences   -
# -----------------------------
X = []
y = []

for i in range(60, len(scaled)):
    X.append(scaled[i-60:i, 0])
    y.append(scaled[i, 0])

X = np.array(X)
y = np.array(y)

# reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# -----------------------------
# Build LSTM model            -
# -----------------------------
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(60, 1)))
model.add(Dropout(0.2))

model.add(LSTM(64))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

print("🚀 Training model...")

# -----------------------------
# Train model                 -
# -----------------------------
model.fit(
    X,
    y,
    epochs=10,
    batch_size=32,
    verbose=1
)

# -----------------------------
# Save model                  -
# -----------------------------
os.makedirs("model", exist_ok=True)

model.save("model/stock_model.keras")

print("✅ Model saved successfully!")