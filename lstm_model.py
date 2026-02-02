import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Veri çek
df = yf.Ticker("AAPL").history(period="5y")
data = df[["Close"]].copy()

# 2. Ölçekleme (0-1 arası)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 3. LSTM veri yapısı oluştur
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i])
        y.append(dataset[i])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)

# 4. Eğitim/test ayır
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. LSTM modelini oluştur
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Eğit
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# 7. Tahmin yap
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_test)

# 8. Grafik
plt.figure(figsize=(12,6))
plt.plot(actual_prices, label='Gerçek Fiyat')
plt.plot(predicted_prices, label='Tahmin Edilen Fiyat')
plt.title("LSTM ile AAPL Kapanış Fiyat Tahmini")
plt.xlabel("Zaman (test seti)")
plt.ylabel("Fiyat (USD)")
plt.legend()
plt.grid()
plt.show()
