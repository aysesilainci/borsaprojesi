import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas_ta as ta

st.set_page_config(page_title="Borsa Tahmin ve Strateji", layout="wide")
st.title("📈 Borsa Analiz Aracı")

# Model Ayarları
st.sidebar.header("🔧 LSTM Model Ayarları")
epochs = st.sidebar.slider("Epoch Sayısı", 1, 50, 10)
batch_size = st.sidebar.slider("Batch Size", 1, 128, 32)
look_back = st.sidebar.slider("Geriye Bakış (look_back)", 1, 60, 20)

# LSTM verisini hazırlama
@st.cache_data
def load_data(look_back):
    df = yf.download("AAPL", start="2015-01-01", end="2024-12-31")
    data = df[["Close"]].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:], scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(actual, predicted):
    fig = plt.figure(figsize=(12,6))
    plt.plot(actual, label="Gerçek Fiyat")
    plt.plot(predicted, label="Tahmin Edilen Fiyat")
    plt.title("LSTM ile AAPL Kapanış Fiyat Tahmini")
    plt.xlabel("Zaman (test seti)")
    plt.ylabel("Fiyat (USD)")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

# --- Sekmeler ---
tab1, tab2 = st.tabs(["📊 LSTM Tahmini", "📈 RSI + MACD Stratejisi"])

# === TAB 1: LSTM ===
with tab1:
    if st.button("📊 LSTM ile Tahmin Yap"):
        st.info("Veriler yükleniyor ve işleniyor...")
        X_train, y_train, X_test, y_test, scaler = load_data(look_back)
        model = build_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        predicted = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted)
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        st.success("Tahmin tamamlandı.")
        plot_predictions(actual_prices, predicted_prices)

# === TAB 2: RSI + MACD ===
with tab2:
    st.subheader("📈 RSI + MACD ile AL/SAT Stratejisi ve Backtest")
    df = yf.Ticker("AAPL").history(period="5y", interval="1d")

    # RSI Hesaplama
    df["RSI"] = ta.rsi(df["Close"], length=14)

    # MACD Hesaplama
    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDs_12_26_9"]

    df.dropna(inplace=True)

    # RSI sinyalleri
    df["RSI_Signal"] = 0
    df.loc[df["RSI"] < 30, "RSI_Signal"] = 1
    df.loc[df["RSI"] > 70, "RSI_Signal"] = -1

    # MACD sinyalleri
    df["MACD_Signal"] = 0
    df.loc[df["MACD"] > df["MACD_signal"], "MACD_Signal"] = 1
    df.loc[df["MACD"] < df["MACD_signal"], "MACD_Signal"] = -1

    # RSI stratejisi getirisi
    df["Return"] = df["Close"].pct_change()
    df["RSI_Strategy"] = df["RSI_Signal"].shift(1) * df["Return"]
    df["MACD_Strategy"] = df["MACD_Signal"].shift(1) * df["Return"]

    df["Cumulative_Market"] = (1 + df["Return"]).cumprod()
    df["Cumulative_RSI"] = (1 + df["RSI_Strategy"]).cumprod()
    df["Cumulative_MACD"] = (1 + df["MACD_Strategy"]).cumprod()

    # AL/SAT noktaları
    rsi_buy = df[df["RSI_Signal"] == 1]
    rsi_sell = df[df["RSI_Signal"] == -1]
    macd_buy = df[(df["MACD_Signal"] == 1) & (df["MACD"].shift(1) < df["MACD_signal"].shift(1))]
    macd_sell = df[(df["MACD_Signal"] == -1) & (df["MACD"].shift(1) > df["MACD_signal"].shift(1))]

    # Grafikler
    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axs[0].plot(df.index, df["Close"], label="Fiyat", color="black")
    axs[0].scatter(rsi_buy.index, rsi_buy["Close"], label="RSI AL", marker="^", color="green", s=100)
    axs[0].scatter(rsi_sell.index, rsi_sell["Close"], label="RSI SAT", marker="v", color="red", s=100)
    axs[0].scatter(macd_buy.index, macd_buy["Close"], label="MACD AL", marker="o", color="blue", s=70)
    axs[0].scatter(macd_sell.index, macd_sell["Close"], label="MACD SAT", marker="x", color="orange", s=70)
    axs[0].set_title("AAPL Fiyat + RSI & MACD AL/SAT Sinyalleri")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(df.index, df["RSI"], label="RSI", color="purple")
    axs[1].axhline(70, color="red", linestyle="--")
    axs[1].axhline(30, color="green", linestyle="--")
    axs[1].set_title("RSI Değerleri")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(df.index, df["MACD"], label="MACD", color="blue")
    axs[2].plot(df.index, df["MACD_signal"], label="Sinyal", color="orange", linestyle="--")
    axs[2].set_title("MACD Göstergesi")
    axs[2].legend()
    axs[2].grid()

    axs[3].plot(df.index, df["Cumulative_Market"], label="Buy & Hold", color="black")
    axs[3].plot(df.index, df["Cumulative_RSI"], label="RSI Stratejisi", color="green")
    axs[3].plot(df.index, df["Cumulative_MACD"], label="MACD Stratejisi", color="blue")
    axs[3].set_title("Stratejilerin Kümülatif Getiri Karşılaştırması")
    axs[3].legend()
    axs[3].grid()

    plt.tight_layout()
    st.pyplot(fig)
