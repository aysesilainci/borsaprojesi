import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt

# 1. Veri çek
data = yf.Ticker("AAPL").history(period="5y", interval="1d")

# 2. RSI hesapla
data["RSI"] = ta.rsi(data["Close"], length=14)
data.dropna(inplace=True)

# 3. Sinyaller
data["Signal"] = 0
data.loc[data["RSI"] < 30, "Signal"] = 1
data.loc[data["RSI"] > 70, "Signal"] = -1
buy_signals = data[data["Signal"] == 1]
sell_signals = data[data["Signal"] == -1]

# 4. Getiriler
data["Return"] = data["Close"].pct_change()
data["Strategy_Return"] = data["Signal"].shift(1) * data["Return"]
data["Cumulative_Market"] = (1 + data["Return"]).cumprod()
data["Cumulative_Strategy"] = (1 + data["Strategy_Return"]).cumprod()

# 5. GRAFİKLERİ OLUŞTUR
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Grafik 1: Fiyat + sinyaller
axs[0].plot(data.index, data["Close"], label="Fiyat", color="steelblue")
axs[0].scatter(buy_signals.index, buy_signals["Close"], label="AL", marker="^", color="green", s=100)
axs[0].scatter(sell_signals.index, sell_signals["Close"], label="SAT", marker="v", color="red", s=100)
axs[0].set_title("AAPL Fiyat Grafiği + AL/SAT Sinyalleri")
axs[0].legend()
axs[0].grid()

# Grafik 2: RSI
axs[1].plot(data.index, data["RSI"], label="RSI", color="orange")
axs[1].axhline(70, color="red", linestyle="--")
axs[1].axhline(30, color="green", linestyle="--")
axs[1].set_title("RSI Grafiği")
axs[1].legend()
axs[1].grid()

# Grafik 3: Strateji vs Buy & Hold
axs[2].plot(data.index, data["Cumulative_Market"], label="Buy & Hold", color="blue")
axs[2].plot(data.index, data["Cumulative_Strategy"], label="RSI Stratejisi", color="green")
axs[2].set_title("Strateji vs Piyasayı Beklemek (Backtest)")
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()
