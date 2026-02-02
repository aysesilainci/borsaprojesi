# 📈 Stock Price Prediction & Trading Strategy with LSTM, RSI, MACD

Bu proje, finansal zaman serisi verileri kullanılarak **LSTM (Long Short-Term Memory)** modeli ile hisse senedi kapanış fiyatı tahmini yapılmasını ve teknik indikatörler (**RSI, MACD**) kullanılarak **al/sat stratejisi** oluşturulmasını amaçlamaktadır. Ayrıca geliştirilen strateji üzerinde **backtest** uygulanarak performans analizi yapılmıştır.

Proje, uçtan uca bir finansal veri bilimi ve algoritmik işlem (algorithmic trading) çalışmasıdır.

---

## 🚀 Projenin Amacı

* Gerçek finans verisi çekmek
* Zaman serisi verisini LSTM modeli ile tahmin etmek
* Teknik analiz indikatörleri ile al/sat sinyalleri üretmek
* Stratejinin geçmiş veriler üzerinde başarımını ölçmek (Backtesting)
* Tahmin ve strateji sonuçlarını görselleştirmek

---

## 📊 Kullanılan Teknolojiler

* Python
* Pandas, Numpy
* Matplotlib
* TensorFlow / Keras (LSTM modeli)
* yfinance (veri çekme)
* Teknik analiz: RSI, MACD

---

## 📥 Veri Kaynağı

Projede hisse senedi verileri **Yahoo Finance API (yfinance)** kullanılarak çekilmiştir.

Örnek:

* AAPL (Apple)
* Tarihsel kapanış fiyatları
* Günlük zaman serisi verisi

---

## 🧠 LSTM Modeli ile Fiyat Tahmini

LSTM modeli, zaman serilerindeki bağımlılıkları öğrenebilme yeteneği sayesinde hisse senedi kapanış fiyatlarını tahmin etmek için kullanılmıştır.

Model adımları:

1. Veri normalize edildi (MinMaxScaler)
2. Zaman pencereleri (time window) oluşturuldu
3. LSTM katmanları ile model kuruldu
4. Eğitim ve test setleri ayrıldı
5. Tahminler görselleştirildi

### 📌 LSTM Tahmin Sonucu

![LSTM Tahmin](images/lstm.jpg)

Grafikte:

* Mavi: Gerçek fiyat
* Turuncu: LSTM tahmini

---

## 📉 RSI ve MACD ile Al/Sat Stratejisi

LSTM tahmininden bağımsız olarak, teknik analiz indikatörleri kullanılarak bir al/sat stratejisi geliştirilmiştir.

### RSI (Relative Strength Index)

* RSI < 30 → Aşırı satım → AL sinyali
* RSI > 70 → Aşırı alım → SAT sinyali

### MACD (Moving Average Convergence Divergence)

* MACD, sinyal çizgisini yukarı keserse → AL
* MACD, sinyal çizgisini aşağı keserse → SAT

Bu iki indikatör birlikte kullanılarak daha güvenilir sinyaller üretilmiştir.

### 📌 Al/Sat Sinyalleri Görselleştirme

![RSI MACD](images/rsi.jpg)

---

## 🔁 Backtesting (Strateji Performansı)

Oluşturulan al/sat stratejisi geçmiş veriler üzerinde test edilmiştir.

Backtest ile:

* Toplam kâr/zarar
* İşlem sayısı
* Başarı oranı
* Strateji performansı

hesaplanmıştır.



## 🧪 Proje Akış Şeması

1. Veri çekme
2. Veri ön işleme
3. LSTM ile tahmin
4. RSI & MACD hesaplama
5. Al/Sat sinyali üretme
6. Backtest
7. Görselleştirme

---

## ▶️ Projeyi Çalıştırma

```bash
pip install -r requirements.txt
python main.py
```

---

## 📌 Kazanımlar

Bu proje ile:

* Zaman serisi analizi
* Deep Learning (LSTM)
* Finansal teknik analiz
* Strateji geliştirme
* Backtesting
* Veri görselleştirme

konularında uçtan uca uygulama gerçekleştirilmiştir.

---

## 👩‍💻 Geliştirici

Ayşe Sıla İnci
Yapay Zeka ve Veri Mühendisliği
