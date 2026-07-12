# 🛍️ Analisis Sentimen Review Shopee

Aplikasi berbasis **Machine Learning** untuk mengklasifikasikan sentimen review pengguna Shopee ke dalam tiga kategori:

- Positif
- Netral
- Negatif

Model dibangun menggunakan **TF-IDF** sebagai feature extraction dan **Linear Support Vector Machine (Linear SVM)** sebagai algoritma klasifikasi. Aplikasi dideploy menggunakan **Flask** sehingga pengguna dapat melakukan prediksi sentimen secara langsung melalui web.

---

## Project Overview

Analisis sentimen merupakan salah satu teknik Text Mining yang digunakan untuk mengetahui opini pengguna terhadap suatu produk atau layanan.

Pada project ini dilakukan klasifikasi sentimen review aplikasi Shopee yang berasal dari Google Play Store sehingga dapat diketahui apakah review yang diberikan pengguna bersifat positif, netral, maupun negatif.

---

## Objectives

- Mengembangkan model klasifikasi sentimen review Shopee.
- Mengimplementasikan preprocessing teks Bahasa Indonesia.
- Membangun model Machine Learning menggunakan Linear SVM.
- Mengembangkan aplikasi web berbasis Flask untuk prediksi sentimen secara real-time.

---

## 🛠️ Tech Stack

- Python
- Flask
- Scikit-Learn
- TF-IDF Vectorizer
- Linear SVM
- Joblib
- Pandas
- NumPy
- NLTK
- Sastrawi

---

## Dataset

Dataset diperoleh dari review aplikasi Shopee pada Google Play Store.

Dataset berisi:

- Review pengguna
- Label sentimen
    - Positif
    - Netral
    - Negatif

---

## Workflow

```
Google Play Review
        │
        ▼
Case Folding
        │
        ▼
Cleaning
        │
        ▼
Slang Normalization
        │
        ▼
Sentiment Labeling (IndoBERT)
        │
        ▼
Tokenization
        │
        ▼
Stopword Removal
        │
        ▼
Stemming
        │
        ▼
Feature Engineering
(TF-IDF Vectorizer)
        │
        ▼
Train Test Split
        │
        ▼
Modeling (Naive Bayes, Logistic Regression, Linear SVM)
        │
        ▼
Evaluation
        │
        ▼
Deployment (Flask)
```

---

## ⚙️ Machine Learning

Model yang dibandingkan:

- Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

Model terbaik dipilih berdasarkan:

- Accuracy
- Precision
- Recall
- F1-Score

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 🚀 Features

- Input review secara langsung
- Preprocessing otomatis
- Prediksi sentimen
- Confidence Score
- Tampilan web interaktif menggunakan Flask

---

## 📁 Project Structure

```
Klasifikasi-Sentimen-Shopee/
│
├── app.py
├── preprocessing.py
├── model_sentimen.joblib
├── tfidf.joblib
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── notebook.ipynb
└── README.md
```

---

## ▶️ Installation

Clone repository

```bash
git clone https://github.com/USERNAME/Klasifikasi-Sentimen-Shopee.git
```

Masuk ke folder project

```bash
cd Sentimen
```

Install library

```bash
pip install -r requirements.txt
```

Jalankan aplikasi

```bash
python app.py
```

Buka browser

```
http://127.0.0.1:5000
```

---

**Web Demo**

```
https://web-production-de25ce.up.railway.app/
```

