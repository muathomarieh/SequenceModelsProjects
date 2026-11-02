# 🔁 Sequence Models — RNN / LSTM Projects (NLP & Time Series)

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗-Datasets-yellow)

Three end-to-end sequence-modelling projects covering **NLP classification, text generation and multivariate forecasting**, built in both **PyTorch** and **TensorFlow/Keras**.

| Project | Task | Framework | Data |
|---|---|---|---|
| [Sentiment Classifier](RNN/sentiment_class/src) | Binary sentiment analysis | **PyTorch** | IMDB — 50,000 movie reviews (Hugging Face) |
| [Char-level Text Generation](RNN/char_based) | Generative language model | TensorFlow | Shakespeare corpus |
| [Climate Forecasting](RNN/temp_time_series) | Multivariate time-series regression | TensorFlow | Daily Delhi climate (Kaggle) |

---

## 1. 🎬 IMDB Sentiment Classifier — PyTorch

Clean, modular PyTorch project (`dataset.py` / `model.py` / `sentiment_classification.py`).

- Custom tokenizer + vocabulary (35k words, sequences padded/truncated to 300 tokens)
- **Pre-trained GloVe embeddings (200-d)** loaded via `gensim`, fine-tuned during training
- **2-layer LSTM** (hidden 128, dropout 0.5) → linear head, `BCEWithLogitsLoss`, Adam
- Custom `Dataset` / `DataLoader`, CUDA/CPU device handling
- **Interactive CLI**: type any review and get *Positive / Negative* live

```bash
cd RNN/sentiment_class/src
python sentiment_classification.py
```

## 2. ✍️ Character-level Shakespeare Generator — TensorFlow

- Embedding(256) → **stacked LSTM (2 × 512)** with dropout → softmax over characters
- `tf.data` pipeline creating 100-char input/target windows
- Text generation with **temperature scaling** and **top-k sampling**

> *Sample output:* “Of this account the golden play of them.”

## 3. 🌡️ Multivariate Climate Forecasting — TensorFlow

- Predicts **4 variables at once** (mean temp, humidity, wind speed, pressure) from the previous **30 days**
- Z-score normalisation fitted on training data only (no leakage)
- `timeseries_dataset_from_array` windowing → **LSTM(256)** → Dense(4)
- Predictions de-normalised and plotted against ground truth on the test set

---

## 🚀 Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---
👤 **Muath Omarieh** — CS student @ University of Jordan · [GitHub](https://github.com/muathomarieh) · [LinkedIn](https://www.linkedin.com/in/muath-omarieh-550679297/)
