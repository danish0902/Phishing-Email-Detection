# PhishGuard-XAI: Explainable Deep Learning for Phishing Email Detection

This repo provides a complete, *from-scratch* implementation to detect phishing emails with LSTM/CNN/BERT,
plus **explainability** (LIME/SHAP/attention) and a **Streamlit** app.

## Quickstart (Baseline: TF‑IDF + Logistic Regression)
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Train baseline model on the tiny sample dataset
python src/models/train_tfidf_lr.py --data data/sample_emails.csv --text_col text --label_col label

# Evaluate and plot confusion matrix
python src/models/evaluate.py --data data/sample_emails.csv --text_col text --label_col label

# Run the app (loads baseline model by default)
streamlit run src/app/streamlit_app.py
```
Artifacts (vectorizer + model) are saved under `artifacts/`.

## Datasets (Recommended for real experiments)
- Hugging Face: `zefang-liu/phishing-email-dataset`
- SpamAssassin, Enron, Nazario, CSDMC2010 (see `src/data/download_datasets.py` for scripts)

## Training Deep Models
- `src/models/train_lstm_keras.py` — LSTM with (optional) GloVe embeddings
- `src/models/train_cnn_keras.py` — 1D CNN
- `src/models/train_bert.py` — DistilBERT fine-tuning (transformers)

## Explainability
- LIME for local word importance (baseline + deep models)
- SHAP for model explanations (baseline + Keras)
- Attention visualization (for BiLSTM with attention)

## App
The Streamlit app provides:
- Email input
- Prediction (phishing / not phishing)
- LIME explanation (highlight tokens)
- URL checks via VirusTotal / PhishTank APIs (optional; set API keys via env vars)

## Project Structure
```
phishguard_xai/
├── data/                    # datasets (sample provided)
├── artifacts/               # saved models & vectorizers
├── src/
│   ├── data/                # dataset loaders / downloaders
│   ├── features/            # preprocessing & utilities
│   ├── models/              # training & evaluation scripts
│   ├── xai/                 # explainability helpers
│   └── app/                 # streamlit app
├── charts/                  # evaluation plots
├── notebooks/               # optional notebooks
├── requirements.txt
└── README.md
```
