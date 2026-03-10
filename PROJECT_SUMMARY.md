# Phishing Email Detection — Project Summary & AI Context Prompt

> **How to use this file**
> Copy everything from the horizontal rule below to the end of this document and paste it into any AI assistant (ChatGPT, Claude, Gemini, etc.). The AI will then have full context to suggest meaningful next steps, improvements, or extensions for this project.

---

---

## CONTEXT PROMPT — Paste the text below into another AI

I have built a **Phishing Email Detection** system as a research / portfolio project. Here is a complete description of everything that has been implemented. Please suggest concrete, actionable next steps I can take to improve or extend this project.

---

### 1. Project Goal

Build a system that automatically classifies email messages as **phishing** (malicious) or **legitimate**, using multiple machine-learning approaches, with explainability so users can understand *why* an email was flagged.

---

### 2. Dataset

| Property | Value |
|---|---|
| File | `Phishing_Email_Cleaned_NO_DUPLICATES.csv` |
| Total emails (after deduplication) | 18,650 |
| Label column | `Email Type` (0 = Legitimate, 1 = Phishing) |
| Split | 80 % train (14,920) / 10 % val (1,865) / 10 % test (1,865) |
| Split strategy | Stratified, fixed seed, indices stored in `split_indices_CLEAN.csv` |

---

### 3. Text Preprocessing (`src/features/preprocess.py`)

```
Raw email text
  → unescape HTML entities
  → strip HTML tags (BeautifulSoup)
  → replace all URLs with the token <URL>
  → remove characters outside [A-Za-z0-9 @ . _ < > space]
  → lowercase + collapse whitespace
```

Key notes:
- URLs in HTML `href` attributes are stripped by BeautifulSoup before URL replacement (they disappear, but never become `<URL>` tokens).
- Allowed special characters: `@`, `.`, `_`, `<`, `>` (to preserve email addresses and the `<URL>` token).

---

### 4. Models Trained

#### Model 1 — Baseline: TF-IDF + Logistic Regression (`src/models/train_tfidf_lr.py`)

- TfidfVectorizer: `ngram_range=(1,2)`, `max_features=5000`
- LogisticRegression: `max_iter=200`
- **Test accuracy: ~96.2 %**
- Artifact: `artifacts/baseline_tfidf_lr.joblib`

#### Model 2 — CNN (`src/models/train_cnn_keras.py`)

- Embedding (vocab 20 k, dim 128) → Conv1D (128 filters, kernel 3) → GlobalMaxPool1D → Dense 64 → Dropout 0.3 → Dense 1 sigmoid
- `max_len=200`, 10 epochs, batch 32, early stopping on val_loss
- **Test accuracy: ~97.4 %**
- Artifact: `artifacts/cnn_keras.h5` + `artifacts/tokenizer.joblib`

#### Model 3 — Bidirectional LSTM (`src/models/train_lstm_keras.py`)

- Embedding → BiLSTM (128 units) → Dense 64 → Dropout 0.3 → Dense 1 sigmoid
- `max_len=200`, 10 epochs, batch 32, early stopping
- **Test accuracy: ~98.1 %**
- Artifact: `artifacts/lstm_keras.h5` + `artifacts/tokenizer.joblib`

#### Model 4 — DistilBERT (`src/models/train_bert.py`)

- Pre-trained: `distilbert-base-uncased`, fine-tuned for 2-class classification
- HuggingFace `Trainer` API, `max_len=256`, 1 epoch, `lr=5e-5`, batch 8
- Metric optimised: F1; evaluated per epoch
- **Test accuracy: ~99.4 %** (best model)
- Artifact: `artifacts/bert_model/` (HuggingFace SavedModel directory)

#### Model 5 — Hybrid CNN+LSTM+URL (`src/models/train_hybrid_model.py`)

- **Text branch**: Embedding → Conv1D (64) → MaxPool → LSTM (64) → Dense 32 → Dropout 0.5
- **URL feature branch** (6 hand-crafted features):
  1. URL length
  2. Number of dots in URL
  3. Has IP address (regex flag)
  4. Count of `@` and `-` characters
  5. Contains suspicious keywords (login / update / verify / bank / secure / account / confirm)
  6. Uses HTTPS (flag)
  - StandardScaler fitted on training set → Dense 16 → Dense 8
- **Fusion**: concatenate both branches → Dense 32 → Dense 16 → Dense 1 sigmoid
- Multi-input Keras `Model`, `max_len=100`, 5 epochs, batch 32, early stopping
- **Test accuracy: ~98.5 %**
- Artifacts: `artifacts/hybrid_cnn_lstm.h5`, `artifacts/tokenizer.joblib`, `artifacts/url_scaler.joblib`, `artifacts/hybrid_config.joblib`

#### Performance Summary

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| TF-IDF + LR (Baseline) | 96.2 % | 95.8 % | 96.5 % | 96.1 % |
| CNN | 97.4 % | 97.1 % | 97.8 % | 97.4 % |
| Bidirectional LSTM | 98.1 % | 97.9 % | 98.3 % | 98.1 % |
| **DistilBERT** | **99.4 %** | **99.3 %** | **99.5 %** | **99.4 %** |
| Hybrid CNN+LSTM+URL | 98.5 % | 98.3 % | 98.7 % | 98.5 % |

---

### 5. Model Serving & Inference (`src/app/model_loader.py`)

A unified `ModelLoader` class provides:

- `predict_tfidf(text)` → probability float
- `predict_cnn(text)` → probability float
- `predict_lstm(text)` → probability float
- `predict_bert(text)` → probability float
- `predict_hybrid(text)` → probability float
- `predict(text, model_type)` → dispatcher
- `predict_ensemble(text, models)` → majority vote over chosen models, returns dict with probability, confidence, per-model votes
- `load_thresholds()` → loads per-model optimal thresholds from `artifacts/optimal_thresholds.joblib` (falls back to 0.5)
- `get_system_info()` → returns list of successfully loaded models
- `evaluate_on_testset(sample_size, models)` → evaluates ensemble on a test CSV sample

---

### 6. Explainability / XAI (`src/xai/`)

#### LIME (all 5 models)

- `lime_baseline.py` — `LimeTextExplainer` wrapping the TF-IDF pipeline
- `lime_keras.py` — LIME for CNN and LSTM Keras models
- `lime_bert.py` — LIME for DistilBERT
- `lime_hybrid.py` — LIME for the hybrid model (text branch only; URL features held constant during perturbation)
- All expose: `explain(text)`, `explain_html(text, save_artifacts=True)`, `explain_to_list(text)` methods
- Saves timestamped HTML + JSON metadata to `artifacts/explanations/`

#### SHAP (baseline model only)

- `shap_baseline.py` — `shap.LinearExplainer` on the TF-IDF+LR model
- Provides exact (non-sampled) Shapley values; fast because the classifier is linear
- Saves self-contained HTML bar chart + JSON metadata
- **SHAP for CNN, LSTM, BERT, and Hybrid is NOT implemented** (computationally expensive; noted as future work)

#### XAI Utilities (`src/xai/utils.py`)

`save_html`, `save_json`, `safe_predict_proba_batch`, `generate_timestamp`, `truncate_text`, `create_explanation_metadata`, `normalize_probabilities`

---

### 7. Streamlit Web Application (`src/app/streamlit_unified.py`)

Two-tab interface:

**Tab 1 — Quick Prediction**
- Sidebar: model selector (tfidf / cnn / lstm / bert / hybrid / ensemble), decision threshold slider
- Input: email text area
- Output: phishing / legitimate verdict, probability gauge, per-model breakdown for ensemble
- URL analysis: extracts URLs from the email, checks them via VirusTotal API (requires `VT_API_KEY` in `.env`)

**Tab 2 — Explainable AI (XAI)**
- Model selector for LIME / SHAP explanations
- Renders explanation HTML inline in the Streamlit page
- Shows top feature weights as a table

Models loaded with `@st.cache_resource` for performance.

---

### 8. URL Analysis (`src/app/urls.py`)

- `extract_urls(text)` — regex extraction, normalisation, trailing-punctuation stripping, skips reserved/test domains
- `check_virustotal(url)` — VirusTotal API v3 (POST scan + GET report), handles rate limiting (429), auth errors (401), timeouts
- `check_phishtank(url)` — PhishTank API check
- Both require API keys in `.env`: `VT_API_KEY`, `PHISHTANK_API_KEY`

---

### 9. REST API (`src/api/main.py`)

Built with **FastAPI**. Run with: `uvicorn src.api.main:app --reload`

| Method | Path | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Lists loaded models |
| GET | `/models` | Lists all supported model IDs |
| POST | `/predict` | Single-email prediction; params: `email_text`, `model` (default: tfidf), `threshold` (default: 0.5) |
| POST | `/predict/batch` | Batch prediction; up to 100 emails; same params |

Error semantics: 422 for invalid input, 503 when model artifact missing, 500 for unexpected errors.
Response includes: `is_phishing`, `probability`, `label`, `model_used`, `cleaned_text_preview`.

---

### 10. Batch Prediction CLI (`src/models/batch_predict.py`)

```
python src/models/batch_predict.py \
    --input data/test_set.csv \
    --output results.csv \
    --model tfidf \
    --text_col "Email Text" \
    --threshold 0.5
```

Appends three columns to the output CSV: `pred_probability`, `pred_label`, `pred_error`.
Supports all 6 model types including ensemble. Shows throughput (emails/s) progress.

---

### 11. Data Utilities

- `src/data/create_train_test_split.py` — stratified 80/10/10 split, saves `train_set.csv`, `val_set.csv`, `test_set.csv`, `split_indices_CLEAN.csv`
- `src/data/download_datasets.py` — HuggingFace `datasets` loader for alternative data sources
- `scripts/generate_dataset_graphs.py` — produces 5 charts (class distribution, email length, data quality, etc.) saved to `charts/dataset_analysis/`
- `scripts/generate_model_comparison_chart.py` — produces performance comparison charts

---

### 12. Unit Tests (`tests/` — 92 tests, all passing)

| File | Covers | Tests |
|---|---|---|
| `test_preprocess.py` | `clean_text`, `strip_html` | 19 |
| `test_urls.py` | URL extraction, VirusTotal, PhishTank (mocked) | 18 |
| `test_xai_utils.py` | File I/O, batching, metadata, normalisation | 24 |
| `test_batch_predict.py` | Batch prediction CLI (mocked model) | 9 |
| `test_api.py` | FastAPI endpoints (mocked model) | 22 |

Run: `pytest tests/`

---

### 13. Dependencies (`requirements.txt`)

```
numpy, pandas, scikit-learn, nltk, tldextract, beautifulsoup4, joblib, matplotlib
tensorflow>=2.15, torch, transformers, datasets
lime, shap
streamlit, requests, python-dotenv
fastapi, uvicorn[standard]
pytest, pytest-cov, httpx
```

---

### 14. What Is NOT Implemented (Known Gaps)

1. **SHAP for deep models** — SHAP explanations only exist for the TF-IDF+LR baseline. CNN, LSTM, BERT, and Hybrid have no SHAP support.
2. **Hyperparameter optimisation** — All models use fixed hyperparameters. No grid search, random search, or Bayesian optimisation.
3. **Cross-validation** — Only a single train/val/test split; no k-fold cross-validation.
4. **Data augmentation** — No text augmentation (back-translation, synonym replacement, etc.).
5. **Class imbalance handling** — No SMOTE, class weights, or oversampling analysis.
6. **Model versioning / registry** — No MLflow, DVC, or similar experiment tracking.
7. **Docker containerisation** — No Dockerfile; deployment requires manual environment setup.
8. **CI/CD pipelines** — No GitHub Actions workflows for automated testing or deployment.
9. **Prediction history / database** — No persistence layer; predictions are stateless.
10. **User authentication** — API and Streamlit app have no auth/rate-limiting.
11. **Monitoring / drift detection** — No production monitoring for model performance over time.
12. **Multilingual support** — Models trained only on English emails.
13. **Header / metadata features** — Only email body text is used; sender, subject, routing headers are ignored.
14. **Threshold optimisation script** — `ModelLoader` loads per-model thresholds from `artifacts/optimal_thresholds.joblib` if it exists, but there is no script that *generates* those optimal thresholds (e.g. via ROC-curve analysis on the validation set). The system falls back to 0.5 for every model until that file is created.
15. **Async processing** — LIME explanations block the Streamlit thread (no background tasks).
16. **Confidence calibration** — Raw sigmoid/softmax probabilities are used without Platt scaling or isotonic regression.
17. **BERT SHAP** — No implementation (architecturally complex; very slow on CPU).
18. **Hybrid SHAP** — Not implemented due to multi-input model architecture complexity.

---

### 15. Repository Metadata Gaps

- No `LICENSE` file — the project has no declared open-source licence.
- No `CONTRIBUTING.md` — no contribution guidelines for collaborators.

---

### 16. Repository Layout

```
Phishing-Email-Detection/
├── data/
│   ├── Phishing_Email_Cleaned_NO_DUPLICATES.csv   (18,650 rows)
│   ├── train_set.csv / val_set.csv / test_set.csv
│   └── split_indices_CLEAN.csv
├── src/
│   ├── api/          main.py            (FastAPI REST API)
│   ├── app/          model_loader.py    (unified inference)
│   │                 streamlit_unified.py (web UI)
│   │                 streamlit_multimodel.py
│   │                 urls.py            (VirusTotal / PhishTank)
│   ├── data/         create_train_test_split.py, download_datasets.py
│   ├── features/     preprocess.py
│   ├── models/       train_tfidf_lr.py, train_cnn_keras.py,
│   │                 train_lstm_keras.py, train_bert.py,
│   │                 train_hybrid_model.py,
│   │                 evaluate.py, evaluate_final.py,
│   │                 batch_predict.py
│   └── xai/          lime_baseline.py, lime_keras.py, lime_bert.py,
│                     lime_hybrid.py, shap_baseline.py, utils.py
├── tests/            test_preprocess.py, test_urls.py,
│                     test_xai_utils.py, test_batch_predict.py,
│                     test_api.py
├── scripts/          generate_dataset_graphs.py,
│                     generate_model_comparison_chart.py
├── charts/           confusion_matrices/, dataset_analysis/
├── requirements.txt
├── GETTING_STARTED.md
└── WORKFLOW_DIAGRAM.md
```

---

**Given all of the above, what are the most valuable next steps I should take to improve, extend, or productionise this phishing email detection project? Please be specific and prioritise your suggestions.**
