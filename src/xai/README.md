# Explainable AI (XAI) Module

This module provides **LIME** and **SHAP** explanations for all phishing detection models in the PhishGuard project.

## 🎯 Features

- **LIME Explanations**: Fast, intuitive explanations for all 5 models
  - Baseline (TF-IDF + Logistic Regression)
  - CNN (Convolutional Neural Network)
  - LSTM (Long Short-Term Memory)
  - BERT (DistilBERT fine-tuned)
  - Hybrid (CNN+LSTM+URL features)

- **SHAP Explanations**: Theoretically-grounded explanations
  - KernelExplainer for baseline model
  - GradientExplainer for Keras models
  - Computationally intensive but more accurate

- **Interactive Streamlit UI**: Web interface for generating and viewing explanations

- **Artifact Management**: All explanations saved as HTML/JSON with metadata

## 📦 Installation

Install required dependencies:

```bash
pip install lime shap matplotlib
```

Existing dependencies (should already be installed):
- tensorflow
- transformers
- torch
- joblib
- numpy
- pandas
- streamlit

## 🚀 Quick Start

### 1. Generate Example Explanations

```bash
python scripts/generate_example_explanations.py
```

This will create LIME explanations for all models and save them to `artifacts/explanations/`.

### 2. Launch XAI Interface

```bash
streamlit run src/app/streamlit_xai.py
```

Open your browser to the displayed URL (typically http://localhost:8501).

### 3. Use in Code

```python
from src.xai.lime_baseline import LimeBaseline

# Initialize explainer
explainer = LimeBaseline()

# Generate explanation
html = explainer.explain_html(
    text="Your account requires verification...",
    num_features=10,
    num_samples=500
)

# View top features
features, prediction, probability = explainer.explain_to_list(
    text="Your account requires verification...",
    num_features=10
)

for feature, weight in features[:5]:
    print(f"{feature:20s}: {weight:+.4f}")
```

## 📚 Module Structure

```
src/xai/
├── __init__.py                 # Package initialization
├── utils.py                    # Common utilities
├── lime_baseline.py            # LIME for TF-IDF+LR
├── lime_keras.py               # LIME for CNN/LSTM
├── lime_bert.py                # LIME for BERT
├── lime_hybrid.py              # LIME for Hybrid model
├── shap_baseline.py            # SHAP for baseline
└── shap_keras.py               # SHAP for Keras models

src/app/
└── streamlit_xai.py            # XAI web interface

scripts/
└── generate_example_explanations.py  # Example generator

tests/
├── test_lime_baseline.py       # Unit tests for LIME
├── test_predict_proba_wrapper.py # Unit tests for predictions
└── run_tests.py                # Test runner
```

## 🔬 LIME Usage

### Baseline Model

```python
from src.xai.lime_baseline import explain_baseline

html = explain_baseline(
    text="Click here to verify your account",
    num_features=10,
    num_samples=500,
    save_artifacts=True
)
```

### CNN/LSTM Models

```python
from src.xai.lime_keras import explain_cnn, explain_lstm

# CNN
html = explain_cnn(text="...", num_features=10, num_samples=500)

# LSTM
html = explain_lstm(text="...", num_features=10, num_samples=500)
```

### BERT Model

```python
from src.xai.lime_bert import explain_bert

html = explain_bert(
    text="...",
    num_features=10,
    num_samples=500
)
```

### Hybrid Model

```python
from src.xai.lime_hybrid import explain_hybrid

html = explain_hybrid(
    text="...",
    num_features=10,
    num_samples=500
)
```

**Note**: For hybrid model, LIME only explains text features. URL features are held constant during perturbation.

## 📈 SHAP Usage

### Baseline Model

```python
from src.xai.shap_baseline import compute_shap_baseline

results = compute_shap_baseline(
    samples=["Email 1", "Email 2"],
    nsamples=200,
    save_artifacts=True
)

print(f"HTML saved to: {results['html_path']}")
```

### Keras Models

```python
from src.xai.shap_keras import compute_shap_cnn, compute_shap_lstm

# CNN
results = compute_shap_cnn(
    samples=["Email 1"],
    method="gradient",  # or "deep"
    save_artifacts=True
)

# LSTM
results = compute_shap_lstm(
    samples=["Email 1"],
    method="gradient",
    save_artifacts=True
)
```

**Warning**: SHAP is computationally expensive. Expect 2-5 minutes per sample.

## ⚙️ Configuration

### LIME Parameters

- **num_features** (default: 10): Number of top features to show
  - More features = more detail but busier visualization
  - Recommended: 5-15

- **num_samples** (default: 500): Number of perturbed samples
  - More samples = more accurate but slower
  - Recommended: 100-1000

### SHAP Parameters

- **nsamples** (default: 200): Number of kernel samples
  - More samples = more accurate but much slower
  - Recommended: 100-500

- **background** (optional): Background dataset for KernelExplainer
  - If None, uses first samples from input
  - Recommended size: 20-50 samples

## 📁 Output Artifacts

All explanations are saved to `artifacts/explanations/` with timestamps:

### LIME Artifacts

- `lime_<model>_<timestamp>.html` - Interactive HTML visualization
- `lime_<model>_<timestamp>_meta.json` - Metadata including:
  - Model name
  - Timestamp
  - Prediction and probability
  - Top contributing features
  - Text preview

### SHAP Artifacts

- `shap_<model>_<timestamp>.html` - Force plot HTML
- `shap_<model>_<timestamp>.png` - Static plot image
- `shap_<model>_<timestamp>_meta.json` - Metadata

### Example Metadata

```json
{
  "model": "baseline_tfidf_lr",
  "timestamp": "2025-11-08T14:30:45.123456",
  "label": "phishing",
  "prediction": 1,
  "probability": 0.9234,
  "text_preview": "Dear Customer, Your account requires immediate...",
  "text_length": 456,
  "num_features": 10,
  "num_samples": 500,
  "top_tokens": [
    ["verify", 0.4251],
    ["immediate", 0.3892],
    ["account", 0.3421],
    ["suspension", 0.2876],
    ["click", 0.2543]
  ]
}
```

## 🧪 Testing

Run unit tests:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test
python -m unittest tests.test_lime_baseline
python -m unittest tests.test_predict_proba_wrapper
```

Tests verify:
- Correct probability format (Nx2, sum to 1, in [0,1])
- HTML generation contains expected elements
- Consistent predictions
- Feature extraction works correctly

## ⚡ Performance Considerations

### LIME
- **Speed**: ~10-30 seconds per explanation
- **Memory**: Low (~100 MB)
- **Recommended samples**: 500-1000
- **Batch size**: Can explain multiple emails sequentially

### SHAP
- **Speed**: 2-5 minutes per sample (baseline), 5-15 minutes (Keras)
- **Memory**: High (~1-2 GB)
- **Recommended samples**: 1-3 at a time
- **Batch size**: Limited by memory

### Optimization Tips

1. **Use LIME for real-time explanations** - Much faster
2. **Use SHAP offline** - Run batch jobs for multiple samples
3. **Reduce num_samples** - For LIME, 100-200 samples often sufficient
4. **Reduce nsamples** - For SHAP, 100 kernel samples acceptable
5. **Use gradient method** - For Keras SHAP, gradient is faster than deep
6. **Limit background size** - Keep to 20-40 samples for SHAP

## 🚨 Known Limitations

1. **Hybrid Model LIME**: Only explains text features, URL features held constant
2. **BERT SHAP**: Not implemented (too computationally expensive)
3. **Hybrid SHAP**: Not implemented (complex multi-input architecture)
4. **CPU Performance**: SHAP on CPU is slow, GPU recommended for production
5. **Memory**: Large SHAP computations may require 8GB+ RAM

## 🛠️ Troubleshooting

### "Model not found" Error
```bash
# Train the model first
python src/models/train_tfidf_lr.py      # Baseline
python src/models/train_cnn_keras.py      # CNN
python src/models/train_lstm_keras.py     # LSTM
python src/models/train_bert.py           # BERT
python src/models/train_hybrid_model.py   # Hybrid
```

### SHAP Taking Too Long
- Reduce `nsamples` to 50-100
- Reduce number of samples to 1-2
- Use LIME instead for faster explanations
- Consider running on GPU

### Out of Memory
- Reduce background dataset size
- Reduce number of samples
- Close other applications
- Use gradient method instead of deep

### HTML Not Displaying
- Check browser console for errors
- Try opening saved HTML files directly
- Ensure JavaScript is enabled
- Try different browser

## 📖 References

- LIME Paper: ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
- SHAP Paper: [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- LIME GitHub: https://github.com/marcotcr/lime
- SHAP GitHub: https://github.com/slundberg/shap

## 📝 License

Same as main project (see root LICENSE file).

## 🤝 Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Update documentation
3. Follow existing code style
4. Test with all model types

## 📧 Support

For issues or questions:
- Open GitHub issue
- Check existing test cases for examples
- Review saved artifacts in `artifacts/explanations/`
