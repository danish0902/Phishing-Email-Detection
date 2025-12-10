# Getting Started with Phishing Email Detection System

Welcome! This guide will help you set up and run the Phishing Email Detection System on your local machine.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the Application](#running-the-application)
4. [Using the Application](#using-the-application)
5. [Training Models (Optional)](#training-models-optional)
6. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

Before you begin, ensure you have the following installed on your system:

### Required Software:
- **Python 3.8 or higher** (Python 3.10 recommended)
  - Download from: https://www.python.org/downloads/
  - ✅ During installation, check "Add Python to PATH"

### System Requirements:
- **RAM**: Minimum 8GB (16GB recommended for BERT model)
- **Storage**: At least 2GB free space
- **OS**: Windows, macOS, or Linux

### Check Your Python Version:
```bash
python --version
```
Expected output: `Python 3.8.x` or higher

---

## 📦 Installation

### Step 1: Clone or Download the Repository

If you have Git installed:
```bash
git clone https://github.com/danish0902/Phishing-Email-Detection.git
cd Phishing-Email-Detection
```

Or download the ZIP file from GitHub and extract it.

---

### Step 2: Create a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your command prompt.

---

### Step 3: Install Required Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⏱️ **This may take 5-10 minutes** depending on your internet speed.

**Expected packages include:**
- TensorFlow/Keras (for CNN, LSTM, Hybrid models)
- Transformers (for BERT model)
- Scikit-learn (for Baseline model)
- Streamlit (for web interface)
- LIME (for explainability)
- And more...

---

### Step 4: Verify Installation

Check if Streamlit is installed:
```bash
streamlit --version
```

Expected output: `Streamlit, version 1.x.x`

---

## 🚀 Running the Application

**⚠️ PREREQUISITE:** You must train all models first (see [Training Models](#training-models-required) section below).

### Quick Start (Recommended)

**On Windows:**
Double-click `launch_app.bat` in the project folder.

**Or manually run:**
```bash
streamlit run src/app/streamlit_unified.py
```

### What Happens Next:

1. Terminal will show:
   ```
   You can now view your Streamlit app in your browser.
   
   Local URL: http://localhost:8501
   Network URL: http://192.168.x.x:8501
   ```

2. Your default web browser will automatically open to `http://localhost:8501`

3. The app interface will load with two tabs:
   - **⚡ Quick Prediction**: Fast phishing detection
   - **🔍 Explainable AI**: Understand model decisions

⏱️ **First launch may take 30-60 seconds** as models are loaded into memory.

---

## 🎯 Using the Application

### Tab 1: Quick Prediction

1. **Select Models** (left sidebar):
   - Choose one or more models: Baseline, CNN, LSTM, BERT, Hybrid
   - All models are selected by default

2. **Adjust Threshold** (optional):
   - Slider: 0.0 (most sensitive) to 1.0 (most conservative)
   - Default: 0.5

3. **Enter Email Content**:
   - Paste the email text you want to analyze
   - Click **"🔍 Analyze Email"**

4. **View Results**:
   - Each model shows: PHISHING or LEGITIMATE
   - Confidence percentage displayed
   - Results appear in ~2-5 seconds

**Example Test Email (Phishing):**
```
Urgent! Your account has been suspended. Click here to verify your identity immediately.
```

**Example Test Email (Legitimate):**
```
Hi there, thanks for your order! Your package will arrive in 2-3 business days.
```

---

### Tab 2: Explainable AI (LIME)

1. **Select Models** (left sidebar):
   - Choose models to explain (baseline, cnn, lstm, bert, hybrid)

2. **Enter Email Content**:
   - Paste the email text
   - Click **"🔍 Generate Explanations"**

3. **View Explanations**:
   - See which words influenced the prediction
   - **Red words**: Indicate phishing
   - **Green words**: Indicate legitimate
   - Feature importance scores show word impact

⏱️ **LIME explanations take 10-30 seconds per model**

---

## 🎓 Training Models (REQUIRED)

**⚠️ IMPORTANT:** The `artifacts/` folder (containing trained models) is NOT included in the Git repository due to large file sizes. You **MUST** train all models before running the application.

### Dataset Location:
- Training data is automatically loaded from `data/` folder
- The dataset files are included in the repository

### Training Commands:

**1. Baseline Model (TF-IDF + Logistic Regression):**
```bash
python src/models/train_tfidf_lr.py
```
⏱️ Time: 2-5 minutes

**2. CNN Model:**
```bash
python src/models/train_cnn_keras.py
```
⏱️ Time: 15-20 minutes

**3. LSTM Model:**
```bash
python src/models/train_lstm_keras.py
```
⏱️ Time: 25-30 minutes

**4. BERT Model:**
```bash
python src/models/train_bert.py
```
⏱️ Time: 20-40 minutes

**5. Hybrid Model (CNN+LSTM+URL):**
```bash
python src/models/train_hybrid_model.py
```
⏱️ Time: 30-40 minutes

### Training Order (Recommended):

**Train models in this order for best results:**

1. **Baseline** (fastest, ~2-5 min)
2. **CNN** (moderate, ~15-20 min) 
3. **LSTM** (moderate, ~25-30 min)
4. **Hybrid** (needs CNN/LSTM components, ~30-40 min)
5. **BERT** (slowest, ~20-40 min)

### After Training:
- Trained models are saved to `artifacts/` folder
- Models are automatically loaded by the app
- The `artifacts/` folder will be created automatically during training
- Total storage needed: ~500MB-1GB for all models

---

## 🔍 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Baseline | 96.2% | 95.8% | 96.5% | 96.1% |
| CNN | 97.4% | 97.1% | 97.8% | 97.4% |
| LSTM | 98.1% | 97.9% | 98.3% | 98.1% |
| **BERT** | **99.4%** | **99.3%** | **99.5%** | **99.4%** |
| Hybrid | 98.5% | 98.3% | 98.7% | 98.5% |

**Best Model:** BERT (DistilBERT) - Recommended for most accurate predictions

---

## 🛠️ Troubleshooting

### Issue 1: "Python not recognized"
**Solution:** 
- Reinstall Python and check "Add Python to PATH"
- Or use full path: `C:\Python310\python.exe`

---

### Issue 2: "Cannot activate virtual environment"
**On Windows PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try activating again.

---

### Issue 3: "Module not found" errors
**Solution:**
```bash
# Ensure virtual environment is activated
pip install -r requirements.txt --force-reinstall
```

---

### Issue 4: App won't load / Port already in use
**Solution:**
```bash
# Stop the current app (Ctrl+C in terminal)
# Run on different port:
streamlit run src/app/streamlit_unified.py --server.port 8502
```

---

### Issue 5: "Out of memory" when loading BERT
**Solution:**
- Close other applications
- Use lighter models (CNN/LSTM) instead
- Or increase system RAM

---

### Issue 6: Slow predictions
**Expected times:**
- Baseline: <1 second
- CNN: 1-2 seconds
- LSTM: 2-3 seconds
- BERT: 3-5 seconds (first time slower)
- Hybrid: 2-4 seconds

**If slower:**
- First prediction is always slower (model loading)
- LIME explanations take 10-30 seconds (normal)

---

## 📂 Project Structure

```
Phishing-Email-Detection/
├── src/
│   ├── app/                    # Streamlit application
│   │   ├── streamlit_unified.py   # Main app file
│   │   └── model_loader.py        # Model management
│   ├── models/                 # Training scripts
│   ├── features/               # Preprocessing
│   └── xai/                    # Explainability (LIME)
├── data/                       # Dataset files
├── artifacts/                  # Trained models (joblib/h5)
├── requirements.txt            # Python dependencies
└── GETTING_STARTED.md         # This file
```

---

## 🎉 Quick Test

Once the app is running, try this phishing email:

```
URGENT SECURITY ALERT

Your PayPal account has been limited due to suspicious activity.

Click here immediately to verify your identity and restore access:
http://verify-paypal-secure.com/login

You have 24 hours before permanent suspension.

PayPal Security Team
```

**Expected Result:** All models should predict **PHISHING** with high confidence (>95%)

---

## 📚 Additional Resources

- **Detailed Workflow:** See `WORKFLOW_DIAGRAM.md` - Complete system architecture and data flow

---

## 🆘 Need Help?

1. Check the **Troubleshooting** section above
2. Review error messages in the terminal
3. Ensure all prerequisites are installed
4. Check Python version compatibility

---

## 🎯 Next Steps

1. ✅ Complete installation and setup
2. ✅ **Train all 5 models** (Baseline → CNN → LSTM → Hybrid → BERT)
3. ✅ Run the application successfully
4. ✅ Test with sample phishing/legitimate emails
5. ✅ Explore different models and compare results
6. ✅ Try LIME explanations to understand decisions

---

## ⚠️ Important Notes

- **Training Required**: You MUST train all models before running the app (artifacts not in repository)
- **Training Time**: Total ~90-120 minutes for all 5 models
- **First Launch**: Takes 30-60 seconds to load all models after training
- **Model Size**: Trained models will use ~500MB-1GB disk space
- **Internet**: Required for initial package installation and downloading BERT pre-trained weights
- **Data Privacy**: All processing happens locally on your machine

---

## 🏁 Summary

**Minimal steps to get started:**

1. Install Python 3.8+
2. Create virtual environment: `python -m venv .venv`
3. Activate: `.\.venv\Scripts\Activate.ps1` (Windows)
4. Install packages: `pip install -r requirements.txt`
5. **Train all models** (see Training Models section - ~90-120 minutes total)
6. Run app: `streamlit run src/app/streamlit_unified.py`
7. Open browser to `http://localhost:8501`
8. Start detecting phishing emails! 🎉

---

**Happy Phishing Detection! 🎯🔍**
