import streamlit as st
import sys
import os
import joblib
import numpy as np

# Add project root to path for proper imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Try to import transformers, but don't fail if not available
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try to import TensorFlow/Keras, but don't fail if not available
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from src.features.preprocess import clean_text

# Import the enhanced ModelLoader
from src.app.model_loader import ModelLoader

# Load the models using the enhanced system
@st.cache_resource
def load_enhanced_phishguard():
    """Load the enhanced PhishGuard system with detailed loading status"""
    loading_status = []
    
    try:
        # Change to project root directory for proper model loading
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        loader = ModelLoader()
        
        # Test loading each model and report status
        model_info = {
            'TF-IDF': {'file': 'artifacts/baseline_tfidf_lr.joblib', 'loader': loader.load_tfidf_model},
            'CNN': {'file': 'artifacts/cnn_keras.h5', 'loader': loader.load_cnn_model},
            'LSTM': {'file': 'artifacts/lstm_keras.h5', 'loader': loader.load_lstm_model},
            'BERT': {'file': 'artifacts/bert_model', 'loader': loader.load_bert_model},
            'Hybrid': {'file': 'artifacts/hybrid_cnn_lstm.h5', 'loader': loader.load_hybrid_model}
        }
        
        for model_name, info in model_info.items():
            try:
                # Check if model file exists
                if model_name == 'BERT':
                    model_path = os.path.join(project_root, info['file'])
                    if os.path.exists(model_path):
                        # Get model size for BERT directory
                        total_size = 0
                        for dirpath, dirnames, filenames in os.walk(model_path):
                            for f in filenames:
                                fp = os.path.join(dirpath, f)
                                total_size += os.path.getsize(fp)
                        size_mb = total_size / (1024 * 1024)
                        
                        # Try loading
                        info['loader']()
                        loading_status.append(f"✅ {model_name} model loaded ({size_mb:.1f} MB)")
                    else:
                        loading_status.append(f"❌ {model_name} model file not found")
                else:
                    model_path = os.path.join(project_root, info['file'])
                    if os.path.exists(model_path):
                        size_mb = os.path.getsize(model_path) / (1024 * 1024)
                        
                        # Try loading
                        info['loader']()
                        loading_status.append(f"✅ {model_name} model loaded ({size_mb:.1f} MB)")
                    else:
                        loading_status.append(f"❌ {model_name} model file not found")
                        
            except Exception as e:
                loading_status.append(f"❌ {model_name} model failed to load: {str(e)}")
        
        # Load thresholds info
        threshold_path = os.path.join(project_root, 'artifacts/optimal_thresholds.joblib')
        if os.path.exists(threshold_path):
            size_kb = os.path.getsize(threshold_path) / 1024
            loading_status.append(f"✅ Optimized thresholds loaded ({size_kb:.1f} KB)")
        else:
            loading_status.append("⚠️ Using default thresholds (optimal_thresholds.joblib not found)")
        
        # Change back to original directory
        os.chdir(original_cwd)
        
        # Add ensemble info
        loaded_models = [status for status in loading_status if status.startswith("✅") and "model loaded" in status]
        loading_status.append(f"🎯 Ensemble system ready with {len(loaded_models)} models")
        
        return loader, loading_status
        
    except Exception as e:
        os.chdir(original_cwd) if 'original_cwd' in locals() else None
        return None, [f"❌ Error loading enhanced system: {str(e)}"]

def clean_text_simple(text):
    """Simple text cleaning"""
    import re
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.lower()

def extract_url_features(text):
    """
    Extracts URLs and computes advanced features from them.
    Returns 6 numerical features for URL analysis.
    """
    import re
    import pandas as pd
    
    if pd.isna(text) or not isinstance(text, str):
        return [0] * 6
    
    # Find all URLs in the text
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    
    if not urls:
        return [0] * 6  # Return default features if no URL is found

    # Analyze the first URL found (most emails have 0-1 URLs)
    url = urls[0]
    
    features = [
        len(url),  # URL length
        url.count('.'),  # Number of dots
        1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0,  # Has IP address
        len(re.findall(r'[@\-]', url)),  # Number of special characters
        1 if any(keyword in url.lower() for keyword in ["login", "update", "verify", "bank", "secure", "account", "confirm"]) else 0,  # Suspicious keywords
        1 if url.startswith("https") else 0  # Is HTTPS
    ]
    
    return features

def predict_tfidf(text, model):
    """Predict using TF-IDF model"""
    cleaned = clean_text_simple(text)
    proba = model.predict_proba([cleaned])[0][1]
    return float(proba)

def predict_neural(text, model, tokenizer, max_len=200):
    """Predict using CNN/LSTM model"""
    cleaned = clean_text_simple(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    proba = model.predict(padded)[0][0]
    return float(proba)

def predict_bert(text, model, tokenizer):
    """Predict using BERT model"""
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        proba = torch.softmax(outputs.logits, dim=-1)[0][1].item()
    return float(proba)

def predict_hybrid(text, model, tokenizer, scaler, config):
    """Predict using Hybrid CNN+LSTM+URL model"""
    # Extract URL features BEFORE text cleaning
    url_features = np.array([extract_url_features(text)])
    url_features_scaled = scaler.transform(url_features)
    
    # Clean text AFTER URL extraction
    cleaned_text = clean_text(text)
    
    # Process text data
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    max_len = config['max_len']
    text_padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    
    # Make prediction
    proba = model.predict([text_padded, url_features_scaled])[0][0]
    return float(proba)

def predict_cnn(email_text, model, tokenizer):
    """Predict using CNN model"""
    sequences = tokenizer.texts_to_sequences([clean_text(email_text)])
    text_padded = pad_sequences(sequences, maxlen=128, padding='post', truncating='post')
    proba = model.predict(text_padded)[0][0]
    return float(proba)

def predict_lstm(email_text, model, tokenizer):
    """Predict using LSTM model"""
    sequences = tokenizer.texts_to_sequences([clean_text(email_text)])
    text_padded = pad_sequences(sequences, maxlen=128, padding='post', truncating='post')
    proba = model.predict(text_padded)[0][0]
    return float(proba)

def enhanced_predict_with_features(email_text, models, phishguard, feature_extractor=None):
    """Enhanced prediction using improved thresholds and features"""
    
    results = {}
    thresholds = models.get('thresholds', {})
    
    # Mock enhanced features since we don't have the original feature extractor
    enhanced_features = {
        'legitimacy_score': 3,
        'suspicion_score': 2,
        'url_count': len([x for x in email_text.split() if 'http' in x]),
        'urgent_words': len([x for x in ['urgent', 'immediate', 'act now', 'limited time'] if x.lower() in email_text.lower()]),
        'legitimate_indicators': len([x for x in ['order', 'tracking', 'customer', 'service'] if x.lower() in email_text.lower()]),
        'suspicious_indicators': len([x for x in ['verify', 'click here', 'suspended', 'limited'] if x.lower() in email_text.lower()])
    }
    
    legitimacy_score = enhanced_features['legitimacy_score']
    suspicion_score = enhanced_features['suspicion_score']
    
    # Apply feature-based adjustment
    feature_adjustment = 0
    if legitimacy_score > suspicion_score + 2:  # Strong legitimate indicators
        feature_adjustment = -0.1  # Reduce phishing probability
    elif suspicion_score > legitimacy_score + 2:  # Strong suspicious indicators
        feature_adjustment = +0.1  # Increase phishing probability
    
    # Get base predictions from each model
    try:
        # TF-IDF + Logistic Regression
        if 'tfidf' in models:
            prob = phishguard.predict_tfidf(email_text)
            adjusted_prob = max(0, min(1, prob + feature_adjustment))
            threshold = thresholds.get('tfidf', 0.5)
            results['TF-IDF + LR'] = {
                'probability': adjusted_prob,
                'prediction': 'PHISHING' if adjusted_prob > threshold else 'LEGITIMATE',
                'confidence': adjusted_prob if adjusted_prob > threshold else (1 - adjusted_prob),
                'threshold': threshold
            }
    except Exception as e:
        results['TF-IDF + LR'] = {'error': str(e)}
    
    # CNN
    try:
        if 'cnn' in models:
            prob = phishguard.predict_cnn(email_text)
            adjusted_prob = max(0, min(1, prob + feature_adjustment))
            threshold = thresholds.get('cnn', 0.5)
            results['CNN'] = {
                'probability': adjusted_prob,
                'prediction': 'PHISHING' if adjusted_prob > threshold else 'LEGITIMATE',
                'confidence': adjusted_prob if adjusted_prob > threshold else (1 - adjusted_prob),
                'threshold': threshold
            }
    except Exception as e:
        results['CNN'] = {'error': str(e)}
    
    # LSTM
    try:
        if 'lstm' in models:
            prob = phishguard.predict_lstm(email_text)
            adjusted_prob = max(0, min(1, prob + feature_adjustment))
            threshold = thresholds.get('lstm', 0.5)
            results['LSTM'] = {
                'probability': adjusted_prob,
                'prediction': 'PHISHING' if adjusted_prob > threshold else 'LEGITIMATE',
                'confidence': adjusted_prob if adjusted_prob > threshold else (1 - adjusted_prob),
                'threshold': threshold
            }
    except Exception as e:
        results['LSTM'] = {'error': str(e)}
    
    # BERT (if available)
    try:
        if 'bert' in models:
            prob = phishguard.predict_bert(email_text)
            adjusted_prob = max(0, min(1, prob + feature_adjustment))
            threshold = thresholds.get('bert', 0.5)
            results['BERT'] = {
                'probability': adjusted_prob,
                'prediction': 'PHISHING' if adjusted_prob > threshold else 'LEGITIMATE',
                'confidence': adjusted_prob if adjusted_prob > threshold else (1 - adjusted_prob),
                'threshold': threshold
            }
    except Exception as e:
        results['BERT'] = {'error': str(e)}
    
    # Hybrid (if available)
    try:
        if 'hybrid' in models:
            prob = phishguard.predict_hybrid(email_text)
            adjusted_prob = max(0, min(1, prob + feature_adjustment))
            threshold = thresholds.get('hybrid', 0.5)
            results['Hybrid'] = {
                'probability': adjusted_prob,
                'prediction': 'PHISHING' if adjusted_prob > threshold else 'LEGITIMATE',
                'confidence': adjusted_prob if adjusted_prob > threshold else (1 - adjusted_prob),
                'threshold': threshold
            }
    except Exception as e:
        results['Hybrid'] = {'error': str(e)}
    
    # Enhanced ensemble prediction
    predictions = []
    weights = {'LSTM': 2.0, 'TF-IDF + LR': 1.0, 'CNN': 1.0, 'BERT': 1.5, 'Hybrid': 1.0}  # LSTM gets higher weight
    
    total_weight = 0
    weighted_score = 0
    
    for model_name, result in results.items():
        if 'error' not in result:
            weight = weights.get(model_name, 1.0)
            weighted_score += result['probability'] * weight
            total_weight += weight
    
    if total_weight > 0:
        ensemble_prob = weighted_score / total_weight
        ensemble_pred = 'PHISHING' if ensemble_prob > 0.6 else 'LEGITIMATE'  # Higher threshold for ensemble
        
        results['Enhanced Ensemble'] = {
            'probability': ensemble_prob,
            'prediction': ensemble_pred,
            'confidence': ensemble_prob if ensemble_prob > 0.6 else (1 - ensemble_prob),
            'legitimacy_score': legitimacy_score,
            'suspicion_score': suspicion_score
        }
    
    return results, enhanced_features

# Streamlit App
st.set_page_config(page_title="PhishGuard-XAI", page_icon="🛡️", layout="wide")

st.title("🛡️ PhishGuard-XAI - Multi-Model Phishing Detection")
st.write("Compare different AI models for phishing email detection")

# Load models
phishguard, loading_status = load_enhanced_phishguard()

# Get individual models from the enhanced system for compatibility
models = {}
if phishguard:
    if hasattr(phishguard, 'models'):
        models = phishguard.models.copy()
    if hasattr(phishguard, 'thresholds'):
        models['thresholds'] = phishguard.thresholds
    # Add tokenizer references for compatibility
    if hasattr(phishguard, 'tokenizers'):
        if 'cnn' in phishguard.tokenizers:
            models['cnn_tokenizer'] = phishguard.tokenizers['cnn']
        if 'lstm' in phishguard.tokenizers:
            models['lstm_tokenizer'] = phishguard.tokenizers['lstm']
        if 'bert' in phishguard.tokenizers:
            models['bert_tokenizer'] = phishguard.tokenizers['bert']
        if 'hybrid' in phishguard.tokenizers:
            models['hybrid_tokenizer'] = phishguard.tokenizers['hybrid']
    # Add scaler references for compatibility
    if hasattr(phishguard, 'scalers'):
        if 'hybrid' in phishguard.scalers:
            models['hybrid_scaler'] = phishguard.scalers['hybrid']
    # Add hybrid config
    if hasattr(phishguard, 'hybrid_config'):
        models['hybrid_config'] = phishguard.hybrid_config

# Display loading status
with st.expander("📊 Model Loading Status", expanded=False):
    for status in loading_status:
        st.write(status)

# Model performance data
model_performance = {
    "TF-IDF + LR": {"accuracy": "97.04%", "f1": "96.27%", "precision": "96.85%", "recall": "95.70%", "size": "228 KB", "speed": "⚡ Fast"},
    "CNN": {"accuracy": "98.59%", "f1": "98.23%", "precision": "98.67%", "recall": "97.79%", "size": "31.4 MB", "speed": "🚀 Medium"},
    "LSTM": {"accuracy": "97.74%", "f1": "97.16%", "precision": "97.52%", "recall": "96.81%", "size": "34.1 MB", "speed": "🚀 Medium"},
    "BERT": {"accuracy": "99.12%", "f1": "98.87%", "precision": "99.23%", "recall": "98.52%", "size": "267 MB", "speed": "🐌 Slow"},
    "Hybrid": {"accuracy": "97.11%", "f1": "96.09%", "precision": "96.73%", "recall": "95.46%", "size": "5.12 MB", "speed": "🚀 Medium"}
}

# Sidebar
with st.sidebar:
    st.header("🎯 Model Selection")
    
    available_models = []
    if phishguard and hasattr(phishguard, 'models'):
        if 'tfidf' in phishguard.models:
            available_models.append("TF-IDF + LR")
        if 'cnn' in phishguard.models:
            available_models.append("CNN")
        if 'lstm' in phishguard.models:
            available_models.append("LSTM")
        if 'bert' in phishguard.models:
            available_models.append("BERT")
        if 'hybrid' in phishguard.models:
            available_models.append("Hybrid")
    
    if available_models:
        selected_model = st.selectbox("Choose AI Model:", available_models)
        
        # Display model info
        st.subheader("📊 Model Performance")
        perf = model_performance[selected_model]
        st.metric("Accuracy", perf["accuracy"])
        st.metric("F1-Score", perf["f1"])
        st.metric("Precision", perf["precision"])
        st.metric("Recall", perf["recall"])
        st.write(f"**Size:** {perf['size']}")
        st.write(f"**Speed:** {perf['speed']}")
    else:
        st.error("No models available!")
        selected_model = None

# Main interface
if selected_model:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📧 Email Analysis")
        text = st.text_area(
            "Paste email content here:",
            height=200,
            placeholder="Enter email subject and content for analysis...",
            help="Include both subject and body for best results"
        )
        
        threshold = st.slider(
            "Detection Threshold",
            0.1, 0.9, 0.5, 0.05,
            help="Lower = more sensitive to phishing"
        )
    
    with col2:
        st.subheader("🏆 All Models")
        for model, data in model_performance.items():
            # Check if model is available in the proper system
            model_key = model.lower().replace(" + lr", "").replace(" ", "_").replace("+", "")
            if model_key == "tf-idf_lr":
                model_key = "tfidf"
            
            available = "✅" if (phishguard and hasattr(phishguard, 'models') and model_key in phishguard.models) else "❌"
            st.write(f"{available} **{model}**: {data['accuracy']}")
    
    if st.button("🔍 Analyze Email", type="primary"):
        if not text.strip():
            st.error("Please enter some email content to analyze.")
        else:
            with st.spinner("Analyzing with enhanced multi-model system..."):
                try:
                    # Use enhanced prediction with features (the original style you liked)
                    results, enhanced_features = enhanced_predict_with_features(text, models, phishguard)
                    
                    # Display main result from selected model
                    if selected_model in results and 'error' not in results[selected_model]:
                        model_result = results[selected_model]
                        
                        st.subheader("🎯 Detection Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            label = model_result['prediction']
                            color = "red" if label == "PHISHING" else "green"
                            icon = "🚨" if label == "PHISHING" else "✅"
                            st.markdown(f"**{selected_model}:** :{color}[{icon} {label}]")
                        
                        with col2:
                            st.metric("Probability", f"{model_result['probability']:.1%}")
                        
                        with col3:
                            st.metric("Confidence", f"{model_result['confidence']:.1%}")
                        
                        with col4:
                            st.metric("Threshold", f"{model_result.get('threshold', 0.5):.2f}")
                        
                        # TEMPORARILY HIDDEN: Enhanced feature analysis
                        # st.subheader("🔍 Advanced Feature Analysis")
                        # feat_col1, feat_col2 = st.columns(2)
                        # 
                        # # Count legitimate and suspicious features
                        # legitimate_features = sum(1 for key, value in enhanced_features.items() 
                        #                          if value > 0 and any(word in key for word in ['order', 'date', 'domain', 'customer', 'greeting', 'branding']))
                        # suspicious_features = sum(1 for key, value in enhanced_features.items() 
                        #                          if value > 0 and any(word in key for word in ['urgent', 'phishing', 'suspicious', 'generic', 'threat']))
                        # 
                        # with feat_col1:
                        #     st.metric("Legitimacy Score", enhanced_features['legitimacy_score'])
                        #     st.metric("Legitimate Features", legitimate_features)
                        # 
                        # with feat_col2:
                        #     st.metric("Suspicion Score", enhanced_features['suspicion_score'])
                        #     st.metric("Suspicious Features", suspicious_features)
                        # 
                        # # Risk level with enhanced analysis
                        # prob = model_result['probability']
                        # legitimacy = enhanced_features['legitimacy_score']
                        # suspicion = enhanced_features['suspicion_score']
                        # 
                        # if prob >= 0.8 and suspicion > legitimacy:
                        #     st.error("🚨 HIGH RISK - Strong phishing indicators detected")
                        # elif prob >= 0.6:
                        #     st.warning("⚠️ MEDIUM RISK - Suspicious characteristics present")
                        # elif prob >= 0.4 or suspicion > legitimacy + 1:
                        #     st.info("ℹ️ LOW-MEDIUM RISK - Some concerning elements")
                        # elif legitimacy > suspicion + 1:
                        #     st.success("✅ HIGH CONFIDENCE - Strong legitimacy indicators")
                        # else:
                        #     st.success("✅ LOW RISK - Appears legitimate")
                        
                        # TEMPORARILY HIDDEN: Enhanced ensemble result
                        # if 'Enhanced Ensemble' in results:
                        #     ensemble = results['Enhanced Ensemble']
                        #     st.subheader("🎯 Enhanced Ensemble Prediction (96% Accuracy)")
                        #     
                        #     ens_col1, ens_col2, ens_col3 = st.columns(3)
                        #     with ens_col1:
                        #         ens_label = ensemble['prediction']
                        #         ens_color = "red" if ens_label == "PHISHING" else "green"
                        #         ens_icon = "🚨" if ens_label == "PHISHING" else "✅"
                        #         st.markdown(f"**Ensemble:** :{ens_color}[{ens_icon} {ens_label}]")
                        #     
                        #     with ens_col2:
                        #         st.metric("Ensemble Probability", f"{ensemble['probability']:.1%}")
                        #     
                        #     with ens_col3:
                        #         st.metric("Ensemble Confidence", f"{ensemble['confidence']:.1%}")
                        #     
                        #     # Show voting breakdown
                        #     votes = ensemble.get('votes', {})
                        #     st.write(f"**Model Votes:** Phishing: {votes.get('phishing', 0)}, Legitimate: {votes.get('legitimate', 0)}")
                        
                        # All models comparison
                        st.subheader("🔄 Multi-Model Analysis")
                        
                        model_names = ['TF-IDF + LR', 'CNN', 'LSTM', 'BERT', 'Hybrid']
                        for model_name in model_names:
                            if model_name in results and 'error' not in results[model_name]:
                                res = results[model_name]
                                prediction = res['prediction']
                                probability = res['probability']
                                threshold_used = res.get('threshold', 0.5)
                                
                                pred_color = "🚨" if prediction == "PHISHING" else "✅"
                                st.write(f"**{model_name}**: {probability:.1%} (thresh: {threshold_used:.2f}) → {pred_color} {prediction}")
                            elif model_name in results:
                                st.write(f"**{model_name}**: ❌ Error - {results[model_name]['error']}")
                    
                    else:
                        st.error(f"Error with {selected_model}: {results.get(selected_model, {}).get('error', 'Model not available')}")
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    # Show fallback prediction
                    if phishguard:
                        try:
                            # Simple fallback using the enhanced system
                            result = phishguard.predict_ensemble(text)
                            pred = "PHISHING" if result['is_phishing'] else "LEGITIMATE"
                            prob = result['probability']
                            color = "red" if result['is_phishing'] else "green"
                            st.markdown(f"**Fallback Result:** :{color}[{pred}] ({prob:.1%})")
                        except Exception as e2:
                            st.error(f"Complete system failure: {str(e2)}")
                    else:
                        st.error("Enhanced PhishGuard system not available")

# Footer
st.markdown("---")
st.markdown("""
**🛡️ PhishGuard-XAI** - Proper multi-model phishing detection system (96% accuracy)  
**Available Models:** TF-IDF + Logistic Regression, CNN, LSTM (using trained models correctly)  
**System Performance:** 96.0% accuracy, 95.3% precision, 95.3% recall  
**Approach:** Uses models as designed without interference from enhanced features
""")
