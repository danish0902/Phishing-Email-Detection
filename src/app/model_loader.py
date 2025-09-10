"""
Model loading utilities for PhishGuard-XAI
Supports loading different trained models (TF-IDF, CNN, LSTM, BERT, Hybrid)
"""

import joblib
import os
import re
import sys
import pandas as pd

# Try to import TensorFlow/Keras, but don't fail if not available
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from sklearn.preprocessing import StandardScaler

# Try to import transformers, but don't fail if not available
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import numpy as np

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.features.preprocess import clean_text

class ModelLoader:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.scalers = {}
        self.thresholds = {}
        self._load_thresholds()
        
    def _load_thresholds(self):
        """Load optimized thresholds for all models"""
        try:
            artifacts_dir = os.path.join(project_root, 'artifacts')
            threshold_path = os.path.join(artifacts_dir, 'optimal_thresholds.joblib')
            if os.path.exists(threshold_path):
                self.thresholds = joblib.load(threshold_path)
            else:
                # Default thresholds if file not found
                self.thresholds = {
                    'tfidf': 0.5, 'cnn': 0.5, 'lstm': 0.5, 
                    'bert': 0.5, 'hybrid': 0.5
                }
        except Exception as e:
            print(f"Warning: Could not load thresholds: {e}")
            self.thresholds = {
                'tfidf': 0.5, 'cnn': 0.5, 'lstm': 0.5, 
                'bert': 0.5, 'hybrid': 0.5
            }
        
    def extract_url_features(self, text):
        """
        Extracts URLs and computes advanced features from them.
        Returns 6 numerical features for URL analysis.
        Same as in hybrid model training.
        """
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
        
    def load_tfidf_model(self):
        """Load TF-IDF + Logistic Regression model"""
        if 'tfidf' not in self.models:
            self.models['tfidf'] = joblib.load("artifacts/baseline_tfidf_lr.joblib")
        return self.models['tfidf']
    
    def load_cnn_model(self):
        """Load CNN model and tokenizer"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        if 'cnn' not in self.models:
            self.models['cnn'] = load_model("artifacts/cnn_keras.h5")
            self.tokenizers['cnn'] = joblib.load("artifacts/tokenizer.joblib")
        return self.models['cnn'], self.tokenizers['cnn']
    
    def load_lstm_model(self):
        """Load LSTM model and tokenizer"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        if 'lstm' not in self.models:
            self.models['lstm'] = load_model("artifacts/lstm_keras.h5")
            self.tokenizers['lstm'] = joblib.load("artifacts/tokenizer.joblib")
        return self.models['lstm'], self.tokenizers['lstm']
    
    def load_bert_model(self):
        """Load BERT model and tokenizer"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Install with: pip install transformers")
        
        if 'bert' not in self.models:
            # Use DistilBERT tokenizer since our model is based on DistilBERT (tokenizer wasn't saved with model)
            self.tokenizers['bert'] = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.models['bert'] = AutoModelForSequenceClassification.from_pretrained("artifacts/bert_model")
        return self.models['bert'], self.tokenizers['bert']
    
    def load_hybrid_model(self):
        """Load Hybrid CNN+LSTM+URL model and associated artifacts"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        if 'hybrid' not in self.models:
            self.models['hybrid'] = load_model("artifacts/hybrid_cnn_lstm.h5")
            self.tokenizers['hybrid'] = joblib.load("artifacts/tokenizer.joblib")  # Use standard tokenizer
            self.scalers['hybrid'] = joblib.load("artifacts/url_scaler.joblib")
            self.hybrid_config = joblib.load("artifacts/hybrid_config.joblib")
        return self.models['hybrid'], self.tokenizers['hybrid'], self.scalers['hybrid']
    
    def predict_tfidf(self, text):
        """Predict using TF-IDF model"""
        model = self.load_tfidf_model()
        proba = model.predict_proba([text])[0][1]
        return proba
    
    def predict_cnn(self, text):
        """Predict using CNN model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        model, tokenizer = self.load_cnn_model()
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
        proba = model.predict(padded)[0][0]
        return float(proba)
    
    def predict_lstm(self, text):
        """Predict using LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        model, tokenizer = self.load_lstm_model()
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
        proba = model.predict(padded)[0][0]
        return float(proba)
    
    def predict_bert(self, text):
        """Predict using BERT model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Install with: pip install transformers")
        
        model, tokenizer = self.load_bert_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            proba = torch.softmax(outputs.logits, dim=-1)[0][1].item()
        return proba
    
    def predict_hybrid(self, text):
        """Predict using Hybrid CNN+LSTM+URL model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        model, tokenizer, scaler = self.load_hybrid_model()
        
        # Extract URL features BEFORE text cleaning
        url_features = np.array([self.extract_url_features(text)])
        url_features_scaled = scaler.transform(url_features)
        
        # Clean text AFTER URL extraction
        cleaned_text = clean_text(text)
        
        # Process text data
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        max_len = self.hybrid_config['max_len']
        text_padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        
        # Make prediction
        proba = model.predict([text_padded, url_features_scaled])[0][0]
        return float(proba)
    
    def predict(self, text, model_type="cnn"):
        """Predict using specified model type"""
        if model_type == "tfidf":
            return self.predict_tfidf(text)
        elif model_type == "cnn":
            return self.predict_cnn(text)
        elif model_type == "lstm":
            return self.predict_lstm(text)
        elif model_type == "bert":
            return self.predict_bert(text)
        elif model_type == "hybrid":
            return self.predict_hybrid(text)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict_ensemble(self, text, models=['tfidf', 'cnn', 'lstm']):
        """
        Ensemble prediction using majority voting with optimized thresholds
        Returns: {prediction, probability, confidence, individual_models, votes, is_phishing}
        """
        predictions = {}
        
        for model_name in models:
            try:
                prob = self.predict(text, model_name)
                threshold = self.thresholds.get(model_name, 0.5)
                predictions[model_name] = {
                    'probability': prob,
                    'prediction': 1 if prob > threshold else 0
                }
            except Exception as e:
                print(f"Warning: Error with {model_name} model: {e}")
                continue
        
        if not predictions:
            raise Exception("No models could make predictions")
        
        # Ensemble decision (majority vote)
        total_models = len(predictions)
        phishing_votes = sum(1 for p in predictions.values() if p['prediction'] == 1)
        avg_probability = sum(p['probability'] for p in predictions.values()) / total_models
        
        final_prediction = 1 if phishing_votes >= (total_models // 2 + 1) else 0
        confidence = avg_probability if final_prediction == 1 else (1 - avg_probability)
        
        return {
            'prediction': final_prediction,
            'probability': avg_probability,
            'confidence': confidence,
            'individual_models': predictions,
            'votes': {'phishing': phishing_votes, 'legitimate': total_models - phishing_votes},
            'is_phishing': final_prediction == 1
        }
    
    def get_system_info(self):
        """Return system information"""
        loaded_models = []
        for model_type in ['tfidf', 'cnn', 'lstm', 'bert', 'hybrid']:
            try:
                # Test if model can be loaded
                if model_type == 'tfidf':
                    self.load_tfidf_model()
                elif model_type == 'cnn':
                    self.load_cnn_model()
                elif model_type == 'lstm':
                    self.load_lstm_model()
                elif model_type == 'bert':
                    self.load_bert_model()
                elif model_type == 'hybrid':
                    self.load_hybrid_model()
                loaded_models.append(model_type)
            except:
                continue
        
        return {
            'status': 'Ready' if loaded_models else 'Not Loaded',
            'models': loaded_models,
            'thresholds': self.thresholds,
            'total_models': len(loaded_models)
        }
    
    def evaluate_on_testset(self, sample_size=100, models=['tfidf', 'cnn', 'lstm']):
        """Evaluate ensemble system on test data"""
        test_file = os.path.join(project_root, 'data', 'test_set.csv')
        if not os.path.exists(test_file):
            print("❌ Test data not found. Please ensure data/test_set.csv exists.")
            return None
        
        print(f"🧪 Evaluating ensemble on {sample_size} test emails...")
        
        test_df = pd.read_csv(test_file)
        test_sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
        
        true_labels = []
        predictions = []
        
        for _, row in test_sample.iterrows():
            try:
                result = self.predict_ensemble(row['Email Text'], models)
                true_labels.append(row['Email Type'])
                predictions.append(result['prediction'])
            except Exception as e:
                print(f"⚠️ Error processing email: {e}")
                continue
        
        if len(predictions) == 0:
            print("❌ No predictions were successful")
            return None
        
        # Calculate metrics
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"Accuracy:  {accuracy*100:.1f}%")
        print(f"Precision: {precision*100:.1f}%")
        print(f"Recall:    {recall*100:.1f}%")
        print(f"F1-Score:  {f1*100:.1f}%")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
