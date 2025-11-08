"""
Final evaluation script using the test set for unbiased performance assessment.
This script should only be run ONCE after all model development is complete.
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.features.preprocess import clean_text

# Try to import ML libraries
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

def evaluate_tfidf_model(test_df, text_col, label_col):
    """Evaluate TF-IDF + Logistic Regression model"""
    print("\n" + "="*50)
    print("EVALUATING TF-IDF + LOGISTIC REGRESSION")
    print("="*50)
    
    try:
        model = joblib.load("artifacts/baseline_tfidf_lr.joblib")
        
        X_test = test_df[text_col].astype(str).apply(clean_text).values
        y_test = test_df[label_col].values
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'model': 'TF-IDF + LR',
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
    except Exception as e:
        print(f"Error evaluating TF-IDF model: {e}")
        return None

def evaluate_cnn_model(test_df, text_col, label_col):
    """Evaluate CNN model"""
    print("\n" + "="*50)
    print("EVALUATING CNN MODEL")
    print("="*50)
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available - skipping CNN evaluation")
        return None
        
    try:
        model = load_model("artifacts/cnn_keras.h5")
        tokenizer = joblib.load("artifacts/tokenizer.joblib")
        
        X_test = test_df[text_col].astype(str).apply(clean_text).values
        y_test = test_df[label_col].values
        
        # Tokenize and pad sequences
        sequences = tokenizer.texts_to_sequences(X_test)
        X_test_padded = pad_sequences(sequences, maxlen=200, padding="post", truncating="post")
        
        y_prob = model.predict(X_test_padded).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'model': 'CNN',
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
    except Exception as e:
        print(f"Error evaluating CNN model: {e}")
        return None

def evaluate_lstm_model(test_df, text_col, label_col):
    """Evaluate LSTM model"""
    print("\n" + "="*50)
    print("EVALUATING LSTM MODEL")
    print("="*50)
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available - skipping LSTM evaluation")
        return None
        
    try:
        model = load_model("artifacts/lstm_keras.h5")
        tokenizer = joblib.load("artifacts/tokenizer.joblib")
        
        X_test = test_df[text_col].astype(str).apply(clean_text).values
        y_test = test_df[label_col].values
        
        # Tokenize and pad sequences
        sequences = tokenizer.texts_to_sequences(X_test)
        X_test_padded = pad_sequences(sequences, maxlen=200, padding="post", truncating="post")
        
        y_prob = model.predict(X_test_padded).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'model': 'LSTM',
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
    except Exception as e:
        print(f"Error evaluating LSTM model: {e}")
        return None

def evaluate_bert_model(test_df, text_col, label_col):
    """Evaluate BERT model"""
    print("\n" + "="*50)
    print("EVALUATING BERT MODEL")
    print("="*50)
    
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers not available - skipping BERT evaluation")
        return None
        
    try:
        model = AutoModelForSequenceClassification.from_pretrained("artifacts/bert_model")
        tokenizer = AutoTokenizer.from_pretrained("artifacts/bert_model")
        
        X_test = test_df[text_col].astype(str).values
        y_test = test_df[label_col].values
        
        # Tokenize
        inputs = tokenizer(list(X_test), truncation=True, padding="max_length", 
                          max_length=256, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            y_prob = torch.softmax(logits, dim=-1)[:, 1].numpy()
            y_pred = np.argmax(logits.numpy(), axis=-1)
        
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'model': 'BERT',
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
    except Exception as e:
        print(f"Error evaluating BERT model: {e}")
        return None

def main(args):
    """Main evaluation function"""
    print("🧪 FINAL MODEL EVALUATION ON TEST SET")
    print("="*60)
    print("⚠️  WARNING: This should only be run ONCE after all development is complete!")
    print("="*60)
    
    # Load test data
    test_df = pd.read_csv(args.test_data)
    print(f"\nTest set size: {len(test_df)} samples")
    print(f"Test set distribution:\n{test_df[args.label_col].value_counts()}")
    
    # Evaluate all models
    results = []
    
    # TF-IDF + LR
    result = evaluate_tfidf_model(test_df, args.text_col, args.label_col)
    if result:
        results.append(result)
    
    # CNN
    result = evaluate_cnn_model(test_df, args.text_col, args.label_col)
    if result:
        results.append(result)
    
    # LSTM
    result = evaluate_lstm_model(test_df, args.text_col, args.label_col)
    if result:
        results.append(result)
    
    # BERT
    result = evaluate_bert_model(test_df, args.text_col, args.label_col)
    if result:
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("📊 FINAL TEST SET RESULTS SUMMARY")
    print("="*60)
    
    if results:
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False, float_format="%.4f"))
        
        # Save results
        os.makedirs("artifacts", exist_ok=True)
        results_df.to_csv("artifacts/final_test_results.csv", index=False)
        print(f"\n✅ Results saved to artifacts/final_test_results.csv")
        
        # Find best model
        best_model = results_df.loc[results_df['accuracy'].idxmax()]
        print(f"\n🏆 Best Model: {best_model['model']} (Accuracy: {best_model['accuracy']:.4f})")
        
    else:
        print("❌ No models could be evaluated")
    
    print("\n🎯 These are your FINAL, UNBIASED test results!")
    print("📝 Use these numbers for reporting and comparison purposes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final evaluation on test set - use only once!")
    parser.add_argument("--test_data", default="data/test_set.csv",
                        help="Path to test dataset")
    parser.add_argument("--text_col", default="Email Text",
                        help="Name of text column")
    parser.add_argument("--label_col", default="Email Type",
                        help="Name of label column")
    
    args = parser.parse_args()
    main(args)
