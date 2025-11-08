"""
Advanced Hybrid Phishing Detection Model
Combines CNN+LSTM for text analysis with URL feature extraction
Adapted for PhishGuard-XAI project structure
"""

import argparse, os, sys
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.features.preprocess import clean_text

def extract_url_features(text):
    """
    Extracts URLs and computes advanced features from them.
    Returns 6 numerical features for URL analysis.
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

def create_hybrid_model(max_words, max_len, num_url_features, embed_dim=128):
    """Creates hybrid model: CNN+LSTM for text, Dense for URL features"""
    # Text Branch (CNN + LSTM)
    text_input = Input(shape=(max_len,), name='text_input')
    embedding = Embedding(input_dim=max_words, output_dim=embed_dim, input_length=max_len)(text_input)
    
    # CNN layers
    conv1d = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding)
    maxpool = MaxPooling1D(pool_size=2)(conv1d)
    
    # LSTM layer
    lstm = LSTM(64, dropout=0.3, recurrent_dropout=0.3)(maxpool)
    text_branch = Dense(32, activation='relu')(lstm)
    text_dropout = Dropout(0.5)(text_branch)

    # URL Feature Branch
    url_input = Input(shape=(num_url_features,), name='url_input')
    url_dense1 = Dense(16, activation='relu')(url_input)
    url_dropout1 = Dropout(0.3)(url_dense1)
    url_branch = Dense(8, activation='relu')(url_dropout1)
    url_dropout2 = Dropout(0.3)(url_branch)

    # Combine both branches
    concatenated = concatenate([text_dropout, url_dropout2])
    
    # Final classification layers
    final_dense1 = Dense(32, activation='relu')(concatenated)
    final_dropout = Dropout(0.5)(final_dense1)
    final_dense2 = Dense(16, activation='relu')(final_dropout)
    output = Dense(1, activation='sigmoid', name='phishing_output')(final_dense2)

    model = Model(inputs=[text_input, url_input], outputs=output)
    return model

def main(args):
    # Load pre-split training and validation data (80/10/10 split)
    train_df = pd.read_csv(args.train_data)
    val_df = pd.read_csv(args.val_data)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Extract URL features BEFORE text cleaning for both sets
    train_url_features = train_df[args.text_col].astype(str).apply(extract_url_features)
    val_url_features = val_df[args.text_col].astype(str).apply(extract_url_features)
    
    X_train_url = np.array(train_url_features.tolist())
    X_val_url = np.array(val_url_features.tolist())
    
    # Clean text using existing preprocessing
    train_df[args.text_col] = train_df[args.text_col].astype(str).apply(clean_text)
    val_df[args.text_col] = val_df[args.text_col].astype(str).apply(clean_text)
    
    X_train_text = train_df[args.text_col].values
    X_val_text = val_df[args.text_col].values
    y_train = train_df[args.label_col].values
    y_val = val_df[args.label_col].values
    
    # Tokenize text data (fit only on training)
    tok = Tokenizer(num_words=args.vocab_size, oov_token="<OOV>")
    tok.fit_on_texts(X_train_text)
    
    # Transform both training and validation text
    seq_train = tok.texts_to_sequences(X_train_text)
    seq_val = tok.texts_to_sequences(X_val_text)
    
    X_train_text_padded = pad_sequences(seq_train, maxlen=args.max_len, padding="post", truncating="post")
    X_val_text_padded = pad_sequences(seq_val, maxlen=args.max_len, padding="post", truncating="post")
    
    # Scale URL features (fit only on training)
    scaler = StandardScaler()
    X_train_url_scaled = scaler.fit_transform(X_train_url)
    X_val_url_scaled = scaler.transform(X_val_url)
    
    # Create and compile model
    num_url_features = X_train_url_scaled.shape[1]
    model = create_hybrid_model(args.vocab_size, args.max_len, num_url_features, args.embed_dim)
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    # Training
    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(
        [X_train_text_padded, X_train_url_scaled], y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=([X_val_text_padded, X_val_url_scaled], y_val),
        callbacks=[es]
    )
    
    # Evaluation on validation set
    y_pred_proba = model.predict([X_val_text_padded, X_val_url_scaled])
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate and display metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(classification_report(y_val, y_pred, target_names=['Legitimate', 'Phishing']))
    
    # Save model and artifacts
    os.makedirs("artifacts", exist_ok=True)
    model.save("artifacts/hybrid_cnn_lstm.h5")
    joblib.dump(tok, "artifacts/tokenizer.joblib")  # Use standard tokenizer name
    joblib.dump(scaler, "artifacts/url_scaler.joblib")
    
    # Save training config
    config = {
        'vocab_size': args.vocab_size,
        'max_len': args.max_len,
        'num_url_features': num_url_features,
        'text_col': args.text_col,
        'label_col': args.label_col,
        'embed_dim': args.embed_dim,
        'model_type': 'hybrid_cnn_lstm'
    }
    joblib.dump(config, "artifacts/hybrid_config.joblib")
    print("Saved artifacts/hybrid_cnn_lstm.h5, tokenizer.joblib, url_scaler.joblib, hybrid_config.joblib")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train hybrid CNN+LSTM+URL model with proper train/validation split")
    p.add_argument("--train_data", default="data/train_set.csv", help="Path to training CSV file")
    p.add_argument("--val_data", default="data/val_set.csv", help="Path to validation CSV file")
    p.add_argument("--text_col", default="Email Text", help="Name of text column")
    p.add_argument("--label_col", default="Email Type", help="Name of label column")
    p.add_argument("--vocab_size", type=int, default=20000, help="Vocabulary size")
    p.add_argument("--max_len", type=int, default=100, help="Max sequence length")
    p.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    args = p.parse_args()
    main(args)
