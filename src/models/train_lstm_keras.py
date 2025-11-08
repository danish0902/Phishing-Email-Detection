import argparse, os, sys
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.features.preprocess import clean_text

def main(args):
    # Load pre-split training and validation data (80/10/10 split)
    train_df = pd.read_csv(args.train_data)
    val_df = pd.read_csv(args.val_data)
    
    # Clean the text data
    train_df[args.text_col] = train_df[args.text_col].astype(str).apply(clean_text)
    val_df[args.text_col] = val_df[args.text_col].astype(str).apply(clean_text)
    
    X_train = train_df[args.text_col].values
    y_train = train_df[args.label_col].values
    X_val = val_df[args.text_col].values
    y_val = val_df[args.label_col].values

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Fit tokenizer on training data only
    tok = Tokenizer(num_words=args.vocab_size, oov_token="<OOV>")
    tok.fit_on_texts(X_train)
    
    # Transform both training and validation data
    seq_train = tok.texts_to_sequences(X_train)
    seq_val = tok.texts_to_sequences(X_val)
    
    X_train = pad_sequences(seq_train, maxlen=args.max_len, padding="post", truncating="post")
    X_val = pad_sequences(seq_val, maxlen=args.max_len, padding="post", truncating="post")

    model = Sequential([
        Embedding(args.vocab_size, args.embed_dim, input_length=args.max_len),
        Bidirectional(LSTM(args.lstm_units, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=[es])

    os.makedirs("artifacts", exist_ok=True)
    model.save("artifacts/lstm_keras.h5")
    import joblib
    joblib.dump(tok, "artifacts/tokenizer.joblib")
    print("Saved artifacts/lstm_keras.h5 and artifacts/tokenizer.joblib")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train LSTM model with proper train/validation split")
    p.add_argument("--train_data", default="data/train_set.csv")
    p.add_argument("--val_data", default="data/val_set.csv")
    p.add_argument("--text_col", default="Email Text")
    p.add_argument("--label_col", default="Email Type")
    p.add_argument("--vocab_size", type=int, default=20000)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--lstm_units", type=int, default=128)
    p.add_argument("--max_len", type=int, default=200)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    main(args)
