import argparse, os, sys
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.features.preprocess import clean_text

def main(args):
    # Load pre-split training data
    train_df = pd.read_csv(args.train_data)
    train_df[args.text_col] = train_df[args.text_col].astype(str).apply(clean_text)
    X_train_full = train_df[args.text_col].values
    y_train_full = train_df[args.label_col].values

    tok = Tokenizer(num_words=args.vocab_size, oov_token="<OOV>")
    tok.fit_on_texts(X_train_full)
    seq_train = tok.texts_to_sequences(X_train_full)
    X_train_full = pad_sequences(seq_train, maxlen=args.max_len, padding="post", truncating="post")

    # Create validation split from training data only
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    model = Sequential([
        Embedding(args.vocab_size, args.embed_dim, input_length=args.max_len),
        Conv1D(filters=128, kernel_size=3, activation="relu"),
        GlobalMaxPool1D(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=[es])

    os.makedirs("artifacts", exist_ok=True)
    model.save("artifacts/cnn_keras.h5")
    import joblib
    joblib.dump(tok, "artifacts/tokenizer.joblib")
    print("Saved artifacts/cnn_keras.h5 and artifacts/tokenizer.joblib")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", default="data/train_set.csv")
    p.add_argument("--text_col", default="Email Text")
    p.add_argument("--label_col", default="Email Type")
    p.add_argument("--vocab_size", type=int, default=20000)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--max_len", type=int, default=200)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    main(args)
