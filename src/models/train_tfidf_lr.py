import argparse
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from src.features.preprocess import clean_text

def main(args):
    # Load pre-split training and validation data (80/10/10 split)
    train_df = pd.read_csv(args.train_data)
    val_df = pd.read_csv(args.val_data)
    
    train_df[args.text_col] = train_df[args.text_col].astype(str).apply(clean_text)
    val_df[args.text_col] = val_df[args.text_col].astype(str).apply(clean_text)
    
    X_train = train_df[args.text_col].values
    y_train = train_df[args.label_col].values
    X_val = val_df[args.text_col].values
    y_val = val_df[args.label_col].values
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=args.max_features)),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred_val = pipe.predict(X_val)
    print("\nValidation Results:")
    print(classification_report(y_val, y_pred_val))
    cm = confusion_matrix(y_val, y_pred_val)
    print("Validation Confusion Matrix:\n", cm)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipe, "artifacts/baseline_tfidf_lr.joblib")
    print("Saved artifacts/baseline_tfidf_lr.joblib")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression with proper train/validation split")
    p.add_argument("--train_data", default="data/train_set.csv")
    p.add_argument("--val_data", default="data/val_set.csv")
    p.add_argument("--text_col", default="Email Text")
    p.add_argument("--label_col", default="Email Type")
    p.add_argument("--max_features", type=int, default=5000)
    args = p.parse_args()
    main(args)
