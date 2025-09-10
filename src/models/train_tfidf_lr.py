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
    # Load pre-split training data
    train_df = pd.read_csv(args.train_data)
    train_df[args.text_col] = train_df[args.text_col].astype(str).apply(clean_text)
    X_train = train_df[args.text_col].values
    y_train = train_df[args.label_col].values
    
    # Load pre-split test data
    test_df = pd.read_csv(args.test_data)
    test_df[args.text_col] = test_df[args.text_col].astype(str).apply(clean_text)
    X_test = test_df[args.text_col].values
    y_test = test_df[args.label_col].values

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=args.max_features)),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipe, "artifacts/baseline_tfidf_lr.joblib")
    print("Saved artifacts/baseline_tfidf_lr.joblib")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", default="data/train_set.csv")
    p.add_argument("--test_data", default="data/test_set.csv")
    p.add_argument("--text_col", default="Email Text")
    p.add_argument("--label_col", default="Email Type")
    p.add_argument("--max_features", type=int, default=5000)
    args = p.parse_args()
    main(args)
