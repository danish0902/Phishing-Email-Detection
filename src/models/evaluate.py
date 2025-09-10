import argparse
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.preprocess import clean_text

def plot_confusion_matrix(cm, labels, save_path):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path)
    print(f"Saved {save_path}")

def main(args):
    df = pd.read_csv(args.data)
    df[args.text_col] = df[args.text_col].astype(str).apply(clean_text)
    X = df[args.text_col].values
    y = df[args.label_col].values

    pipe = joblib.load("artifacts/baseline_tfidf_lr.joblib")
    probs = pipe.predict_proba(X)[:,1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y, probs)
    except Exception:
        auc = float("nan")

    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(y, preds)
    plot_confusion_matrix(cm, labels=["ham","phish"], save_path="charts/confusion_matrix.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample_emails.csv")
    p.add_argument("--text_col", default="text")
    p.add_argument("--label_col", default="label")
    args = p.parse_args()
    main(args)
