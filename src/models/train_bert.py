import argparse, os
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    try:
        probs = 1/(1+np.exp(-logits[:,1]))
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": float((preds==labels).mean()), "precision":prec, "recall":rec, "f1":f1, "roc_auc":auc}

def main(args):
    # Load pre-split training and validation data (80/10/10 split)
    train_df = pd.read_csv(args.train_data)[[args.text_col, args.label_col]].rename(columns={args.text_col:"text", args.label_col:"label"})
    val_df = pd.read_csv(args.val_data)[[args.text_col, args.label_col]].rename(columns={args.text_col:"text", args.label_col:"label"})
    
    train_df["label"] = train_df["label"].astype(int)
    train_df["text"] = train_df["text"].astype(str)
    val_df["label"] = val_df["label"].astype(int)
    val_df["text"] = val_df["text"].astype(str)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Convert to Dataset
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def tok(batch):
        # Convert to list if it's not already
        texts = [str(text) for text in batch["text"]]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=args.max_len)
    
    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)
    
    rem_cols = ["text"]
    if "__index_level_0__" in train_ds.column_names:
        rem_cols.append("__index_level_0__")
    
    train_ds = train_ds.remove_columns(rem_cols)
    val_ds = val_ds.remove_columns([col for col in rem_cols if col in val_ds.column_names])
    
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    args_tr = TrainingArguments(
        output_dir="artifacts/bert",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("artifacts/bert_model")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train BERT model with proper train/validation split")
    p.add_argument("--train_data", default="data/train_set.csv")
    p.add_argument("--val_data", default="data/val_set.csv")
    p.add_argument("--text_col", default="Email Text")
    p.add_argument("--label_col", default="Email Type")
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    args = p.parse_args()
    main(args)
