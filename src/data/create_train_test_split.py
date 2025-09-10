"""
Create consistent train/test split for all models to ensure fair comparison.
This script creates the same 80/20 split that all models can use.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def main(args):
    print("Creating consistent train/test split...")
    
    # Load the dataset
    df = pd.read_csv(args.data)
    print(f"Loaded dataset with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Label distribution:\n{df[args.label_col].value_counts()}")
    
    # Prepare features and labels
    X = df[args.text_col].values
    y = df[args.label_col].values
    
    # Create the split with the same random state used in training scripts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"\nSplit summary:")
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Create DataFrames for train and test sets
    train_df = pd.DataFrame({
        args.text_col: X_train,
        args.label_col: y_train
    })
    
    test_df = pd.DataFrame({
        args.text_col: X_test,
        args.label_col: y_test
    })
    
    print(f"\nTraining set label distribution:")
    print(train_df[args.label_col].value_counts())
    print(f"\nTest set label distribution:")
    print(test_df[args.label_col].value_counts())
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save the splits
    train_path = "data/train_set.csv"
    test_path = "data/test_set.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved training set to: {train_path}")
    print(f"Saved test set to: {test_path}")
    
    # Also save indices for reference
    indices_df = pd.DataFrame({
        'original_index': range(len(X)),
        'split': ['train'] * len(X_train) + ['test'] * len(X_test)
    })
    indices_path = "data/split_indices.csv"
    indices_df.to_csv(indices_path, index=False)
    print(f"Saved split indices to: {indices_path}")
    
    print(f"\nNow all models can use:")
    print(f"- Training: {train_path}")
    print(f"- Testing: {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create consistent train/test split")
    parser.add_argument("--data", default="data/Phishing_Email_Cleaned.csv",
                        help="Path to the dataset")
    parser.add_argument("--text_col", default="Email Text",
                        help="Name of the text column")
    parser.add_argument("--label_col", default="Email Type", 
                        help="Name of the label column")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data for test set")
    
    args = parser.parse_args()
    main(args)
