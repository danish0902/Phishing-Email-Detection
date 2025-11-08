"""
Create consistent train/validation/test split for all models to ensure fair comparison.
This script creates the proper 80/10/10 split following ML best practices.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def main(args):
    print("Creating consistent train/validation/test split (80/10/10)...")
    
    # Load the dataset
    df = pd.read_csv(args.data)
    print(f"Loaded dataset with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Label distribution:\n{df[args.label_col].value_counts()}")
    
    # Prepare features and labels
    X = df[args.text_col].values
    y = df[args.label_col].values
    
    # First split: 80% train, 20% temp (which will become 10% val + 10% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: Split the 20% temp into 10% validation and 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nSplit summary:")
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Create DataFrames for train, validation, and test sets
    train_df = pd.DataFrame({
        args.text_col: X_train,
        args.label_col: y_train
    })
    
    val_df = pd.DataFrame({
        args.text_col: X_val,
        args.label_col: y_val
    })
    
    test_df = pd.DataFrame({
        args.text_col: X_test,
        args.label_col: y_test
    })
    
    print(f"\nTraining set label distribution:")
    print(train_df[args.label_col].value_counts())
    print(f"\nValidation set label distribution:")
    print(val_df[args.label_col].value_counts())
    print(f"\nTest set label distribution:")
    print(test_df[args.label_col].value_counts())
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save the splits
    train_path = "data/train_set.csv"
    val_path = "data/val_set.csv"
    test_path = "data/test_set.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved training set to: {train_path}")
    print(f"Saved validation set to: {val_path}")
    print(f"Saved test set to: {test_path}")
    
    # Also save indices for reference
    train_indices = list(range(len(X_train)))
    val_indices = list(range(len(X_train), len(X_train) + len(X_val)))
    test_indices = list(range(len(X_train) + len(X_val), len(X_train) + len(X_val) + len(X_test)))
    
    indices_df = pd.DataFrame({
        'original_index': train_indices + val_indices + test_indices,
        'split': ['train'] * len(X_train) + ['validation'] * len(X_val) + ['test'] * len(X_test)
    })
    indices_path = "data/split_indices_CLEAN.csv"
    indices_df.to_csv(indices_path, index=False)
    print(f"Saved split indices to: {indices_path}")
    
    print(f"\nNow all models can use:")
    print(f"- Training: {train_path}")
    print(f"- Validation: {val_path}")
    print(f"- Testing: {test_path}")
    print(f"\n🎯 This follows ML best practices with proper 80/10/10 split!")
    print(f"📊 Use validation set for hyperparameter tuning and early stopping")
    print(f"🧪 Use test set ONLY for final evaluation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create consistent train/validation/test split (80/10/10)")
    parser.add_argument("--data", default="data/Phishing_Email_Cleaned_NO_DUPLICATES.csv",
                        help="Path to the dataset")
    parser.add_argument("--text_col", default="Email Text",
                        help="Name of the text column")
    parser.add_argument("--label_col", default="Email Type", 
                        help="Name of the label column")
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="Proportion of data for validation set")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Proportion of data for test set")
    
    args = parser.parse_args()
    main(args)
