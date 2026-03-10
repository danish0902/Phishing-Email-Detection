"""
Batch Prediction Script for Phishing Email Detection

Reads a CSV file containing email texts, runs predictions using the chosen
model, and writes a new CSV with the results appended.

Usage:
    python src/models/batch_predict.py --input data/test_set.csv --model tfidf

    # Save results to a specific output file
    python src/models/batch_predict.py \
        --input emails.csv \
        --output predictions.csv \
        --model tfidf \
        --text_col "Email Text" \
        --threshold 0.5
"""

import argparse
import os
import sys
import time

import pandas as pd

# Add project root to path so src.* imports resolve correctly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.app.model_loader import ModelLoader

# Maximum number of characters to include in truncated error / preview strings.
_MAX_TRUNCATE_LEN = 120


# ---------------------------------------------------------------------------
# Core batch prediction logic
# ---------------------------------------------------------------------------

def batch_predict(
    input_path: str,
    output_path: str,
    model_type: str = "tfidf",
    text_col: str = "Email Text",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run batch prediction on a CSV file of emails.

    Args:
        input_path:  Path to the input CSV file.
        output_path: Path to save the output CSV file with predictions.
        model_type:  Model identifier (tfidf | cnn | lstm | bert | hybrid | ensemble).
        text_col:    Column name that contains email body text.
        threshold:   Decision threshold — probability >= threshold → phishing.

    Returns:
        DataFrame with added prediction columns.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"📂  Loading data from {input_path} …")
    df = pd.read_csv(input_path)

    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found in {input_path}. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"📧  Found {len(df):,} emails. Using model: {model_type}")

    loader = ModelLoader()

    probabilities: list = []
    labels: list = []
    errors: list = []

    start = time.time()
    for i, text in enumerate(df[text_col].fillna(""), start=1):
        try:
            if model_type == "ensemble":
                result = loader.predict_ensemble(str(text))
                prob = result["probability"]
            else:
                prob = loader.predict(str(text), model_type=model_type)

            is_phishing = prob >= threshold
            probabilities.append(round(float(prob), 6))
            labels.append("phishing" if is_phishing else "legitimate")
            errors.append("")
        except Exception as exc:
            probabilities.append(None)
            labels.append("error")
            errors.append(str(exc)[:_MAX_TRUNCATE_LEN])

        if i % 100 == 0 or i == len(df):
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  Processed {i:,}/{len(df):,} emails  ({rate:.1f} emails/s)", end="\r")

    print()  # newline after \r

    # Add result columns to the DataFrame
    df["pred_probability"] = probabilities
    df["pred_label"] = labels
    df["pred_error"] = errors

    # Summary stats
    n_phishing = (df["pred_label"] == "phishing").sum()
    n_legit = (df["pred_label"] == "legitimate").sum()
    n_error = (df["pred_label"] == "error").sum()

    print(f"\n📊  Results summary:")
    print(f"    Phishing    : {n_phishing:,} ({n_phishing/len(df)*100:.1f}%)")
    print(f"    Legitimate  : {n_legit:,} ({n_legit/len(df)*100:.1f}%)")
    if n_error:
        print(f"    Errors      : {n_error:,}")

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅  Saved predictions to {output_path}")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch phishing email prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output CSV file (defaults to <input>_predictions.csv)",
    )
    parser.add_argument(
        "--model",
        default="tfidf",
        choices=["tfidf", "cnn", "lstm", "bert", "hybrid", "ensemble"],
        help="Model to use for prediction",
    )
    parser.add_argument(
        "--text_col",
        default="Email Text",
        help="Column name containing email body text",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold (0–1); probability >= threshold → phishing",
    )

    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_predictions{ext}"

    batch_predict(
        input_path=args.input,
        output_path=args.output,
        model_type=args.model,
        text_col=args.text_col,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
