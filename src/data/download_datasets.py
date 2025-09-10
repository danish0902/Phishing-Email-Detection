"""
Scripts to help you fetch datasets.

1) Hugging Face (recommended)
-----------------------------
pip install datasets
python -c "from datasets import load_dataset; ds=load_dataset('zefang-liu/phishing-email-dataset'); print(ds)"

2) SpamAssassin / Enron / Nazario / CSDMC2010
---------------------------------------------
- Download archives from their sources (links in README or search).
- Unpack and point preprocess scripts to their locations.
"""

import os
import glob
import pandas as pd
from typing import List

def load_hf_export_if_available(save_csv: str = "data/hf_phishing.csv"):
    """
    If you've already downloaded the HF dataset locally via `datasets`,
    export it to a flat CSV for training with scikit/keras.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        print("Install `datasets` to load HF datasets: pip install datasets")
        raise

    ds = load_dataset("zefang-liu/phishing-email-dataset")
    df = pd.DataFrame(ds['train'])
    text_col = 'text' if 'text' in df.columns else 'email_text'
    if text_col not in df.columns:
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break
    label_col = 'label' if 'label' in df.columns else 'is_phishing'
    if label_col not in df.columns:
        if 'target' in df.columns:
            label_col = 'target'
        else:
            raise ValueError("Could not locate label column; please adjust manually.")
    out = df[[text_col, label_col]].rename(columns={text_col:'text', label_col:'label'})
    out.to_csv(save_csv, index=False)
    print(f"Saved {len(out)} rows to {save_csv}")
    return save_csv

def merge_multiple_csv(inputs: List[str], save_csv: str):
    frames = []
    for path in inputs:
        if not os.path.exists(path):
            print(f"Skip missing {path}")
            continue
        frames.append(pd.read_csv(path))
    out = pd.concat(frames, ignore_index=True)
    out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out.to_csv(save_csv, index=False)
    print(f"Merged {len(out)} rows -> {save_csv}")
    return save_csv
