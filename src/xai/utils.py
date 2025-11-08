"""
Utility functions for XAI module
Handles file operations, batching, and common helpers
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Callable, Dict, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, create if not
    
    Args:
        path: Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_html(path: str, html: str) -> None:
    """
    Save HTML content to file
    
    Args:
        path: File path to save HTML
        html: HTML content as string
    """
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"Saved HTML explanation to {path}")


def save_json(path: str, data: Dict[Any, Any]) -> None:
    """
    Save JSON metadata
    
    Args:
        path: File path to save JSON
        data: Dictionary to serialize
    """
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved metadata to {path}")


def safe_predict_proba_batch(
    fn: Callable,
    texts: List[str],
    batch_size: int = 32
) -> np.ndarray:
    """
    Safely batch predict probabilities with error handling
    
    Args:
        fn: Prediction function that takes list of texts
        texts: List of text samples
        batch_size: Batch size for processing
        
    Returns:
        Nx2 array of probabilities [prob_class_0, prob_class_1]
    """
    if not texts:
        return np.array([]).reshape(0, 2)
    
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_probs = fn(batch)
            # Ensure shape is (N, 2)
            if len(batch_probs.shape) == 1:
                batch_probs = np.column_stack([1 - batch_probs, batch_probs])
            results.append(batch_probs)
        except Exception as e:
            logger.error(f"Error in batch {i//batch_size}: {e}")
            # Return neutral probabilities for failed batch
            results.append(np.ones((len(batch), 2)) * 0.5)
    
    return np.vstack(results)


def generate_timestamp() -> str:
    """Generate timestamp string for filenames"""
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def truncate_text(text: str, max_len: int = 120) -> str:
    """Truncate text for logging (privacy)"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def get_artifacts_dir() -> str:
    """Get artifacts directory path"""
    project_root = Path(__file__).parent.parent.parent
    return str(project_root / "artifacts")


def get_explanations_dir() -> str:
    """Get explanations directory path"""
    explanations_dir = os.path.join(get_artifacts_dir(), "explanations")
    ensure_dir(explanations_dir)
    return explanations_dir


def create_explanation_metadata(
    model_name: str,
    text: str,
    prediction: int,
    probability: float,
    top_features: List[tuple] = None,
    num_features: int = 10,
    num_samples: int = 500
) -> Dict[str, Any]:
    """
    Create metadata dictionary for explanation
    
    Args:
        model_name: Name of the model
        text: Input text (will be truncated)
        prediction: Predicted label (0 or 1)
        probability: Prediction probability
        top_features: List of (feature, weight) tuples
        num_features: Number of features used
        num_samples: Number of samples used for LIME
        
    Returns:
        Metadata dictionary
    """
    label = "phishing" if prediction == 1 else "legitimate"
    
    metadata = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "prediction": int(prediction),
        "probability": float(probability),
        "text_preview": truncate_text(text, 120),
        "text_length": len(text),
        "num_features": num_features,
        "num_samples": num_samples,
    }
    
    if top_features:
        metadata["top_tokens"] = [
            [str(token), float(weight)] 
            for token, weight in top_features[:10]
        ]
    
    return metadata


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    """
    Ensure probabilities sum to 1 across classes
    
    Args:
        probs: Nx2 array of probabilities
        
    Returns:
        Normalized Nx2 array
    """
    row_sums = probs.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return probs / row_sums
