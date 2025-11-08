"""
LIME Explainer for Keras models (CNN, LSTM)
"""

import os
import sys
import joblib
import numpy as np
from typing import List, Tuple
from pathlib import Path
from lime.lime_text import LimeTextExplainer

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from src.features.preprocess import clean_text
from src.xai.utils import (
    save_html, save_json, generate_timestamp,
    get_explanations_dir, create_explanation_metadata,
    logger
)


class LimeKeras:
    """
    LIME explainer for Keras models (CNN, LSTM)
    """
    
    def __init__(self, model_path: str = None, model_type: str = "lstm", maxlen: int = 200):
        """
        Initialize LIME Keras explainer
        
        Args:
            model_path: Path to Keras .h5 model file
            model_type: Type of model ('cnn' or 'lstm')
            maxlen: Maximum sequence length for padding
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow not available. Install with: pip install tensorflow"
            )
        
        self.model_type = model_type.lower()
        self.maxlen = maxlen
        
        # Default paths
        if model_path is None:
            model_path = os.path.join(
                project_root,
                "artifacts",
                f"{self.model_type}_keras.h5"
            )
        
        tokenizer_path = os.path.join(project_root, "artifacts", "tokenizer.joblib")
        
        # Check files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Keras model not found at {model_path}. "
                f"Please train the model first using train_{self.model_type}_keras.py"
            )
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                "Please ensure tokenizer was saved during training."
            )
        
        logger.info(f"Loading {self.model_type.upper()} model from {model_path}")
        self.model = load_model(model_path)
        
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = joblib.load(tokenizer_path)
        
        self.explainer = LimeTextExplainer(class_names=['legitimate', 'phishing'])
    
    def _prepare(self, texts: List[str]) -> np.ndarray:
        """
        Tokenize and pad texts
        
        Args:
            texts: List of text samples
            
        Returns:
            Padded sequences array
        """
        # Clean texts
        cleaned = [clean_text(text) for text in texts]
        
        # Tokenize
        sequences = self.tokenizer.texts_to_sequences(cleaned)
        
        # Pad
        padded = pad_sequences(
            sequences,
            maxlen=self.maxlen,
            padding='post',
            truncating='post'
        )
        
        return padded
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities for texts
        
        Args:
            texts: List of text samples
            
        Returns:
            Nx2 array of probabilities [prob_legitimate, prob_phishing]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Prepare input
        X = self._prepare(texts)
        
        # Predict (returns probability of phishing class)
        phishing_probs = self.model.predict(X, verbose=0).flatten()
        
        # Convert to Nx2 format [legitimate, phishing]
        legitimate_probs = 1 - phishing_probs
        probs = np.column_stack([legitimate_probs, phishing_probs])
        
        return probs
    
    def explain(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 500
    ) -> Tuple[object, int, float]:
        """
        Generate LIME explanation for text
        
        Args:
            text: Input text to explain
            num_features: Number of top features to show
            num_samples: Number of perturbed samples for LIME
            
        Returns:
            Tuple of (explanation_object, prediction, probability)
        """
        logger.info(
            f"Generating LIME explanation for {self.model_type.upper()} "
            f"(features={num_features}, samples={num_samples})"
        )
        
        # Get prediction
        proba = self.predict_proba([text])[0]
        prediction = int(np.argmax(proba))
        probability = float(proba[prediction])
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            text,
            self.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,)  # Explain phishing class
        )
        
        return explanation, prediction, probability
    
    def explain_html(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 500,
        save_artifacts: bool = True
    ) -> str:
        """
        Generate LIME explanation as HTML
        
        Args:
            text: Input text to explain
            num_features: Number of top features to show
            num_samples: Number of perturbed samples
            save_artifacts: Whether to save HTML and metadata files
            
        Returns:
            HTML string of explanation
        """
        explanation, prediction, probability = self.explain(
            text, num_features, num_samples
        )
        
        # Get HTML
        html = explanation.as_html()
        
        # Save artifacts if requested
        if save_artifacts:
            timestamp = generate_timestamp()
            explanations_dir = get_explanations_dir()
            
            # Save HTML
            html_path = os.path.join(
                explanations_dir,
                f"lime_{self.model_type}_{timestamp}.html"
            )
            save_html(html_path, html)
            
            # Get top features
            top_features = explanation.as_list(label=1)
            
            # Save metadata
            metadata = create_explanation_metadata(
                model_name=f"{self.model_type}_keras",
                text=text,
                prediction=prediction,
                probability=probability,
                top_features=top_features,
                num_features=num_features,
                num_samples=num_samples
            )
            
            metadata_path = os.path.join(
                explanations_dir,
                f"lime_{self.model_type}_{timestamp}_meta.json"
            )
            save_json(metadata_path, metadata)
            
            logger.info(f"Saved explanation artifacts with timestamp {timestamp}")
        
        return html
    
    def explain_to_list(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 500
    ) -> Tuple[List[Tuple[str, float]], int, float]:
        """
        Generate LIME explanation as list of (feature, weight) tuples
        
        Args:
            text: Input text to explain
            num_features: Number of top features
            num_samples: Number of perturbed samples
            
        Returns:
            Tuple of (feature_list, prediction, probability)
        """
        explanation, prediction, probability = self.explain(
            text, num_features, num_samples
        )
        
        features = explanation.as_list(label=1)
        return features, prediction, probability


# Convenience functions
def explain_cnn(
    text: str,
    num_features: int = 10,
    num_samples: int = 500,
    save_artifacts: bool = True
) -> str:
    """Quick function to generate CNN LIME explanation"""
    explainer = LimeKeras(model_type="cnn")
    return explainer.explain_html(text, num_features, num_samples, save_artifacts)


def explain_lstm(
    text: str,
    num_features: int = 10,
    num_samples: int = 500,
    save_artifacts: bool = True
) -> str:
    """Quick function to generate LSTM LIME explanation"""
    explainer = LimeKeras(model_type="lstm")
    return explainer.explain_html(text, num_features, num_samples, save_artifacts)


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Dear Customer,
    
    Your account requires immediate verification. Click here to verify your identity
    and prevent account suspension: http://secure-verify-account.com/login
    
    This link expires in 24 hours.
    
    Best regards,
    Security Team
    """
    
    print("Generating LIME explanation for LSTM model...")
    explainer = LimeKeras(model_type="lstm")
    html = explainer.explain_html(sample_text, num_features=10, num_samples=500)
    print(f"Generated explanation HTML ({len(html)} chars)")
    print("\nTop features:")
    features, pred, prob = explainer.explain_to_list(sample_text, num_features=10)
    for feature, weight in features[:5]:
        print(f"  {feature:20s}: {weight:+.4f}")
