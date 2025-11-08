"""
LIME Explainer for Hybrid CNN+LSTM+URL model
Note: LIME only explains text features, not URL features
"""

import os
import sys
import joblib
import numpy as np
import re
import pandas as pd
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


class LimeHybrid:
    """
    LIME explainer for Hybrid CNN+LSTM+URL model
    Note: Explains text features only, URL features are kept constant
    """
    
    def __init__(self, model_path: str = None, maxlen: int = 200):
        """
        Initialize LIME Hybrid explainer
        
        Args:
            model_path: Path to hybrid model .h5 file
            maxlen: Maximum sequence length for padding
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow not available. Install with: pip install tensorflow"
            )
        
        self.maxlen = maxlen
        
        # Default paths
        if model_path is None:
            model_path = os.path.join(
                project_root,
                "artifacts",
                "hybrid_cnn_lstm.h5"
            )
        
        tokenizer_path = os.path.join(project_root, "artifacts", "tokenizer.joblib")
        scaler_path = os.path.join(project_root, "artifacts", "url_scaler.joblib")
        config_path = os.path.join(project_root, "artifacts", "hybrid_config.joblib")
        
        # Check files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hybrid model not found at {model_path}. "
                "Please train the model first using train_hybrid_model.py"
            )
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"URL scaler not found at {scaler_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Hybrid config not found at {config_path}")
        
        logger.info(f"Loading Hybrid model from {model_path}")
        self.model = load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)
        self.scaler = joblib.load(scaler_path)
        self.config = joblib.load(config_path)
        
        self.maxlen = self.config.get('max_len', maxlen)
        
        # Store original text for URL feature extraction
        self.original_text = None
        
        self.explainer = LimeTextExplainer(class_names=['legitimate', 'phishing'])
    
    def extract_url_features(self, text: str) -> np.ndarray:
        """
        Extract URL features from text (same as in model_loader.py)
        
        Args:
            text: Input text
            
        Returns:
            6-dimensional URL feature vector
        """
        if pd.isna(text) or not isinstance(text, str):
            return np.array([[0] * 6])
        
        # Find all URLs
        urls = re.findall(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            text
        )
        
        if not urls:
            return np.array([[0] * 6])
        
        url = urls[0]  # Use first URL
        
        features = [
            len(url),  # URL length
            url.count('.'),  # Number of dots
            1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0,  # Has IP
            len(re.findall(r'[@\-]', url)),  # Special characters
            1 if any(kw in url.lower() for kw in ["login", "update", "verify", "bank", "secure", "account", "confirm"]) else 0,
            1 if url.startswith("https") else 0  # Is HTTPS
        ]
        
        return np.array([features])
    
    def _prepare(self, texts: List[str], original_text: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenize and pad texts, extract URL features
        
        Args:
            texts: List of text samples (possibly perturbed by LIME)
            original_text: Original text for URL feature extraction
            
        Returns:
            Tuple of (padded_sequences, url_features)
        """
        # For LIME perturbations, use original text's URL features
        if original_text is not None:
            url_features = self.extract_url_features(original_text)
            url_features_scaled = self.scaler.transform(url_features)
            # Repeat for all texts in batch
            url_features_batch = np.repeat(url_features_scaled, len(texts), axis=0)
        else:
            # Extract URL features from each text
            url_features_list = [self.extract_url_features(text) for text in texts]
            url_features = np.vstack(url_features_list)
            url_features_batch = self.scaler.transform(url_features)
        
        # Clean texts AFTER URL extraction
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
        
        return padded, url_features_batch
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities for texts
        Uses original text's URL features for LIME perturbations
        
        Args:
            texts: List of text samples
            
        Returns:
            Nx2 array of probabilities [prob_legitimate, prob_phishing]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Prepare inputs (use original text for URL features)
        text_input, url_input = self._prepare(texts, self.original_text)
        
        # Predict
        phishing_probs = self.model.predict(
            [text_input, url_input],
            verbose=0
        ).flatten()
        
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
        Note: Only explains text features, URL features held constant
        
        Args:
            text: Input text to explain
            num_features: Number of top features to show
            num_samples: Number of perturbed samples for LIME
            
        Returns:
            Tuple of (explanation_object, prediction, probability)
        """
        logger.info(
            f"Generating LIME explanation for Hybrid model "
            f"(features={num_features}, samples={num_samples})"
        )
        logger.info("Note: Only text features explained, URL features held constant")
        
        # Store original text for URL feature extraction
        self.original_text = text
        
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
        
        # Add note about URL features
        note = """
        <div style="padding: 10px; background-color: #fff3cd; border: 1px solid #ffc107; margin: 10px 0;">
            <strong>Note:</strong> This explanation shows text feature contributions only. 
            URL features (length, special chars, etc.) are held constant during LIME perturbation.
        </div>
        """
        html = note + html
        
        # Save artifacts if requested
        if save_artifacts:
            timestamp = generate_timestamp()
            explanations_dir = get_explanations_dir()
            
            # Save HTML
            html_path = os.path.join(
                explanations_dir,
                f"lime_hybrid_{timestamp}.html"
            )
            save_html(html_path, html)
            
            # Get top features
            top_features = explanation.as_list(label=1)
            
            # Save metadata
            metadata = create_explanation_metadata(
                model_name="hybrid_cnn_lstm_url",
                text=text,
                prediction=prediction,
                probability=probability,
                top_features=top_features,
                num_features=num_features,
                num_samples=num_samples
            )
            metadata["note"] = "Text features only, URL features held constant"
            
            metadata_path = os.path.join(
                explanations_dir,
                f"lime_hybrid_{timestamp}_meta.json"
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


# Convenience function
def explain_hybrid(
    text: str,
    num_features: int = 10,
    num_samples: int = 500,
    save_artifacts: bool = True
) -> str:
    """Quick function to generate Hybrid LIME explanation"""
    explainer = LimeHybrid()
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
    
    print("Generating LIME explanation for Hybrid model...")
    explainer = LimeHybrid()
    html = explainer.explain_html(sample_text, num_features=10, num_samples=500)
    print(f"Generated explanation HTML ({len(html)} chars)")
    print("\nTop features:")
    features, pred, prob = explainer.explain_to_list(sample_text, num_features=10)
    for feature, weight in features[:5]:
        print(f"  {feature:20s}: {weight:+.4f}")
