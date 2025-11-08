"""
LIME Explainer for Baseline TF-IDF + Logistic Regression model
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

from src.features.preprocess import clean_text
from src.xai.utils import (
    save_html, save_json, generate_timestamp,
    get_explanations_dir, create_explanation_metadata,
    logger
)


class LimeBaseline:
    """
    LIME explainer for baseline TF-IDF + Logistic Regression model
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize LIME baseline explainer
        
        Args:
            model_path: Path to baseline model joblib file
        """
        if model_path is None:
            model_path = os.path.join(
                project_root, 
                "artifacts", 
                "baseline_tfidf_lr.joblib"
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Baseline model not found at {model_path}. "
                "Please train the model first using train_tfidf_lr.py"
            )
        
        logger.info(f"Loading baseline model from {model_path}")
        self.pipeline = joblib.load(model_path)
        self.explainer = LimeTextExplainer(class_names=['legitimate', 'phishing'])
        
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities for texts
        
        Args:
            texts: List of text samples
            
        Returns:
            Nx2 array of probabilities [prob_legitimate, prob_phishing]
        """
        # Clean texts (baseline pipeline expects cleaned text)
        cleaned = [clean_text(text) for text in texts]
        probs = self.pipeline.predict_proba(cleaned)
        return probs
    
    def predict_proba_single(self, texts: List[str]) -> np.ndarray:
        """Wrapper for LIME (which may pass single strings)"""
        if isinstance(texts, str):
            texts = [texts]
        return self.predict_proba(texts)
    
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
        logger.info(f"Generating LIME explanation (features={num_features}, samples={num_samples})")
        
        # Get prediction
        proba = self.predict_proba([text])[0]
        prediction = int(np.argmax(proba))
        probability = float(proba[prediction])
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            text,
            self.predict_proba_single,
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
                f"lime_baseline_{timestamp}.html"
            )
            save_html(html_path, html)
            
            # Get top features
            top_features = explanation.as_list(label=1)
            
            # Save metadata
            metadata = create_explanation_metadata(
                model_name="baseline_tfidf_lr",
                text=text,
                prediction=prediction,
                probability=probability,
                top_features=top_features,
                num_features=num_features,
                num_samples=num_samples
            )
            
            metadata_path = os.path.join(
                explanations_dir,
                f"lime_baseline_{timestamp}_meta.json"
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
def explain_baseline(
    text: str,
    num_features: int = 10,
    num_samples: int = 500,
    save_artifacts: bool = True
) -> str:
    """
    Quick function to generate baseline LIME explanation
    
    Args:
        text: Input text
        num_features: Number of features
        num_samples: Number of samples
        save_artifacts: Whether to save files
        
    Returns:
        HTML string
    """
    explainer = LimeBaseline()
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
    
    print("Generating LIME explanation for baseline model...")
    explainer = LimeBaseline()
    html = explainer.explain_html(sample_text, num_features=10, num_samples=500)
    print(f"Generated explanation HTML ({len(html)} chars)")
    print("\nTop features:")
    features, pred, prob = explainer.explain_to_list(sample_text, num_features=10)
    for feature, weight in features[:5]:
        print(f"  {feature:20s}: {weight:+.4f}")
