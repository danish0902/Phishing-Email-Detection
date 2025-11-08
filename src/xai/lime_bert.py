"""
LIME Explainer for BERT (DistilBERT) model
"""

import os
import sys
import numpy as np
from typing import List, Tuple
from pathlib import Path
from lime.lime_text import LimeTextExplainer

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from src.xai.utils import (
    save_html, save_json, generate_timestamp,
    get_explanations_dir, create_explanation_metadata,
    logger
)


class LimeBERT:
    """
    LIME explainer for fine-tuned DistilBERT model
    """
    
    def __init__(self, model_path: str = None, max_length: int = 256):
        """
        Initialize LIME BERT explainer
        
        Args:
            model_path: Path to fine-tuned BERT model directory
            max_length: Maximum sequence length
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library not available. "
                "Install with: pip install transformers torch"
            )
        
        self.max_length = max_length
        
        # Default path
        if model_path is None:
            model_path = os.path.join(project_root, "artifacts", "bert_model")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"BERT model not found at {model_path}. "
                "Please train the model first using train_bert.py"
            )
        
        logger.info(f"Loading BERT model from {model_path}")
        
        # Load tokenizer (use base distilbert since we didn't save custom tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Load fine-tuned model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode
        
        self.explainer = LimeTextExplainer(class_names=['legitimate', 'phishing'])
    
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
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).numpy()
        
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
            f"Generating LIME explanation for BERT "
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
                f"lime_bert_{timestamp}.html"
            )
            save_html(html_path, html)
            
            # Get top features
            top_features = explanation.as_list(label=1)
            
            # Save metadata
            metadata = create_explanation_metadata(
                model_name="bert_distilbert",
                text=text,
                prediction=prediction,
                probability=probability,
                top_features=top_features,
                num_features=num_features,
                num_samples=num_samples
            )
            
            metadata_path = os.path.join(
                explanations_dir,
                f"lime_bert_{timestamp}_meta.json"
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
def explain_bert(
    text: str,
    num_features: int = 10,
    num_samples: int = 500,
    save_artifacts: bool = True
) -> str:
    """
    Quick function to generate BERT LIME explanation
    
    Args:
        text: Input text
        num_features: Number of features
        num_samples: Number of samples
        save_artifacts: Whether to save files
        
    Returns:
        HTML string
    """
    explainer = LimeBERT()
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
    
    print("Generating LIME explanation for BERT model...")
    explainer = LimeBERT()
    html = explainer.explain_html(sample_text, num_features=10, num_samples=500)
    print(f"Generated explanation HTML ({len(html)} chars)")
    print("\nTop features:")
    features, pred, prob = explainer.explain_to_list(sample_text, num_features=10)
    for feature, weight in features[:5]:
        print(f"  {feature:20s}: {weight:+.4f}")
