"""
SHAP Explainer for Baseline TF-IDF + Logistic Regression model

SHAP (SHapley Additive exPlanations) provides globally consistent feature
attributions, complementing the LIME explanations already in this project.

Usage:
    from src.xai.shap_baseline import ShapBaseline

    explainer = ShapBaseline()
    features, pred, prob = explainer.explain_to_list("Dear user, verify your account...")
    html = explainer.explain_html("Dear user, verify your account...")
"""

import os
import sys
import joblib
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

import shap

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features.preprocess import clean_text
from src.xai.utils import (
    save_html,
    save_json,
    generate_timestamp,
    get_explanations_dir,
    create_explanation_metadata,
    logger,
)


class ShapBaseline:
    """
    SHAP explainer for the TF-IDF + Logistic Regression baseline model.

    Uses ``shap.LinearExplainer`` which works directly with the sparse TF-IDF
    matrix, giving fast and exact Shapley values for linear models.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise the SHAP baseline explainer.

        Args:
            model_path: Path to the baseline model joblib file.
                        Defaults to artifacts/baseline_tfidf_lr.joblib.
        """
        if model_path is None:
            model_path = os.path.join(
                project_root, "artifacts", "baseline_tfidf_lr.joblib"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Baseline model not found at {model_path}. "
                "Please train the model first using train_tfidf_lr.py"
            )

        logger.info(f"Loading baseline model from {model_path}")
        self.pipeline = joblib.load(model_path)
        self.vectorizer = self.pipeline.named_steps["tfidf"]
        self.classifier = self.pipeline.named_steps["clf"]

        # Build the SHAP explainer on the sparse TF-IDF representation.
        # masker=shap.maskers.Independent samples the background from the
        # training distribution encoded in the vectorizer vocabulary.
        self._explainer: Optional[shap.LinearExplainer] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_explainer(self) -> shap.LinearExplainer:
        """Lazily build the SHAP LinearExplainer (once per instance)."""
        if self._explainer is None:
            logger.info("Building SHAP LinearExplainer …")
            # Use the zero-vector as a simple background (fast, deterministic)
            background = np.zeros((1, len(self.vectorizer.vocabulary_)))
            self._explainer = shap.LinearExplainer(
                self.classifier,
                background,
                feature_perturbation="interventional",
            )
        return self._explainer

    def _transform(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clean text and transform to TF-IDF feature vector.

        Returns:
            dense_matrix (1 × vocab_size),  feature_names array
        """
        cleaned = clean_text(text)
        sparse = self.vectorizer.transform([cleaned])
        dense = sparse.toarray()
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        return dense, feature_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_proba(self, text: str) -> Tuple[float, float]:
        """
        Return (prob_legitimate, prob_phishing) for a single text.
        """
        cleaned = clean_text(text)
        probs = self.pipeline.predict_proba([cleaned])[0]
        return float(probs[0]), float(probs[1])

    def explain(
        self,
        text: str,
        num_features: int = 10,
    ) -> Tuple[List[Tuple[str, float]], int, float]:
        """
        Compute SHAP values and return top contributing features.

        Args:
            text: Raw email text.
            num_features: Number of top features to return.

        Returns:
            (feature_list, prediction, probability)
            where feature_list is a list of (token, shap_value) sorted by
            absolute impact (descending).
        """
        dense, feature_names = self._transform(text)
        explainer = self._get_explainer()

        shap_values = explainer.shap_values(dense)
        # LinearExplainer returns shape (1, n_features) for binary models
        # Use the phishing class (index 1) if multi-output, else use single array
        if isinstance(shap_values, list):
            sv = shap_values[1][0]   # phishing class
        else:
            sv = shap_values[0]      # already for positive class

        # Build (token, shap_value) pairs and sort by |shap_value|
        pairs = list(zip(feature_names, sv))
        # Filter out zero-contribution features (not present in the text)
        pairs = [(tok, float(val)) for tok, val in pairs if val != 0.0]
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_pairs = pairs[:num_features]

        # Derive prediction from pipeline
        prob_legit, prob_phish = self.predict_proba(text)
        prediction = 1 if prob_phish >= 0.5 else 0
        probability = prob_phish if prediction == 1 else prob_legit

        return top_pairs, prediction, probability

    def explain_to_list(
        self,
        text: str,
        num_features: int = 10,
    ) -> Tuple[List[Tuple[str, float]], int, float]:
        """Alias for ``explain`` — mirrors the LIME module interface."""
        return self.explain(text, num_features)

    def explain_html(
        self,
        text: str,
        num_features: int = 10,
        save_artifacts: bool = True,
    ) -> str:
        """
        Generate an HTML bar-chart showing the top SHAP features.

        Args:
            text: Raw email text.
            num_features: Number of features to display.
            save_artifacts: Whether to save HTML/JSON files to disk.

        Returns:
            HTML string.
        """
        top_pairs, prediction, probability = self.explain(text, num_features)
        label = "phishing" if prediction == 1 else "legitimate"
        html = _render_shap_html(top_pairs, label, probability)

        if save_artifacts:
            timestamp = generate_timestamp()
            explanations_dir = get_explanations_dir()

            html_path = os.path.join(
                explanations_dir, f"shap_baseline_{timestamp}.html"
            )
            save_html(html_path, html)

            metadata = create_explanation_metadata(
                model_name="baseline_tfidf_lr_shap",
                text=text,
                prediction=prediction,
                probability=probability,
                top_features=top_pairs,
                num_features=num_features,
                num_samples=0,  # SHAP is exact, not sample-based
            )
            meta_path = os.path.join(
                explanations_dir, f"shap_baseline_{timestamp}_meta.json"
            )
            save_json(meta_path, metadata)
            logger.info(f"Saved SHAP explanation artifacts with timestamp {timestamp}")

        return html


# ---------------------------------------------------------------------------
# HTML rendering helper
# ---------------------------------------------------------------------------

# Scale factor for mapping SHAP values to a bar width percentage.
# A SHAP value of 1/300 ≈ 0.0033 maps to 1% width; values above 1/3 are
# capped at 100% so the chart always fits within the table cell.
_SHAP_DISPLAY_SCALE = 300


def _render_shap_html(
    features: List[Tuple[str, float]],
    label: str,
    probability: float,
) -> str:
    """Render a minimal self-contained HTML bar chart for SHAP values."""
    bar_rows = []
    for token, shap_val in features:
        pct = min(abs(shap_val) * _SHAP_DISPLAY_SCALE, 100)  # scale for display
        colour = "#d62728" if shap_val > 0 else "#1f77b4"
        bar_rows.append(
            f"<tr>"
            f"<td style='padding:2px 8px;text-align:right;font-family:monospace'>{token}</td>"
            f"<td style='padding:2px 4px'>"
            f"<div style='background:{colour};width:{pct:.1f}%;height:16px;"
            f"border-radius:3px'></div></td>"
            f"<td style='padding:2px 8px;font-size:0.85em;color:{colour}'>"
            f"{shap_val:+.4f}</td>"
            f"</tr>"
        )
    rows_html = "\n".join(bar_rows)

    header_colour = "#d62728" if label == "phishing" else "#2ca02c"
    return f"""<!DOCTYPE html>
<html>
<head><meta charset='utf-8'><title>SHAP Explanation</title></head>
<body style='font-family:Arial,sans-serif;max-width:640px;margin:24px auto'>
  <h2 style='color:{header_colour}'>
    Prediction: <em>{label}</em>
    &nbsp;<small style='font-size:0.7em'>(p = {probability:.4f})</small>
  </h2>
  <p style='font-size:0.9em'>
    Red bars push the prediction toward <strong>phishing</strong>;
    blue bars push toward <strong>legitimate</strong>.
  </p>
  <table style='border-collapse:collapse;width:100%'>
    <thead>
      <tr>
        <th style='text-align:right;padding:4px 8px'>Token</th>
        <th style='text-align:left;padding:4px 8px'>SHAP value</th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def explain_shap_baseline(
    text: str,
    num_features: int = 10,
    save_artifacts: bool = True,
) -> str:
    """
    Quick helper to generate a SHAP explanation for the baseline model.

    Args:
        text: Raw email body.
        num_features: Number of top features.
        save_artifacts: Whether to save HTML/JSON files.

    Returns:
        HTML string.
    """
    explainer = ShapBaseline()
    return explainer.explain_html(text, num_features, save_artifacts)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE = """
    Dear Customer,

    Your account requires immediate verification. Click here to verify your
    identity and prevent account suspension:
    http://secure-verify-account.com/login

    This link expires in 24 hours.

    Best regards,
    Security Team
    """

    print("Generating SHAP explanation for baseline model …")
    explainer = ShapBaseline()
    features, pred, prob = explainer.explain_to_list(SAMPLE, num_features=10)
    print(f"\nPrediction : {'phishing' if pred == 1 else 'legitimate'} (p = {prob:.4f})")
    print("\nTop SHAP features:")
    for token, val in features:
        bar = "█" * int(abs(val) * 50)
        direction = "→phishing" if val > 0 else "→legit   "
        print(f"  {token:25s} {direction}  {val:+.4f}  {bar}")
