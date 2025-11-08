"""
Explainable AI (XAI) module for PhishGuard
Provides LIME and SHAP explanations for all model types
"""

from .utils import ensure_dir, save_html, safe_predict_proba_batch

__all__ = [
    'ensure_dir',
    'save_html', 
    'safe_predict_proba_batch',
]
