"""
FastAPI REST API for Phishing Email Detection
Exposes model predictions as HTTP endpoints.

Usage:
    uvicorn src.api.main:app --reload

Endpoints:
    GET  /           — API info
    GET  /health     — Health check
    GET  /models     — List available models
    POST /predict    — Single-email prediction
    POST /predict/batch — Batch prediction from a list of emails
"""

import os
import sys
import time
from typing import List, Optional

# Add project root so that src.* imports work
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.features.preprocess import clean_text
from src.app.model_loader import ModelLoader

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Phishing Email Detection API",
    description=(
        "REST API for detecting phishing emails using multiple ML models "
        "(TF-IDF + LR, CNN, LSTM, BERT, Hybrid CNN+LSTM+URL)."
    ),
    version="1.0.0",
)

# Lazily loaded model loader — populated on first request
_loader: Optional[ModelLoader] = None


def get_loader() -> ModelLoader:
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader


VALID_MODELS = {"tfidf", "cnn", "lstm", "bert", "hybrid", "ensemble"}

# Maximum characters for inline text previews and error messages in responses.
_PREVIEW_LENGTH = 120


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    email_text: str = Field(..., min_length=1, description="Raw email body text")
    model: str = Field(
        default="tfidf",
        description="Model to use: tfidf | cnn | lstm | bert | hybrid | ensemble",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Decision threshold (probability above this → phishing)",
    )


class PredictResponse(BaseModel):
    is_phishing: bool
    probability: float
    label: str
    model_used: str
    cleaned_text_preview: str


class BatchPredictRequest(BaseModel):
    emails: List[str] = Field(
        ..., min_length=1, description="List of raw email body texts"
    )
    model: str = Field(default="tfidf")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class BatchPredictItem(BaseModel):
    index: int
    is_phishing: bool
    probability: float
    label: str


class BatchPredictResponse(BaseModel):
    model_used: str
    count: int
    results: List[BatchPredictItem]


class HealthResponse(BaseModel):
    status: str
    available_models: List[str]


class ModelsResponse(BaseModel):
    supported_models: List[str]
    default_model: str
    note: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "Phishing Email Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """Return API health and list of successfully loaded models."""
    loader = get_loader()
    info = loader.get_system_info()
    return HealthResponse(
        status=info.get("status", "Unknown"),
        available_models=info.get("models", []),
    )


@app.get("/models", response_model=ModelsResponse, tags=["Info"])
def list_models():
    """Return the list of supported model identifiers."""
    return ModelsResponse(
        supported_models=sorted(VALID_MODELS),
        default_model="tfidf",
        note=(
            "Models require pre-trained artifacts in the artifacts/ directory. "
            "Run the training scripts first. "
            "'ensemble' combines tfidf, cnn, and lstm via majority vote."
        ),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Classify a single email as phishing or legitimate.

    - **email_text**: raw email body (HTML or plain text)
    - **model**: which model to use (default: tfidf)
    - **threshold**: decision threshold (default: 0.5)
    """
    if request.model not in VALID_MODELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown model '{request.model}'. Choose from: {sorted(VALID_MODELS)}",
        )

    loader = get_loader()
    cleaned = clean_text(request.email_text)

    try:
        if request.model == "ensemble":
            result = loader.predict_ensemble(request.email_text)
            probability = result["probability"]
        else:
            probability = loader.predict(request.email_text, model_type=request.model)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifact not found. Train the model first. Details: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        )

    is_phishing = probability >= request.threshold
    label = "phishing" if is_phishing else "legitimate"
    preview = cleaned[:_PREVIEW_LENGTH] + ("..." if len(cleaned) > _PREVIEW_LENGTH else "")

    return PredictResponse(
        is_phishing=is_phishing,
        probability=round(float(probability), 6),
        label=label,
        model_used=request.model,
        cleaned_text_preview=preview,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictRequest):
    """
    Classify a batch of emails.

    - **emails**: list of raw email body strings (max 100)
    - **model**: which model to use (default: tfidf)
    - **threshold**: decision threshold (default: 0.5)
    """
    if len(request.emails) > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Batch size cannot exceed 100 emails per request.",
        )

    if request.model not in VALID_MODELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown model '{request.model}'. Choose from: {sorted(VALID_MODELS)}",
        )

    loader = get_loader()
    results: List[BatchPredictItem] = []

    for idx, email_text in enumerate(request.emails):
        try:
            if request.model == "ensemble":
                result = loader.predict_ensemble(email_text)
                probability = result["probability"]
            else:
                probability = loader.predict(email_text, model_type=request.model)

            is_phishing = probability >= request.threshold
            results.append(
                BatchPredictItem(
                    index=idx,
                    is_phishing=is_phishing,
                    probability=round(float(probability), 6),
                    label="phishing" if is_phishing else "legitimate",
                )
            )
        except Exception as exc:
            # Skip failed items but surface the error in the label
            results.append(
                BatchPredictItem(
                    index=idx,
                    is_phishing=False,
                    probability=0.0,
                    label=f"error: {str(exc)[:_PREVIEW_LENGTH]}",
                )
            )

    return BatchPredictResponse(
        model_used=request.model,
        count=len(results),
        results=results,
    )
