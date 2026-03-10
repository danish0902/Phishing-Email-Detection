"""
Unit tests for src/api/main.py  (FastAPI REST endpoints)
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Patch ModelLoader before importing the app so no artifact files are needed
# ---------------------------------------------------------------------------

mock_loader = MagicMock()
mock_loader.get_system_info.return_value = {
    "status": "Ready",
    "models": ["tfidf"],
    "thresholds": {"tfidf": 0.5},
    "total_models": 1,
}
mock_loader.predict.return_value = 0.92
mock_loader.predict_ensemble.return_value = {
    "probability": 0.88,
    "is_phishing": True,
    "prediction": 1,
    "confidence": 0.88,
    "individual_models": {"tfidf": {"probability": 0.88, "prediction": 1}},
    "votes": {"phishing": 1, "legitimate": 0},
}

with patch("src.api.main.ModelLoader", return_value=mock_loader):
    from src.api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestRoot:
    def test_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_contains_api_name(self):
        resp = client.get("/")
        assert "Phishing" in resp.json()["name"]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self):
        with patch("src.api.main.get_loader", return_value=mock_loader):
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_has_status_and_models(self):
        with patch("src.api.main.get_loader", return_value=mock_loader):
            resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "available_models" in data


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

class TestModels:
    def test_returns_200(self):
        resp = client.get("/models")
        assert resp.status_code == 200

    def test_lists_supported_models(self):
        resp = client.get("/models")
        data = resp.json()
        assert "supported_models" in data
        for m in ["tfidf", "cnn", "lstm", "bert", "hybrid", "ensemble"]:
            assert m in data["supported_models"]


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_valid_request_returns_200(self):
        with patch("src.api.main.get_loader", return_value=mock_loader):
            resp = client.post(
                "/predict",
                json={"email_text": "Your account has been compromised. Click here."},
            )
        assert resp.status_code == 200

    def test_response_fields_present(self):
        with patch("src.api.main.get_loader", return_value=mock_loader):
            resp = client.post(
                "/predict",
                json={"email_text": "Verify your account now."},
            )
        data = resp.json()
        for field in ("is_phishing", "probability", "label", "model_used"):
            assert field in data

    def test_phishing_label_when_above_threshold(self):
        mock_loader.predict.return_value = 0.92
        with patch("src.api.main.get_loader", return_value=mock_loader):
            resp = client.post(
                "/predict",
                json={"email_text": "Click to verify", "threshold": 0.5},
            )
        data = resp.json()
        assert data["label"] == "phishing"
        assert data["is_phishing"] is True

    def test_legitimate_label_when_below_threshold(self):
        mock_loader.predict.return_value = 0.1
        with patch("src.api.main.get_loader", return_value=mock_loader):
            resp = client.post(
                "/predict",
                json={"email_text": "Hello, hope you are well.", "threshold": 0.5},
            )
        data = resp.json()
        assert data["label"] == "legitimate"
        assert data["is_phishing"] is False

    def test_unknown_model_returns_422(self):
        with patch("src.api.main.get_loader", return_value=mock_loader):
            resp = client.post(
                "/predict",
                json={"email_text": "test", "model": "nonexistent"},
            )
        assert resp.status_code == 422

    def test_empty_email_text_rejected(self):
        resp = client.post("/predict", json={"email_text": ""})
        assert resp.status_code == 422

    def test_ensemble_model(self):
        with patch("src.api.main.get_loader", return_value=mock_loader):
            resp = client.post(
                "/predict",
                json={"email_text": "Verify now", "model": "ensemble"},
            )
        assert resp.status_code == 200
        assert resp.json()["model_used"] == "ensemble"

    def test_model_load_error_returns_503(self):
        bad_loader = MagicMock()
        bad_loader.predict.side_effect = FileNotFoundError("artifact missing")
        with patch("src.api.main.get_loader", return_value=bad_loader):
            resp = client.post(
                "/predict",
                json={"email_text": "hello", "model": "tfidf"},
            )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_valid_request_returns_200(self):
        mock_loader.predict.return_value = 0.85
        with patch("src.api.main.get_loader", return_value=mock_loader):
            resp = client.post(
                "/predict/batch",
                json={"emails": ["Email one", "Email two"]},
            )
        assert resp.status_code == 200

    def test_response_count_matches_input(self):
        mock_loader.predict.return_value = 0.5
        with patch("src.api.main.get_loader", return_value=mock_loader):
            resp = client.post(
                "/predict/batch",
                json={"emails": ["a", "b", "c"]},
            )
        data = resp.json()
        assert data["count"] == 3
        assert len(data["results"]) == 3

    def test_exceeding_100_emails_returns_422(self):
        with patch("src.api.main.get_loader", return_value=mock_loader):
            resp = client.post(
                "/predict/batch",
                json={"emails": ["email"] * 101},
            )
        assert resp.status_code == 422

    def test_empty_email_list_rejected(self):
        resp = client.post("/predict/batch", json={"emails": []})
        assert resp.status_code == 422

    def test_result_indices_are_correct(self):
        mock_loader.predict.return_value = 0.6
        with patch("src.api.main.get_loader", return_value=mock_loader):
            resp = client.post(
                "/predict/batch",
                json={"emails": ["first", "second", "third"]},
            )
        results = resp.json()["results"]
        indices = [r["index"] for r in results]
        assert indices == [0, 1, 2]

    def test_unknown_model_returns_422(self):
        resp = client.post(
            "/predict/batch",
            json={"emails": ["test"], "model": "unknown"},
        )
        assert resp.status_code == 422
