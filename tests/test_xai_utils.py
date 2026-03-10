"""
Unit tests for src/xai/utils.py
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from src.xai.utils import (
    ensure_dir,
    save_html,
    save_json,
    safe_predict_proba_batch,
    generate_timestamp,
    truncate_text,
    create_explanation_metadata,
    normalize_probabilities,
)


# ---------------------------------------------------------------------------
# ensure_dir
# ---------------------------------------------------------------------------

class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        new_dir = str(tmp_path / "nested" / "dir")
        ensure_dir(new_dir)
        assert os.path.isdir(new_dir)

    def test_does_not_fail_if_dir_exists(self, tmp_path):
        ensure_dir(str(tmp_path))  # Already exists – should not raise


# ---------------------------------------------------------------------------
# save_html
# ---------------------------------------------------------------------------

class TestSaveHtml:
    def test_writes_html_file(self, tmp_path):
        path = str(tmp_path / "explanation.html")
        html_content = "<html><body>Test</body></html>"
        save_html(path, html_content)
        with open(path, "r", encoding="utf-8") as f:
            assert f.read() == html_content

    def test_creates_parent_directory(self, tmp_path):
        path = str(tmp_path / "subdir" / "explanation.html")
        save_html(path, "<html/>")
        assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# save_json
# ---------------------------------------------------------------------------

class TestSaveJson:
    def test_writes_valid_json(self, tmp_path):
        path = str(tmp_path / "meta.json")
        data = {"model": "baseline", "accuracy": 0.96, "label": "phishing"}
        save_json(path, data)
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_creates_parent_directory(self, tmp_path):
        path = str(tmp_path / "nested" / "meta.json")
        save_json(path, {"key": "value"})
        assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# safe_predict_proba_batch
# ---------------------------------------------------------------------------

class TestSafePredictProbaBatch:
    def _make_fn(self):
        """Returns a dummy predict_proba function."""
        def fn(texts):
            n = len(texts)
            return np.column_stack([np.zeros(n), np.ones(n)])
        return fn

    def test_returns_empty_array_for_empty_input(self):
        result = safe_predict_proba_batch(self._make_fn(), [])
        assert result.shape == (0, 2)

    def test_returns_correct_shape(self):
        result = safe_predict_proba_batch(self._make_fn(), ["a", "b", "c"])
        assert result.shape == (3, 2)

    def test_batches_large_input(self):
        texts = [f"text_{i}" for i in range(100)]
        result = safe_predict_proba_batch(self._make_fn(), texts, batch_size=10)
        assert result.shape == (100, 2)

    def test_handles_prediction_error_gracefully(self):
        def bad_fn(texts):
            raise RuntimeError("Model error")

        result = safe_predict_proba_batch(bad_fn, ["text1", "text2"])
        # Should return neutral 0.5 probabilities
        assert result.shape == (2, 2)
        assert np.allclose(result, 0.5)


# ---------------------------------------------------------------------------
# generate_timestamp
# ---------------------------------------------------------------------------

class TestGenerateTimestamp:
    def test_returns_non_empty_string(self):
        ts = generate_timestamp()
        assert isinstance(ts, str)
        assert len(ts) > 0

    def test_format_matches_expected(self):
        import re
        ts = generate_timestamp()
        assert re.match(r"^\d{8}T\d{6}$", ts), f"Unexpected timestamp format: {ts}"

    def test_two_successive_calls_are_unique_or_equal(self):
        ts1 = generate_timestamp()
        ts2 = generate_timestamp()
        # They could be equal if run within the same second – that's fine
        assert isinstance(ts1, str) and isinstance(ts2, str)


# ---------------------------------------------------------------------------
# truncate_text
# ---------------------------------------------------------------------------

class TestTruncateText:
    def test_short_text_is_unchanged(self):
        text = "Hello"
        assert truncate_text(text, max_len=120) == text

    def test_long_text_is_truncated(self):
        text = "A" * 200
        result = truncate_text(text, max_len=120)
        assert len(result) < 200
        assert result.endswith("...")

    def test_text_at_exact_limit_is_unchanged(self):
        text = "A" * 120
        assert truncate_text(text, max_len=120) == text


# ---------------------------------------------------------------------------
# create_explanation_metadata
# ---------------------------------------------------------------------------

class TestCreateExplanationMetadata:
    def test_returns_dict(self):
        meta = create_explanation_metadata(
            model_name="baseline",
            text="Hello World",
            prediction=1,
            probability=0.95,
        )
        assert isinstance(meta, dict)

    def test_phishing_label(self):
        meta = create_explanation_metadata("m", "text", 1, 0.9)
        assert meta["label"] == "phishing"

    def test_legitimate_label(self):
        meta = create_explanation_metadata("m", "text", 0, 0.1)
        assert meta["label"] == "legitimate"

    def test_contains_expected_keys(self):
        meta = create_explanation_metadata("baseline", "some text", 1, 0.85)
        for key in ["model", "timestamp", "label", "prediction", "probability"]:
            assert key in meta

    def test_top_tokens_serialised(self):
        features = [("click", 0.4), ("verify", 0.3), ("account", 0.2)]
        meta = create_explanation_metadata("m", "text", 1, 0.9, top_features=features)
        assert "top_tokens" in meta
        assert len(meta["top_tokens"]) == 3

    def test_probability_stored_as_float(self):
        meta = create_explanation_metadata("m", "text", 1, np.float32(0.95))
        assert isinstance(meta["probability"], float)


# ---------------------------------------------------------------------------
# normalize_probabilities
# ---------------------------------------------------------------------------

class TestNormalizeProbabilities:
    def test_rows_sum_to_one(self):
        probs = np.array([[0.3, 0.7], [0.6, 0.4], [0.9, 0.1]])
        result = normalize_probabilities(probs)
        np.testing.assert_allclose(result.sum(axis=1), 1.0)

    def test_handles_zero_row(self):
        probs = np.array([[0.0, 0.0], [0.5, 0.5]])
        result = normalize_probabilities(probs)
        # Zero rows should not produce NaN
        assert not np.any(np.isnan(result))

    def test_already_normalised_unchanged(self):
        probs = np.array([[0.3, 0.7], [0.8, 0.2]])
        result = normalize_probabilities(probs)
        np.testing.assert_allclose(result, probs)
