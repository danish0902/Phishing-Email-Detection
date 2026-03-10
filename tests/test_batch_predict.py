"""
Unit tests for src/models/batch_predict.py
"""
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
from src.models.batch_predict import batch_predict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(tmp_path, rows, col="Email Text"):
    """Write a small CSV to a temp file and return its path."""
    df = pd.DataFrame({col: rows})
    path = str(tmp_path / "input.csv")
    df.to_csv(path, index=False)
    return path


def _mock_loader_predict(self, text, model_type="tfidf"):
    """Deterministic stub: returns 0.9 if 'phish' in text, else 0.1."""
    return 0.9 if "phish" in text.lower() else 0.1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBatchPredict:
    def test_raises_if_input_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            batch_predict(
                input_path=str(tmp_path / "missing.csv"),
                output_path=str(tmp_path / "out.csv"),
            )

    def test_raises_if_text_col_missing(self, tmp_path):
        path = _make_csv(tmp_path, ["hello"], col="body")
        with pytest.raises(ValueError, match="not found"):
            with patch(
                "src.models.batch_predict.ModelLoader.predict",
                _mock_loader_predict,
            ):
                batch_predict(
                    input_path=path,
                    output_path=str(tmp_path / "out.csv"),
                    text_col="Email Text",  # wrong column
                )

    def test_output_csv_created(self, tmp_path):
        path = _make_csv(tmp_path, ["normal email text", "phishing click here"])
        out = str(tmp_path / "out.csv")
        with patch(
            "src.models.batch_predict.ModelLoader.predict",
            _mock_loader_predict,
        ):
            batch_predict(input_path=path, output_path=out)
        assert os.path.isfile(out)

    def test_output_has_prediction_columns(self, tmp_path):
        path = _make_csv(tmp_path, ["hello world", "verify account phish"])
        out = str(tmp_path / "out.csv")
        with patch(
            "src.models.batch_predict.ModelLoader.predict",
            _mock_loader_predict,
        ):
            df = batch_predict(input_path=path, output_path=out)
        for col in ("pred_probability", "pred_label", "pred_error"):
            assert col in df.columns

    def test_correct_label_assignment(self, tmp_path):
        rows = ["totally normal email", "PHISH click here now"]
        path = _make_csv(tmp_path, rows)
        out = str(tmp_path / "out.csv")
        with patch(
            "src.models.batch_predict.ModelLoader.predict",
            _mock_loader_predict,
        ):
            df = batch_predict(input_path=path, output_path=out, threshold=0.5)
        assert df.iloc[0]["pred_label"] == "legitimate"
        assert df.iloc[1]["pred_label"] == "phishing"

    def test_custom_threshold(self, tmp_path):
        rows = ["marginal email"]  # stub returns 0.1 → below 0.05 threshold
        path = _make_csv(tmp_path, rows)
        out = str(tmp_path / "out.csv")
        with patch(
            "src.models.batch_predict.ModelLoader.predict",
            _mock_loader_predict,
        ):
            df = batch_predict(input_path=path, output_path=out, threshold=0.05)
        assert df.iloc[0]["pred_label"] == "phishing"  # 0.1 >= 0.05

    def test_model_error_recorded_in_pred_error(self, tmp_path):
        path = _make_csv(tmp_path, ["some text"])
        out = str(tmp_path / "out.csv")

        def _bad_predict(self, text, model_type="tfidf"):
            raise RuntimeError("model exploded")

        with patch("src.models.batch_predict.ModelLoader.predict", _bad_predict):
            df = batch_predict(input_path=path, output_path=out)
        assert df.iloc[0]["pred_label"] == "error"
        assert "model exploded" in df.iloc[0]["pred_error"]

    def test_custom_text_column(self, tmp_path):
        df_in = pd.DataFrame({"body": ["hello", "click phish link"]})
        path = str(tmp_path / "custom.csv")
        df_in.to_csv(path, index=False)
        out = str(tmp_path / "out.csv")
        with patch(
            "src.models.batch_predict.ModelLoader.predict",
            _mock_loader_predict,
        ):
            df = batch_predict(input_path=path, output_path=out, text_col="body")
        assert "pred_label" in df.columns

    def test_handles_nan_text_gracefully(self, tmp_path):
        df_in = pd.DataFrame({"Email Text": ["normal text", None, "phish now"]})
        path = str(tmp_path / "nan.csv")
        df_in.to_csv(path, index=False)
        out = str(tmp_path / "out.csv")
        with patch(
            "src.models.batch_predict.ModelLoader.predict",
            _mock_loader_predict,
        ):
            df = batch_predict(input_path=path, output_path=out)
        assert len(df) == 3
