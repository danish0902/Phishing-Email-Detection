"""
Unit tests for src/app/urls.py  (URL extraction and VirusTotal helpers)
"""
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.app.urls import extract_urls, check_virustotal, check_phishtank


# ---------------------------------------------------------------------------
# extract_urls
# ---------------------------------------------------------------------------

class TestExtractUrls:
    def test_returns_empty_list_for_none(self):
        assert extract_urls(None) == []

    def test_returns_empty_list_for_empty_string(self):
        assert extract_urls("") == []

    def test_extracts_http_url(self):
        urls = extract_urls("Visit http://phish-site.com/page for details.")
        assert len(urls) == 1
        assert urls[0] == "http://phish-site.com/page"

    def test_extracts_https_url(self):
        urls = extract_urls("Go to https://secure-bank.com/login now")
        assert len(urls) == 1
        assert "https://secure-bank.com/login" in urls[0]

    def test_extracts_www_url_and_adds_protocol(self):
        urls = extract_urls("Visit www.phishing-site.org/verify")
        assert len(urls) == 1
        assert urls[0].startswith("http://")

    def test_extracts_multiple_urls(self):
        text = "Go to http://first.com or https://second.com for info"
        urls = extract_urls(text)
        assert len(urls) == 2

    def test_strips_trailing_punctuation(self):
        urls = extract_urls("Check http://phish-site.com/page.")
        assert urls[0].endswith("/page")

    def test_strips_trailing_parenthesis(self):
        urls = extract_urls("Link (http://phish-site.com/page)")
        assert ")" not in urls[0]

    def test_skips_reserved_domains(self):
        urls = extract_urls("Go to http://example.com and http://test.com")
        assert len(urls) == 0

    def test_no_urls_in_plain_text(self):
        urls = extract_urls("This is just plain text with no links at all.")
        assert urls == []

    def test_extracts_url_with_query_params(self):
        urls = extract_urls("Click http://phish.net/verify?user=abc&token=xyz")
        assert len(urls) == 1
        assert urls[0].startswith("http://phish.net/")


# ---------------------------------------------------------------------------
# check_virustotal (mocked – no real API calls)
# ---------------------------------------------------------------------------

class TestCheckVirusTotal:
    def test_returns_error_when_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove VT_API_KEY if set
            os.environ.pop("VT_API_KEY", None)
            result = check_virustotal("http://evil.com")
        assert result["ok"] is False
        assert "VT_API_KEY" in result["reason"]

    def test_returns_rate_limit_error_on_429(self):
        mock_response = MagicMock()
        mock_response.status_code = 429
        with patch.dict(os.environ, {"VT_API_KEY": "fakekey"}):
            with patch("src.app.urls.requests.post", return_value=mock_response):
                result = check_virustotal("http://evil.com")
        assert result["ok"] is False
        assert "rate limit" in result["reason"].lower()

    def test_returns_auth_error_on_401(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        with patch.dict(os.environ, {"VT_API_KEY": "badkey"}):
            with patch("src.app.urls.requests.post", return_value=mock_response):
                result = check_virustotal("http://evil.com")
        assert result["ok"] is False
        assert "api key" in result["reason"].lower()

    def test_handles_connection_error(self):
        import requests as req
        with patch.dict(os.environ, {"VT_API_KEY": "fakekey"}):
            with patch("src.app.urls.requests.post", side_effect=req.exceptions.ConnectionError("fail")):
                result = check_virustotal("http://evil.com")
        assert result["ok"] is False
        assert "connection" in result["reason"].lower()

    def test_handles_timeout(self):
        import requests as req
        with patch.dict(os.environ, {"VT_API_KEY": "fakekey"}):
            with patch("src.app.urls.requests.post", side_effect=req.exceptions.Timeout("timed out")):
                result = check_virustotal("http://evil.com")
        assert result["ok"] is False
        assert "timeout" in result["reason"].lower()


# ---------------------------------------------------------------------------
# check_phishtank (mocked – no real API calls)
# ---------------------------------------------------------------------------

class TestCheckPhishTank:
    def test_returns_error_when_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PHISHTANK_API_KEY", None)
            result = check_phishtank("http://evil.com")
        assert result["ok"] is False
        assert "PHISHTANK_API_KEY" in result["reason"]

    def test_returns_ok_on_200(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": {"valid": True}}
        with patch.dict(os.environ, {"PHISHTANK_API_KEY": "fakekey"}):
            with patch("src.app.urls.requests.post", return_value=mock_response):
                result = check_phishtank("http://evil.com")
        assert result["ok"] is True
        assert "data" in result

    def test_returns_error_on_non_200(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        with patch.dict(os.environ, {"PHISHTANK_API_KEY": "fakekey"}):
            with patch("src.app.urls.requests.post", return_value=mock_response):
                result = check_phishtank("http://evil.com")
        assert result["ok"] is False
        assert result["status"] == 500
