"""
Unit tests for src/features/preprocess.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.features.preprocess import clean_text, strip_html


# ---------------------------------------------------------------------------
# strip_html
# ---------------------------------------------------------------------------

class TestStripHtml:
    def test_returns_empty_string_for_none(self):
        assert strip_html(None) == ""

    def test_returns_empty_string_for_empty(self):
        assert strip_html("") == ""

    def test_strips_basic_html_tags(self):
        result = strip_html("<p>Hello <b>World</b></p>")
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Hello" in result
        assert "World" in result

    def test_unescapes_html_entities(self):
        result = strip_html("&lt;div&gt; &amp; &quot;hello&quot; &apos;world&apos;")
        assert "&lt;" not in result
        assert "&amp;" not in result

    def test_preserves_plain_text(self):
        text = "This is plain text with no HTML."
        result = strip_html(text)
        assert result == text

    def test_strips_script_tags(self):
        result = strip_html("<script>alert('xss')</script>Normal text")
        assert "alert" not in result
        assert "Normal text" in result


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_returns_empty_for_none(self):
        assert clean_text(None) == ""

    def test_returns_empty_for_empty_string(self):
        assert clean_text("") == ""

    def test_lowercases_text(self):
        result = clean_text("HELLO WORLD")
        assert result == result.lower()

    def test_replaces_url_with_token(self):
        text = "Click here: https://phishing-site.com/login?user=abc"
        result = clean_text(text)
        assert "<url>" in result
        assert "phishing-site.com" not in result

    def test_replaces_http_url(self):
        text = "Visit http://example.com for details"
        result = clean_text(text)
        assert "<url>" in result

    def test_replaces_www_url(self):
        text = "Visit www.example.com for details"
        result = clean_text(text)
        assert "<url>" in result

    def test_strips_html_before_processing(self):
        text = "<p>Your <b>account</b> is at risk!</p>"
        result = clean_text(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "account" in result

    def test_normalises_whitespace(self):
        result = clean_text("hello   world\t\nfoo")
        assert "  " not in result

    def test_removes_most_special_characters(self):
        result = clean_text("hello! world# $100 great?")
        # Exclamation, hash, dollar sign, and question mark should be removed
        for char in ["!", "#", "$", "?"]:
            assert char not in result

    def test_preserves_allowed_special_characters(self):
        # Allowed: alphanumeric, @, ., _, <, >, space
        result = clean_text("user@example.com some_word")
        assert "@" in result
        assert "." in result
        assert "_" in result

    def test_multiple_urls_all_replaced(self):
        text = "Go to https://a.com or http://b.com for help"
        result = clean_text(text)
        assert "a.com" not in result
        assert "b.com" not in result

    def test_phishing_email_example_plain_url(self):
        # URL in plain text (not inside an HTML attribute) is replaced
        text = "Your account will be suspended. Visit http://bad-site.com/verify now!"
        result = clean_text(text)
        assert "bad-site.com" not in result
        assert "<url>" in result
        assert "account" in result
        assert "suspended" in result

    def test_phishing_email_example_href_stripped(self):
        # URL only in an HTML href attribute: BeautifulSoup strips it before
        # URL replacement, so the domain is removed but no <URL> token appears.
        text = (
            "<p>Dear User,</p>"
            "<p>Your account will be suspended. "
            "Click <a href='http://bad-site.com/verify'>HERE</a> now!</p>"
        )
        result = clean_text(text)
        # The domain is not present (stripped by BeautifulSoup)
        assert "bad-site.com" not in result
        # Visible text content is preserved
        assert "account" in result
        assert "suspended" in result
