import re
import html
from bs4 import BeautifulSoup

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)

def strip_html(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(" ", strip=True)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = strip_html(text)
    text = re.sub(URL_PATTERN, " <URL> ", text)
    text = re.sub(r'[^A-Za-z0-9<>@._\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text
