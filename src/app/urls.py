import os, re, requests

URL_RE = re.compile(r'(https?://\S+|www\.\S+)', re.IGNORECASE)

def extract_urls(text: str):
    return URL_RE.findall(text or "")

def check_virustotal(url: str):
    api_key = os.getenv("VT_API_KEY")
    if not api_key:
        return {"ok": False, "reason": "Set VT_API_KEY in env to enable checks."}
    headers = {"x-apikey": api_key}
    r = requests.post("https://www.virustotal.com/api/v3/urls", headers=headers, data={"url": url})
    if r.status_code in (200, 201):
        data = r.json()
        return {"ok": True, "data": data}
    return {"ok": False, "status": r.status_code, "text": r.text}

def check_phishtank(url: str):
    api_key = os.getenv("PHISHTANK_API_KEY")
    if not api_key:
        return {"ok": False, "reason": "Set PHISHTANK_API_KEY in env to enable checks."}
    payload = {"url": url, "format": "json", "app_key": api_key}
    r = requests.post("https://checkurl.phishtank.com/checkurl/", data=payload)
    if r.status_code == 200:
        return {"ok": True, "data": r.json()}
    return {"ok": False, "status": r.status_code, "text": r.text}
