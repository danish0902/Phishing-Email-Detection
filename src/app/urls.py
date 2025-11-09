import os, re, requests
import time
import base64
from pathlib import Path

# Load .env file from project root - OVERRIDE any system env vars
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env", override=True)  # Force override system variables

URL_RE = re.compile(r'(https?://\S+|www\.\S+)', re.IGNORECASE)

# Reserved/invalid domains that VirusTotal can't analyze
RESERVED_DOMAINS = [
    '.example', '.test', '.invalid', '.localhost',
    'example.com', 'example.org', 'example.net',
    'test.com', 'localhost'
]

def extract_urls(text: str):
    """Extract and normalize URLs from text"""
    urls = URL_RE.findall(text or "")
    normalized_urls = []
    
    for url in urls:
        # Remove trailing punctuation that's not part of the URL
        url = url.rstrip('.,;:!?)\'"')
        
        # Add http:// if URL starts with www.
        if url.lower().startswith('www.') and not url.lower().startswith('http'):
            url = 'http://' + url
        
        # Basic validation - must have a protocol and domain
        if '://' in url and '.' in url:
            # Skip reserved/invalid domains
            is_reserved = any(reserved in url.lower() for reserved in RESERVED_DOMAINS)
            if not is_reserved:
                normalized_urls.append(url)
    
    return normalized_urls

def check_virustotal(url: str, get_report=True):
    """
    Check URL with VirusTotal API v3.
    
    Args:
        url: URL to check
        get_report: If True, also fetch the analysis report (requires additional API call)
        
    Returns:
        dict with analysis results
    """
    api_key = os.getenv("VT_API_KEY")
    if not api_key:
        return {"ok": False, "reason": "Set VT_API_KEY in env to enable checks."}
    
    headers = {"x-apikey": api_key}
    
    try:
        # Submit URL for analysis with timeout
        r = requests.post(
            "https://www.virustotal.com/api/v3/urls", 
            headers=headers, 
            data={"url": url},
            timeout=10  # 10 second timeout
        )
        
        # Check for rate limiting
        if r.status_code == 429:
            return {"ok": False, "reason": "Rate limit exceeded. Free tier: 4 requests/min, 500/day. Please wait."}
        
        # Check for authentication errors
        if r.status_code == 401:
            return {"ok": False, "reason": "Invalid API key. Check your VT_API_KEY."}
        
        # Check for other errors
        if r.status_code not in (200, 201):
            error_detail = "Unknown error"
            try:
                error_json = r.json()
                error_detail = error_json.get("error", {}).get("message", r.text)
                
                # Special handling for URL canonicalization error
                if "canonicalize" in error_detail.lower():
                    return {"ok": False, "reason": f"Invalid URL format: {url}. URL must be complete with protocol (http:// or https://)"}
            except:
                error_detail = r.text[:200]  # First 200 chars
            return {"ok": False, "status": r.status_code, "reason": f"API Error: {error_detail}"}
        
        data = r.json()
        analysis_id = data.get("data", {}).get("id", "")
        
        if not get_report:
            return {"ok": True, "data": data, "analysis_id": analysis_id}
        
        # Get the analysis report
        # Convert URL to base64 URL ID format for querying
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        
        # Wait a moment for analysis to complete (VT needs time)
        time.sleep(2)
        
        # Fetch the report
        report_url = f"https://www.virustotal.com/api/v3/urls/{url_id}"
        report_r = requests.get(report_url, headers=headers, timeout=10)
        
        # Check for rate limiting on report fetch
        if report_r.status_code == 429:
            return {"ok": False, "reason": "Rate limit exceeded while fetching report. Please wait."}
        
        if report_r.status_code == 200:
            report_data = report_r.json()
            stats = report_data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
            
            return {
                "ok": True,
                "analysis_id": analysis_id,
                "url": url,
                "stats": stats,
                "malicious": stats.get("malicious", 0),
                "suspicious": stats.get("suspicious", 0),
                "harmless": stats.get("harmless", 0),
                "undetected": stats.get("undetected", 0),
                "total_engines": sum(stats.values()),
                "report_data": report_data
            }
        
        # If report not ready, return submission info
        return {"ok": True, "data": data, "analysis_id": analysis_id, "report_pending": True}
        
    except requests.exceptions.Timeout:
        return {"ok": False, "reason": "Request timeout. VirusTotal server not responding."}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "reason": "Connection error. Check your internet connection."}
    except requests.exceptions.RequestException as e:
        return {"ok": False, "reason": f"Network error: {str(e)}"}
    except Exception as e:
        return {"ok": False, "reason": f"Unexpected error: {str(e)}"}

def check_phishtank(url: str):
    api_key = os.getenv("PHISHTANK_API_KEY")
    if not api_key:
        return {"ok": False, "reason": "Set PHISHTANK_API_KEY in env to enable checks."}
    payload = {"url": url, "format": "json", "app_key": api_key}
    r = requests.post("https://checkurl.phishtank.com/checkurl/", data=payload)
    if r.status_code == 200:
        return {"ok": True, "data": r.json()}
    return {"ok": False, "status": r.status_code, "text": r.text}
