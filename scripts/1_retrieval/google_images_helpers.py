"""
google_images_helpers.py
Browser settings, CAPTCHA handling and metadata writer used by 01_search_google_images.py.
The settings are those of crawler_config.py used for the paper; the functions are copied
unchanged from the project's earlier crawler (crawl_drug.py), whose own crawl function is not used.
"""
import os, re, time, urllib.parse
from datetime import datetime

# ── Browser settings used for the paper (from crawler_config.py on the lab PC) ──
HEADLESS = False
LANGUAGE = "en-US"
IMPLICIT_WAIT = 3
PAGE_LOAD_TIMEOUT = 30
CHROME_OPTIONS = ["--disable-notifications", "--log-level=3", "--no-sandbox",
                  "--disable-dev-shm-usage", "--start-maximized", "--window-size=1400,1000"]
EXCLUDE_SWITCHES = ["enable-automation", "enable-logging"]
USE_AUTOMATION_EXTENSION = False

# ── File helpers ──────────────────────────────────────────────────────────────
IMG_RE = re.compile(r'\.(jpg|jpeg|png|webp)$', re.I)
NUM_RE = re.compile(r'^(\d{5})\.(jpg|jpeg|png|webp)$', re.I)

def _list_images(d):
    return [f for f in os.listdir(d) if IMG_RE.search(f)] if os.path.isdir(d) else []

def _image_count(d): return len(_list_images(d))

def _next_num(d):
    nums = [int(NUM_RE.match(f).group(1))
            for f in _list_images(d) if NUM_RE.match(f)]
    return 0 if not nums else max(nums) + 1

def _ext(hint):
    s = str(hint).lower()
    if ".png" in s or "image/png" in s:  return ".png"
    if ".webp" in s or "image/webp" in s: return ".jpg"  # convert webp→jpg
    return ".jpg"

def _domain(url):
    try:    return urllib.parse.urlparse(url).netloc.lower()
    except: return ""

def _clean(v):
    return str(v or "").replace("\t"," ").replace("\r"," ").replace("\n"," ").strip()

# ── reCAPTCHA detection ───────────────────────────────────────────────────────
def _is_captcha(driver):
    try:
        url = driver.current_url or ""
        if "sorry/index" in url or "recaptcha" in url.lower(): return True
        if "unusual traffic" in driver.page_source: return True
    except: pass
    return False

def _handle_captcha(driver):
    if _is_captcha(driver):
        print("  [CAPTCHA] Solve in the browser window, then press Enter...")
        if not HEADLESS:
            input("  Press Enter after solving... ")
            time.sleep(2.0)
            return True
        else:
            print("  [SKIP] Headless — cannot solve CAPTCHA.")
            return False
    return True

# ── Metadata writers ──────────────────────────────────────────────────────────
METADATA_COLS = ["no", "image_url", "source_page_url", "source_page_title",
                 "site_name", "image_domain", "alt_text", "w", "h", "timestamp"]

def _write_metadata(out_dir, numstr, image_url, source_page_url,
                    source_page_title, alt_text, w, h):
    meta = os.path.join(out_dir, "metadata.tsv")
    write_header = not os.path.exists(meta)
    site_name = _domain(source_page_url)
    row = [numstr, image_url, source_page_url, source_page_title,
           site_name, _domain(image_url), alt_text, str(w), str(h),
           datetime.now().isoformat()]
    with open(meta, "a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(METADATA_COLS) + "\n")
        f.write("\t".join(_clean(x) for x in row) + "\n")
    # GUI-compatible list.tsv
    with open(os.path.join(out_dir, "list.tsv"), "a", encoding="utf-8") as f:
        f.write(f"{numstr}\t{image_url}\n")
