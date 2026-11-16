"""
01_search_google_images.py
===============
Fresh Google Images crawl with CORRECT image -> source-page attribution.

Differences from the earlier crawler (June run):
  * The image URL and the source-page URL are read from the SAME preview element
    in one step (the preview <img> and the link wrapped around it), after waiting
    until the preview has really changed. No page-wide link scoring, no .gov bonus.
  * Provenance check: the source page is fetched and we record whether its HTML
    contains the image (provenance.csv). Stage 3 can report results with/without it.
  * Same output format as before (metadata.tsv, list.tsv, crawl_log.csv,
    run_summary.json, 00000.jpg ...), so LightGlue and Stage 3 scripts work unchanged.

Usage (from the repository root):
    python scripts/1_retrieval/01_search_google_images.py --run_id fresh_oct --only Q003,Q038      # test
    python scripts/1_retrieval/01_search_google_images.py --run_id fresh_oct                         # all
    python scripts/1_retrieval/01_search_google_images.py --run_id fresh_oct_form --expand form     # expanded
    python scripts/1_retrieval/01_search_google_images.py --run_id fresh_oct_tz --country tz        # Tanzania setting
Resumable: products whose folder already has run_summary.json are skipped.
A product with no thumbnails (e.g. CAPTCHA not cleared) gets crawl_failed.json
instead of run_summary.json, so it is retried on the next run.
Re-crawl: products in --only are re-crawled; an existing folder is never appended
to - it is moved to crawl/_superseded/ and logged in tables/rerun_log.csv with
--reason.
"""
import argparse, json, os, sys, time, random, urllib.parse, ntpath, re, shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd, requests, certifi
from PIL import Image

EXP_ROOT    = Path(__file__).resolve().parents[2]  # repository root
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import google_images_helpers as cd                    # June helpers: search URL, CAPTCHA, metadata writer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

CHROMEDRIVER = EXP_ROOT / "tools" / "chromedriver"
COLLECT_NUM  = 10
MAX_THUMBS   = 60
MIN_SIDE     = 200
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# ── driver (explicit chromedriver path) ─────────────────────────────────────
def new_driver():
    p = Path(CHROMEDRIVER)
    exe = p if p.is_file() else next(iter(sorted(p.rglob("chromedriver.exe")) or sorted(p.rglob("chromedriver"))))
    opts = webdriver.ChromeOptions()
    if cd.HEADLESS: opts.add_argument("--headless=new")
    opts.add_argument(f"--lang={cd.LANGUAGE}")
    for a in cd.CHROME_OPTIONS: opts.add_argument(a)
    opts.add_experimental_option("excludeSwitches", cd.EXCLUDE_SWITCHES)
    opts.add_experimental_option("useAutomationExtension", cd.USE_AUTOMATION_EXTENSION)
    drv = webdriver.Chrome(service=Service(str(exe)), options=opts)
    drv.implicitly_wait(cd.IMPLICIT_WAIT)
    try: drv.set_page_load_timeout(cd.PAGE_LOAD_TIMEOUT)
    except Exception: pass
    return drv

# ── JS: thumbnails, and the current preview (image + its own link) ──────────
JS_THUMBS = r"""
return [...document.querySelectorAll('img')].filter(im => {
  const s = im.currentSrc || im.src || '';
  const r = im.getBoundingClientRect();
  return (s.startsWith('data:image') || s.includes('encrypted-tbn')) && r.width >= 60 && r.height >= 60;
});
"""
# The preview = the largest visible full-size (http, non-Google) image on screen.
# Its source page = the link wrapped around that same image (or nearest ancestor link).
JS_PREVIEW = r"""
const bad = h => /(^|\.)(google\.|gstatic\.|googleusercontent\.)/.test(h);
let best = null, area = 0;
for (const im of document.querySelectorAll('img')) {
  const s = im.currentSrc || im.src || '';
  if (!s.startsWith('http')) continue;
  let host; try { host = new URL(s).hostname; } catch(e) { continue; }
  if (bad(host) || s.includes('encrypted-tbn')) continue;
  const r = im.getBoundingClientRect();
  const vis = r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0 &&
              r.top < innerHeight && r.left < innerWidth;
  if (!vis || r.width < 150) continue;
  if (r.width * r.height > area) { area = r.width * r.height; best = im; }
}
if (!best) return null;
let href = '', how = '';
const a = best.closest('a[href]');
const ok = u => u && u.startsWith('http') && !bad(new URL(u).hostname);
if (a && ok(a.href)) { href = a.href; how = 'image_anchor'; }
let el = best.parentElement, d = 0;
while (!href && el && d < 8) {
  const hs = [...el.querySelectorAll('a[href]')].map(x => x.href).filter(ok);
  if (hs.length) { href = hs[0]; how = 'ancestor_' + d; }
  el = el.parentElement; d++;
}
return JSON.stringify({src: best.currentSrc || best.src, href: href, how: how, alt: best.alt || ''});
"""

def read_preview(driver, prev_src, timeout=4.0):
    """Wait until the preview shows a NEW image, then return image+link from that element."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            raw = driver.execute_script(JS_PREVIEW)
            if raw:
                p = json.loads(raw)
                if p["src"] != prev_src:
                    return p
        except Exception:
            pass
        time.sleep(0.25)
    return None

def download(url):
    try:
        r = requests.get(url, timeout=10, verify=certifi.where(), headers=UA)
        return (r.content, r.status_code) if r.status_code == 200 else (None, r.status_code)
    except Exception:
        return (None, None)

def page_has_image(page_url, image_url, timeout=15):
    if not page_url: return "no_url"
    try:
        r = requests.get(page_url, headers=UA, timeout=timeout)
        if r.status_code != 200: return f"http_{r.status_code}"
        html = r.text
    except Exception as e:
        return f"error_{type(e).__name__}"
    path = urllib.parse.urlparse(image_url).path
    stem = ntpath.splitext(ntpath.basename(path))[0]
    if image_url in html or (len(path) > 8 and path in html): return "found_url"
    if len(stem) >= 6 and stem in html: return "found_filename"
    return "not_found"

# ── one product ─────────────────────────────────────────────────────────────
def _on_results(driver):
    u = driver.current_url or ""
    return ("tbm=isch" in u or "udm=2" in u) and not cd._is_captcha(driver)

COUNTRY = "us"     # Google country setting (gl); "us" = the setting used in June and in fresh_oct

def search_url(q):
    """Same search as the earlier crawler (English, large images), with the
    country setting made explicit so runs can differ only in this parameter."""
    return ("https://www.google.com/search?hl=en&gl=" + COUNTRY +
            "&safe=off&tbm=isch&tbs=isz:l&q=" + urllib.parse.quote(q))

def open_results(driver, drug, attempts=2):
    """Open the image results page; handle CAPTCHAs (also a second one after a
    reload). Returns (thumbnails, captcha_seen, attempts_used)."""
    captcha_seen = False
    for a in range(1, attempts + 1):
        driver.get(search_url(drug)); time.sleep(2.0)
        for _ in range(3):                         # a CAPTCHA can come back after solving
            if not cd._is_captcha(driver): break
            captcha_seen = True
            if not cd._handle_captcha(driver):
                return [], captcha_seen, a
            time.sleep(3.0)
        if not _on_results(driver):
            driver.get(search_url(drug)); time.sleep(2.5)
            if cd._is_captcha(driver):
                captcha_seen = True
                cd._handle_captcha(driver); time.sleep(3.0)
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);"); time.sleep(1.0)
        driver.execute_script("window.scrollTo(0, 0);"); time.sleep(0.5)
        thumbs = driver.execute_script(JS_THUMBS) or []
        if thumbs:
            return thumbs, captcha_seen, a
        if a < attempts:
            print(f"  thumbnails: 0 (attempt {a}/{attempts}) -> reloading")
            time.sleep(10)
    return [], captcha_seen, attempts

def crawl_one(driver, drug, out_dir, collect_num=COLLECT_NUM, expand="none"):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    thumbs, captcha_seen, n_att = open_results(driver, drug)
    print(f"  thumbnails: {len(thumbs)}")
    if not thumbs:
        # no run_summary.json -> the product counts as NOT done and is retried on resume
        fail = {"drug_name": drug, "expand": expand, "country": COUNTRY, "error": "no_thumbnails",
                "captcha_seen": captcha_seen, "attempts": n_att,
                "time": datetime.now().isoformat(timespec="seconds")}
        with open(os.path.join(out_dir, "crawl_failed.json"), "w") as f:
            json.dump(fail, f, indent=2)
        print("  [FAILED] no thumbnails - not marked as done")
        return {"images_saved": 0, "error": "no_thumbnails", "captcha_seen": captcha_seen}
    events, prov, seen, added, prev = [], [], set(), 0, None
    iii = cd._next_num(out_dir)
    for th in thumbs[:MAX_THUMBS]:
        if added >= collect_num: break
        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", th)
            time.sleep(0.2 + random.random() * 0.3)
            try: th.click()
            except Exception: driver.execute_script("arguments[0].click();", th)
            p = read_preview(driver, prev)
            if not p: continue
            prev = p["src"]
            img_url, src_url = p["src"], p["href"]
            if img_url in seen: continue
            seen.add(img_url)
            content, status = download(img_url)
            if content is None: continue
            try:
                im = Image.open(BytesIO(content)).convert("RGB"); w, h = im.size
            except Exception:
                continue
            if w < MIN_SIDE or h < MIN_SIDE: continue
            numstr = f"{iii:05d}"
            im.save(os.path.join(out_dir, f"{numstr}.jpg"), "JPEG", quality=92)
            cd._write_metadata(out_dir, numstr, img_url, src_url, "", p["alt"], w, h)
            events.append({"no": numstr, "image_url": img_url, "source_page_url": src_url,
                           "source_page_title": "", "image_domain": cd._domain(img_url),
                           "w": w, "h": h, "status": status})
            prov.append({"no": numstr, "image_url": img_url, "source_page_url": src_url,
                         "link_how": p["how"], "alt_text": p["alt"]})
            iii += 1; added += 1
            print(f"  {numstr}: {cd._domain(img_url)} -> {cd._domain(src_url) or 'none'} ({p['how']})")
        except Exception:
            continue

    # provenance check (after the browser part, so the panel is not delayed)
    for r in prov:
        r["page_has_image"] = page_has_image(r["source_page_url"], r["image_url"])
        r["checked_at"] = datetime.now().isoformat(timespec="seconds")
    pd.DataFrame(events).to_csv(os.path.join(out_dir, "crawl_log.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(prov).to_csv(os.path.join(out_dir, "provenance.csv"), index=False, encoding="utf-8-sig")
    summary = {"drug_name": drug, "expand": expand, "country": COUNTRY, "images_saved": added,
               "thumbnails": len(thumbs), "captcha_seen": captcha_seen, "attempts": n_att,
               "source_urls_captured": sum(1 for e in events if e["source_page_url"]),
               "page_contains_image": sum(1 for r in prov if str(r["page_has_image"]).startswith("found")),
               "crawl_date": datetime.now().date().isoformat(),
               "duration_sec": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved {added} | links {summary['source_urls_captured']} | page contains image {summary['page_contains_image']}")
    return summary

# ── query construction ──────────────────────────────────────────────────────
FORM_WORDS = ("tablet", "capsule", "injection", "syrup", "suspension", "cream", "gel",
              "ointment", "drops", "spray", "solution", "powder", "sachet", "inhaler")

def build_query(r, expand):
    name = str(r.product_name).strip()
    if expand == "none":
        return name
    if expand == "medicine":
        return f"{name} medicine"
    form = str(getattr(r, "form", "") or "").strip().lower()
    if not form:
        return f"{name} medicine"
    if any(w in name.lower() for w in FORM_WORDS) or form in name.lower():
        return name                      # form already in the name (e.g. "EXTOCIN Injection")
    return f"{name} {form}"

# ── batch ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="fresh_oct")
    ap.add_argument("--queries", default="config/queries.csv")
    ap.add_argument("--collect_num", type=int, default=COLLECT_NUM)
    ap.add_argument("--only", default="")
    ap.add_argument("--reason", default="rerun",
                    help="why products in --only are re-crawled (written to tables/rerun_log.csv), "
                         "e.g. captcha_no_results or query_name_corrected")
    ap.add_argument("--country", default="us",
                    help="Google country setting gl (default us = June/fresh_oct setting; e.g. tz)")
    ap.add_argument("--expand", choices=["none", "form", "medicine"], default="none",
                    help="none: name only (June protocol); form: name + dosage form from the "
                         "'form' column (falls back to 'medicine'); medicine: name + 'medicine'")
    args = ap.parse_args()
    global COUNTRY
    COUNTRY = args.country.strip().lower()
    print(f"[INFO] Google country setting gl={COUNTRY}")

    df = pd.read_csv(EXP_ROOT / args.queries, dtype=str).fillna("")
    if "include_in_exp" in df.columns:
        df = df[df["include_in_exp"].str.strip().str.lower().isin({"1", "true", "yes", "y"})]
    only = {q.strip() for q in args.only.split(",") if q.strip()}
    if only: df = df[df["query_id"].isin(only)]

    root = EXP_ROOT / "runs" / args.run_id
    crawl_root, tables = root / "crawl", root / "tables"
    crawl_root.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)

    driver, rows = new_driver(), []
    try:
        for i, r in enumerate(df.itertuples(), 1):
            slug = str(getattr(r, "slug", "") or r.product_name.replace(" ", "_")).strip()
            out_dir = crawl_root / f"{r.query_id}_{slug}"
            if (out_dir / "run_summary.json").exists() and r.query_id not in only:
                print(f"[{i}/{len(df)}] {r.query_id} done, skipping"); continue
            query = build_query(r, args.expand)
            print(f"\n[{i}/{len(df)}] {r.query_id}: {query}")
            if out_dir.exists():
                # never append to an old folder: move it aside and log why
                old = {}
                for fn in ("run_summary.json", "crawl_failed.json"):
                    if (out_dir / fn).exists():
                        old = json.loads((out_dir / fn).read_text()); break
                sup = crawl_root / "_superseded"; sup.mkdir(exist_ok=True)
                dest = sup / f"{out_dir.name}__{datetime.now():%Y%m%d_%H%M%S}"
                shutil.move(str(out_dir), str(dest))
                log = tables / "rerun_log.csv"
                pd.DataFrame([{"query_id": r.query_id,
                               "reason": args.reason if r.query_id in only else "previous_attempt_failed",
                               "old_query": old.get("drug_name", ""),
                               "old_images_saved": old.get("images_saved", 0),
                               "old_error": old.get("error", ""),
                               "moved_to": str(dest),
                               "time": datetime.now().isoformat(timespec="seconds")}]
                             ).to_csv(log, mode="a", header=not log.exists(), index=False)
                print(f"  old folder moved to {dest.name}")
            s = crawl_one(driver, query, str(out_dir), args.collect_num, args.expand)
            s.update({"query_id": r.query_id, "drug": r.product_name, "query": query,
                      "expand": args.expand, "country": COUNTRY, "out_dir": str(out_dir)})
            rows.append(s)
            if i < len(df): time.sleep(20 + random.random() * 20)
    finally:
        driver.quit()
    if rows:
        pd.DataFrame(rows).to_csv(tables / f"crawl_summary_{datetime.now():%Y%m%d_%H%M}.csv", index=False)

if __name__ == "__main__":
    main()
