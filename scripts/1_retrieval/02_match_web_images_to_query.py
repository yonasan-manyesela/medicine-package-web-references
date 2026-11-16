"""
02_match_web_images_to_query.py
=====================
Visual filtering (DISK + LightGlue)
'inliers' in this script and its outputs = number of LightGlue matches (no RANSAC or other geometric verification). and source-page fetch for one crawl run.

Same method and parameters as the earlier version of this script (June run) (DISK 2048 keypoints, LightGlue,
MIN_INLIERS = 50, top-3, same page-text extraction), with these fixes:

  1. One summary row per product, also when no image passes the filter
     -> tables/match_summary.csv  (needed for the coverage funnel and for
        comparing plain vs expanded queries on the same products)
  2. Top-k is taken over DISTINCT source pages (two images from the same page
     no longer count as two pieces of evidence); tracking parameters such as
     ?srsltid=... are ignored when comparing URLs
  3. Provenance from the crawl (link_how, page_has_image) is carried into the
     output so Stage 3 can report / filter on it
  4. Raw HTML of each fetched page is saved with the fetch time (frozen inputs)
  5. Matching errors are logged and counted instead of silently scored as 0
  6. metadata.tsv is parsed without quote handling (titles with quotes no
     longer shift columns)
  7. Query image features are extracted once per product (faster, same result)

Usage (from the repository root):
    python scripts/1_retrieval/02_match_web_images_to_query.py --run_id fresh_oct
    python scripts/1_retrieval/02_match_web_images_to_query.py --run_id fresh_oct_form
Outputs in runs/<run_id>/:
    pipeline/<qid>_<slug>/rank{k}_page.txt, rank{k}_page.html, top3_summary.csv
    matching/<qid>_<slug>/scores.csv          (all images, inlier counts)
    tables/match_summary.csv                  (one row per product)
    tables/pipeline_top3_all.csv              (one row per selected page)
"""
import argparse, csv, json, os, sys, time, urllib.parse
from datetime import datetime
from pathlib import Path

import pandas as pd, requests, certifi
from bs4 import BeautifulSoup

EXP_ROOT = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(EXP_ROOT / "scripts" / "LightGlue"))
sys.path.insert(0, str(EXP_ROOT / "scripts"))

import torch
from lightglue import LightGlue, DISK
from lightglue.utils import load_image, rbd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
HEADERS  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
TRACKING_PARAMS = {"srsltid", "fbclid", "gclid"}

# ── models ──────────────────────────────────────────────────────────────────
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DISK(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="disk").eval().to(device)
    print(f"[INFO] DISK + LightGlue ready on {device}")
    return extractor, matcher, device

@torch.inference_mode()
def match_folder(query_img, images, extractor, matcher, device):
    f0 = extractor.extract(load_image(str(query_img)).to(device))
    results, errors = [], []
    for db in images:
        try:
            f1 = extractor.extract(load_image(str(db)).to(device))
            out = rbd(matcher({"image0": f0, "image1": f1}))
            results.append((db, int(out["matches"].shape[0])))
        except Exception as e:
            results.append((db, 0))
            errors.append(f"{db.name}:{type(e).__name__}")
    results.sort(key=lambda x: x[1], reverse=True)
    return results, errors

# ── crawl metadata ──────────────────────────────────────────────────────────
def read_tsv(p):
    return pd.read_csv(p, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                       on_bad_lines="skip", encoding="utf-8").fillna("")

def load_crawl_info(folder):
    """no -> {image_url, source_page_url, alt_text, link_how, page_has_image}"""
    info = {}
    meta = folder / "metadata.tsv"
    if meta.exists():
        for _, r in read_tsv(meta).iterrows():
            no = str(r.get("no", "")).strip().zfill(5)
            info[no] = {"image_url": r.get("image_url", ""),
                        "source_page_url": r.get("source_page_url", "").strip(),
                        "alt_text": r.get("alt_text", "")}
    prov = folder / "provenance.csv"
    if prov.exists():
        for _, r in pd.read_csv(prov, dtype=str, encoding="utf-8-sig").fillna("").iterrows():
            no = str(r.get("no", "")).strip().zfill(5)
            info.setdefault(no, {})
            info[no]["link_how"] = r.get("link_how", "")
            info[no]["page_has_image"] = r.get("page_has_image", "")
    return info

def crawl_query(folder):
    p = folder / "run_summary.json"
    try:
        return json.loads(p.read_text()).get("drug_name", "")
    except Exception:
        return ""

def norm_url(u):
    if not u:
        return ""
    p = urllib.parse.urlsplit(u.strip())
    q = [(k, v) for k, v in urllib.parse.parse_qsl(p.query)
         if k.lower() not in TRACKING_PARAMS and not k.lower().startswith("utm_")]
    return urllib.parse.urlunsplit((p.scheme.lower(), p.netloc.lower().removeprefix("www."),
                                    p.path.rstrip("/"), urllib.parse.urlencode(q), ""))

# ── page fetch (same text extraction as the earlier version) ─────────────────
def fetch_page(url, timeout=12):
    if not url or not url.startswith("http"):
        return "", "", None, "no_url"
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, verify=certifi.where())
        if r.status_code != 200:
            return "", "", r.status_code, f"http_{r.status_code}"
        ct = r.headers.get("Content-Type", "").lower()
        if "html" not in ct and "text" not in ct:
            return "", "", r.status_code, "non_html"
        html = r.text
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = " ".join(soup.get_text().split())
        qual = "ok" if len(text.split()) >= 50 else "low_content"
        return text[:8000], html, r.status_code, qual
    except requests.exceptions.Timeout:
        return "", "", None, "timeout"
    except Exception as e:
        return "", "", None, f"error_{type(e).__name__}"

# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--queries", default="config/queries.csv")
    ap.add_argument("--min_inliers", type=int, default=50)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    run = EXP_ROOT / "runs" / args.run_id
    crawl_dir, pipe_dir = run / "crawl", run / "pipeline"
    match_dir, tables = run / "matching", run / "tables"
    for d in (pipe_dir, match_dir, tables):
        d.mkdir(parents=True, exist_ok=True)

    df_q = pd.read_csv(EXP_ROOT / args.queries, dtype=str).fillna("")
    if "include_in_exp" in df_q.columns:
        df_q = df_q[df_q["include_in_exp"].str.strip().str.lower().isin({"1", "true", "yes", "y"})]
    only = {q.strip() for q in args.only.split(",") if q.strip()}
    if only:
        df_q = df_q[df_q["query_id"].isin(only)]

    extractor, matcher, device = load_models()
    summary, all_rows = [], []

    for _, row in df_q.iterrows():
        qid, drug = row["query_id"].strip(), row["product_name"].strip()
        slug = (row.get("slug", "") or drug.replace(" ", "_")).strip()
        qimg = row.get("query_image_abspath", "") or str(EXP_ROOT / row.get("query_image_path", ""))
        folder = crawl_dir / f"{qid}_{slug}"
        s = {"query_id": qid, "drug": drug, "query_used": "", "status": "",
             "n_images": 0, "n_errors": 0, "best_inliers": 0,
             "n_valid_images": 0, "n_valid_pages": 0, "has_match": False,
             "min_inliers": args.min_inliers}
        print(f"\n[{qid}] {drug}")

        if not folder.is_dir():
            s["status"] = "no_crawl_folder"; summary.append(s); print("  no crawl folder"); continue
        if not qimg or not os.path.exists(qimg):
            s["status"] = "no_query_image"; summary.append(s); print(f"  query image missing: {qimg}"); continue
        s["query_used"] = crawl_query(folder)
        images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
        s["n_images"] = len(images)
        if not images:
            s["status"] = "no_images"; summary.append(s); print("  no images"); continue

        results, errors = match_folder(qimg, images, extractor, matcher, device)
        s["n_errors"], s["best_inliers"] = len(errors), results[0][1]
        if errors:
            print(f"  [WARN] {len(errors)} matching errors: {errors[:3]}")
        info = load_crawl_info(folder)

        (match_dir / f"{qid}_{slug}").mkdir(exist_ok=True)
        pd.DataFrame([{"image": p.name, "inliers": c,
                       "source_page_url": info.get(p.stem.zfill(5), {}).get("source_page_url", "")}
                      for p, c in results]).to_csv(match_dir / f"{qid}_{slug}" / "scores.csv", index=False)

        valid = [(p, c) for p, c in results if c >= args.min_inliers]
        s["n_valid_images"] = len(valid)
        # distinct pages, keep the best-matching image per page
        chosen, seen = [], set()
        for p, c in valid:
            m = info.get(p.stem.zfill(5), {})
            key = norm_url(m.get("source_page_url", "")) or f"noimgurl:{p.name}"
            if key in seen:
                continue
            seen.add(key); chosen.append((p, c, m))
        s["n_valid_pages"] = len(chosen)
        s["has_match"] = len(chosen) > 0
        if not chosen:
            s["status"] = "no_match"; summary.append(s)
            print(f"  no image >= {args.min_inliers} inliers (best {results[0][1]})"); continue
        s["status"] = "ok"; summary.append(s)

        out_dir = pipe_dir / f"{qid}_{slug}"; out_dir.mkdir(exist_ok=True)
        rows = []
        for rank, (p, c, m) in enumerate(chosen[:args.top_k], 1):
            url = m.get("source_page_url", "")
            text, html, status, qual = fetch_page(url)
            fetched_at = datetime.now().isoformat(timespec="seconds")
            words = len(text.split()) if text else 0
            page_file, html_file = out_dir / f"rank{rank}_page.txt", out_dir / f"rank{rank}_page.html"
            with open(page_file, "w", encoding="utf-8") as f:
                f.write(f"QUERY_ID: {qid}\nDRUG: {drug}\nRANK: {rank}\nMATCH_COUNT: {c}\n"
                        f"IMAGE: {p.name}\nURL: {url}\nSTATUS: {status}\nQUALITY: {qual}\n"
                        f"WORDS: {words}\nFETCHED_AT: {fetched_at}\n" + "=" * 60 + "\n" + text)
            if html:
                html_file.write_text(html, encoding="utf-8", errors="replace")
            r = {"query_id": qid, "drug": drug, "query_used": s["query_used"], "rank": rank,
                 "image": str(p), "match_count": c, "image_url": m.get("image_url", ""),
                 "source_page_url": url, "alt_text": m.get("alt_text", ""),
                 "link_how": m.get("link_how", ""), "page_has_image": m.get("page_has_image", ""),
                 "status_code": status, "content_quality": qual, "word_count": words,
                 "fetched_at": fetched_at, "page_file": str(page_file),
                 "html_file": str(html_file) if html else ""}
            rows.append(r); all_rows.append(r)
            print(f"  rank{rank}: {c} inliers | {qual} {words}w | {url[:70]}")
            time.sleep(0.5)
        pd.DataFrame(rows).to_csv(out_dir / "top3_summary.csv", index=False)

    sm = pd.DataFrame(summary)
    top = pd.DataFrame(all_rows)
    if only:
        # re-run of some products: replace their rows, keep all other products
        for name, new in (("match_summary.csv", sm), ("pipeline_top3_all.csv", top)):
            p = tables / name
            if p.exists():
                old = pd.read_csv(p, dtype=str)
                old = old[~old["query_id"].isin(only)]
                new = pd.concat([old, new.astype(str)], ignore_index=True).sort_values(
                    ["query_id"] + (["rank"] if "rank" in new.columns else []))
            new.to_csv(p, index=False)
        print(f"updated {len(only)} products in the run tables")
    else:
        sm.to_csv(tables / "match_summary.csv", index=False)
        if all_rows:
            top.to_csv(tables / "pipeline_top3_all.csv", index=False)
    print("\n" + "=" * 60)
    print(f"products: {len(sm)} | with >=1 match: {int(sm['has_match'].sum())} | "
          f"selected pages: {len(all_rows)}")
    print(sm["status"].value_counts().to_string())

if __name__ == "__main__":
    main()
