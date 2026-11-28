"""
00_export_retrieval_records.py
Exports the retrieval records of each search run into data/1_retrieval/ (run in the ORIGINAL working folder,
which holds runs/<run_id>/crawl and runs/<run_id>/matching; the repository itself holds no images).

For each run (fresh_oct = name only, fresh_oct_form = name + dosage form, fresh_oct_tz = name only, Tanzania region):
  candidates_<run>.csv      every downloaded candidate: query, image file, download position (search order among
                            downloadable images), image URL, source-page URL, number of LightGlue matches
  match_summary_<run>.csv   one row per product (copied from runs/<run>/tables)
  top3_<run>.csv            the selected top-three images per product, without local file paths
Also copies the duplicate check (runs/fresh_oct/duplicate_audit/pairs.csv, duplicate_confirmed.csv).

Usage: python scripts/4_revision_checks/00_export_retrieval_records.py --src <working folder> --out data/1_retrieval
"""
import argparse, csv, glob, os
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", default="data/1_retrieval")
a = ap.parse_args()
src, out = Path(a.src), Path(a.out)
out.mkdir(parents=True, exist_ok=True)

for run in ("fresh_oct", "fresh_oct_form", "fresh_oct_tz"):
    rows = []
    for d in sorted(glob.glob(str(src / "runs" / run / "crawl" / "Q*"))):
        qid = os.path.basename(d)[:4]
        md = Path(d) / "metadata.tsv"
        if not md.exists():
            continue
        with open(md, encoding="utf-8", newline="") as f:      # titles may contain quotes: no quote handling
            r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            head = next(r)
            meta = [dict(zip(head, x)) for x in r]
        sc = src / "runs" / run / "matching" / os.path.basename(d) / "scores.csv"
        m = dict(pd.read_csv(sc)[["image", "inliers"]].values) if sc.exists() else {}
        for x in meta:
            img = f"{x['no']}.jpg"
            rows.append({"query_id": qid, "image": img, "download_pos": int(x["no"]) + 1, "image_url": x.get("image_url", ""),
                         "source_page_url": x.get("source_page_url", ""), "matches": m.get(img, "")})
    pd.DataFrame(rows).to_csv(out / f"candidates_{run}.csv", index=False)
    t = src / "runs" / run / "tables"
    pd.read_csv(t / "match_summary.csv").to_csv(out / f"match_summary_{run}.csv", index=False)
    p = pd.read_csv(t / "pipeline_top3_all.csv", dtype=str)
    p["image"] = p["image"].map(lambda s: os.path.basename(str(s).replace("\\", "/")))
    p.drop(columns=[c for c in ("page_file", "html_file") if c in p]).to_csv(out / f"top3_{run}.csv", index=False)
    print(run, len(rows), "candidates")

dup = src / "runs" / "fresh_oct" / "duplicate_audit"
for f in ("pairs.csv", "duplicate_confirmed.csv"):
    pd.read_csv(dup / f).to_csv(out / f"duplicate_check_{f}", index=False)
print("written:", out)
