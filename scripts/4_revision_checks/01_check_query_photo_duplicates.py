"""
audit_query_duplicates.py
Does the web search return the query photograph itself? Every downloaded candidate image of the name-only run
(runs/fresh_oct/crawl, up to 10 per product) is compared with its query image.
Similarity: difference hash (64 bit) and gradient hash on the whole image and on the central 80%, normalised
correlation of 32x32 grey thumbnails, and the LightGlue match count from matching/*/scores.csv.
Candidates (any of: hash distance <= 12, correlation >= 0.90, matches >= 900) are written with a contact sheet
for visual confirmation; the confirmed list is entered by hand in duplicate_confirmed.csv (query_id, image, verdict).
Outputs: runs/fresh_oct/duplicate_audit/pairs.csv, candidates.csv, contact_sheet_*.jpg
"""
from pathlib import Path
import glob, os, re
import numpy as np, pandas as pd
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "fresh_oct"; OUT = RUN / "duplicate_audit"; OUT.mkdir(exist_ok=True)

def load(p):
    im = Image.open(p); im = ImageOps.exif_transpose(im); return im.convert("L")

def crop80(im):
    w, h = im.size; return im.crop((int(.1 * w), int(.1 * h), int(.9 * w), int(.9 * h)))

def dhash(im, n=8):
    a = np.asarray(im.resize((n + 1, n), Image.LANCZOS), float); return (a[:, 1:] > a[:, :-1]).ravel()

def vhash(im, n=8):
    a = np.asarray(im.resize((n, n + 1), Image.LANCZOS), float); return (a[1:, :] > a[:-1, :]).ravel()

def thumb(im):
    a = np.asarray(im.resize((32, 32), Image.LANCZOS), float).ravel(); a -= a.mean(); return a / (np.linalg.norm(a) + 1e-9)

def feats(im):
    return {v: (dhash(x), vhash(x), thumb(x)) for v, x in (("full", im), ("c80", crop80(im)))}

q = pd.read_csv(ROOT / "config" / "queries.csv", dtype=str)
rows = []
for d in sorted(glob.glob(str(RUN / "crawl" / "Q*"))):
    if "_superseded" in d: continue
    qid = os.path.basename(d)[:4]
    r = q[q.query_id == qid].iloc[0]
    fq = feats(load(ROOT / r.query_image_path))
    sc = RUN / "matching" / os.path.basename(d) / "scores.csv"
    m = dict(pd.read_csv(sc)[["image", "inliers"]].values) if sc.exists() else {}
    for f in sorted(glob.glob(os.path.join(d, "*.jpg"))):
        fi = feats(load(f)); best = {"ham": 99, "corr": -1}
        for a in fq.values():
            for b in fi.values():
                best["ham"] = min(best["ham"], int((a[0] != b[0]).sum() + (a[1] != b[1]).sum()) // 2)
                best["corr"] = max(best["corr"], float(a[2] @ b[2]))
        rows.append({"query_id": qid, "dataset": r.dataset, "image": os.path.basename(f), "matches": m.get(os.path.basename(f)),
                     "hash_distance": best["ham"], "thumb_corr": round(best["corr"], 3)})
p = pd.DataFrame(rows); p.to_csv(OUT / "pairs.csv", index=False)
c = p[(p.hash_distance <= 12) | (p.thumb_corr >= 0.90) | (p.matches >= 900)].sort_values(["thumb_corr"], ascending=False)
c.to_csv(OUT / "candidates.csv", index=False)
print(len(p), "pairs;", len(c), "candidates"); print(c.to_string(index=False))

# contact sheets: query | candidate, 12 pairs per sheet
W = 220
for s in range(0, len(c), 12):
    part = c.iloc[s:s + 12]; sheet = Image.new("RGB", (2 * W + 10, len(part) * (W + 24)), "white"); dr = ImageDraw.Draw(sheet)
    for k, x in enumerate(part.itertuples()):
        qp = ROOT / q[q.query_id == x.query_id].iloc[0].query_image_path
        cp = glob.glob(str(RUN / "crawl" / f"{x.query_id}_*" / x.image))[0]
        for j, path in enumerate((qp, cp)):
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGB"); im.thumbnail((W, W))
            sheet.paste(im, (j * (W + 10), k * (W + 24) + 20))
        dr.text((2, k * (W + 24) + 2), f"{x.query_id} {x.dataset} {x.image} m={x.matches} h={x.hash_distance} r={x.thumb_corr}", fill="black")
    sheet.save(OUT / f"contact_sheet_{s // 12 + 1:02d}.jpg", quality=85)
