#!/usr/bin/env python3
"""Generate offline, blinded image-labeling pages for pharmaceutical product matching.

Annotations follow the study's v7 operational rule (v6 product-vs-pack + W flag): brand first, followed by
active ingredient(s), strength per unit, and dosage form. Packaging presentation
is recorded separately, without assessing authenticity.

Run from the repository root::

    python scripts/2_product_identity/03_build_identity_annotation_page.py --runs fresh_oct,fresh_oct_form,fresh_oct_tz

Place labeling_template.html next to this script. The researcher receives an
all-item page; a separate ZIP contains *only* the second annotator's subset.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repository root
TEMPLATE_FILE = Path(__file__).with_name("labeling_template.html")
OUTPUT_DIR = PROJECT_ROOT / "runs" / "labeling"
PREVIEW_DIR = OUTPUT_DIR / "img"
MAX_PREVIEW_SIDE = 1600
JPEG_QUALITY = 90
TEST_IDS_FILE = OUTPUT_DIR.parent / "labeling_test_ids.csv"   # practice items, never in the second subset
SECOND_MANIFEST = OUTPUT_DIR.parent / "labeling_second_sample_manifest.csv"  # the fixed second-annotator subset
GUIDE_VERSION = "v7"  # v7 = v6 product-vs-pack rule + W flag (watermark/overlay)


# ==================== Inputs and image processing ====================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="Comma-separated pipeline run IDs")
    parser.add_argument("--queries", default="config/queries.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--band-min", "--band_min", type=int, default=25)
    parser.add_argument("--band-per-product", "--band_per_product", type=int, default=2)
    parser.add_argument("--second-n", "--second_n", type=int, default=100)
    parser.add_argument("--second-accepted-share", "--second_accepted_share", type=float, default=0.8)
    parser.add_argument("--redraw-second", action="store_true",
                        help="draw the second-annotator subset again (ONLY before anyone has received it)")
    parser.add_argument("--test-n", "--test_n", type=int, default=0,
                        help="build a practice package of this many items for testing the page; "
                             "the chosen items are saved and excluded from the second annotator's subset")
    args = parser.parse_args()
    if args.band_min < 0 or args.band_per_product < 0:
        parser.error("Near-threshold settings must be non-negative")
    if args.second_n < 1 or not 0 <= args.second_accepted_share <= 1:
        parser.error("Invalid second-annotator sample size or accepted share")
    if not TEMPLATE_FILE.is_file():
        parser.error(f"Missing HTML template: {TEMPLATE_FILE}")
    return args


def resolve_path(value: str, run_dir: Path | None = None) -> Path:
    path = Path(str(value).strip())
    if path.is_absolute():
        return path
    if run_dir is not None and (run_dir / path).is_file():
        return run_dir / path
    return PROJECT_ROOT / path


def file_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


def save_preview(source: Path, destination: Path) -> bool:
    """Create a resized display copy; do not modify the source image."""
    if not source.is_file():
        return False
    if destination.is_file() and destination.stat().st_size > 0:
        if destination.stat().st_mtime >= source.stat().st_mtime:
            return True
    try:
        with Image.open(source) as original:
            image = original.convert("RGB")
            image.thumbnail((MAX_PREVIEW_SIDE, MAX_PREVIEW_SIDE))
            image.save(destination, "JPEG", quality=JPEG_QUALITY)
        return True
    except (OSError, UnidentifiedImageError) as exc:
        print(f"  [unreadable image] {source}: {exc}")
        return False


# ==================== Build retrieval candidate population ====================
def candidate_rows(run_ids: list[str], args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    for run_id in run_ids:
        run_dir = PROJECT_ROOT / "runs" / run_id
        top_path = run_dir / "tables" / "pipeline_top3_all.csv"
        summary_path = run_dir / "tables" / "match_summary.csv"
        if not top_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Required retrieval tables are missing: {run_id}")

        top = pd.read_csv(top_path, dtype=str).fillna("")
        for _, r in top.iterrows():
            rows.append({
                "run": run_id, "query_id": r["query_id"], "kind": "selected",
                "rank": r.get("rank", ""), "inliers": r.get("match_count", ""),
                "image_path": str(resolve_path(r["image"], run_dir)),
                "image_url": r.get("image_url", ""),
                "source_page_url": r.get("source_page_url", ""),
            })

        summary = pd.read_csv(summary_path, dtype=str).fillna("")
        for _, r in summary.iterrows():
            qid = r["query_id"]
            matching_dirs = sorted((run_dir / "matching").glob(f"{qid}_*"))
            crawl_dirs = sorted((run_dir / "crawl").glob(f"{qid}_*"))
            if len(matching_dirs) != 1 or len(crawl_dirs) != 1:
                print(f"  [warning] {run_id}/{qid}: ambiguous or missing matching/crawl folder; skip below-threshold candidates")
                continue
            scores_path = matching_dirs[0] / "scores.csv"
            if not scores_path.is_file():
                continue
            scores = pd.read_csv(scores_path, dtype=str).fillna("")
            if scores.empty or not {"inliers", "image"}.issubset(scores.columns):
                continue
            scores["_score"] = pd.to_numeric(scores["inliers"], errors="coerce").fillna(0)
            scores.sort_values("_score", ascending=False, kind="stable", inplace=True)
            threshold = int(float(r.get("min_inliers") or 50))
            near = scores[(scores["_score"] >= args.band_min) & (scores["_score"] < threshold)]
            picks = [(v, "near_threshold") for _, v in near.head(args.band_per_product).iterrows()]
            if r.get("has_match", "").lower() == "false" and not picks:
                below = scores[scores["_score"] < threshold]
                if not below.empty:
                    picks = [(below.iloc[0], "best_below_threshold")]
            for v, kind in picks:
                rows.append({
                    "run": run_id, "query_id": qid, "kind": kind, "rank": "",
                    "inliers": v["inliers"],
                    "image_path": str(resolve_path(str(v["image"]), crawl_dirs[0])),
                    "image_url": v.get("image_url", ""),
                    "source_page_url": v.get("source_page_url", ""),
                })
    return rows


def build_items(rows: list[dict], queries: pd.DataFrame, seed: int) -> tuple[list[dict], pd.DataFrame]:
    """Deduplicate by (query_id, source image bytes) across all retrieval runs."""
    items: dict[str, dict] = {}
    metadata: list[dict] = []
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for row in rows:
        candidate = Path(row["image_path"])
        qid = row["query_id"]
        if not candidate.is_file() or qid not in queries.index:
            print(f"  [skip] missing candidate or query: {qid} {candidate}")
            continue
        item_id = f"{qid}_{file_hash(candidate)}"
        if item_id not in items:
            qr = queries.loc[qid]
            query_source = resolve_path(qr["query_image_abspath"])
            q_preview = PREVIEW_DIR / f"query_{qid}.jpg"
            c_preview = PREVIEW_DIR / f"{item_id}.jpg"
            if not save_preview(query_source, q_preview) or not save_preview(candidate, c_preview):
                print(f"  [skip] unable to prepare image pair: {item_id}")
                continue
            # No drug/product name, ranking, inlier score, or source URL in the HTML.
            items[item_id] = {
                "id": item_id, "qid": qid,
                "q": f"img/{q_preview.name}", "c": f"img/{c_preview.name}",
            }
        try:
            with Image.open(candidate) as original:
                width, height = original.size
        except (OSError, UnidentifiedImageError):
            width = height = ""
        metadata.append({
            "item_id": item_id, "query_product_name": queries.loc[qid].get("product_name", ""),
            "width_px": width, "height_px": height, **row,
        })

    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in items.values():
        grouped[item["qid"]].append(item)
    rng = random.Random(seed)
    ordered = []
    for qid in sorted(grouped):
        rng.shuffle(grouped[qid])
        ordered.extend(grouped[qid])
    return ordered, pd.DataFrame(metadata)


# ==================== Freeze independent 80/20 draw ====================
def choose_second_sample(
    items: list[dict], metadata: pd.DataFrame, n: int, accepted_share: float, seed: int,
    exclude: set[str] = frozenset(), redraw: bool = False
) -> tuple[list[str], pd.DataFrame]:
    """Stratify by acceptance in ANY run. No human annotation is consulted.
    Practice (test) items are excluded: someone who tried the page must not meet them again.
    Once drawn, the subset is FIXED: later builds (e.g. after adding a run) reuse the saved
    manifest, because annotators may already have received it."""
    if SECOND_MANIFEST.is_file() and not redraw:
        saved = pd.read_csv(SECOND_MANIFEST, dtype=str)
        present = {it["id"] for it in items}
        missing = sorted(set(saved["item_id"]) - present)
        if missing:
            raise RuntimeError(f"{len(missing)} fixed second-annotator items are no longer in the build, "
                               f"e.g. {missing[:3]}. Use the same runs plus new ones, never fewer.")
        keep = set(saved["item_id"])
        return [it["id"] for it in items if it["id"] in keep], saved
    accepted_ids = set(metadata.loc[metadata["kind"] == "selected", "item_id"])
    pool = [it for it in items if it["id"] not in exclude]
    acc = [it["id"] for it in pool if it["id"] in accepted_ids]
    rej = [it["id"] for it in pool if it["id"] not in accepted_ids]
    total = min(n, len(pool))
    n_acc = min(len(acc), round(total * accepted_share))
    n_rej = min(len(rej), total - n_acc)
    # Fill shortages from the other stratum while keeping sample size exact.
    shortfall = total - n_acc - n_rej
    extra_acc = min(shortfall, len(acc) - n_acc)
    n_acc += extra_acc
    n_rej += min(shortfall - extra_acc, len(rej) - n_rej)
    if n_acc + n_rej != total:
        raise AssertionError("Second-annotator stratified sampling failed")
    rng = random.Random(seed + 1)
    chosen = set(rng.sample(acc, n_acc) + rng.sample(rej, n_rej))
    sampled = [item["id"] for item in items if item["id"] in chosen]
    manifest = pd.DataFrame([
        {"item_id": item_id,
         "sampling_stratum": "accepted" if item_id in accepted_ids else "below_threshold",
         "sampling_seed": seed + 1}
        for item_id in sampled
    ], columns=["item_id", "sampling_stratum", "sampling_seed"])
    return sampled, manifest


# ==================== Practice (test) items ====================
def test_items(items: list[dict], metadata: pd.DataFrame, n: int, seed: int) -> list[str]:
    """Fixed practice items for trying the page. Chosen once and saved, so later builds
    (e.g. after adding a run) keep the same items and keep them out of the second subset."""
    present = {it["id"] for it in items}
    if TEST_IDS_FILE.is_file():
        saved = pd.read_csv(TEST_IDS_FILE, dtype=str)["item_id"].tolist()
        return [i for i in saved if i in present]
    if n <= 0:
        return []
    accepted_ids = set(metadata.loc[metadata["kind"] == "selected", "item_id"])
    acc = [it["id"] for it in items if it["id"] in accepted_ids]
    rej = [it["id"] for it in items if it["id"] not in accepted_ids]
    rng = random.Random(seed + 2)
    n_acc = min(len(acc), (n + 1) // 2)
    chosen = rng.sample(acc, n_acc) + rng.sample(rej, min(len(rej), n - n_acc))
    pd.DataFrame({"item_id": chosen}).to_csv(TEST_IDS_FILE, index=False, encoding="utf-8-sig")
    return chosen


def write_package(items: list[dict], mode: str, build_id: str, zip_path: Path) -> Path:
    """Zip with ONE self-contained page (images embedded) and the README.
    Embedding avoids the common failure of opening the page from inside the zip,
    where Windows extracts only the HTML file and the img/ folder is missing."""
    images_used = sorted({it["q"] for it in items} | {it["c"] for it in items})
    images = {rel: "data:image/jpeg;base64," + base64.b64encode((OUTPUT_DIR / rel).read_bytes()).decode()
              for rel in images_used}
    html = render_page(items, mode, build_id, images)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("labeling/label_page.html", html)
        archive.writestr("labeling/README_labelling.txt", ANNOTATOR_README)
    print(f"  {zip_path.name}: {len(items)} pairs, page {len(html) / 1e6:.1f} MB")
    return zip_path


# ==================== Create independent delivery packages ====================
def safe_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")


def render_page(items: list[dict], mode: str, build_id: str, images: dict | None = None) -> str:
    template = TEMPLATE_FILE.read_text(encoding="utf-8")
    return (template
            .replace("__ITEMS_JSON__", safe_json(items))
            .replace("__IMAGES_JSON__", safe_json(images or {}))
            .replace("__MODE__", mode)
            .replace("__GUIDE_VERSION__", GUIDE_VERSION)
            .replace("__BUILD_ID__", build_id))


def write_outputs(items: list[dict], meta: pd.DataFrame, second: list[str],
                  manifest: pd.DataFrame, run_ids: list[str], seed: int) -> tuple[Path, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Researcher-only metadata is deliberately outside the distributable folder.
    meta.to_csv(OUTPUT_DIR.parent / "labeling_items.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(OUTPUT_DIR.parent / "labeling_second_sample_manifest.csv", index=False,
                    encoding="utf-8-sig")
    build_id = hashlib.sha256(safe_json([it["id"] for it in items]).encode()).hexdigest()[:12]
    (OUTPUT_DIR / "label_page.html").write_text(render_page(items, "primary", build_id), encoding="utf-8")
    (OUTPUT_DIR / "README_labelling.txt").write_text(ANNOTATOR_README, encoding="utf-8")
    (OUTPUT_DIR.parent / "labeling_build_manifest.json").write_text(
        json.dumps({
            "build_id": build_id, "guide_version": GUIDE_VERSION,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "runs": run_ids, "random_seed": seed,
            "unique_items": len(items), "second_items": len(second),
            "second_strata": manifest["sampling_stratum"].value_counts().to_dict(),
        }, indent=2), encoding="utf-8")

    chosen = set(second)
    second_items = [it for it in items if it["id"] in chosen]
    return write_package(second_items, "second", build_id,
                         OUTPUT_DIR.parent / "labeling_for_second_annotator.zip"), build_id


def main() -> None:
    args = parse_args()
    run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    if not run_ids:
        raise ValueError("At least one pipeline run ID must be specified")
    queries = pd.read_csv(resolve_path(args.queries), dtype=str).fillna("")
    if queries["query_id"].duplicated().any():
        raise ValueError("config/queries.csv has duplicated query_id values")
    queries = queries.set_index("query_id")
    rows = candidate_rows(run_ids, args)
    items, metadata = build_items(rows, queries, args.seed)
    if not items:
        raise RuntimeError("No query–candidate pairs were found; check image paths")
    tests = test_items(items, metadata, args.test_n, args.seed)
    reused = SECOND_MANIFEST.is_file() and not args.redraw_second
    second, manifest = choose_second_sample(
        items, metadata, args.second_n, args.second_accepted_share, args.seed, exclude=set(tests),
        redraw=args.redraw_second
    )
    print("Second-annotator subset: " + ("FIXED, reused from " + SECOND_MANIFEST.name if reused
                                         else "drawn now and saved to " + SECOND_MANIFEST.name))
    package, build_id = write_outputs(items, metadata, second, manifest, run_ids, args.seed)
    if tests:
        test_zip = write_package([it for it in items if it["id"] in set(tests)], "test", build_id,
                                 OUTPUT_DIR.parent / "labeling_test_package.zip")
        print(f"Test package ({len(tests)} practice pairs, not in the second subset): {test_zip}")
    counts = manifest["sampling_stratum"].value_counts()
    print(f"Unique image pairs: {len(items)} ({len(metadata)} underlying retrieval records)")
    print(f"Second annotator: {len(second)} = {counts.get('accepted', 0)} accepted + "
          f"{counts.get('below_threshold', 0)} below threshold")
    if len(second) < args.second_n:
        print(f"  [note] only {len(items)} unique pairs available")
    print(f"Primary page: {OUTPUT_DIR / 'label_page.html'}")
    print(f"Second annotator ZIP (100-pair subset only): {package}")
    print(f"Researcher metadata: {OUTPUT_DIR.parent / 'labeling_items.csv'}")


ANNOTATOR_README = """WEB REFERENCE ANNOTATION GUIDE  (version v7)
==============================

Purpose
-------
Annotate whether the retrieved image (right) shows packaging of the same
medicine as the query image (left).

Matching criteria
-----------------
Compare the information visible on both packages:
  1. Brand name
  2. Active ingredient(s)
  3. Strength per unit (e.g., 500 mg per tablet or 125 mg per 5 mL)
  4. Dosage form (e.g., tablet, dispersible tablet, capsule, suspension)

How to decide
-------------
  Same medicine      All four criteria match.                      -> 1, 1
  Different variant  The brand matches, but at least one of the
                     other three criteria differs.                 -> 1, 2
  Different product  The brand names differ, or no medicine
                     package is shown (e.g. only a leaflet,
                     a document or an unrelated object).           -> 2
  Cannot determine   No difference is identified, but the
                     information needed to confirm a match is
                     missing or unreadable.                        -> 3, or 1, 3

Optional flags
--------------
  M - Packaging difference
      Select when the packages differ in pack size, unit count, printed
      age/weight instructions, language, country, logo, colour, or design.
      Use only when the medicine matches or cannot be fully determined.

  C - Part of a larger image
      Select when the retrieved package appears as part of a collage,
      document, advertisement, or wider scene, rather than as the main
      subject of the image.
      Do not select C simply because the package is held in a hand or shown
      alongside its own bottle or blister.

  W - Watermark or overlay
      Select when text or graphics have been added over the retrieved
      photograph, such as a stock-photo watermark, website logo, price, or
      banner.
      Do not select W for text printed directly on the package.

  Multiple flags may be selected when applicable.

Important
---------
Use only the two displayed images. Do not use external searches or
additional information.

How to use the page
-------------------
1. Unzip (right-click > Extract All). The page is one file; images are inside it.
2. Open label_page.html in Chrome or Edge. No internet connection is needed;
   nothing is sent anywhere.
3. Enter your initials. Answer Step 1 (brand), then Step 2 when the brand is
   the same (keys 1/2/3 or click). Add flags or a note, then press Enter.
4. Press "Save CSV" about every 50 images and at the end. Send the CSV file
   exactly as downloaded. Do not open and re-save it in Excel (Excel changes it).
"""


if __name__ == "__main__":
    main()
