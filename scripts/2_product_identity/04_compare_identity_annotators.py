"""
04_compare_identity_annotators.py
===============
Agreement between the main annotator and the second annotator, then a sheet
for resolving their disagreements.

Step A (before any discussion):
    python scripts/2_product_identity/04_compare_identity_annotators.py --main runs\\labeling\\labels_YZ_....csv
                                    --second runs\\labeling\\labels_second_PH_....csv
  -> prints % agreement and Cohen's kappa at two levels, on the items both labelled:
       brand level : Step 1 answer (same / different / unreadable)
       primary     : 4 classes (same, variant, different, cannot_determine)
       full        : 5 classes (same, variant, brand_only, different, unclear)
       flags       : M, C, W separately
     also separately for items the filter accepted vs near/below-threshold items
  -> writes runs\\labeling_agreement\\agreement.txt        (numbers for the paper)
            runs\\labeling_agreement\\disagreements.csv    (one row per disagreement)
            runs\\labeling_agreement\\disagreements.html   (images side by side, for the discussion)

Step B (discussion): open disagreements.html together, decide, and fill the
column final_label in disagreements.csv (same / variant / brand_only / different / unclear).

Step C:
    python scripts/2_product_identity/04_compare_identity_annotators.py --main ... --second ... --resolved runs\\labeling_agreement\\disagreements.csv
  -> writes runs\\labeling_agreement\\labels_final.csv: your labels, with the agreed
     label for every resolved disagreement (and a column 'resolved' = 1 there).
The agreement numbers are always computed from the original, unresolved labels.
"""
import argparse
import html
import io
from collections import Counter
from pathlib import Path

import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[2]  # repository root

def read_labels(path):
    """Labels CSV from the page: '#' comment lines (who, when, rules) are skipped.
    Not pandas' comment='#', which would also cut notes that contain '#'."""
    text = Path(path).read_text(encoding="utf-8-sig")
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    return pd.read_csv(io.StringIO(body), dtype=str).fillna("")

def kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return po, k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--second", required=True)
    ap.add_argument("--resolved", default="")
    args = ap.parse_args()

    out = EXP_ROOT / "runs" / "labeling_agreement"; out.mkdir(parents=True, exist_ok=True)
    A = read_labels(args.main)
    B = read_labels(args.second)
    meta = pd.read_csv(EXP_ROOT / "runs" / "labeling_items.csv", dtype=str).fillna("")
    accepted = set(meta.loc[meta.kind == "selected", "item_id"])

    m = A.merge(B, on="item_id", suffixes=("_1", "_2"))
    m["accepted"] = m.item_id.isin(accepted)
    lines = [f"main annotator: {A.annotator.iloc[0] if len(A) else ''} ({len(A)} items) | "
             f"second annotator: {B.annotator.iloc[0] if len(B) else ''} ({len(B)} items)",
             f"items labelled by both: {len(m)} "
             f"(accepted by the filter: {int(m.accepted.sum())}, near/below threshold: {int((~m.accepted).sum())})", ""]
    # primary label: the two "cannot determine" outcomes folded into one (4 classes)
    fold = {"unclear": "cannot_determine", "brand_only": "cannot_determine"}
    for k in ("1", "2"):
        m[f"primary_{k}"] = m[f"label_{k}"].map(lambda v: fold.get(v, v))
    for name, sub in [("all items", m), ("accepted by the filter", m[m.accepted]),
                      ("near/below threshold", m[~m.accepted])]:
        po_b, k_b = kappa(sub.brand_1, sub.brand_2)
        po_p, k_p = kappa(sub.primary_1, sub.primary_2)
        po_f, k_f = kappa(sub.label_1, sub.label_2)
        lines.append(f"{name:24s} n={len(sub):3d} | brand: {po_b:.1%}, kappa {k_b:.2f}"
                     f" | primary (4 classes): {po_p:.1%}, kappa {k_p:.2f}"
                     f" | full (5 classes): {po_f:.1%}, kappa {k_f:.2f}")
    for flag in ("packaging_differs", "part_of_larger_image", "watermark_or_overlay"):
        if f"{flag}_1" in m and f"{flag}_2" in m:
            po, k = kappa(m[f"{flag}_1"], m[f"{flag}_2"])
            lines.append(f"flag {flag:22s}: agreement {po:.1%}, kappa {k:.2f}")
    lines += ["", "full-annotation confusion (rows = main, columns = second):",
              pd.crosstab(m.label_1, m.label_2).to_string()]
    txt = "\n".join(lines)
    (out / "agreement.txt").write_text(txt, encoding="utf-8"); print(txt)

    d = m[m.label_1 != m.label_2].copy()
    d = d[["item_id", "query_id_1", "label_1", "label_2", "note_1", "note_2", "accepted"]].rename(
        columns={"query_id_1": "query_id", "label_1": "label_main", "label_2": "label_second",
                 "note_1": "note_main", "note_2": "note_second"})
    if not args.resolved:
        d["final_label"] = ""
        d.to_csv(out / "disagreements.csv", index=False, encoding="utf-8-sig")
        rows = "".join(
            f"<tr><td>{html.escape(r.item_id)}</td>"
            f"<td><img src='../labeling/img/query_{html.escape(r.query_id)}.jpg'></td>"
            f"<td><img src='../labeling/img/{html.escape(r.item_id)}.jpg'></td>"
            f"<td>main: <b>{html.escape(r.label_main)}</b><br>{html.escape(r.note_main)}<br><br>"
            f"second: <b>{html.escape(r.label_second)}</b><br>{html.escape(r.note_second)}</td></tr>"
            for r in d.itertuples())
        (out / "disagreements.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Disagreements</title>"
            "<style>body{font-family:system-ui;margin:16px}td{border:1px solid #ccc;padding:6px;vertical-align:top}"
            "img{max-width:420px;max-height:320px}</style>"
            f"<h2>{len(d)} disagreements</h2><table>{rows}</table>", encoding="utf-8")
        print(f"\n{len(d)} disagreements -> {out / 'disagreements.csv'} and disagreements.html")
        return

    R = pd.read_csv(args.resolved, dtype=str).fillna("")
    R = R[R.final_label.str.strip() != ""]
    valid = {"same", "variant", "brand_only", "different", "unclear"}
    bad = R[~R.final_label.isin(valid)]
    if len(bad):
        raise SystemExit(f"invalid final_label values: {sorted(set(bad.final_label))}")
    F = A.copy(); F["resolved"] = 0
    fix = dict(zip(R.item_id, R.final_label))
    hit = F.item_id.isin(fix)
    F.loc[hit, "label"] = F.loc[hit, "item_id"].map(fix); F.loc[hit, "resolved"] = 1
    F.to_csv(out / "labels_final.csv", index=False, encoding="utf-8-sig")
    print(f"\nresolved {int(hit.sum())} of {len(d)} disagreements -> {out / 'labels_final.csv'}")

if __name__ == "__main__":
    main()
