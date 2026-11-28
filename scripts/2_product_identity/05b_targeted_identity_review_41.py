#!/usr/bin/env python3
"""Build/apply a traceable identity-label quality-control review.

Do NOT overwrite the independent annotations, original adjudication, or
labels_final_checked.csv. This script creates a new HTML review and writes a
new checked-QC CSV only when --apply is explicitly requested.

From web_reference_experiments/:
    python scripts/15_identity_qc_review.py
    # Open runs/labeling_agreement/identity_qc/identity_review_41_v13.html
    # Export the reviewed CSV from Chrome, then:
    python scripts/15_identity_qc_review.py --apply PATH_TO_EXPORTED_REVIEW.csv

Optional:
    python scripts/15_identity_qc_review.py --scope all
    python scripts/15_identity_qc_review.py --root PATH_TO_EXPERIMENTS

Requires: pandas. Browser: Chrome/Edge, offline.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

VERSION = "identity_qc_v1_2026_11"
LABELS = {"same", "variant", "different", "unclear", "brand_only"}
FLAG_COLUMNS = ["pack_quantity", "branding", "layout_color", "language_market", "guidance", "other"]
ATTRS = ["name", "ingredient", "strength", "dosage_form", "formulation"]

def read(path: Path) -> pd.DataFrame:
    body = "\n".join(s for s in path.read_text(encoding="utf-8-sig").splitlines() if not s.startswith("#"))
    return pd.read_csv(io.StringIO(body), dtype=str, keep_default_na=False).fillna("")

def root_from_here() -> Path:
    for p in (Path.cwd().resolve(), *Path(__file__).resolve().parents):
        if (p / "runs/labeling_agreement/labels_final_checked.csv").exists():
            return p
    raise SystemExit("Cannot locate the experiments root. Pass --root PATH_TO_web_reference_experiments")

def find_optional(root: Path, *names: str) -> Path | None:
    places = [root / "runs/labeling_agreement", root / "runs/labeling", root / "scripts", Path(__file__).resolve().parent]
    for base in places:
        for name in names:
            p = base / name
            if p.is_file():
                return p
    return None

def index_rows(p: Path | None) -> dict:
    if p is None:
        return {}
    df = read(p)
    if "item_id" not in df:
        raise SystemExit(f"{p} missing item_id column")
    if df.item_id.duplicated().any():
        raise SystemExit(f"Duplicate item_id in {p}")
    return {r["item_id"]: r for r in df.to_dict("records")}

def build(root: Path, scope: str, flagged_file: Path | None, output: Path | None, ids_file: Path | None = None) -> None:
    ag = root / "runs/labeling_agreement"
    labels_path = ag / "labels_final_checked.csv"
    if not labels_path.is_file():
        raise SystemExit(f"Missing finalized labels: {labels_path}")
    final = read(labels_path)
    if final.item_id.duplicated().any():
        raise SystemExit("Duplicate IDs in labels_final_checked.csv")
    if not {"item_id", "label", "packaging_differs", "query_id"} <= set(final.columns):
        raise SystemExit("Final label CSV lacks required identity or M columns")
    byid = {r["item_id"]: r for r in final.to_dict("records")}
    disagreements = index_rows(ag / "disagreements_resolved.csv" if (ag / "disagreements_resolved.csv").is_file() else None)
    if scope == "review" and not disagreements:
        raise SystemExit("Missing runs/labeling_agreement/disagreements_resolved.csv")

    if flagged_file is None:
        flagged_file = next((p for p in (
            root / "runs/labeling_agreement/Comment3_16_variant_flags_FINAL.csv",
            root / "runs/labeling/Comment3_16_variant_flags_FINAL.csv",
            root / "scripts/Comment3_16_variant_flags_FINAL.csv",
        ) if p.is_file()), None)
    flagged = index_rows(flagged_file)
    if scope == "review" and not flagged:
        raise SystemExit("STOP: Cannot find Comment3_16_variant_flags_FINAL.csv. "
                         "The 41-case QC review must include the 16 flagged images. "
                         "Place the CSV in runs/labeling_agreement/ or use --flags PATH; then rebuild.")

    # Original independent annotations are reference evidence, not ground truth.
    original = index_rows(find_optional(root, "labels_primary_Yona_2026-10-22.csv", "labels_primary_Yona_2026-10-22 (1).csv"))
    ei = index_rows(find_optional(root, "labels_second_EI_2026-10-22(1).csv", "labels_second_EI_2026-10-22.csv"))
    rm = index_rows(find_optional(root, "labels_second_RM_2026-11-03 (2).csv", "labels_second_RM_2026-11-03.csv"))

    if ids_file is not None:
        target = read(ids_file)
        if "item_id" not in target.columns:
            raise SystemExit(f"Custom review list must have an item_id column: {ids_file}")
        ids = list(dict.fromkeys(target.item_id.tolist()))
    elif scope == "all":
        ids = final.item_id.tolist()
    else:
        # Original 28 first; the 16 Comment3 flags add non-overlapping cases.
        ids = list(disagreements) + [iid for iid in flagged if iid not in disagreements]
    if scope == "review" and ids_file is None:
        expected = len(set(disagreements) | set(flagged))
        if expected != 41:
            raise SystemExit(f"STOP: Expected 41 unique cases (28 prior disagreements + 16 flagged - 3 overlaps); found {expected}. Verify the input files.")
    unknown = [iid for iid in ids if iid not in byid]
    if unknown:
        raise SystemExit(f"Review IDs missing from finalized dataset: {unknown[:5]}")
    outdir = (output or (ag / "identity_qc")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    img = root / "runs/labeling/img"
    rows = []
    for iid in ids:
        r = byid[iid]
        qid = r["query_id"]
        dis = disagreements.get(iid, {})
        flag = flagged.get(iid, {})
        qpath = img / f"query_{qid}.jpg"
        mpath = img / f"{iid}.jpg"
        def lb(src):
            v = src.get("label", "")
            return v if v in LABELS else ""
        rows.append({
            "id": iid, "qid": qid,
            "focus": ("Comment 3 flagged" if iid in flagged else "") + (" + prior disagreement" if iid in flagged and iid in disagreements else ("Prior disagreement" if iid in disagreements else "")),
            "card": flag.get("card", ""), "medicine": flag.get("medicine", ""),
            "baseline_label": r["label"], "baseline_m": str(r["packaging_differs"]),
            "baseline_note": r.get("note", ""),
            "qimage": os.path.relpath(qpath, outdir).replace("\\", "/"),
            "mimage": os.path.relpath(mpath, outdir).replace("\\", "/"),
            "a1_label": lb(original.get(iid, {})) or dis.get("label_main", ""),
            "ei_label": lb(ei.get(iid, {})) or dis.get("label_second", ""),
            "rm_label": lb(rm.get(iid, {})),
            "ei_note": ei.get(iid, {}).get("note", "") or dis.get("note_second", ""),
            "rm_note": rm.get(iid, {}).get("note", ""),
            "prior_resolution": dis.get("reason", ""),
            "comment3_note": flag.get("note", ""),
            "review_focus": flag.get("review_focus", ""),
        })
    fingerprint = hashlib.sha256((VERSION+"\n"+"\n".join(f'{r["id"]}|{r["baseline_label"]}|{r["baseline_m"]}' for r in rows)).encode()).hexdigest()[:16]
    template = HTML_TEMPLATE.replace("__DATA__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"))
    template = template.replace("__FINGERPRINT__", fingerprint).replace("__VERSION__", VERSION)
    (outdir / "identity_review_41_v13.html").write_text(template, encoding="utf-8")
    (outdir / "identity_review_manifest.csv").write_text(
        pd.DataFrame([{k:r.get(k,"") for k in ["id","qid","baseline_label","baseline_m","focus","qimage","mimage"]} for r in rows]).rename(columns={"id":"item_id", "qid":"query_id"}).to_csv(index=False), encoding="utf-8-sig")
    missing_images = sum(not (img / f'{r["id"]}.jpg').exists() or not (img / f'query_{r["qid"]}.jpg').exists() for r in rows)
    print(f"Generated {len(rows)} cases: {len(disagreements)} original disagreements, {len(flagged)} flagged cases; overlap {len(set(disagreements)&set(flagged))}.")
    print(f"Baseline source: {labels_path}; baseline fingerprint: {fingerprint}")
    print(f"Images missing in target project: {missing_images}/{len(rows)} pairs.")
    print(f"Open ONLY this page in Chrome: {outdir / 'identity_review_41_v13.html'}")
    print("No source label files were modified.")


def apply(root: Path, csv_path: Path, outdir: Path | None = None) -> None:
    ag = root / "runs/labeling_agreement"
    p = ag / "labels_final_checked.csv"
    final = read(p)
    rev = read(csv_path)
    required = {"item_id", "query_id", "baseline_label", "baseline_m", "revised_label", "revised_m", "packaging_observed", "reviewer", "reviewed_at", "review_reason", "review_status", "version", "fingerprint"}
    missing = required - set(rev.columns)
    if missing: raise SystemExit(f"Review CSV missing: {sorted(missing)}")
    if rev.empty: raise SystemExit("Review CSV has no completed reviews")
    if rev.item_id.duplicated().any(): raise SystemExit("Duplicate item IDs in review CSV")
    if set(rev.version) != {VERSION} or len(set(rev.fingerprint)) != 1:
        raise SystemExit("Wrong review format or mixed builds")
    manifest = (outdir or (ag / "identity_qc")) / "identity_review_manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"Missing the review manifest: {manifest}. Rebuild the review tool first.")
    man = read(manifest)
    expected_fp = hashlib.sha256((VERSION+"\n"+"\n".join(f'{r["item_id"]}|{r["baseline_label"]}|{r["baseline_m"]}' for r in man.to_dict("records"))).encode()).hexdigest()[:16]
    if set(rev.fingerprint) != {expected_fp} or not set(rev.item_id).issubset(set(man.item_id)):
        raise SystemExit("Review CSV does not match the current QC manifest/fingerprint")
    if not rev.review_status.isin({"confirmed", "unresolved"}).all():
        raise SystemExit("Review CSV contains unknown review status")
    ix = final.set_index("item_id")
    unknown = set(rev.item_id) - set(ix.index)
    if unknown: raise SystemExit(f"Unknown item IDs: {list(unknown)[:5]}")
    audit = []
    for _, r in rev.iterrows():
        oid = r.item_id
        old = ix.loc[oid]
        if old.label != r.baseline_label or old.packaging_differs != r.baseline_m or old.query_id != r.query_id:
            raise SystemExit(f"Baseline conflict for {oid}: do not overwrite a later correction. Rebuild review.")
        if not r.reviewer.strip() or not r.reviewed_at.strip():
            raise SystemExit(f"Missing reviewer/time for {oid}")
        if r.review_status == "unresolved":
            continue  # Preserve baseline labels; unresolved cases enter a separate log.
        if r.revised_label not in LABELS or r.revised_m not in {"0", "1"}:
            raise SystemExit(f"Bad label/M value for {oid}")
        if r.packaging_observed not in {"yes", "no"}:
            raise SystemExit(f"Packaging observation missing/ambiguous for confirmed {oid}")
        cats = {x for x in r.get("packaging_categories", "").split(";") if x}
        if r.packaging_observed == "yes" and not cats:
            raise SystemExit(f"Choose a packaging-difference category for {oid}")
        if r.packaging_observed == "no" and cats:
            raise SystemExit(f"Conflicting packaging categories for {oid}")
        if not cats.issubset(set(FLAG_COLUMNS)):
            raise SystemExit(f"Unknown packaging categories for {oid}: {cats-set(FLAG_COLUMNS)}")
        name = r.get("name_status", "")
        attrs = [r.get(x+"_status", "") for x in ("ingredient", "strength", "dosage_form", "formulation")]
        if name not in {"same", "base", "different", "unreadable"}:
            raise SystemExit(f"Missing/invalid name comparison for {oid}")
        if name in {"same", "base"} and any(v not in {"same", "different", "only_one", "unreadable", "neither"} for v in attrs):
            raise SystemExit(f"Incomplete pharmaceutical comparisons for {oid}")
        suggestion = ("different" if name == "different" else "brand_only" if name == "unreadable" else
                      "variant" if "different" in attrs else
                      "unclear" if name == "base" or any(v in {"only_one", "unreadable"} for v in attrs) else "same")
        if r.revised_label != suggestion:
            if r.get("rule_override", "") != "1":
                raise SystemExit(f"Missing explicit rule override flag for {oid}")
            # The full structured evidence and override indicator are preserved;
            # a free-text note is optional for speed. Overrides remain visible in the audit.
        elif r.get("rule_override", "") != "0":
            raise SystemExit(f"Unexpected rule override flag for {oid}")
        if r.revised_label in {"variant", "different"} and r.revised_m == "1":
            raise SystemExit(f"M=1 for {oid}: original M definition is restricted to same/unclear. Detailed packaging differences should be retained in observation columns instead.")
        expected_m = "0" if r.revised_label in {"variant", "different"} else ("1" if r.packaging_observed == "yes" else "0")
        if r.revised_m != expected_m:
            raise SystemExit(f"M flag does not match packaging selections for {oid}")
        audit.append({"item_id": oid, "query_id": r.query_id, "old_label": r.baseline_label,
                      "new_label": r.revised_label, "old_m": r.baseline_m, "new_m": r.revised_m,
                      "label_changed": int(r.revised_label != r.baseline_label),
                      "m_changed": int(r.revised_m != r.baseline_m),
                      "review_reason": r.review_reason, "reviewer": r.reviewer, "reviewed_at": r.reviewed_at,
                      "name_status": r.get("name_status", ""), "ingredient_status": r.get("ingredient_status", ""),
                      "strength_status": r.get("strength_status", ""), "dosage_form_status": r.get("dosage_form_status", ""),
                      "formulation_status": r.get("formulation_status", ""),
                      "packaging_observed": r.get("packaging_observed", ""),
                      "comparison_status": r.get("comparison_status", ""),
                      "rule_override": r.get("rule_override", "")})
    # Write only a NEW version; preserve all original columns and unchanged rows.
    revised = final.copy()
    if "qc_reviewed" not in revised: revised["qc_reviewed"] = "0"
    if "qc_review_date" not in revised: revised["qc_review_date"] = ""
    if "qc_decision_note" not in revised: revised["qc_decision_note"] = ""
    for _,r in rev[rev.review_status == "confirmed"].iterrows():
        hit = revised.item_id == r.item_id
        revised.loc[hit,"label"] = r.revised_label
        revised.loc[hit,"primary_label"] = "cannot_determine" if r.revised_label in {"brand_only", "unclear"} else r.revised_label
        revised.loc[hit,"packaging_differs"] = r.revised_m
        revised.loc[hit,"qc_reviewed"] = "1"
        revised.loc[hit,"qc_review_date"] = r.reviewed_at
        revised.loc[hit,"qc_decision_note"] = r.review_reason
    # Avoid overwrites if previous QC already exists.
    stem = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ag / f"labels_final_checked_identity_QC_{stem}.csv"
    log_path = ag / f"identity_QC_changes_{stem}.csv"
    sum_path = ag / f"identity_QC_summary_{stem}.txt"
    unresolved_path = ag / f"identity_QC_unresolved_{stem}.csv"
    revised.to_csv(output, index=False, encoding="utf-8-sig")
    rev[rev.review_status == "unresolved"].to_csv(unresolved_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(log_path, index=False, encoding="utf-8-sig")
    changed = [a for a in audit if a["label_changed"] or a["m_changed"]]
    summary = f"QC confirmed: {len(audit)}; unresolved: {int((rev.review_status=='unresolved').sum())}; identity changes: {sum(x['label_changed'] for x in audit)}; M-only changes: {sum(x['m_changed'] and not x['label_changed'] for x in audit)}; any change: {len(changed)}.\nOriginal labels unchanged: {p}.\nNew full labels: {output}.\nNew detailed audit: {log_path}.\nUnresolved records: {unresolved_path}.\n"
    sum_path.write_text(summary,encoding="utf-8")
    print(summary)

HTML_TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Identity QC · 41 cases · v1.3</title>
<style>
:root{--bg:#111723;--panel:#1b2434;--panel2:#141e2e;--line:#33425b;--ink:#e9eef6;--muted:#a2b2c8;--accent:#62a6d1;--good:#65bdab;--warn:#f2b364;--bad:#e28d7d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,Segoe UI,Arial,sans-serif}header{position:sticky;top:0;z-index:10;background:#101827;border-bottom:1px solid var(--line);padding:9px 15px;display:flex;align-items:center;gap:9px;flex-wrap:wrap}
header strong{font-size:17px;margin-right:auto}button,input,textarea{font:inherit;color:inherit}button{cursor:pointer;border:1px solid var(--line);border-radius:5px;background:var(--panel2);padding:7px 10px}button:hover{border-color:var(--accent)}button.sel{background:#185780;color:white;border-color:#7ebbe0}button.primary{background:#287ca8;border-color:#287ca8;color:white;font-weight:700}button:focus-visible,input:focus-visible,textarea:focus-visible{outline:2px solid var(--accent);outline-offset:2px}button:disabled{opacity:.4;cursor:not-allowed}
input[type=text],textarea{background:#101726;border:1px solid var(--line);border-radius:5px;padding:7px 8px}input[type=text]{width:100%}textarea{width:100%;min-height:62px}#search{max-width:145px}#who{width:75px}main{max-width:1650px;margin:12px auto;padding:0 15px 26px}.card{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:11px 13px;margin-bottom:10px}.meta{color:var(--muted);font-size:12px}.tag{color:var(--warn);font-size:12px}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.imcard{text-align:center;background:#0d1521;border:1px solid var(--line);padding:7px;border-radius:5px}.imcard img{width:100%;height:280px;object-fit:contain;cursor:zoom-in}.imcard b{display:block;margin-bottom:4px}.hint{color:var(--muted);font-size:12.5px;margin:5px 0}.topdata{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:9px}.topdata>div{background:var(--panel2);border:1px solid var(--line);border-radius:5px;padding:8px}.topdata b{display:block;font-size:12.5px;color:#cde3f6}.topdata p{margin:5px 0 0}.line{border-bottom:1px solid var(--line);margin:10px 0}h2{font-size:16px;margin:0 0 8px}h3{font-size:14px;margin:0 0 6px}.attrtable{width:100%;border-collapse:collapse;table-layout:fixed}.attrtable th{font-size:12px;font-weight:600;color:var(--muted);text-align:left}.attrtable td,.attrtable th{padding:6px;border-bottom:1px solid #314056;vertical-align:middle}.attrtable th:first-child,.attrtable td:first-child{width:17%}.attrtable th:nth-child(2),.attrtable td:nth-child(2){width:20%}.attrtable th:nth-child(3),.attrtable td:nth-child(3){width:20%}.attrtable th:last-child,.attrtable td:last-child{width:43%}.statusrow{display:flex;gap:4px;flex-wrap:wrap}.statusrow button{font-size:12px;padding:6px 7px}.bottomrow{display:grid;grid-template-columns:1fr 1fr;gap:10px}.opts{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}.opts button{font-size:13px}.packchecks{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin:7px 0}.packchecks label{background:var(--panel2);border:1px solid var(--line);padding:7px;border-radius:5px}.recommend{padding:8px;background:#103048;border:1px solid #367499;border-radius:5px;font-size:13px}.bad{color:var(--bad)}.warn{color:var(--warn)}.good{color:var(--good)}.small{font-size:12px;color:var(--muted)}#msg{min-height:22px;margin-top:7px}.modal{position:fixed;inset:0;z-index:30;display:none;background:#05070bea;grid-template-columns:1fr 1fr;gap:5px;padding:35px 8px 8px}.modal.show{display:grid}.modal img{width:100%;height:calc(100vh - 50px);object-fit:contain}.modal button{position:absolute;right:12px;top:7px}.disabled{opacity:.6}details summary{cursor:pointer;color:var(--accent)}@media(max-width:1050px){.bottomrow,.topdata,.pair{grid-template-columns:1fr}.attrtable{table-layout:auto}.attrtable td,.attrtable th{width:auto!important}.imcard img{height:220px}}@media(max-width:620px){header{position:static}.attrtable{display:block;overflow-x:auto}.packchecks{grid-template-columns:1fr 1fr}}

/* Simple, paper-like review interface. Optional transcription fields are hidden initially. */
:root{--bg:#f6f8fb;--panel:#fff;--panel2:#f1f5f9;--line:#ccd6e2;--ink:#162235;--muted:#52657d;--accent:#246a94;--good:#15724d;--warn:#a74c16;--bad:#b13f36}
body{background:var(--bg);color:var(--ink)}
header{background:#fff;box-shadow:0 1px 5px #1d3c5a13}
button{background:#fff;color:var(--ink)}button:hover{background:#eef6fb}button.sel{background:#246a94;color:white;border-color:#246a94}
button.primary{background:#1b779f;border-color:#1b779f;color:white}
input[type=text],textarea{background:#fff;color:var(--ink)}
.imcard{background:#f8fafc}.topdata>div{background:#f8fafc}.topdata b{color:#355577}
.attrtable td,.attrtable th{border-color:#e5eaf0}.recommend{background:#eef6fb;border-color:#c5dce9}
.meta,.hint,.small{color:var(--muted)}.tag{color:#875015}details summary{color:#205b82}
#wordingToggle{padding:4px 9px;font-size:12px;color:#246a94;border-color:#cbddeb;background:#fff}
.attrtable th:first-child,.attrtable td:first-child{width:24%}
.attrtable th:last-child,.attrtable td:last-child{width:76%}
.attrtable th:nth-child(2),.attrtable td:nth-child(2),.attrtable th:nth-child(3),.attrtable td:nth-child(3){display:none}
body.showwords .attrtable th:first-child,body.showwords .attrtable td:first-child{width:17%}
body.showwords .attrtable th:nth-child(2),body.showwords .attrtable td:nth-child(2),body.showwords .attrtable th:nth-child(3),body.showwords .attrtable td:nth-child(3){display:table-cell;width:20%}
body.showwords .attrtable th:last-child,body.showwords .attrtable td:last-child{width:43%}
#referenceStatus{color:#30485a;font-size:12.5px;margin-top:5px}
#reasonDetails{margin:9px 0}#reasonDetails textarea{min-height:44px}#msg{min-height:0;margin-top:3px}
.statusnote{font-size:12px;min-height:17px}
</style></head><body>
<header><strong>IDENTITY QC v1.3 — <span id="index" class="meta"></span></strong><label class="meta">Reviewer <input id="who" value="A1"></label><input type="text" id="search" placeholder="Find ID or Q034"><button id="goSearch">Find</button><button id="prev">← Previous</button><button id="next">Next →</button><button id="todo">Next to review</button><button id="import">Import review CSV</button><input type="file" id="fileinput" accept=".csv" hidden><button class="primary" id="export">Save QC CSV</button><span class="meta" id="progress"></span></header>
<main><section class="card"><div><b id="title"></b> <span class="tag" id="focus"></span><span class="meta" id="product"></span></div>
<div class="pair" style="margin-top:8px"><div class="imcard"><b>Query package</b><img id="qi" alt="Query package; click to enlarge"><span class="meta">Click to enlarge</span></div><div class="imcard"><b>Retrieved image</b><img id="mi" alt="Retrieved image; click to enlarge"><span class="meta">Click to enlarge</span></div></div>
<details style="margin-top:9px"><summary>Original independent annotations and previous decisions (for context)</summary><div class="topdata"><div><b>Independent labels</b><p id="independent"></p><p id="indnotes" class="small"></p></div><div><b>Earlier adjudication</b><p id="adjudication"></p></div><div><b>Finalized baseline</b><p id="baseline"></p><p id="baselineNote" class="small"></p></div></div><p class="small" id="c3note"></p></details>
</section>
<section class="card"><h2>1 · Medicine information <button type="button" id="wordingToggle">+ Printed wording (optional)</button></h2>
<p class="hint">Compare the two packages only. If information is missing, do not assume it matches.</p>
<table class="attrtable"><thead><tr><th>Attribute</th><th>Query wording</th><th>Retrieved wording</th><th>Comparison</th></tr></thead><tbody id="attributes"></tbody></table>
<div class="recommend" style="margin-top:9px"><b>Identity suggested:</b> <span id="suggestion"></span> <span id="guideMessage" class="small"></span></div>
</section>
<section class="bottomrow"><div class="card"><h2>2 · Packaging differences</h2><div class="opts" id="packaging"></div><div class="packchecks" id="packchecks"></div>
<details><summary>Pack quantities (optional)</summary><div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:6px"><label class="small">Query<input id="packQ" type="text" placeholder="e.g. 24 tablets"></label><label class="small">Retrieved<input id="packR" type="text" placeholder="e.g. 6 tablets"></label></div></details></div>
<div class="card"><h2>3 · Decision</h2><div class="opts" id="labels"></div><p id="logic" class="hint"></p><div id="referenceStatus"></div>
<details id="reasonDetails"><summary>Optional note</summary><textarea id="reason" placeholder="Only if clarification is useful"></textarea></details>
<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:7px"><button class="primary" id="confirm">✓ Save &amp; next</button><button id="unresolved">Cannot decide</button><span id="status" class="statusnote"></span></div><div id="msg" class="warn"></div></div></section>
<p class="hint">Save QC CSV regularly. Original annotations are preserved.</p></main>
<div class="modal" id="modal"><button id="closezoom">Close ✕</button><img id="zq"><img id="zm"></div>
<script>
"use strict";
const DATA=__DATA__, FP='__FINGERPRINT__', VER='__VERSION__';
const KEY='iecv_identity_qc_'+FP;const $=x=>document.getElementById(x);
let ST={},i=0;try{ST=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(_){}
const attrNames=[['name','Printed medicine name'],['ingredient','Active ingredient(s)'],['strength','Strength per unit'],['dosage_form','Dosage form'],['formulation','Formulation (e.g., XR, dispersible)']];
const nameOpts=[['same','Same'],['base','Same base name'],['different','Different'],['unreadable','Unreadable']];
const attOpts=[['same','Same'],['different','Different'],['only_one','Only on one'],['unreadable','Unreadable'],['neither','Neither printed']];
const packOpts=[['yes','Difference observed'],['no','None observed'],['unclear','Cannot tell']];
const labelOpts=[['same','Same'],['variant','Variant'],['different','Different'],['unclear','Cannot determine'],['brand_only','Name unreadable']];
const packCats=[['pack_quantity','Pack quantity'],['branding','Manufacturer / logo'],['layout_color','Design / color'],['language_market','Language / market'],['guidance','Age / weight guidance'],['other','Other']];
function fresh(it){return {attrs:{},printed:{},packaging:'',pack_cats:[],pack_query:'',pack_retrieved:'',label:'',reason:'',confirmed:false,unresolved:false};}
function st(it){return ST[it.id]||fresh(it)}
function save(){try{localStorage.setItem(KEY,JSON.stringify(ST))}catch(e){console.warn(e)}}
function update(changes){const it=DATA[i];ST[it.id]={...st(it),...changes,confirmed:false,unresolved:false};save();render()}
function setAttr(k,choice){const s=st(DATA[i]);update({attrs:{...s.attrs,[k]:choice}})}
function setPrinted(k,which,value){const s=st(DATA[i]);update({printed:{...s.printed,[k+'_'+which]:value}})}
function optionButtons(opts,selected,onClick){return opts.map(([v,l])=>`<button type="button" data-value="${v}" class="${selected===v?'sel':''}">${l}</button>`).join('')}
function render(){const it=DATA[i],s=st(it);$('index').textContent=`${i+1}/${DATA.length}`;
 $('title').textContent=it.id; $('focus').textContent=it.focus; $('product').textContent=`  ${it.medicine||it.qid}${it.card?' · Card '+it.card:''}`;
 $('qi').src=it.qimage;$('mi').src=it.mimage;
 $('independent').textContent=`A1: ${it.a1_label||'not available'} · E.I.: ${it.ei_label||'not sampled'} · R.M.: ${it.rm_label||'not sampled'}`;
 $('indnotes').textContent=[it.ei_note?'E.I.: '+it.ei_note:'',it.rm_note?'R.M.: '+it.rm_note:''].filter(Boolean).join(' · ')||'No supporting notes';
 $('adjudication').textContent=it.prior_resolution||'Not part of the original 28 disagreements';
 $('baseline').textContent=`Identity: ${it.baseline_label} · M=${it.baseline_m}`;$('baselineNote').textContent=it.baseline_note||'';
 $('c3note').textContent=it.comment3_note?`Comment 3 record (not changed): ${it.comment3_note}`:'';
 $('attributes').innerHTML=attrNames.map(([k,l])=>`<tr><td><b>${l}</b></td><td><input type="text" data-printed="${k}:q" placeholder="As printed" /></td><td><input type="text" data-printed="${k}:r" placeholder="As printed" /></td><td><div class="statusrow" data-attr="${k}">${optionButtons(k==='name'?nameOpts:attOpts,(s.attrs||{})[k]||'')}</div></td></tr>`).join('');
 $('attributes').querySelectorAll('[data-printed]').forEach(el=>{const [k,side]=el.dataset.printed.split(':');el.value=(s.printed||{})[k+'_'+side]||'';el.oninput=()=>{const now=st(it);ST[it.id]={...now,printed:{...now.printed,[k+'_'+side]:el.value},confirmed:false,unresolved:false};save();$('status').textContent='';};});
 $('attributes').querySelectorAll('[data-attr]').forEach(parent=>parent.querySelectorAll('button').forEach(b=>b.onclick=()=>setAttr(parent.dataset.attr,b.dataset.value)));
 $('packaging').innerHTML=optionButtons(packOpts,s.packaging,()=>{});$('packaging').querySelectorAll('button').forEach(b=>b.onclick=()=>update({packaging:b.dataset.value,pack_cats:b.dataset.value==='yes'?st(it).pack_cats:[]}));
 $('packchecks').innerHTML=packCats.map(([v,l])=>`<label><input type="checkbox" data-packcat="${v}" ${(s.pack_cats||[]).includes(v)?'checked':''}> ${l}</label>`).join('');
 $('packchecks').querySelectorAll('input').forEach(el=>el.onchange=()=>{const old=st(it);const arr=new Set(old.pack_cats||[]);el.checked?arr.add(el.dataset.packcat):arr.delete(el.dataset.packcat);update({pack_cats:Array.from(arr),packaging:arr.size?'yes':old.packaging==='yes'?'':old.packaging})});
 $('packQ').value=s.pack_query||'';$('packR').value=s.pack_retrieved||'';
 $('labels').innerHTML=optionButtons(labelOpts,s.label||suggest(s).label,()=>{});$('labels').querySelectorAll('button').forEach(b=>b.onclick=()=>update({label:b.dataset.value}));
 $('reason').value=s.reason||'';
 const sg=suggest(s); $('suggestion').textContent=sg.label?sg.label:'Select attributes';$('guideMessage').textContent='';$('referenceStatus').textContent=comparisonStatus(s);
 const newM=effectiveM(s);$('logic').textContent=`Final label: ${s.label||sg.label||'—'} · Packaging flag: ${newM||'—'} (was ${it.baseline_m})`;
 $('status').textContent=s.confirmed?'✓ Saved':s.unresolved?'Needs review':'';$('status').className=s.confirmed?'good':'warn';
 $('progress').textContent=DATA.filter(x=>st(x).confirmed).length+' confirmed · '+DATA.filter(x=>st(x).unresolved).length+' unresolved / '+DATA.length;
 $('msg').textContent='';$('reasonDetails').open=!!s.reason;
}
function suggest(s){const a=s.attrs||{};if(!a.name)return {label:'',note:'First compare printed names.'};
 if(a.name==='different')return {label:'different',note:'Different printed medicine names.'};
 if(a.name==='unreadable')return {label:'brand_only',note:'Name unreadable; do not assume pharmaceutical identity.'};
 for(const k of ['ingredient','strength','dosage_form','formulation'])if(!a[k])return {label:'',note:'Complete all four pharmaceutical comparisons.'};
 if(['ingredient','strength','dosage_form','formulation'].some(k=>a[k]==='different'))return {label:'variant',note:'At least one explicit pharmaceutical/formulation difference is selected. Optional: record the printed wording.'};
 if(a.name==='base'||['ingredient','strength','dosage_form','formulation'].some(k=>a[k]==='only_one'||a[k]==='unreadable'))return {label:'unclear',note:'The available printed information does not establish equivalence. Do not infer unspecified formulation terms.'};
 return {label:'same',note:'All comparable pharmaceutical attributes agree; criteria absent from both packages are not contrasted.'};}
function comparisonCode(s){const a=s.attrs||{},sg=suggest(s);
 if(!sg.label)return 'not_assessed';
 if(['variant','different'].includes(sg.label))return 'pharmaceutical_difference';
 if(sg.label==='unclear'||sg.label==='brand_only'||['ingredient','strength','dosage_form','formulation'].some(k=>a[k]==='neither'))return 'formulation_or_identity_unconfirmed';
 if(s.packaging==='yes')return 'formulation_supported_packaging_differs';
 if(s.packaging==='no')return 'potential_comparison_candidate';
 return 'packaging_not_assessed';}
function comparisonStatus(s){return ({not_assessed:'Reference comparison: not assessed',pharmaceutical_difference:'Reference comparison: pharmaceutical mismatch',formulation_or_identity_unconfirmed:'Reference comparison: formulation/identity not established',formulation_supported_packaging_differs:'Reference comparison: formulation supported; packaging differs',potential_comparison_candidate:'Reference comparison: potential candidate (not proof of authenticity)',packaging_not_assessed:'Reference comparison: assess packaging differences'})[comparisonCode(s)];}
function effectiveM(s){if(!s.packaging||s.packaging==='unclear')return '';
 if((s.label||suggest(s).label)==='variant'||(s.label||suggest(s).label)==='different')return '0';
 return s.packaging==='yes'?'1':'0';}
// Summarize the structured comparisons automatically so routine corrections do not require typing.
function autoEvidence(s){
 const a=s.attrs||{};
 const words={same:'same',base:'same base name',different:'different',unreadable:'unreadable',only_one:'printed on one package only',neither:'not printed on either package'};
 const parts=attrNames.filter(([k])=>a[k]).map(([k,l])=>l+': '+(words[a[k]]||a[k]));
 if(s.packaging==='yes'){
  const ks=(s.pack_cats||[]).map(k=>packCats.find(x=>x[0]===k)?.[1]||k);
  parts.push('Packaging difference: '+ks.join(', '));
 }else if(s.packaging==='no')parts.push('Packaging: no observed difference');
 else if(s.packaging==='unclear')parts.push('Packaging: cannot tell');
 if(s.pack_query||s.pack_retrieved)parts.push('Pack quantities: query '+(s.pack_query||'not stated')+', retrieved '+(s.pack_retrieved||'not stated'));
 for(const [k,l] of attrNames){const q=(s.printed||{})[k+'_q'],r=(s.printed||{})[k+'_r'];if(q||r)parts.push(l+' wording: '+(q||'?')+' / '+(r||'?'));}
 return 'Structured evidence — '+(parts.join('; ')||'review unresolved; evidence insufficient');
}
function check(){const it=DATA[i],s=st(it),sg=suggest(s);
 if(!sg.label)return 'Complete the pharmaceutical attribute comparisons before confirming.';
 if(!(s.label||sg.label)||!labelOpts.some(([k])=>k===(s.label||sg.label)))return 'Choose a final identity label.';
 if(!s.packaging||s.packaging==='unclear')return 'Choose Difference observed or None observed. Use Mark unresolved if packaging cannot be assessed.';
 if(s.packaging==='yes'&&!(s.pack_cats||[]).length)return 'Select at least one packaging-difference category.';
 if(s.packaging==='no'&&(s.pack_cats||[]).length)return 'Clear packaging categories or choose Difference observed.';
 // Decisions and any overrides are captured by structured selections.
 // Free-text notes are always optional; rule overrides are flagged in the audit.
 return '';}
$('wordingToggle').onclick=()=>{document.body.classList.toggle('showwords');$('wordingToggle').textContent=document.body.classList.contains('showwords')?'− Hide printed wording':'+ Printed wording (optional)'};
$('prev').onclick=()=>{i=Math.max(0,i-1);render();scrollTo(0,0)};
$('next').onclick=()=>{i=Math.min(DATA.length-1,i+1);render();scrollTo(0,0)};
$('todo').onclick=()=>{let j=DATA.findIndex((x,k)=>k>i&&!st(x).confirmed&&!st(x).unresolved);if(j<0)j=DATA.findIndex(x=>!st(x).confirmed&&!st(x).unresolved);if(j>=0){i=j;render();scrollTo(0,0)}};
$('goSearch').onclick=()=>{const q=$('search').value.trim().toLowerCase();const j=DATA.findIndex(x=>x.id.toLowerCase().includes(q)||x.qid.toLowerCase()===q||x.card===q||x.medicine.toLowerCase().includes(q));if(j>=0){i=j;render();scrollTo(0,0)}else alert('No matching item in this review set.')};
$('search').onkeydown=e=>{if(e.key==='Enter'){$('goSearch').click();e.preventDefault()}};
$('packQ').oninput=e=>{const it=DATA[i];ST[it.id]={...st(it),pack_query:e.target.value,confirmed:false,unresolved:false};save();$('status').textContent=''};
$('packR').oninput=e=>{const it=DATA[i];ST[it.id]={...st(it),pack_retrieved:e.target.value,confirmed:false,unresolved:false};save();$('status').textContent=''};
$('reason').oninput=e=>{const it=DATA[i];ST[it.id]={...st(it),reason:e.target.value,confirmed:false,unresolved:false};save();$('status').textContent=''};
$('confirm').onclick=()=>{const issue=check();if(issue){$('msg').textContent=issue;return;}const it=DATA[i],s=st(it);
 ST[it.id]={...s,label:s.label||suggest(s).label,confirmed:true,unresolved:false,at:new Date().toISOString(),reviewer:$('who').value.trim()||'A1'};save();if(i<DATA.length-1)i++;render();scrollTo(0,0);};
$('unresolved').onclick=()=>{const it=DATA[i],s=st(it);
 // Unresolved cases are preserved in a separate report; the original identity label is not changed.
 ST[it.id]={...s,confirmed:false,unresolved:true,at:new Date().toISOString(),reviewer:$('who').value.trim()||'A1'};save();render();$('msg').textContent='';if(i<DATA.length-1){i++;render();scrollTo(0,0)};};
function csvQ(x){const v=String(x==null?'':x);return /[",\n\r]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v}
function parseCSV(str){const out=[];let row=[],field='',quote=false;str=str.replace(/^\ufeff/,'');for(let j=0;j<str.length;j++){let c=str[j];if(quote){if(c==='"'){if(str[j+1]==='"'){field+='"';j++}else quote=false}else field+=c}else if(c==='"'){quote=true}else if(c===','){row.push(field);field=''}else if(c==='\n'){row.push(field.replace(/\r$/,''));out.push(row);row=[];field=''}else field+=c}if(field!==''||row.length){row.push(field);out.push(row)}return out}
const COLS=['version','fingerprint','item_id','query_id','baseline_label','baseline_m','revised_label','revised_m','packaging_observed','packaging_categories','name_status','ingredient_status','strength_status','dosage_form_status','formulation_status',...attrNames.flatMap(([k])=>[k+'_query_text',k+'_retrieved_text']),'pack_query_text','pack_retrieved_text','suggested_label','comparison_status','rule_override','review_reason','reviewer','reviewed_at','review_status'];
function resultRow(it,s){const obj={version:VER,fingerprint:FP,item_id:it.id,query_id:it.qid,baseline_label:it.baseline_label,baseline_m:it.baseline_m,
 revised_label:s.label,revised_m:effectiveM(s),packaging_observed:s.packaging,packaging_categories:(s.pack_cats||[]).join(';'),
 name_status:s.attrs?.name||'',ingredient_status:s.attrs?.ingredient||'',strength_status:s.attrs?.strength||'',dosage_form_status:s.attrs?.dosage_form||'',formulation_status:s.attrs?.formulation||'',
 pack_query_text:s.pack_query||'',pack_retrieved_text:s.pack_retrieved||'',suggested_label:suggest(s).label,comparison_status:comparisonCode(s),rule_override:s.label!==suggest(s).label?'1':'0',
 review_reason:(s.reason||autoEvidence(s)),reviewer:s.reviewer||'A1',reviewed_at:s.at||'',review_status:s.unresolved?'unresolved':'confirmed'};
 for(const [k] of attrNames){obj[k+'_query_text']=(s.printed||{})[k+'_q']||'';obj[k+'_retrieved_text']=(s.printed||{})[k+'_r']||'';}return obj;}
$('export').onclick=()=>{const rows=DATA.filter(it=>st(it).confirmed||st(it).unresolved).map(it=>resultRow(it,st(it)));
 if(!rows.length){alert('Confirm or mark at least one case unresolved before exporting.');return;}
 const lines=[COLS.join(','),...rows.map(r=>COLS.map(k=>csvQ(r[k])).join(','))];const data='\ufeff'+lines.join('\r\n');const url=URL.createObjectURL(new Blob([data],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='identity_qc_review_'+new Date().toISOString().slice(0,10)+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);};
$('import').onclick=()=>$('fileinput').click();
$('fileinput').onchange=async e=>{const file=e.target.files[0];if(!file)return;let rows=parseCSV(await file.text());let header=rows.shift();let pos={};header.forEach((h,k)=>pos[h]=k);
 if(!['version','fingerprint','item_id','revised_label','review_status'].every(k=>k in pos)){alert('This is not an identity-QC review CSV. Original labeling CSV files cannot be imported here.');return;}
 const get=(r,k)=>r[pos[k]]||'';if(rows.some(r=>get(r,'fingerprint')!==FP||get(r,'version')!==VER)){alert('Review file is from another build or source baseline. Not imported.');return;}
 if(!confirm('Import '+rows.length+' saved QC decisions? Matching existing drafts will be replaced.'))return;
 let n=0;for(const r of rows){const id=get(r,'item_id'),it=DATA.find(x=>x.id===id);if(!it)continue;if(it.baseline_label!==get(r,'baseline_label')||it.baseline_m!==get(r,'baseline_m'))continue;
 const a={},p={};for(const [k] of attrNames){a[k]=get(r,k+'_status');p[k+'_q']=get(r,k+'_query_text');p[k+'_r']=get(r,k+'_retrieved_text')}
 ST[id]={attrs:a,printed:p,packaging:get(r,'packaging_observed'),pack_cats:(get(r,'packaging_categories')||'').split(';').filter(Boolean),pack_query:get(r,'pack_query_text'),pack_retrieved:get(r,'pack_retrieved_text'),label:get(r,'revised_label'),reason:get(r,'review_reason').startsWith('Structured evidence —')?'':get(r,'review_reason'),reviewer:get(r,'reviewer'),at:get(r,'reviewed_at'),confirmed:get(r,'review_status')==='confirmed',unresolved:get(r,'review_status')==='unresolved'};n++}
 save();render();alert('Imported '+n+' QC reviews.');e.target.value='';};
$('qi').onclick=()=>{$('zq').src=DATA[i].qimage;$('zm').src=DATA[i].mimage;$('modal').classList.add('show')};$('mi').onclick=$('qi').onclick;$('closezoom').onclick=()=>$('modal').classList.remove('show');document.onkeydown=e=>{if(e.key==='Escape')$('modal').classList.remove('show')};
render();
</script></body></html>'''

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=None, help="web_reference_experiments directory")
    p.add_argument("--scope", choices=["review","all"], default="review")
    p.add_argument("--ids-file", type=Path, default=None, help="Optional CSV with item_id for a targeted review; overrides --scope")
    p.add_argument("--flags", type=Path, default=None, help="Comment3_16_variant_flags_FINAL.csv")
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--apply", type=Path, default=None, help="QC CSV exported by the browser")
    args = p.parse_args()
    root = args.root.resolve() if args.root else root_from_here()
    apply(root,args.apply,args.outdir) if args.apply else build(root,args.scope,args.flags,args.outdir,args.ids_file)
if __name__ == "__main__": main()
