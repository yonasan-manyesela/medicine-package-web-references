"""
09_evaluate_against_human_reference.py
Evaluation of the source-context step (v3: quote-backed signals) against the human evidence
annotation (labelling page ctx_v8). Human annotations are a reference for agreement and
overlap, not ground truth. Human and model labels come from the SAME rule (context_label_rule.py).

Inputs (runs/<run_id>/):  human files can be given with a full path or as a name in source_context_v3/annotation/
  --human1   CSV of the first annotator (all pages)            context_labels_<code>_all_<date>.csv
  --human2   CSV of the second annotator (second set)          context_labels_<code>_second_<date>.csv
  --resolved CSV saved in compare mode (adjudicated second set); replaces human1 on those pages
  --coding   warning_coding.csv: item_id, annotator, quote, category (quality_regulatory | clinical_caution).
             Warning highlights coded clinical_caution do not meet W1 and are dropped before the rule.
  --manifest second_set_manifest.csv (column stratum) for per-stratum agreement
  source_context_v3/models/context_<A|B>__<model>.csv   every model run of sc2 (no __subset files)
  --baseline tables/llm_results_cot.csv  the original S1-S6 GPT-4o run (labels mapped to the new names)
  --identity labels_final_checked.csv and --items labeling_items.csv (in annotation/): product-identity labels of the
             matched images; joined to the pages by query + rank (source URL checked). Skipped if absent.
Outputs in runs/<run_id>/source_context_v3/eval/:
  human_pages.csv            human signals and derived labels (reference)
  inter_annotator.csv        agreement between the two annotators (second set, before adjudication): page label and
                             signals, after and before warning coding, pooled and per sampling stratum, bootstrap 95% CI
  reference_distribution.csv page and product context labels of the human reference (main and sensitivity)
  label_agreement.csv        page and product label agreement of each run with the reference; second_set_* = the same on the
                             pages both human annotators labelled (like-for-like with the inter-annotator agreement)
  signal_agreement.csv       R, P1-P5, W1-W3 agreement of each pipeline-A run
  grounding.csv              quote grounding per run and signal
  evidence_overlap.csv       position-based, per Pipeline-A run and signal: highlight coverage (located human highlights
                             overlapped by a uniquely located model quote of the same signal) and quote support
                             (uniquely located model quotes overlapping a located human highlight); explicit denominators;
                             plus same value: pages where human and model both give the signal and a model quote and a
                             human highlight contain each other after normalisation (case, spaces, punctuation; >= 3 chars)
  paired_comparison.csv      kappa difference between two runs on their common valid pages (signal-rule vs direct-rule per
                             model; model vs model per pipeline), paired product-cluster bootstrap 95% CI (same resamples for both runs)
  human_highlights_mapped.csv  every human highlight: text, archived position, context, page position,
                             and its location in the model's input text (mapped / by context / ambiguous / not in text)
  highlight_positions.csv    where on the page and in the model text the evidence sits (scroll question)
  completeness.csv           missing / unfinished / duplicate pages per annotator file (stops if any)
  identity_context_images.csv  one row per matched image: rank, inliers, identity, source-page context (human reference)
  identity_context_table.csv   identity x context counts (context labels annotate the image; nothing is filtered)
  identity_summary.csv       query level (Wilson 95% CI): visual match, same medicine, same packaging, same packaging +
                             product-information page; image level: same-packaging share by rank (Wilson) and by page
                             context (query-cluster bootstrap 95% CI); Cramer's V of same packaging x context
  human_evidence_location.csv  are the human highlights in the extracted text?
  review_<run>.csv           two-way disagreement list, checked by hand on the page text:
                             model_only = model quote matched by no human highlight (incl. human "not on page"),
                             human_only = human highlight matched by no model quote.
                             Fill supports_signal, belongs_to_product, verdict (human_miss / model_error / both_ok)
Notes: warning coding applies to the passage (item + quote), whoever highlighted it. Only valid model results
(status ok / empty / not_fetched) are evaluated; completion is reported. Products = queries with matched pages.
Usage:
  python scripts/3_source_context/09_evaluate_against_human_reference.py --human1 <file> --human2 <file> [--resolved <file>]
"""
import argparse, json, math, re, sys
from pathlib import Path, PureWindowsPath

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_label_rule import P_KEYS, norm, page_label, sensitivity_label, product_label, full_text_file

EXP_ROOT = Path(__file__).resolve().parents[2]   # repository root
TYPE2P = {"ingredient": "ingredient", "strength": "strength", "form": "form", "manufacturer": "manufacturer"}
OLD = {"suggests suspicious reference": "warning", "supports normal reference": "product_information",
       "insufficient evidence": "insufficient"}
PAGE_LABELS = ["warning", "product_information", "insufficient", "not_evaluable"]


# ---------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    if not n: return (None, None)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def cluster_share(df, flag, B=2000, seed=1):
    """Share of rows with `flag`, 95% CI by resampling queries (images are clustered in queries)."""
    import numpy as np
    g = df.groupby("query_id")[flag].agg(["sum", "size"]).to_numpy(float)
    i = np.random.default_rng(seed).integers(0, len(g), (B, len(g)))
    r = 100 * g[i, 0].sum(1) / g[i, 1].sum(1)
    return round(float(np.percentile(r, 2.5)), 1), round(float(np.percentile(r, 97.5)), 1)


def cramers_v(t):
    o = t.values.astype(float); e = o.sum(1, keepdims=True) * o.sum(0, keepdims=True) / o.sum()
    return round(math.sqrt(((o - e) ** 2 / e).sum() / (o.sum() * (min(o.shape) - 1))), 3)


def kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0 or len(set(a)) < 2 or len(set(b)) < 2:   # kappa only where both raters use both values
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def agree(a, b):
    a, b = list(a), list(b)
    return {"n": len(a), "agreement_pct": round(100 * sum(x == y for x, y in zip(a, b)) / max(len(a), 1), 1),
            "kappa": round(kappa(a, b), 3)}


def clusters(n, groups):
    """Item indices per cluster; without groups every item is its own cluster (plain item bootstrap)."""
    cl = {}
    for j, g in enumerate(groups if groups is not None else range(n)):
        cl.setdefault(g, []).append(j)
    return list(cl.values())


def qids(ids):
    """Product (query) of each page item_id 'run|Qxxx|rank': the bootstrap cluster."""
    return [str(i).split("|")[1] for i in ids]


def boot(a, b, B=2000, seed=1, groups=None):
    """Bootstrap 95% CI of kappa and raw agreement; whole products are resampled when groups is given."""
    import random
    a, b, n, rng, ks, ps = list(a), list(b), len(a), random.Random(seed), [], []
    cl = clusters(n, groups)
    for _ in range(B if n else 0):
        i = [j for _ in cl for j in cl[rng.randrange(len(cl))]]
        x, y = [a[j] for j in i], [b[j] for j in i]
        k = kappa(x, y)
        if k == k:
            ks.append(k)
        ps.append(100 * sum(u == v for u, v in zip(x, y)) / len(i))
    q = lambda v, f: round(sorted(v)[int(f * (len(v) - 1))], 3) if v else float("nan")
    return {"kappa_lo": q(ks, .025), "kappa_hi": q(ks, .975), "agreement_lo": q(ps, .025), "agreement_hi": q(ps, .975)}


def same_value(hs, qs):
    """Human highlight and model quote name the same value: one contains the other after normalisation."""
    f = lambda t: re.sub(r"[\W_]+", "", norm(t))
    return any(len(a) >= 3 and len(b) >= 3 and (a in b or b in a) for a in map(f, hs) for b in map(f, qs))


def paired(ref, m1, m2, B=2000, seed=1, groups=None):
    import random
    a, b, c, n, rng, d = list(ref), list(m1), list(m2), len(ref), random.Random(seed), []
    cl = clusters(n, groups)
    for _ in range(B if n else 0):
        i = [j for _ in cl for j in cl[rng.randrange(len(cl))]]
        x = kappa([a[j] for j in i], [b[j] for j in i]) - kappa([a[j] for j in i], [c[j] for j in i])
        if x == x: d.append(x)
    q = lambda f: round(sorted(d)[int(f * (len(d) - 1))], 3) if d else float("nan")
    return {"kappa_diff": round(kappa(a, b) - kappa(a, c), 3), "diff_lo": q(.025), "diff_hi": q(.975)}


def text_index(text):
    """Normalized model input text (the coordinate system of model quote offsets) and a
    whitespace-free copy with a map back to it: page text and extracted text differ in spacing."""
    n = norm(text or "")
    idx = [i for i, ch in enumerate(n) if not ch.isspace()]
    return n, "".join(n[i] for i in idx), idx


def squeeze(s):
    return re.sub(r"\s+", "", norm(s))


def map_highlight(m, tix):
    """Locate one human highlight in the model's input text.
    Returns state, start, end (in normalized model text), number of occurrences.
    state: mapped | mapped_by_context | ambiguous | not_in_model_text | translated.
    A highlight made on translated text is searched like any other (names and numbers usually survive
    translation); only when it is not found is it "translated" (cannot be located) instead of "not_in_model_text"."""
    n, sq, idx = tix
    key = squeeze(m.get("t", ""))
    missing = "translated" if m.get("tr") else "not_in_model_text"
    if not key:
        return missing, None, None, 0
    occ, k = [], sq.find(key)
    while k >= 0:
        occ.append(k); k = sq.find(key, k + 1)
    if not occ:
        return missing, None, None, 0
    state, best = "mapped", occ[0]
    if len(occ) > 1:                                 # several occurrences: choose by surrounding text
        pre, post = squeeze(m.get("pre", ""))[-30:], squeeze(m.get("post", ""))[:30]
        def score(o):
            a = sq[max(0, o - len(pre)):o]; b = sq[o + len(key):o + len(key) + len(post)]
            sa = sum(1 for x, y in zip(reversed(a), reversed(pre)) if x == y)
            sb = sum(1 for x, y in zip(b, post) if x == y)
            return sa + sb
        sc = [score(o) for o in occ]
        top = max(sc)
        best = occ[sc.index(top)]
        state = "mapped_by_context" if sc.count(top) == 1 and top > 0 else "ambiguous"
    return state, idx[best], idx[best + len(key) - 1] + 1, len(occ)


def find_all(quote, tix):
    """All locations of a model quote in the model text (same whitespace-free search as for highlights).
    A quote that occurs more than once has no single position: the model does not say which one it meant."""
    n, sq, idx = tix
    parts = [squeeze(x) for x in re.split(r"\.\.\.|\u2026", norm(quote).strip(" \"'")) if squeeze(x)]
    if not parts:
        return []
    key, out, k = parts[0], [], sq.find(parts[0])
    while k >= 0:
        out.append((idx[k], idx[k + len(key) - 1] + 1)); k = sq.find(key, k + 1)
    return out


def locate_model_quote(x, tix):
    """Position of a model quote: inside its context line when the line occurs once in the text,
    otherwise the quote's own occurrences (one = located, several = no single position)."""
    n, sq, idx = tix
    c = x.get("context") or ""
    if c:
        cl = find_all(c, tix)
        if len(cl) == 1:
            a, b = cl[0]
            inner = [s for s in find_all(x["quote"], tix) if s[0] >= a and s[1] <= b]
            if inner:
                return [inner[0]]
    return find_all(x["quote"], tix)


def spans_overlap(a, b):
    return a[0] is not None and b[0] is not None and a[0] < b[1] and b[0] < a[1]


STEP2SIG = {"name": "product_named", "ingredient": "ingredient", "strength": "strength", "form": "form",
            "manufacturer": "manufacturer", "warn": "warning"}


def read_page_text(page_file, pipe_dir):
    p = full_text_file(page_file, Path(pipe_dir).parent)   # full text from sc0, as sent to the models
    if not p.exists():
        p = Path(page_file)
    if not p.exists():
        w = PureWindowsPath(page_file)
        p = pipe_dir / w.parent.name / w.name
    try:
        lines = p.read_text(encoding="utf-8").split("\n")
    except Exception:
        return None
    for k, line in enumerate(lines):
        if line.startswith("=" * 10):
            return "\n".join(lines[k + 1:])
    return "\n".join(lines)


# ---------------------------------------------------------------- human
def completeness(path, ann_dir, report):
    """Every page of the annotator's set must be present and finished (all steps answered)."""
    d = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    who = d["annotator"].iloc[0] if len(d) else "?"
    sets = sorted(set(d["set"]))
    want = set()
    for st in sets:
        f = ann_dir / f"context_items_{st}.csv"
        if f.exists():
            want |= set(pd.read_csv(f, dtype=str)["item_id"])
    have = set(d["item_id"])
    done = set(d.loc[d.get("complete", pd.Series("1", index=d.index)).isin(["1", ""]), "item_id"])
    sec = pd.to_numeric(d["seconds_active"], errors="coerce")
    row = {"file": Path(path).name, "annotator": who, "set": "+".join(sets), "pages_in_set": len(want),
           "rows": len(d), "duplicates": int(d["item_id"].duplicated().sum()),
           "missing": len(want - have) if want else "", "unfinished": len(have - done),
           "builds": "+".join(sorted(set(d["build_id"]))), "median_seconds": float(sec.median()) if len(sec) else "",
           "pages_under_10s": int((sec < 10).sum()),
           "absent_answers_without_scrolling": int((((d.get("W1_warning", "") == "no") | (d.get("P1_named", "") == "no"))
                                                    & (pd.to_numeric(d.get("scroll_page_pct", 100), errors="coerce").fillna(100) < 50)
                                                    & (d.get("shown_as_text", d.get("text_view_used", "0")) != "1")).sum()) if "scroll_page_pct" in d else "",
           "name_only_in_picture": int((d.get("name_only_in_picture", "0") == "1").sum()) if "name_only_in_picture" in d else "",
           "text_view_by_annotator": int((d.get("text_view_by_annotator", "0") == "1").sum()) if "text_view_by_annotator" in d else "",
           "median_scroll_page_pct": float(pd.to_numeric(d.get("scroll_page_pct", pd.Series(dtype=str)), errors="coerce").median())
           if "scroll_page_pct" in d else ""}
    if want and want - have:
        row["missing_ids"] = " ".join(sorted(want - have))
    if have - done:
        row["unfinished_ids"] = " ".join(sorted(have - done))
    report.append(f"{who}: {row['rows']} rows | missing {row['missing']} | unfinished {row['unfinished']} | "
                  f"duplicates {row['duplicates']} | median {row['median_seconds']} s | under 10 s: {row['pages_under_10s']}")
    return row


def human_pages(path, drop=frozenset()):
    """drop: (item_id, normalised quote) of warning highlights that do not meet W1 (see --coding)."""
    d = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    if "complete" in d:
        d = d[d["complete"] == "1"]                       # unfinished pages never enter the reference
    rows = []
    for _, x in d.iterrows():
        marks = json.loads(x["marks_json"] or "[]")
        R = "no" if x["R_readable"] == "no" else "yes"
        hl = {k: [] for k in P_KEYS}
        for m in marks:
            if m["st"] == "name":
                hl["product_named"].append(m["t"])
            elif m["st"] in TYPE2P:                      # ctx_v8.1: one sub-step per detail
                hl[TYPE2P[m["st"]]].append(m["t"])
            elif m["st"] == "detail":                     # ctx_v8.0: ticks on a detail highlight
                for t in m.get("ty", []):
                    hl[TYPE2P[t]].append(m["t"])
        warns = [m for m in marks if m["st"] == "warn" and (x["item_id"], norm(m["t"])) not in drop]
        p = {k: bool(hl[k]) for k in P_KEYS}
        if not p["product_named"]:                      # details only count for a named product
            p.update({k: False for k in P_KEYS[1:]})
        targets = [w.get("tg", "") for w in warns]
        lab, unclear = page_label(R, p, targets)
        r = {"item_id": x["item_id"], "annotator": x["annotator"], "R": R, "label": lab, "target_unclear": unclear,
             "label_sensitivity": sensitivity_label(lab, unclear), "translated": x.get("translated_detected", "0"),
             "W1": "yes" if warns else "no", "W2_this": int("this" in targets), "W2_cant_tell": int("cant_tell" in targets),
             "W3_official": int(any(w.get("of") == "yes" for w in warns)), "warn_quotes": json.dumps([w["t"] for w in warns]),
             "marks_json": x["marks_json"]}
        for k in P_KEYS:
            r[k] = "yes" if p[k] else "no"
            r[f"{k}_quotes"] = json.dumps(hl[k])
        rows.append(r)
    return pd.DataFrame(rows)


def sig_table(h):
    """Comparable signal values (yes/no) for a human or model page table."""
    out = pd.DataFrame({"item_id": h["item_id"], "R": h["R"]})
    for k in P_KEYS:
        out[k] = h[k]
    out["W1"] = h["W1"]
    out["W2_this"] = h["W2_this"].astype(int).map({1: "yes", 0: "no"})
    out["W2_cant_tell"] = h["W2_cant_tell"].astype(int).map({1: "yes", 0: "no"})
    out["W3_official"] = (h["W3_official"].astype(int) > 0).map({True: "yes", False: "no"})
    return out


def model_A_signals(m):
    s = pd.DataFrame({"item_id": m["item_id"], "R": m["R"].fillna("yes")})
    for k in P_KEYS:
        s[k] = (m[f"{k}_state"].fillna("not_asserted") != "not_asserted").map({True: "yes", False: "no"})
    s.loc[s["R"] == "no", P_KEYS] = "no"
    s["W1"] = m["W1"].fillna("no")
    s["W2_this"] = (m["W2_this"].fillna(0).astype(int) > 0).map({True: "yes", False: "no"})
    s["W2_cant_tell"] = (m["W2_cant_tell"].fillna(0).astype(int) > 0).map({True: "yes", False: "no"})
    s["W3_official"] = (m["W3_official"].fillna(0).astype(int) > 0).map({True: "yes", False: "no"})
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="fresh_oct")
    ap.add_argument("--human1", required=True)
    ap.add_argument("--human2", default="")
    ap.add_argument("--resolved", default="")
    ap.add_argument("--baseline", default="llm_results_cot.csv")
    ap.add_argument("--coding", default="warning_coding.csv")
    ap.add_argument("--manifest", default="second_set_manifest.csv")
    ap.add_argument("--identity", default="labels_final_checked.csv")
    ap.add_argument("--items", default="labeling_items.csv")
    ap.add_argument("--identity_annotator", default="Yona", help="code of the first author in the identity file")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--allow_incomplete", action="store_true", help="evaluate finished pages only, even if some are missing")
    args = ap.parse_args()
    run = EXP_ROOT / "runs" / args.run_id
    tables, sc = run / "tables", run / "source_context_v3"
    out, models, ann = sc / "eval", sc / "models", sc / "annotation"
    fp = lambda f: (Path(f) if Path(f).exists() else ann / f) if f else ""
    out.mkdir(parents=True, exist_ok=True)
    pages = pd.read_csv(tables / "pipeline_top3_all.csv", dtype=str).fillna("")
    pages["item_id"] = args.run_id + "|" + pages["query_id"] + "|" + pages["rank"].astype(int).astype(str)
    report = []

    # ---- completeness: stop unless every page is present and finished
    comp = [completeness(fp(f), ann, report) for f in (args.human1, args.human2, args.resolved) if f]
    pd.DataFrame(comp).to_csv(out / "completeness.csv", index=False)
    bad = [c for c in comp if c["missing"] or c["unfinished"] or c["duplicates"] or "+" in c["builds"]]
    if bad and not args.allow_incomplete:
        print("\n".join(report))
        raise SystemExit("[STOP] annotation incomplete (missing / unfinished / duplicate pages, or mixed page builds). "
                         f"See {out / 'completeness.csv'}. Finish the pages (Next unfinished) or rerun with --allow_incomplete.")

    # ---- warning coding (W1 definition applied to every annotator's warning highlights)
    cf, DROP = fp(args.coding), frozenset()
    if cf and Path(cf).exists():
        c = pd.read_csv(cf, dtype=str).fillna("")
        DROP = frozenset((r.item_id, norm(r.quote)) for r in c.itertuples() if r.category == "clinical_caution")
        report.append(f"warning coding: {len(DROP)} highlights coded clinical caution (not W1), from {Path(cf).name}")

    # ---- human reference
    h1 = human_pages(fp(args.human1), DROP)
    ref = h1.set_index("item_id")
    SECOND = set()
    if args.human2:
        mf = fp(args.manifest)
        strata = pd.read_csv(mf, dtype=str).set_index("item_id")["stratum"].to_dict() if mf and Path(mf).exists() else {}
        if not strata:
            report.append("inter-annotator: no second_set_manifest.csv -> pooled only (no strata)")
        ia = []
        for coding, drop in (("after", DROP), ("before", frozenset())):
            g1 = human_pages(fp(args.human1), drop).set_index("item_id")
            h2 = human_pages(fp(args.human2), drop).set_index("item_id")
            common = h2.index.intersection(g1.index)
            if coding == "after":
                SECOND = set(common)
            s1, s2 = sig_table(g1.loc[common].reset_index()), sig_table(h2.loc[common].reset_index())
            s1["label"], s2["label"] = list(g1.loc[common, "label"]), list(h2.loc[common, "label"])
            for st in ["all"] + sorted(set(strata.values())):
                ids = [i for i in range(len(common)) if st == "all" or strata.get(common[i]) == st]
                for c in ["label"] + [c for c in s1.columns if c not in ("item_id", "label")]:
                    a, b = s1[c].iloc[ids], s2[c].iloc[ids]
                    if coding == "before" and not c.startswith(("label", "W")):
                        continue                                   # coding only changes warnings
                    ia.append({"measure": c, "warning_coding": coding, "stratum": st, **agree(a, b), **boot(a, b, args.boot, groups=qids([common[i] for i in ids])),
                               "rater1_yes" if c != "label" else "rater1_values": (a == "yes").sum() if c != "label" else "; ".join(f"{k} {v}" for k, v in a.value_counts().items()),
                               "rater2_yes" if c != "label" else "rater2_values": (b == "yes").sum() if c != "label" else "; ".join(f"{k} {v}" for k, v in b.value_counts().items())})
        pd.DataFrame(ia).to_csv(out / "inter_annotator.csv", index=False)
        top = [r for r in ia if r["measure"] == "label" and r["stratum"] == "all"]
        for r in top:
            report.append(f"inter-annotator page label ({r['warning_coding']} warning coding, n={r['n']}): {r['agreement_pct']}%, "
                          f"kappa {r['kappa']} [{r['kappa_lo']}, {r['kappa_hi']}]")
    if args.resolved:
        res = human_pages(fp(args.resolved), DROP).set_index("item_id")
        ref.loc[res.index.intersection(ref.index)] = res.loc[res.index.intersection(ref.index), ref.columns]
        report.append(f"adjudicated pages replaced in the reference: {len(res.index.intersection(ref.index))}")
    ref = ref.reset_index()
    ref.to_csv(out / "human_pages.csv", index=False)

    # ---- map every human highlight onto the text the model receives (the extracted page text)
    texts = {r.item_id: read_page_text(r.page_file, run / "pipeline") for r in pages.itertuples()}
    tix = {i: text_index(t) for i, t in texts.items() if t is not None}
    hmap, hrows = {}, []                                # (item, signal) -> [(text, (start, end))]
    for x in ref.itertuples():
        if x.item_id not in tix:
            continue
        nlen = max(1, len(tix[x.item_id][0]))
        for m in json.loads(x.marks_json or "[]"):
            if m["st"] == "warn" and (x.item_id, norm(m["t"])) in DROP:
                continue                                  # clinical caution: not W1 evidence
            st, a, b, n_occ = map_highlight(m, tix[x.item_id])
            for sig in ([TYPE2P[t] for t in m.get("ty", [])] if m["st"] == "detail" else [STEP2SIG.get(m["st"], m["st"])]):
                hmap.setdefault((x.item_id, sig), []).append((m["t"], (a, b), st))
            hrows.append({"item_id": x.item_id, "signal": STEP2SIG.get(m["st"], m["st"]), "text": m["t"], "view": m.get("v", ""),
                          "archived_start": m.get("s"), "archived_end": m.get("e"), "context_before": m.get("pre", ""),
                          "context_after": m.get("post", ""), "y_pct_of_page": m.get("y"), "translated": m.get("tr", 0),
                          "map_state": st, "occurrences": n_occ, "model_text_start": a, "model_text_end": b,
                          "position_pct_of_model_text": round(100 * a / nlen, 1) if a is not None else None})
    hm = pd.DataFrame(hrows)
    hm.to_csv(out / "human_highlights_mapped.csv", index=False)
    if len(hm):
        tab = hm.groupby(["signal", "map_state"]).size().unstack(fill_value=0)
        tab.to_csv(out / "human_evidence_location.csv")
        yv = pd.to_numeric(hm["y_pct_of_page"], errors="coerce")
        hm.assign(lower_half=yv > 50).groupby("signal").agg(highlights=("text", "size"), median_y_pct=("y_pct_of_page", "median"),
            in_lower_half=("lower_half", "sum"), median_pos_in_model_text=("position_pct_of_model_text", "median")).to_csv(out / "highlight_positions.csv")
        report.append(f"highlights in the lower half of the page: {int((yv > 50).sum())}/{int(yv.notna().sum())}")
        report.append("human highlights mapped to the model's input text: "
                      + ", ".join(f"{k} {int(v)}" for k, v in hm["map_state"].value_counts().items()))
    report.append("reference page labels: " + ", ".join(f"{k} {v}" for k, v in ref["label"].value_counts().items())
                  + f" | target unclear {int(ref['target_unclear'].sum())}")

    # ---- products = queries with matched pages; excluded (pre-screen) and unsaved pages are not_evaluable on both sides
    pre_f0 = ann / "prescreen.csv"
    p0 = pd.read_csv(pre_f0, dtype=str).fillna("") if pre_f0.exists() else pd.DataFrame(columns=["item_id", "decision"])
    EXCL, SAVED = set(p0.loc[p0.decision == "exclude", "item_id"]), set(p0.item_id)
    QPAGES = {q: list(g.item_id) for q, g in pages.groupby("query_id")}
    def products(lab):
        """Product label per query; None when a saved, annotated page has no valid result (product left out)."""
        out_ = {}
        for q, ids in QPAGES.items():
            if all(i in EXCL or (SAVED and i not in SAVED) for i in ids):
                continue                                   # no annotated page: not evaluable for every rater by rule, left out
            labs = ["not_evaluable" if (i in EXCL or (SAVED and i not in SAVED)) else lab.get(i) for i in ids]
            out_[q] = None if None in labs else product_label(labs)
        return pd.Series(out_)
    ref_prod, ref_prod_s = products(dict(zip(ref.item_id, ref.label))), products(dict(zip(ref.item_id, ref.label_sensitivity)))
    dist = [{"level": "page", "version": v, **ref[c].value_counts().to_dict(), "n": len(ref)} for v, c in (("main", "label"), ("sensitivity", "label_sensitivity"))]
    dist += [{"level": "product", "version": v, **sr.dropna().value_counts().to_dict(), "n": int(sr.notna().sum())} for v, sr in (("main", ref_prod), ("sensitivity", ref_prod_s))]
    pd.DataFrame(dist).fillna(0).to_csv(out / "reference_distribution.csv", index=False)
    report.append("reference product labels (queries with pages): " + ", ".join(f"{k} {v}" for k, v in ref_prod.dropna().value_counts().items()))

    # ---- product identity x source context (the page label annotates the image; nothing is filtered)
    idf, itf = fp(args.identity), fp(args.items)
    if idf and itf and Path(idf).exists() and Path(itf).exists():
        L = pd.read_csv(idf, dtype=str).fillna(""); L = L[L.annotator == args.identity_annotator]
        it = pd.read_csv(itf, dtype=str).fillna(""); it = it[(it.kind == "selected") & (it.run == args.run_id)]
        im = it.merge(L.drop(columns=["query_id"]), on="item_id", how="left", validate="1:1")
        im["page_item"] = args.run_id + "|" + im.query_id + "|" + im["rank"].astype(int).astype(str)
        url = dict(zip(pages.item_id, pages.source_page_url.str.strip()))
        bad_url = (im.page_item.map(url) != im.source_page_url.str.strip()).sum()
        if bad_url or im.label.eq("").any():
            raise SystemExit(f"[STOP] identity join: {bad_url} images whose source URL differs from the page; {int(im.label.eq('').sum())} unlabelled")
        im["identity"] = im.apply(lambda r: ("same, no packaging difference" if r.packaging_differs == "0" else "same, packaging difference")
                                  if r.label == "same" else {"brand_only": "name only", "unclear": "cannot determine"}.get(r.label, r.label), axis=1)
        lab = dict(zip(ref.item_id, ref.label))
        im["context"] = [lab.get(i) or ("no readable text" if i in EXCL else "not saved") for i in im.page_item]
        im["good"], im["pi"] = im.identity.eq("same, no packaging difference"), im.context.eq("product_information")
        im[["page_item", "query_id", "rank", "inliers", "identity", "context", "image_url", "source_page_url"]].to_csv(out / "identity_context_images.csv", index=False)
        pd.crosstab(im.identity, im.context, margins=True).to_csv(out / "identity_context_table.csv")
        nq = len(pd.read_csv(tables / "match_summary.csv", dtype=str))
        qa = im.groupby("query_id").agg(same=("label", lambda s: s.eq("same").any()), good=("good", "any"),
                                         both=("good", lambda s: (s & im.loc[s.index, "pi"]).any()))
        rows = [{"level": "query", "measure": m, "k": int(k), "n": nq, "pct": round(100 * k / nq, 1), **dict(zip(("lo", "hi"), wilson(k, nq)))}
                for m, k in (("visual match", len(qa)), ("same medicine", qa.same.sum()), ("same, no packaging difference", qa.good.sum()),
                             ("same, no packaging difference + product-information page", qa.both.sum()))]
        for rk_, g in im.groupby(im["rank"].astype(int)):
            rows.append({"level": "image", "measure": f"same, no packaging difference | rank {rk_}", "k": int(g.good.sum()), "n": len(g),
                         "pct": round(100 * g.good.mean(), 1), **dict(zip(("lo", "hi"), wilson(int(g.good.sum()), len(g))))})
        for c, g in im.groupby("context"):
            rows.append({"level": "image", "measure": f"same, no packaging difference | page {c}", "k": int(g.good.sum()), "n": len(g),
                         "pct": round(100 * g.good.mean(), 1), **dict(zip(("lo", "hi"), cluster_share(g, "good", args.boot)))})
        rd = im[im.context.isin(PAGE_LABELS)]
        rows.append({"level": "image", "measure": "Cramer's V: same packaging x context (all contexts)", "pct": cramers_v(pd.crosstab(im.good, im.context))})
        rows.append({"level": "image", "measure": "Cramer's V: same packaging x context (readable pages)", "pct": cramers_v(pd.crosstab(rd.good, rd.context))})
        pd.DataFrame(rows).astype({"k": "Int64", "n": "Int64"}).to_csv(out / "identity_summary.csv", index=False)
        report.append("identity (queries of " + str(nq) + "): " + " -> ".join(f"{r['measure']} {r['k']}" for r in rows[:4]))
        report.append("same packaging by page context: " + "; ".join(f"{r['measure'].split('| page ')[1]} {r['k']}/{r['n']}" for r in rows if "| page " in r["measure"]))
    else:
        report.append(f"identity x context: skipped ({args.identity} / {args.items} not found in {ann})")

    # ---- runs (only valid model results)
    runs = {f.stem.removeprefix("context_"): pd.read_csv(f, dtype={"item_id": str}) for f in sorted(models.glob("context_[AB]__*.csv")) if not f.name.endswith("__subset.csv")}
    bl = tables / args.baseline
    if bl.exists():
        b = pd.read_csv(bl, dtype=str).fillna("")
        b["item_id"] = args.run_id + "|" + b["query_id"] + "|" + b["rank"].astype(float).astype(int).astype(str)
        unk = b.loc[~b["label"].isin(OLD) & (b["label"] != ""), "label"]
        if len(unk):
            report.append(f"[CHECK] baseline: {len(unk)} unmapped labels ({', '.join(sorted(set(unk)))[:120]}) -> left out")
        b = b[b["label"].isin(OLD) | (b["label"] == "")]
        b["label"] = b["label"].map(OLD).fillna("not_evaluable"); b["label_sensitivity"] = b["label"]; b["status"] = "ok"
        runs["baseline_S1-S6__" + bl.stem.removeprefix("llm_results_")] = b
    VALID = {"ok", "empty", "not_fetched"}
    rk = ref.set_index("item_id")
    lab_rows, sig_rows, grd_rows, ov_rows, VAL = [], [], [], [], {}
    for name, m in runs.items():
        mm = m[m.get("status", pd.Series("ok", index=m.index)).isin(VALID)].set_index("item_id")
        ids = [i for i in ref.item_id if i in mm.index]
        r, ml = rk.loc[ids], mm.loc[ids, "label"]
        VAL[name] = ml
        row = {"run": name, "pages_valid": len(ids), "pages_reference": len(ref), **agree(r["label"], ml), **boot(r["label"], ml, args.boot, groups=qids(ids))}
        for lab in PAGE_LABELS:
            row.update({f"{lab}_ref": int((r["label"] == lab).sum()), f"{lab}_model": int((ml == lab).sum()), f"{lab}_both": int(((r["label"] == lab) & (ml == lab)).sum())})
        i2 = [i for i in ids if i in SECOND]
        if i2:
            a2, b2 = agree(rk.loc[i2, "label"], mm.loc[i2, "label"]), boot(rk.loc[i2, "label"], mm.loc[i2, "label"], args.boot, groups=qids(i2))
            row.update({"second_set_n": a2["n"], "second_set_agreement_pct": a2["agreement_pct"], "second_set_kappa": a2["kappa"],
                        "second_set_kappa_lo": b2["kappa_lo"], "second_set_kappa_hi": b2["kappa_hi"]})
        row["kappa_sensitivity"] = agree(r["label_sensitivity"], mm.loc[ids, "label_sensitivity"].fillna(ml))["kappa"]
        if "label_grounded" in mm:
            row["kappa_grounded_only"] = agree(r["label"], mm.loc[ids, "label_grounded"].fillna(ml))["kappa"]
        mp = products(dict(zip(mm.index, mm["label"])))
        both = ref_prod.notna() & mp.notna()
        pb = boot(ref_prod[both], mp[both], args.boot)
        pa = agree(ref_prod[both], mp[both])
        row.update({"product_n": pa["n"], "product_agreement_pct": pa["agreement_pct"], "product_kappa": pa["kappa"],
                    "product_kappa_lo": pb["kappa_lo"], "product_kappa_hi": pb["kappa_hi"],
                    "product_kappa_sensitivity": agree(ref_prod_s[both], products(dict(zip(mm.index, mm["label_sensitivity"].fillna(mm["label"]))))[both])["kappa"]})
        pd.crosstab(r["label"].rename("human"), ml.rename("model")).to_csv(out / f"confusion_{name}.csv")
        lab_rows.append(row)

        if name.startswith("A__"):              # signal agreement, grounding, position-based evidence
            hs, ms = sig_table(r.reset_index()), model_A_signals(mm.loc[ids].reset_index())
            for c in hs.columns[1:]:
                sig_rows.append({"run": name, "signal": c, **agree(hs[c], ms[c]), "human_yes": int((hs[c] == "yes").sum()),
                                 "model_yes": int((ms[c] == "yes").sum()), "both_yes": int(((hs[c] == "yes") & (ms[c] == "yes")).sum())})
            def mquotes(i, k):
                v = mm.loc[i].get("warnings_json" if k == "warning" else f"{k}_quotes_json")
                return [x for x in json.loads(v) if x.get("quote")] if isinstance(v, str) and v.strip() else []
            manual = []
            for k in P_KEYS + ["warning"]:
                st = [x["state"] for i in mm.index for x in mquotes(i, k)]
                grd_rows.append({"run": name, "signal": k, "quotes": len(st), "grounded": st.count("grounded"), "ungrounded": st.count("ungrounded"),
                                 "ungrounded_pct": round(100 * st.count("ungrounded") / max(len(st), 1), 1)})
                hl_loc = hl_cov = hl_unloc = q_loc = q_sup = q_multi = both_q = same_v = 0
                for i in ids:
                    H = hmap.get((i, k), [])
                    if H and mquotes(i, k):            # same value: needs no position, so also pages without model text
                        both_q += 1; same_v += same_value([t for t, _, _ in H], [x["quote"] for x in mquotes(i, k)])
                    if i not in tix:
                        continue
                    hsp = [(t, sp) for t, sp, s_ in H if s_ in ("mapped", "mapped_by_context")]   # uniquely located highlights
                    hl_unloc += len(H) - len(hsp)
                    Q = [(x["quote"], L_[0]) for x in mquotes(i, k) for L_ in [locate_model_quote(x, tix[i])] if len(L_) == 1]
                    q_multi += sum(len(locate_model_quote(x, tix[i])) > 1 for x in mquotes(i, k))
                    hl_loc += len(hsp); q_loc += len(Q)
                    for t, sp in hsp:
                        hit = any(spans_overlap(sp, qs) for _, qs in Q); hl_cov += hit
                        if not hit:
                            pack = k == "strength" and re.search(r"\d\s*(ml|g|gm|tablets?|caps?)\b", t, re.I) and not re.search(r"%|/|mg\b|mcg|iu", t, re.I)
                            manual.append({"direction": "human_only", "item_id": i, "signal": k, "human_highlight": t, "model_quote": "",
                                           "note": "looks like a pack size" if pack else "", "verdict": ""})
                    for q, qs in Q:
                        hit = any(spans_overlap(qs, sp) for _, sp in hsp); q_sup += hit
                        if not hit:
                            manual.append({"direction": "model_only", "item_id": i, "signal": k, "human_highlight": "", "model_quote": q,
                                           "note": "page can't be read" if r.loc[i, "R"] == "no" else ("human: other text" if H else "human: not on page / none"), "verdict": ""})
                ov_rows.append({"run": name, "signal": k, "human_highlights_located": hl_loc, "highlights_covered": hl_cov,
                                "highlight_coverage_pct": round(100 * hl_cov / max(hl_loc, 1), 1), "human_highlights_not_located": hl_unloc,
                                "model_quotes_located": q_loc, "quotes_on_human_evidence": q_sup, "quote_support_pct": round(100 * q_sup / max(q_loc, 1), 1),
                                "model_quotes_more_than_once": q_multi, "pages_both_quoted": both_q, "pages_same_value": same_v,
                                "same_value_pct": round(100 * same_v / max(both_q, 1), 1)})
            pd.DataFrame(manual).to_csv(out / f"review_{name}.csv", index=False)
        elif name.startswith("B__") and "evidence_json" in mm:
            st = [e["state"] for es in mm["evidence_json"].dropna() for e in json.loads(es)]
            n_as = sum(s != "not_asserted" for s in st)
            grd_rows.append({"run": name, "signal": "evidence (all roles)", "quotes": n_as, "grounded": st.count("grounded"),
                             "ungrounded": st.count("ungrounded"), "ungrounded_pct": round(100 * st.count("ungrounded") / max(n_as, 1), 1)})

    pd.DataFrame(lab_rows).to_csv(out / "label_agreement.csv", index=False)
    mods = sorted({n.split("__", 1)[1] for n in VAL if n[:3] in ("A__", "B__")})
    pairs = [(f"A__{m}", f"B__{m}") for m in mods] + [(f"{p}__{x}", f"{p}__{y}") for p in "AB" for i_, x in enumerate(mods) for y in mods[i_ + 1:]]
    pc = []
    for r1, r2 in pairs:
        if r1 in VAL and r2 in VAL:
            com = [i for i in VAL[r1].index if i in VAL[r2].index]
            pc.append({"run_1": r1, "run_2": r2, "pages_common": len(com), **paired(rk.loc[com, "label"], VAL[r1][com], VAL[r2][com], args.boot, groups=qids(com))})
    if pc:
        pd.DataFrame(pc).to_csv(out / "paired_comparison.csv", index=False)
        report.append("paired kappa differences (common pages, 95% CI): " + "; ".join(f"{x['run_1']} - {x['run_2']} = {x['kappa_diff']} [{x['diff_lo']}, {x['diff_hi']}] (n={x['pages_common']})" for x in pc))
    pd.DataFrame(sig_rows).to_csv(out / "signal_agreement.csv", index=False)
    pd.DataFrame(grd_rows).to_csv(out / "grounding.csv", index=False)
    pd.DataFrame(ov_rows).to_csv(out / "evidence_overlap.csv", index=False)

    pre_f = ann / "prescreen.csv"                   # pages excluded before annotation: what did the models say?
    if pre_f.exists():
        pre = pd.read_csv(pre_f, dtype=str).fillna("")
        ex = set(pre.loc[pre.decision == "exclude", "item_id"])
        rows = [{"run": n, **m[m.item_id.isin(ex)]["label"].value_counts().to_dict()} for n, m in runs.items()]
        pd.DataFrame(rows).to_csv(out / "excluded_pages_model_labels.csv", index=False)
        report.append(f"prescreen: {len(ex)} pages excluded (no usable content); model labels on them in excluded_pages_model_labels.csv")
    print("\n".join(report))
    if lab_rows:
        print(pd.DataFrame(lab_rows)[["run", "pages_valid", "agreement_pct", "kappa", "kappa_lo", "kappa_hi", "product_n", "product_kappa"]].to_string(index=False))
        if SECOND:
            print("\nsame pages as the two human annotators (like-for-like):")
            print(pd.DataFrame(lab_rows)[["run", "second_set_n", "second_set_agreement_pct", "second_set_kappa", "second_set_kappa_lo", "second_set_kappa_hi"]].to_string(index=False))
    if grd_rows:
        g = pd.DataFrame(grd_rows).groupby("run")[["quotes", "ungrounded"]].sum()
        print("\nungrounded quotes (%):\n" + (100 * g.ungrounded / g.quotes.clip(lower=1)).round(1).to_string())
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
