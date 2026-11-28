"""
13_prevalence_adjusted_agreement_and_verified_yield.py
Two additional checks requested in review (run from the repository root):

1. Prevalence-adjusted agreement of page labels. Cohen's kappa is low when one label dominates (160 of 195 reference
   pages are product information), so Gwet's AC1 and PABAK are reported next to kappa for
   - each model run vs the human reference (valid outputs, as in Table 4), and
   - Annotator 1 vs the pharmacist annotator (57 pages, after the W1 re-assessment).
   95% CIs: 2000 bootstrap resamples of products (same scheme as kappa).
     PABAK = (k * p_o - 1) / (k - 1),  k = 4 page labels
     AC1   = (p_o - p_e) / (1 - p_e),  p_e = sum_c pi_c (1 - pi_c) / (k - 1),  pi_c = mean of the two raters' shares
2. HTML-verified reference-candidate yield: reference candidates whose saved page HTML contains the image URL or
   file name (page_has_image starts with "found").
Outputs: results/prevalence_adjusted_agreement.csv, results/verified_reference_candidate_yield.csv
"""
import importlib.util, random, sys
from pathlib import Path
import pandas as pd

D, R = Path("data"), Path("results")
K = 4
LABELS = ["product_information", "insufficient", "warning", "not_evaluable"]


def stats(a, b):
    a, b = list(a), list(b); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa = {c: a.count(c) / n for c in LABELS}; pb = {c: b.count(c) / n for c in LABELS}
    pe_k = sum(pa[c] * pb[c] for c in LABELS)
    pi = {c: (pa[c] + pb[c]) / 2 for c in LABELS}
    pe_g = sum(p * (1 - p) for p in pi.values()) / (K - 1)
    kap = (po - pe_k) / (1 - pe_k) if pe_k < 1 else float("nan")
    return {"agreement": po, "kappa": kap, "ac1": (po - pe_g) / (1 - pe_g), "pabak": (K * po - 1) / (K - 1)}


def boot(a, b, groups, B=2000, seed=1):
    a, b = list(a), list(b); cl = {}
    for j, g in enumerate(groups):
        cl.setdefault(g, []).append(j)
    cl = list(cl.values()); rng = random.Random(seed); out = {"ac1": [], "pabak": []}
    for _ in range(B):
        i = [j for _ in cl for j in cl[rng.randrange(len(cl))]]
        s = stats([a[j] for j in i], [b[j] for j in i])
        out["ac1"].append(s["ac1"]); out["pabak"].append(s["pabak"])
    q = lambda v, f: sorted(v)[int(f * (len(v) - 1))]
    return {f"{m}_lo": q(v, .025) for m, v in out.items()} | {f"{m}_hi": q(v, .975) for m, v in out.items()}


ref = pd.read_csv(R / "from_step09/human_pages.csv", dtype=str).set_index("item_id")
rows = []
for run in ("A__gpt-4o-2024-08-06", "B__gpt-4o-2024-08-06", "A__webref-qwen3-8b", "B__webref-qwen3-8b"):
    m = pd.read_csv(D / f"3_source_context/models/context_{run}.csv", dtype=str).set_index("item_id")
    ids = [i for i in ref.index if i in m.index and m.loc[i, "status"] in ("ok", "empty", "not_fetched")]
    a, b = ref.loc[ids, "label"], m.loc[ids, "label"]
    rows.append({"comparison": run, "n": len(ids), **stats(a, b), **boot(a, b, [i.split("|")[1] for i in ids])})

# Annotator 1 vs pharmacist (labels derived with the same rule and warning coding as step 09)
spec = importlib.util.spec_from_file_location("ev", "scripts/3_source_context/09_evaluate_against_human_reference.py")
ev = importlib.util.module_from_spec(spec); sys.path.insert(0, "scripts/3_source_context"); spec.loader.exec_module(ev)
wc = pd.read_csv(D / "3_source_context/annotations/warning_coding.csv", dtype=str).fillna("")
drop = frozenset((r.item_id, ev.norm(r.quote)) for r in wc.itertuples() if r.category == "clinical_caution")
a1 = ev.human_pages(D / "3_source_context/annotations/annotations_FirstAuthor_all_2026-11-14_final.csv", drop).set_index("item_id")
ph = ev.human_pages(D / "3_source_context/annotations/annotations_PH01_main_2026-11-01_completed_2026-11-26.csv", drop).set_index("item_id")
ids = [i for i in ph.index if i in a1.index]
a, b = a1.loc[ids, "label"], ph.loc[ids, "label"]
rows.append({"comparison": "Annotator 1 vs pharmacist annotator", "n": len(ids), **stats(a, b), **boot(a, b, [i.split("|")[1] for i in ids])})
out = pd.DataFrame(rows).round(3)
out.to_csv(R / "prevalence_adjusted_agreement.csv", index=False)
print(out.to_string(index=False))

# HTML-verified reference-candidate yield
q = pd.read_csv(D / "2_product_identity/retrieval_qc_2026-11-28_246_images.csv", dtype=str)
t = pd.read_csv(D / "1_retrieval/top3_fresh_oct.csv", dtype=str)
q["rank"] = q["rank"].astype(int); t["rank"] = t["rank"].astype(int)
x = q.merge(t[["query_id", "rank", "page_has_image"]], on=["query_id", "rank"])
x["item"] = "fresh_oct|" + x.query_id + "|" + x["rank"].astype(str)
x["context"] = x["item"].map(ref["label"])
x["ref"] = x.revised_primary.eq("same") & x.revised_M.eq("0") & x.context.eq("product_information")
x["verified"] = x.page_has_image.fillna("").str.startswith("found")
y = pd.DataFrame([{"reference_candidates": int(x[x.ref].query_id.nunique()),
                   "reference_candidates_html_verified": int(x[x.ref & x.verified].query_id.nunique()),
                   "candidate_images": int(x.ref.sum()), "candidate_images_html_verified": int((x.ref & x.verified).sum())}])
y.to_csv(R / "verified_reference_candidate_yield.csv", index=False)
print(y.to_string(index=False))
