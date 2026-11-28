"""
11_reproduce_reported_numbers.py
Recomputes the numbers reported in the manuscript from the files in this repository and compares each with the
value printed in the paper. Run from the repository root:

    python scripts/4_revision_checks/11_reproduce_reported_numbers.py

Inputs: data/ (annotations, labels, model outputs, retrieval records) and results/from_step09/ (outputs of
scripts/3_source_context/09_evaluate_against_human_reference.py, which needs the saved page texts that are not
redistributed). Output: results/reported_numbers.csv with section, quantity, recomputed value, value in the
paper, status (match / differs / record only) and the source file. Every row of RESULTS_EVIDENCE.md points here.
"""
import itertools, math, random
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import numpy as np
import pandas as pd

D, R = Path("data"), Path("results")
ROWS = []


def add(section, quantity, value, paper, source, note=""):
    v, p = str(value), str(paper)
    status = "record only" if paper is None else ("match" if v == p else "differs")
    ROWS.append({"section": section, "quantity": quantity, "recomputed": v, "paper": "" if paper is None else p,
                 "status": status, "source": source, "note": note})


def pct(k, n, d=1):
    return f"{100 * k / n:.{d}f}"


def wilson(k, n, z=1.96):
    p, den = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / den, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return f"{100 * (c - h):.1f}--{100 * (c + h):.1f}"


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def kappa(a, b):
    a, b = list(a), list(b); n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe)


def boot_kappa(a, b, groups, B=2000, seed=1):
    a, b = list(a), list(b); rng = random.Random(seed)
    cl = {}
    for j, g in enumerate(groups):
        cl.setdefault(g, []).append(j)
    cl = list(cl.values()); ks = []
    for _ in range(B):
        i = [j for _ in cl for j in cl[rng.randrange(len(cl))]]
        x, y = [a[j] for j in i], [b[j] for j in i]
        if len(set(x)) > 1 and len(set(y)) > 1:
            ks.append(kappa(x, y))
    ks.sort()
    q = lambda f: ks[int(f * (len(ks) - 1))]
    return f"{q(.025):.2f} to {q(.975):.2f}"


def read_labels(f):
    return pd.read_csv(f, dtype=str, comment="#", encoding="utf-8-sig").dropna(axis=1, how="all").fillna("")


# ------------------------------------------------------------------ inputs
qc = pd.read_csv(D / "2_product_identity/retrieval_qc_2026-11-28_246_images.csv", dtype=str).fillna("")
qc["rank"] = qc["rank"].astype(int)
items = pd.read_csv(D / "2_product_identity/labeling_items.csv", dtype=str).fillna("")
lab = pd.read_csv(D / "2_product_identity/labels_final_checked_qc_2026-11-28.csv", dtype=str).fillna("")
queries = pd.read_csv("config/queries.csv", dtype=str).fillna("")
ms = pd.read_csv(D / "1_retrieval/match_summary_fresh_oct.csv")
ctx = pd.read_csv(R / "from_step09/identity_context_images.csv", dtype=str)
ctx["rank"] = ctx["rank"].astype(int)
img = qc.merge(ctx[["query_id", "rank", "context"]], on=["query_id", "rank"], how="left", validate="1:1")
img["same"] = img.revised_primary.eq("same"); img["m0"] = img.same & img.revised_M.eq("0")
img["ref"] = img.m0 & img.context.eq("product_information")
SRC = "data/2_product_identity/retrieval_qc_2026-11-28_246_images.csv"

# ------------------------------------------------------------------ 4.1 product identity
g = img.groupby("query_id")
add("4.1", "products with >=1 matched image", int(ms.has_match.sum()), 88, "data/1_retrieval/match_summary_fresh_oct.csv")
add("4.1", "products with an image of the same printed name", int(g.revised_primary.apply(lambda s: s.ne("different").any()).sum()), 83, SRC,
    "same, variant or cannot determine (name agreed)")
s3, s3m0, yld = int(g.same.any().sum()), int(g.m0.any().sum()), int(g.ref.any().sum())
add("4.1", "Success@3 (products)", s3, 64, SRC)
add("4.1", "Success@3 Wilson 95% CI", wilson(s3, 100), "54.2--72.7", SRC)
add("4.1", "products: same medicine, no packaging difference", s3m0, 48, SRC)
add("4.2", "reference-candidate yield (products)", yld, 35, SRC + " + results/from_step09/identity_context_images.csv")
add("4.2", "reference-candidate yield Wilson 95% CI", wilson(yld, 100), "26.4--44.7", SRC)
vc = img.revised_primary.value_counts()
for k, name, n_paper, p_paper in (("same", "same medicine", 118, "48.0"), ("variant", "variant", 53, "21.5"),
                                  ("different", "different product", 38, "15.4"), ("cannot_determine", "cannot determine", 37, "15.0")):
    add("4.1", f"images: {name}", int(vc.get(k, 0)), n_paper, SRC)
    add("4.1", f"images: {name} %", pct(vc.get(k, 0), 246), p_paper, SRC)
add("4.1", "images: same, no packaging difference", int(img.m0.sum()), 70, SRC)
add("4.1", "images: same, packaging difference", int((img.same & img.revised_M.eq("1")).sum()), 48, SRC)
for r_, paper in ((1, "54/88"), (2, "41/83"), (3, "23/75")):
    x = img[img["rank"] == r_]
    add("4.1", f"same medicine at rank {r_}", f"{int(x.same.sum())}/{len(x)}", paper, SRC)

# below threshold (name only)
it = items.merge(lab[["item_id", "label"]], on="item_id", how="left")
bt = it[(it.run == "fresh_oct") & (it.kind != "selected")]
add("3.3", "name-only images below threshold", len(bt), 102, "data/2_product_identity/labeling_items.csv")
for k, paper in (("different", 73), ("variant", 24), ("same", 2)):
    add("4.1", f"below threshold: {k}", int((bt.label == k).sum()), paper, "data/2_product_identity/labels_final_checked_qc_2026-11-28.csv")
sel = it[it.kind == "selected"]
a, b = set(sel[sel.run == "fresh_oct"].item_id), set(sel[sel.run == "fresh_oct_form"].item_id)
add("3.3", "distinct top-three images, both configurations", len(a | b), 383, "data/2_product_identity/labeling_items.csv")
add("3.3", "top-three images shared by both configurations", len(a & b), 112, "data/2_product_identity/labeling_items.csv")
add("3.3", "distinct below-threshold images", it[it.kind != "selected"].item_id.nunique(), 187, "data/2_product_identity/labeling_items.csv")
add("3.3", "annotated image pairs", lab.item_id.nunique(), 570, "data/2_product_identity/labels_final_checked_qc_2026-11-28.csv")

# strict attribute evidence
att = pd.read_csv(D / "2_product_identity/attribute_check_A1_2026-11-18.csv", dtype=str).fillna("")
sx = img[img.same].merge(att[["item_id", "outcome"]], on="item_id", how="left")
strict = sx.outcome.eq("strict") | sx.qc_evidence_state.eq("supported_formulation_agreement")
add("4.1", "same-medicine images with complete attribute agreement", f"{int(strict.sum())}/{len(sx)}", "111/118", "data/2_product_identity/attribute_check_A1_2026-11-18.csv")
add("4.1", "Success@3 with complete attribute agreement", sx[strict].query_id.nunique(), 60, "data/2_product_identity/attribute_check_A1_2026-11-18.csv")

# by source
queries["src"] = queries.dataset.str.extract(r"^(field|kaggle|mendeley|public|alert|nafdac)", expand=False).replace({"mendeley": "kaggle", "public": "kaggle", "nafdac": "alert"})
img2 = img.merge(queries[["query_id", "src"]], on="query_id")
for ds, n, paper in (("field", 44, "24/44"), ("kaggle", 49, "36/49"), ("alert", 7, "4/7")):
    add("4.1", f"Success@3, {ds} queries", f"{img2[img2.src == ds].groupby('query_id').same.any().sum()}/{n}", paper, SRC + " + config/queries.csv")

# duplicate check
dc = pd.read_csv(D / "1_retrieval/duplicate_check_duplicate_confirmed.csv")
top = pd.read_csv(D / "1_retrieval/top3_fresh_oct.csv", dtype=str)
top["rank"] = top["rank"].astype(int)
imgd = img.merge(top[["query_id", "rank", "image"]], on=["query_id", "rank"])
for verdicts, s_paper, y_paper in ((["same_photograph"], 63, 35), (["same_photograph", "possible_same_photograph"], 63, 34)):
    k = set(map(tuple, dc[dc.verdict.isin(verdicts)][["query_id", "image"]].values))
    x = imgd[[(q, i) not in k for q, i in zip(imgd.query_id, imgd.image)]].groupby("query_id")
    add("4.1", f"Success@3 excluding {'+'.join(verdicts)}", int(x.same.any().sum()), s_paper, "data/1_retrieval/duplicate_check_duplicate_confirmed.csv")
    add("4.1", f"yield excluding {'+'.join(verdicts)}", int(x.ref.any().sum()), y_paper, "data/1_retrieval/duplicate_check_duplicate_confirmed.csv")
add("3.2", "downloaded candidates compared with query", len(pd.read_csv(D / "1_retrieval/duplicate_check_pairs.csv")), 950, "data/1_retrieval/duplicate_check_pairs.csv")

# match count: ROC and cutoffs
ann = it[it.run == "fresh_oct"].copy()
ann["m"] = ann.inliers.astype(float); ann["same"] = ann.label.eq("same")
pos, neg = ann.loc[ann.same, "m"].to_numpy(), ann.loc[~ann.same, "m"].to_numpy()
auc = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
add("4.1", "AUC of match count (same medicine)", f"{auc:.2f}", "0.90", "data/2_product_identity/labeling_items.csv + labels_final_checked_qc_2026-11-28.csv")
add("4.1", "share of same-medicine images kept at 50 matches", f"{100 * (pos >= 50).mean():.0f}", "98", "same")
img["m"] = img.merge(top[["query_id", "rank", "match_count"]], on=["query_id", "rank"]).match_count.astype(float).values
for c, p_paper, s_paper, y_paper in ((50, "48.0", 64, 35), (200, "62.1", 62, 35)):
    x = img[img.m >= c]; gg = x.groupby("query_id")
    add("4.1", f"cutoff {c}: same-medicine share of retained images", pct(x.same.sum(), len(x)), p_paper, SRC)
    add("4.1", f"cutoff {c}: Success@3", int(gg.same.any().sum()), s_paper, SRC)
    add("4.1", f"cutoff {c}: reference-candidate yield", int(gg.ref.any().sum()), y_paper, SRC)
add("4.1", "share of same-medicine images kept at 200 matches", f"{100 * (pos >= 200).mean():.0f}", "92", "same")
for c, paper in ((300, 55), (400, 51)):
    add("4.1", f"cutoff {c}: Success@3", int(img[img.m >= c].groupby("query_id").same.any().sum()), paper, SRC)
for c, n_, p_, s_, y_ in ((100, 220, "53.2", 63, 35), (300, 151, "68.2", 55, 32), (400, 114, "76.3", 51, 28), (50, 246, "48.0", 64, 35), (200, 177, "62.1", 62, 35)):
    x = img[img.m >= c]; gg = x.groupby("query_id")
    add("Table threshold", f"{c} matches: images kept / same-medicine share / Success@3 / yield",
        f"{len(x)} / {pct(x.same.sum(), len(x))} / {int(gg.same.any().sum())} / {int(gg.ref.any().sum())}", f"{n_} / {p_} / {s_} / {y_}", SRC)
pos_ = imgd.image.str.extract(r"(\d+)")[0].astype(int) + 1
add("4.1", "retained images from download positions 4-10", int((pos_ >= 4).sum()), 130, "data/1_retrieval/top3_fresh_oct.csv")
first = imgd.assign(p=pos_)[imgd.same].groupby("query_id").p.min()
add("4.1", "products whose same-medicine images all came from positions 4-10", int((first >= 4).sum()), 11, "data/1_retrieval/top3_fresh_oct.csv")

# dosage-form configuration and Tanzania region
fo = it[(it.run == "fresh_oct_form") & (it.kind == "selected")]
add("4.1", "dosage-form queries: top-three images", len(fo), 249, "data/2_product_identity/labeling_items.csv")
add("4.1", "dosage-form queries: variants", f"{int((fo.label == 'variant').sum())} ({pct((fo.label == 'variant').sum(), len(fo))}%)", "39 (15.7%)", "labels_final_checked_qc_2026-11-28.csv")
add("4.1", "dosage-form queries: different products", f"{int((fo.label == 'different').sum())} ({pct((fo.label == 'different').sum(), len(fo))}%)", "50 (20.1%)", "labels_final_checked_qc_2026-11-28.csv")
sf = set(fo[fo.label == "same"].query_id); sn = set(img[img.same].query_id)
add("4.1", "dosage-form queries: Success@3", len(sf), 68, "labels_final_checked_qc_2026-11-28.csv")
fm0 = fo[(fo.label == "same") & (fo.packaging_differs_y if "packaging_differs_y" in fo else True)]
add("4.1", "McNemar discordant (name only / form only)", f"{len(sn - sf)}/{len(sf - sn)}", "6/10", "labels_final_checked_qc_2026-11-28.csv")
add("4.1", "McNemar exact p (name vs form)", f"{mcnemar_exact(len(sn - sf), len(sf - sn)):.2f}", "0.45", "computed")
tz = pd.read_csv(D / "1_retrieval/match_summary_fresh_oct_tz.csv")
mm = ms[["query_id", "has_match"]].merge(tz[["query_id", "has_match"]], on="query_id", suffixes=("_us", "_tz"))
gain, lost = int((~mm.has_match_us & mm.has_match_tz).sum()), int((mm.has_match_us & ~mm.has_match_tz).sum())
add("5.5", "Tanzania region: products with a visual match", int(tz.has_match.sum()), 87, "data/1_retrieval/match_summary_fresh_oct_tz.csv")
add("5.5", "Tanzania vs US exact McNemar p", f"{mcnemar_exact(gain, lost):.2f}", "1.00", "match_summary_fresh_oct*.csv")
t_us, t_tz = pd.read_csv(D / "1_retrieval/top3_fresh_oct.csv", dtype=str), pd.read_csv(D / "1_retrieval/top3_fresh_oct_tz.csv", dtype=str)
ku = set(zip(t_us.query_id, t_us.source_page_url.str.split("?").str[0])); kt = set(zip(t_tz.query_id, t_tz.source_page_url.str.split("?").str[0]))
add("5.5", "selected pages shared US/Tanzania", f"{len(ku & kt)} of {len(kt)}", "93 of 245", "data/1_retrieval/top3_fresh_oct*.csv")

# identity agreement (before reconciliation)
A = {k: read_labels(D / f"2_product_identity/{f}") for k, f in (("A1", "labels_primary_Yona_2026-10-22.csv"), ("A2", "labels_second_EI_2026-10-22.csv"), ("A3", "labels_second_RM_2026-11-03.csv"))}
s = A["A2"][["item_id", "label"]].rename(columns={"label": "A2"}).merge(A["A3"][["item_id", "label"]].rename(columns={"label": "A3"}), on="item_id") \
    .merge(A["A1"][["item_id", "label"]].rename(columns={"label": "A1"}), on="item_id")
tab = pd.get_dummies(s[["A1", "A2", "A3"]].stack()).groupby(level=0).sum().to_numpy(float)
P = ((tab ** 2).sum(axis=1) - 3) / 6; pj = tab.sum(axis=0) / tab.sum()
fl = (P.mean() - (pj ** 2).sum()) / (1 - (pj ** 2).sum())
add("4.1", "identity subset: all three agree", int((s.A1.eq(s.A2) & s.A2.eq(s.A3)).sum()), 60, "data/2_product_identity/labels_*_2026-10-22/11-03.csv")
add("4.1", "Fleiss kappa", f"{fl:.2f}", "0.58", "same")
pw = sorted(kappa(s[x], s[y]) for x, y in itertools.combinations(["A1", "A2", "A3"], 2))
add("4.1", "pairwise Cohen kappa range", f"{pw[0]:.2f} to {pw[-1]:.2f}", "0.51 to 0.65", "same", f"paper corrected from 0.52; exact values {[round(v, 4) for v in pw]}")

# reconciliation
man = pd.read_csv(D / "2_product_identity/identity_review_manifest_41.csv")
add("3.3", "targeted review: images", len(man), 41, "data/2_product_identity/identity_review_manifest_41.csv")
add("3.3", "targeted review: earlier disagreements", int(man.focus.str.contains("prior", case=False).sum()), 28, "same")
add("3.3", "targeted review: flagged in attribute check", int(man.focus.str.contains("Comment 3").sum()), 16, "same")
add("3.3", "targeted review: images among the 246", int(man.item_id.isin(qc.item_id).sum()), 27, "same")
q21 = pd.read_csv(D / "2_product_identity/retrieval_qc_2026-11-21_246_images.csv", dtype=str).fillna("")
tr = (qc.original_primary != qc.revised_primary)
add("3.3", "identity labels changed among the 246 (review + consistency check)", int(tr.sum()), 17, SRC)
add("3.3", "packaging-difference flags changed", int(q21.M_changed.eq("1").sum()), 18, "data/2_product_identity/retrieval_qc_2026-11-21_246_images.csv")

# ------------------------------------------------------------------ 3.5 / 4.2 source context
pre = pd.read_csv(D / "3_source_context/annotations/prescreen.csv", dtype=str).fillna("")
add("3.5", "saved pages", len(pre), 228, "data/3_source_context/annotations/prescreen.csv")
add("3.5", "pages excluded before annotation", int((pre.decision == "exclude").sum()), 33, "same")
hp = pd.read_csv(R / "from_step09/human_pages.csv", dtype=str)
add("3.5", "annotated pages", len(hp), 195, "results/from_step09/human_pages.csv")
add("3.5", "products with annotated pages", hp.item_id.str.split("|").str[1].nunique(), 86, "same")
for k, paper in (("product_information", 160), ("insufficient", 31), ("warning", 2), ("not_evaluable", 2)):
    add("4.2", f"reference page labels: {k}", int((hp.label == k).sum()), paper, "same")
add("4.2", "pages with a medicine-quality warning (W1)", int((hp.W1 == "yes").sum()), 5, "same")
add("4.2", "warning pages attributed to an authority (W3)", int((hp.W3_official.astype(int) > 0).sum()), 5, "same")
ph = top.copy(); ph["item_id"] = "fresh_oct|" + ph.query_id + "|" + ph["rank"].astype(str)
ph = ph[ph.item_id.isin(hp.item_id)]
add("4.2", "pages whose HTML contains the image URL or file name", f"{int(ph.page_has_image.str.startswith('found').sum())}/{len(ph)}", "163/195", "data/1_retrieval/top3_fresh_oct.csv")
ia = pd.read_csv(R / "from_step09/inter_annotator.csv")
x = ia[(ia.measure == "label") & (ia.stratum == "all")].set_index("warning_coding")
add("4.2", "Annotator 1 vs pharmacist: n, agreement, kappa", f"{x.loc['after', 'n']}, {x.loc['after', 'agreement_pct']}, {x.loc['after', 'kappa']:.2f}", "57, 82.5, 0.41", "results/from_step09/inter_annotator.csv")
add("4.2", "Annotator 1 vs pharmacist: kappa 95% CI", f"{x.loc['after', 'kappa_lo']:.2f}--{x.loc['after', 'kappa_hi']:.2f}", "0.06--0.62", "same")
add("3.5", "Annotator 1 vs pharmacist before warning re-assessment", f"{x.loc['before', 'agreement_pct']}, {x.loc['before', 'kappa']:.2f}", "45.6, 0.09", "same")
v = pd.read_csv(D / "3_source_context/annotations/PH01_W1_verified_2026-11-26.csv", dtype=str)
add("3.5", "pharmacist re-assessed highlights: total / quality warning", f"{len(v)} / {int((v.W1_is_quality_warning == 'yes').sum())}", "44 / 1", "data/3_source_context/annotations/PH01_W1_verified_2026-11-26.csv")

# identity x page label (Fig. 8b) and attribute consistency (Fig. 8c)
rd = img[img.context.isin(["product_information", "insufficient", "warning", "not_evaluable"])]
grp = np.where(rd.same, np.where(rd.revised_M == "0", "same_m0", "same_m1"), rd.revised_primary)
for k, paper in (("same_m0", "48/52"), ("variant", "35/37"), ("same_m1", "34/40"), ("different", "12/31")):
    y = rd[grp == k]
    add("4.2", f"product-information pages among {k} images", f"{int((y.context == 'product_information').sum())}/{len(y)}", paper, SRC + " + identity_context_images.csv")
nopage = img[~img.context.isin(["product_information", "insufficient", "warning", "not_evaluable"])]
add("4.2", "matched images without an annotated page", len(nopage), 51, "same")
add("4.2", "  of which same, no packaging difference", int(nopage.m0.sum()), 18, "same")
pc = pd.read_csv(D / "3_source_context/attribute_consistency/panel_c_reconciled_2026-11-28.csv", dtype=str)
add("4.2", "attribute check: contradictions among variants", f"{int(((pc.new_group == 'Variant') & (pc.outcome == 'Contradiction')).sum())}/{int((pc.new_group == 'Variant').sum())}", "30/35", "data/3_source_context/attribute_consistency/panel_c_reconciled_2026-11-28.csv")
y = pc[pc.new_group == "Same, M=0"]
add("4.2", "attribute check: contradictions among same (no pkg diff)", f"{int((y.outcome == 'Contradiction').sum())}/{len(y)}", "0/48", "same")
add("4.2", "attribute check: incomplete among same (no pkg diff)", f"{int((y.outcome == 'Incomplete evidence').sum())}/{len(y)}", "20/48", "same")
add("4.2", "attribute check: contradiction / agree / incomplete (100)", "/".join(str(int((pc.outcome == o).sum())) for o in ("Contradiction", "All three agree", "Incomplete evidence")), "31/40/29", "same")

# ------------------------------------------------------------------ 4.3 automated annotation
la = pd.read_csv(R / "from_step09/label_agreement.csv").set_index("run")
for run, paper in (("A__gpt-4o-2024-08-06", "195, 86.7, 0.53, 0.34 to 0.67, 0.72 (0.41 to 0.94)"), ("B__gpt-4o-2024-08-06", "194, 75.8, 0.39, 0.26 to 0.51, 0.64 (0.35 to 0.86)"),
                   ("A__webref-qwen3-8b", "190, 82.1, 0.42, 0.28 to 0.55, 0.41 (0.19 to 0.62)"), ("B__webref-qwen3-8b", "193, 84.5, 0.48, 0.31 to 0.61, 0.48 (0.23 to 0.70)"),
                   ("baseline_S1-S6__cot", "195, 65.1, 0.10, -0.07 to 0.24, 0.35 (0.07 to 0.59)")):
    r_ = la.loc[run]
    h = lambda x: str(Decimal(str(x)).quantize(Decimal("0.01"), ROUND_HALF_UP))   # CSV holds 3 decimals; round half up
    add("Table 4", run, f"{int(r_.pages_valid)}, {r_.agreement_pct}, {h(r_.kappa)}, {h(r_.kappa_lo)} to {h(r_.kappa_hi)}, {h(r_.product_kappa)} ({h(r_.product_kappa_lo)} to {h(r_.product_kappa_hi)})", paper, "results/from_step09/label_agreement.csv")
    if not run.startswith("baseline"):
        add("4.3", f"{run} on the 57 shared pages: kappa", f"{r_.second_set_kappa:.2f}", {"A__gpt-4o-2024-08-06": "0.67", "B__gpt-4o-2024-08-06": "0.54", "A__webref-qwen3-8b": "0.27", "B__webref-qwen3-8b": "0.45"}[run], "same")
ref = hp.set_index("item_id")
fail_paper = {"A__gpt-4o-2024-08-06": ("0", "0.53"), "B__gpt-4o-2024-08-06": ("1", "0.38"), "A__webref-qwen3-8b": ("5", "0.39"), "B__webref-qwen3-8b": ("2", "0.46")}
for run, (nf, kp) in fail_paper.items():
    m = pd.read_csv(D / f"3_source_context/models/context_{run}.csv", dtype=str).set_index("item_id")
    ml = [m.loc[i, "label"] if m.loc[i, "status"] in ("ok", "empty", "not_fetched") else "failed" for i in ref.index]
    add("4.3", f"{run}: failed outputs / kappa with failures as errors", f"{ml.count('failed')} / {kappa(ref.label, ml):.2f}", f"{nf} / {kp}", f"data/3_source_context/models/context_{run}.csv")
gr = pd.read_csv(R / "from_step09/grounding.csv"); gq = gr.groupby("run")[["quotes", "ungrounded"]].sum()
lo, hi = (100 * (1 - gq.ungrounded / gq.quotes)).min(), (100 * (1 - gq.ungrounded / gq.quotes)).max()
add("4.3", "automatic quotation grounding range (%)", f"{lo:.1f}--{hi:.1f}", "97.2--99.5", "results/from_step09/grounding.csv")

# quotation review
e = pd.read_csv(D / "3_source_context/quotation_review/expanded_model_quote_results.csv", dtype=str).fillna("")
au = pd.read_csv(D / "3_source_context/quotation_review/audit_merged_CH_with_completed_A1.csv", dtype=str).fillna("")
qm = e.merge(au, left_on=["group_case_id", "group_quote_id"], right_on=["case_id", "quote_id"], how="left")
qm = qm[qm.stratum != "human_only"]
add("3.7", "reviewed quotation instances GPT-4o / Qwen3 8B", "/".join(str(int((qm.model == mm_).sum())) for mm_ in ("gpt-4o-2024-08-06", "webref-qwen3-8b")), "226/305", "data/3_source_context/quotation_review/expanded_model_quote_results.csv")
add("4.3", "reviewed quotations grounded GPT-4o / Qwen3 8B (%)", "/".join(pct(qm[qm.model == mm_].located.eq("yes").sum(), (qm.model == mm_).sum()) for mm_ in ("gpt-4o-2024-08-06", "webref-qwen3-8b")), "98.7/95.7",
    "data/3_source_context/quotation_review/audit_merged_CH_with_completed_A1.csv")
qs = pd.read_csv(R / "quotation_review_final_summary.csv").set_index(["model", "signal"])
G, W = "gpt-4o-2024-08-06", "webref-qwen3-8b"
add("4.3", "grounded + supporting GPT-4o / Qwen3 8B (%)", f"{qs.loc[(G, 'all'), 'supported_pct']}/{qs.loc[(W, 'all'), 'supported_pct']}", "83.2/69.8",
    "results/quotation_review_final_summary.csv", "rebuilt by scripts/3_source_context/quotation_review/16_final_quotation_support.py")
add("4.3", "grounded + supporting + product-relevant GPT-4o / Qwen3 8B (%)", f"{qs.loc[(G, 'all'), 'relevant_pct']}/{qs.loc[(W, 'all'), 'relevant_pct']}", "81.4/68.2", "results/quotation_review_final_summary.csv")
add("4.3", "P5 quotations supported GPT-4o / Qwen3 8B", f"{qs.loc[(G, 'P5'), 'supported']}/{qs.loc[(G, 'P5'), 'n']}, {qs.loc[(W, 'P5'), 'supported']}/{qs.loc[(W, 'P5'), 'n']}", "43/60, 45/77",
    "results/quotation_review_final_summary.csv", "FINAL_FINDINGS.md table")
add("4.3", "W1 quotations describing genuine warnings GPT-4o / Qwen3 8B", f"{qs.loc[(G, 'W1'), 'supported']}/{qs.loc[(G, 'W1'), 'n']}, {qs.loc[(W, 'W1'), 'supported']}/{qs.loc[(W, 'W1'), 'n']}", "6/7, 8/29",
    "results/quotation_review_final_summary.csv")
add("4.3", "P5 disagreement pages with a supporting model quotation", "18/53", "18/53", "data/3_source_context/quotation_review/FINAL_FINDINGS.md", "text record only")

out = pd.DataFrame(ROWS)
out.to_csv(R / "reported_numbers.csv", index=False)
print(out.status.value_counts().to_string())
print(out[out.status != "match"][["section", "quantity", "recomputed", "paper"]].to_string(index=False))
