"""
fig_match_count_roc.py
How well the LightGlue match count separates same-medicine images, and what the 50-match cutoff trades.
Uses existing labels only (name-only run, fresh_oct): the final QC identity labels
(runs/labeling_agreement/identity_qc/labels_final_checked_qc_2026-11-28.csv, final guide) for the 246 selected
images (>= 50 matches) and the 102 annotated images below the cutoff (20-49 matches).
Image order: crawl files are numbered in the order they were downloaded, i.e. search order among the
images that could be downloaded (thumbnails that failed are skipped and their search position is not kept).
(a) ROC of the match count among annotated images, for same medicine and for same medicine with no
    observed packaging difference; cutoffs 50 and 200 marked.
(b) Effect of raising the cutoff from 50 (exact: a higher cutoff only removes selected images):
    share of retained images that are the same medicine, products with a same-medicine image (Success@3),
    and products with a reference candidate (same, no packaging difference, product-information page).
Outputs: runs/fresh_oct/source_context_v3/eval/fig4_match_count_roc.{pdf,png}, match_count_roc.csv,
         ../../tables/threshold_sensitivity_revised_identity.csv
Usage: python source_context_v3\\scripts\\fig_match_count_roc.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "fresh_oct"
ANN, EV = RUN / "source_context_v3" / "annotation", RUN / "source_context_v3" / "eval"
QC = ROOT / "source_context_v3" / "scripts" / "product_identity" / "data" / "retrieval_qc_2026-11-28_246_images.csv"
LABELS_QC = ROOT / "runs" / "labeling_agreement" / "identity_qc" / "labels_final_checked_qc_2026-11-28.csv"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.linewidth": 0.7, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "pdf.fonttype": 42, "legend.frameon": False})
GREEN, NAVY, GREY, INK = "#1baf7a", "#1f4e79", "#8a8984", "#0b0b0b"


def roc(score, y):
    """ROC points over all distinct thresholds (score >= t is positive) and the AUC (ties count half)."""
    t = np.r_[np.inf, np.sort(np.unique(score))[::-1]]
    P, N = y.sum(), (~y).sum()
    tpr = np.array([(y & (score >= x)).sum() / P for x in t]); fpr = np.array([(~y & (score >= x)).sum() / N for x in t])
    pos, neg = score[y], score[~y]
    auc = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (P * N)
    return fpr, tpr, auc


def main():
    it = pd.read_csv(ANN / "labeling_items.csv", dtype=str).fillna("")
    it = it[it.run == "fresh_oct"]
    lab = pd.read_csv(LABELS_QC, dtype=str).fillna("")
    qc = pd.read_csv(QC, dtype=str).fillna("")
    d = it.merge(lab[["item_id", "label", "packaging_differs"]], on="item_id", how="left", validate="1:1") \
          .merge(qc[["item_id", "revised_primary", "revised_M"]], on="item_id", how="left", validate="1:1")
    if d.label.isna().any() or d.label.eq("").any():
        raise SystemExit("[STOP] an annotated image has no final QC label")
    sel = d.kind == "selected"                      # consistency with the 246-image retrieval file used in the paper
    same_name = d.label.replace({"unclear": "cannot_determine", "brand_only": "cannot_determine"})   # naming in the retrieval file
    if not (same_name[sel] == d.loc[sel, "revised_primary"]).all() or not (d.loc[sel, "packaging_differs"] == d.loc[sel, "revised_M"]).all():
        raise SystemExit("[STOP] QC label file and retrieval_qc file disagree on a selected image")
    d["matches"] = d.inliers.astype(float)
    d["same"] = d.label.eq("same"); d["same_m0"] = d.same & d.packaging_differs.eq("0")
    d["download_pos"] = d.image_path.str.extract(r"(\d{5})\.\w+$")[0].astype(int) + 1
    s3 = d[sel & d.same]; first = s3.groupby("query_id").download_pos.min()
    s0 = d[sel & d.same_m0].groupby("query_id").download_pos.min()
    pd.DataFrame([{"selected_images": int(sel.sum()), "selected_from_download_pos_4_10": int((d.loc[sel, "download_pos"] >= 4).sum()),
                   "products_same": len(first), "products_same_only_pos_4_10": int((first >= 4).sum()),
                   "products_same_m0": len(s0), "products_same_m0_only_pos_4_10": int((s0 >= 4).sum())}]
                 ).to_csv(EV / "reranking_download_order.csv", index=False)

    # ---- (a) ROC
    out, curves = [], {}
    for k, name in (("same", "Same medicine"), ("same_m0", "Same, no packaging difference")):
        fpr, tpr, auc = roc(d.matches.to_numpy(), d[k].to_numpy())
        curves[k] = (fpr, tpr, auc, name)
        for c in (50, 200):
            y = d[k]; s = d.matches >= c
            out.append({"outcome": name, "auc": round(auc, 3), "cutoff": c, "tpr": round((y & s).sum() / y.sum(), 3),
                        "fpr": round((~y & s).sum() / (~y).sum(), 3), "positives": int(y.sum()), "negatives": int((~y).sum())})
    pd.DataFrame(out).to_csv(EV / "match_count_roc.csv", index=False)

    # ---- (b) cutoff sensitivity on the selected images (exact for cutoffs >= 50)
    ctx = pd.read_csv(EV / "identity_context_images.csv", dtype=str)
    ctx["rank"] = ctx["rank"].astype(int).astype(str); qc["rank"] = qc["rank"].astype(int).astype(str)
    s_ = ctx.merge(qc[["query_id", "rank", "revised_primary", "revised_M"]], on=["query_id", "rank"], validate="1:1")
    s_["m"] = s_.inliers.astype(float); s_["same"] = s_.revised_primary.eq("same")
    s_["ref"] = s_.same & s_.revised_M.eq("0") & s_.context.eq("product_information")
    s_["m0"] = s_.same & s_.revised_M.eq("0")
    rows = []
    for c in range(50, 801, 10):
        x = s_[s_.m >= c]; g = x.groupby("query_id")
        rows.append({"cutoff": c, "images": len(x), "precision_same_pct": 100 * x.same.mean() if len(x) else np.nan,
                     "precision_same_m0_pct": 100 * x.m0.mean() if len(x) else np.nan,
                     "success3_same": int(g.same.any().sum()), "success3_same_m0": int(g.m0.any().sum()),
                     "reference_candidates": int(g.ref.any().sum()), "products_no_image": 100 - x.query_id.nunique()})
    sens = pd.DataFrame(rows)
    sens.round(1).to_csv(RUN / "tables" / "threshold_sensitivity_revised_identity.csv", index=False)
    assert sens.iloc[0][["success3_same", "success3_same_m0", "reference_candidates"]].tolist() == [64, 48, 35]

    # ---- precision-recall of the match count (image level, annotated name-only images)
    #  precision(t) = same-medicine share of annotated images with >= t matches
    #  recall(t)    = share of the annotated same-medicine images that have >= t matches
    #  Exact for t >= 50 (all top-three images are annotated); 20-49 rests on the annotated near-threshold sample.
    pr = []
    for t in sorted(d.matches.unique()):
        if t < 20:
            continue
        k_ = d.matches >= t
        pr.append({"threshold": t, "retained": int(k_.sum()), "precision": d.loc[k_, "same"].mean(),
                   "recall": (d.same & k_).sum() / d.same.sum()})
    pr = pd.DataFrame(pr)
    pr_s = pr.sort_values("recall")
    ap = float(((pr_s.recall.diff().fillna(pr_s.recall.iloc[0])) * pr_s.precision).sum())
    pr.round(4).to_csv(EV / "match_count_precision_recall.csv", index=False)

    # ---- figure: (a) precision-recall of the match count, thresholds marked; (b) products covered against the threshold
    MUTED = "#52514e"
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.32, "width_ratios": [1, 1.15]})
    pv = pr[pr.retained >= 20]                   # drop the far tail where fewer than 20 images remain
    a.plot(pv.recall, pv.precision, color=GREEN, lw=1.8, drawstyle="steps-post")
    base = d.same.mean()
    a.axhline(base, color="#c9c8c3", lw=0.7, ls="--")
    a.text(0.02, base - 0.03, f"no threshold: all annotated images ({100 * base:.0f}%)", fontsize=6.8, color=MUTED, va="top")
    a.text(0.03, 0.97, "Labels: minimum number of matches", fontsize=6.8, color=MUTED, va="top", transform=a.transAxes)
    for t, lab, pos in ((50, "50 (used)", (0.86, 0.40)), (100, "100", (0.86, 0.49)), (200, "200", (0.82, 0.58)), (400, "400", (0.63, 0.69))):
        r_ = pr[pr.threshold >= t].iloc[0]
        a.plot(r_.recall, r_.precision, "o", ms=5.5, color=GREEN, mec="white", mew=1.2, zorder=3)
        a.annotate(lab, (r_.recall, r_.precision), xytext=pos, fontsize=7.2, color=INK, ha="right", va="center",
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=3))
    a.set_xlim(0, 1.02); a.set_ylim(0, 1.02)
    a.set_xticks([0, 0.25, 0.5, 0.75, 1]); a.set_xticklabels(["0", "25", "50", "75", "100"])
    a.set_yticks([0, 0.25, 0.5, 0.75, 1]); a.set_yticklabels(["0", "25", "50", "75", "100"])
    a.set_xlabel("Recall: same-medicine images kept (%)"); a.set_ylabel("Precision: same medicine\namong kept images (%)")
    a.grid(color="#eeeeea", lw=0.6); a.set_axisbelow(True); a.tick_params(labelsize=7.5)
    a.set_title("(a) Images kept by the match count", loc="left", fontsize=8.5)

    XMAX = 500                                   # few images remain above this; the curve gets noisy
    sv = sens[sens.cutoff <= XMAX].set_index("cutoff")
    b.axvline(50, color=MUTED, lw=0.7, ls=":")
    b.text(56, 97, "threshold used", fontsize=6.8, color=MUTED, va="top")
    b.plot(sv.index, sv.success3_same, color=NAVY, lw=1.8)
    b.plot(sv.index, sv.reference_candidates, color=GREY, lw=1.8)
    for ser, color, dy in ((sv.success3_same, NAVY, 5), (sv.reference_candidates, GREY, -5)):
        for x_ in (50, 200):
            y_ = float(ser.loc[x_])
            b.plot(x_, y_, "o", ms=5, color=color, mec="white", mew=1.0, zorder=3)
            b.text(x_ + 8, y_ + dy, f"{y_:.0f}", fontsize=7.3, color=INK, va="center")
    b.text(XMAX, 76, "with a same-medicine\nimage (Success@3)", fontsize=6.8, color=NAVY, ha="right", va="bottom")
    b.text(XMAX, 16, "with a reference\ncandidate", fontsize=6.8, color=GREY, ha="right", va="top")
    b.set_xlim(40, XMAX); b.set_xticks([50, 200, 300, 400, 500]); b.set_ylim(0, 100)
    b.set_xlabel("Minimum number of matches"); b.set_ylabel("Products (of 100)")
    b.grid(axis="y", color="#e6e5e1", lw=0.6); b.set_axisbelow(True); b.tick_params(labelsize=7.5)
    b.set_title("(b) Products: coverage vs threshold", loc="left", fontsize=8.5)
    print(f"average precision (t >= 20): {ap:.3f}")
    print(pr[pr.threshold.isin([pr[pr.threshold >= t].threshold.iloc[0] for t in (20, 25, 50, 100, 200, 400)])].round(3).to_string(index=False))
    for ext in ("pdf", "png"):
        fig.savefig(EV / f"fig4_match_count_roc.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.05)
    print(pd.DataFrame(out).to_string(index=False)); print(pd.read_csv(EV / "reranking_download_order.csv").T.to_string())
    print(sens[sens.cutoff.isin([50, 100, 200, 300, 400])].round(1).to_string(index=False))
    print("written:", EV / "fig4_match_count_roc.pdf")


if __name__ == "__main__":
    main()
