"""
10_plot_result_figures.py
Results figures drawn only from sc3 output (runs/<run_id>/source_context_v3/eval/identity_context_images.csv),
so a rerun of sc3 updates them. Titles and messages belong in the captions, not in the figures.
  fig1_identity_by_rank    share of matched images that show the same medicine (no observed packaging difference),
                           a variant, or a different product, at visual rank 1-3; Wilson 95% CI (one image per query
                           and rank)
  fig2_label_by_identity   images whose source page could be read and labelled (195 of 246; 18 pages not saved, 33 without
                           readable text): share labelled product information per identity, with counts; cannot determine
                           (26 images) reported in the text only
  fig3_model_agreement     page-level Cohen's kappa of each model run and the previous pipeline with the human reference,
                           bootstrap 95% CI (label_agreement.csv); colour = model, open marker = direct-rule; last row =
                           observed agreement of the two human annotators (inter_annotator.csv), its own row and CI, not a
                           threshold. Product kappa, warnings and the like-for-like 56-page kappa: Result 3 table
Usage: python scripts/3_source_context/10_plot_result_figures.py [--run_id fresh_oct]
"""
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import math

def wilson(k, n, z=1.96):  # same as in 09_evaluate_against_human_reference.py
    if not n: return (None, None)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)

EXP_ROOT = Path(__file__).resolve().parents[2]
ID_LAB = {"same, no packaging difference": "Same", "same, packaging difference": "Same, packaging differs",
          "variant": "Variant", "different": "Different", "undetermined": "Cannot determine"}   # short names, defined in the caption
SHOWN = ["same, no packaging difference", "same, packaging difference", "variant", "different"]    # cannot determine: text only
READABLE = ["product_information", "insufficient", "warning", "not_evaluable"]
CONTEXTS = READABLE + ["no readable text", "not saved"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.linewidth": 0.7, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "pdf.fonttype": 42, "legend.frameon": False})


def save(fig, ev, name):
    for ext in ("pdf", "png"):
        fig.savefig(ev / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run_id", default="fresh_oct"); args = ap.parse_args()
    ev = EXP_ROOT / "runs" / args.run_id / "source_context_v3" / "eval"
    im = pd.read_csv(ev / "identity_context_images.csv")
    im["key"] = im.identity.replace({"cannot determine": "undetermined", "name only": "undetermined"})
    bad = (set(im.key) - set(ID_LAB)) | (set(im.context) - set(CONTEXTS))
    if bad or im.duplicated(["query_id", "rank"]).any():
        raise SystemExit(f"[STOP] unknown labels {bad} or duplicate query/rank rows in identity_context_images.csv")

    # Figure 1: identity by visual rank
    ranks = sorted(im["rank"].unique()); n = [int((im["rank"] == r).sum()) for r in ranks]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for (k, lab, col, mk), dx in zip([("same, no packaging difference", "Same", "#1baf7a", "o"), ("variant", "Variant", "#eda100", "s"),
                                      ("different", "Different", "#e34948", "^")], (-0.06, 0, 0.06)):
        kk = [int(((im["rank"] == r) & (im.key == k)).sum()) for r in ranks]
        p = np.array([100 * a / b for a, b in zip(kk, n)]); ci = np.array([wilson(a, b) for a, b in zip(kk, n)])
        ax.errorbar(np.array(ranks) + dx, p, yerr=[p - ci[:, 0], ci[:, 1] - p], color=col, marker=mk, ms=6, lw=2, capsize=3, label=lab)
    ax.set_xticks(ranks, [f"Rank {r}\n(n = {m})" for r, m in zip(ranks, n)]); ax.set_xlim(0.7, 3.3)
    ax.set_ylim(0, 60); ax.set_ylabel("Share of matched images (%)"); ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8.5, ncol=3)
    save(fig, ev, "fig1_identity_by_rank")

    # Figure 2: share of images whose readable source page is labelled product information, per image identity
    rd = im[im.context.isin(READABLE)]                    # pages that could be read and labelled; nothing is filtered from the results
    fig, ax = plt.subplots(figsize=(4.6, 2.3)); y = np.arange(len(SHOWN))
    for i, k in enumerate(SHOWN):
        g = rd[rd.key == k]; a_, n_ = int(g.context.eq("product_information").sum()), len(g)
        ax.barh(i, 100 * a_ / n_, height=0.58, color="#2a78d6")
        ax.text(100 * a_ / n_ + 1.5, i, f"{100 * a_ / n_:.0f}%  ({a_}/{n_})", va="center", fontsize=8.5, color="#0b0b0b")
    ax.set_yticks(y, [ID_LAB[k] for k in SHOWN]); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 115); ax.set_xticks(range(0, 101, 20)); ax.spines["bottom"].set_bounds(0, 100); ax.set_xlabel("Source page labelled product information (%)")
    save(fig, ev, "fig2_label_by_identity")
    # Figure 3: model agreement with the human reference (page level); colour = model, filled = signal-rule, open = direct-rule
    la = pd.read_csv(ev / "label_agreement.csv").set_index("run")
    ia = pd.read_csv(ev / "inter_annotator.csv"); ia = ia[(ia.measure == "label") & (ia.warning_coding == "after") & (ia.stratum == "all")]
    mods = sorted({r.split("__", 1)[1] for r in la.index if r[:3] in ("A__", "B__")}, key=lambda m: ("qwen" in m, m))
    show = lambda m: "GPT-4o" if m.startswith("gpt-4o") else "Qwen3 8B" if "qwen" in m else m
    rows, ticks, y0 = [], [], 0.0                          # rows: (run, colour, filled, y); ticks: (y, label, colour)
    for c_, m in zip(["#2a78d6", "#eb6834", "#6250d6"], mods):
        ys = []
        for p_ in "AB":
            if f"{p_}__{m}" in la.index: rows.append((f"{p_}__{m}", c_, p_ == "A", y0)); ys.append(y0); y0 += 0.7
        ticks.append((sum(ys) / len(ys), show(m), c_)); y0 += 0.5
    for r in la.index:
        if r.startswith("baseline"): rows.append((r, "#8a8984", True, y0)); ticks.append((y0, "Previous pipeline", "#52514e")); y0 += 1.1
    if len(ia):                                            # observed human-human agreement: its own row, not a threshold line
        h = ia.iloc[0]; rows.append(("human", "#0b0b0b", True, y0)); ticks.append((y0, "Two annotators", "#0b0b0b"))
        la.loc["human", ["kappa", "kappa_lo", "kappa_hi"]] = [h.kappa, h.kappa_lo, h.kappa_hi]
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    for r, c_, filled, y in rows:
        x = la.loc[r]
        ax.errorbar(x.kappa, y, xerr=[[x.kappa - x.kappa_lo], [x.kappa_hi - x.kappa]], fmt="D" if r == "human" else "o", ms=6 if r == "human" else 7,
                    color=c_, mfc=c_ if filled else "white", mew=1.8, capsize=3, lw=1.6)
        ax.text(x.kappa_hi + 0.02, y, f"{x.kappa:.2f}", va="center", fontsize=8, color="#0b0b0b")
    if len(ia):
        ax.axhline(rows[-1][3] - 0.55, color="#c9c8c3", lw=0.6)      # separates the human comparison from the runs
    ax.axvline(0, color="#8a8984", lw=0.6)
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks]); ax.set_ylim(rows[-1][3] + 0.5, -0.5); ax.tick_params(axis="y", length=0)
    for t, tk in zip(ax.get_yticklabels(), ticks): t.set_color(tk[2])
    ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=6, color="#52514e", label="signal–rule"),
                       plt.Line2D([], [], ls="", marker="o", ms=6, color="#52514e", mfc="white", mew=1.5, label="direct-rule")],
              loc="center right", bbox_to_anchor=(1.0, 0.36), fontsize=7.5, handletextpad=0.2, borderaxespad=0.2)
    ax.set_xlim(-0.15, 1.0); ax.set_xlabel("Cohen's κ (page level)")
    ax.grid(axis="x", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    save(fig, ev, "fig3_model_agreement")
    print(f"written: {ev / 'fig1_identity_by_rank.pdf'} | {ev / 'fig2_label_by_identity.pdf'} "
          f"| readable pages {len(rd)} of {len(im)} images; no saved or readable page: {len(im) - len(rd)}")


if __name__ == "__main__":
    main()
