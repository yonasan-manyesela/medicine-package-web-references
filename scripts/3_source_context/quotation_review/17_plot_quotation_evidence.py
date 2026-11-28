"""
17_plot_quotation_evidence.py
Figure of textual grounding, evidence support and product relevance of the reviewed model quotations,
drawn from results/quotation_review_final_summary.csv (made by 16_final_quotation_support.py).
Output: results/figures/fig_quotation_evidence.pdf / .png
Usage (repository root): python scripts/3_source_context/quotation_review/17_plot_quotation_evidence.py
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
s = pd.read_csv(R / "quotation_review_final_summary.csv")
s = s[s.signal == "all"].set_index("model")
models = [("gpt-4o-2024-08-06", "GPT-4o", "#246199"), ("webref-qwen3-8b", "Qwen3 8B", "#d47c3e")]
measures = [("grounded_pct", "Textual grounding"), ("supported_pct", "Evidence support"), ("relevant_pct", "Product relevance")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.6, 3.2))
h = 0.33
for j, (col, label) in enumerate(measures):
    for k, (m, name, color) in enumerate(models):
        y = -j - k * (h + 0.04)
        v = float(s.loc[m, col])
        ax.barh(y, v, height=h, color=color, label=name if j == 0 else None)
        ax.text(v + 1, y, f"{v:.1f}%", va="center", fontsize=10.5, color="#2f3640")
ax.set_yticks([-j - (h + 0.04) / 2 for j in range(len(measures))])
ax.set_yticklabels([m[1] for m in measures])
ax.set_xlim(0, 110)
ax.set_xticks(range(0, 101, 20))
ax.grid(axis="x", color="#e1e4e8", lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#b8c2cc")
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
fig.tight_layout()
(R / "figures").mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(R / "figures" / f"fig_quotation_evidence.{ext}", dpi=300, bbox_inches="tight")
print("written results/figures/fig_quotation_evidence.pdf")
