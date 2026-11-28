"""
relabel_2026-11-28.py
Consistency correction of one product-identity label (28 Nov 2026).

Q092_1a315d036c3dea9b (EZAMOL-C, name-only rank 2) was labelled "same medicine", but the strict attribute
check (runs/labeling/attribute_check_A1_2026-11-18_summary.csv) records the dosage form as visible on only
one package. Under the identity guide, an attribute visible on only one package gives "cannot determine".
The label is therefore changed to cannot determine; the packaging-difference flag (0) is kept.

Originals are not modified. Written next to them, dated 2026-11-28:
  source_context_v3/scripts/product_identity/data/retrieval_qc_2026-11-28_246_images.csv
  source_context_v3/scripts/product_identity/data/panel_c_reconciled_2026-11-28.csv
  runs/labeling_agreement/identity_qc/labels_final_checked_qc_2026-11-28.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "source_context_v3" / "scripts" / "product_identity" / "data"
Q = ROOT / "runs" / "labeling_agreement" / "identity_qc"
ITEM, NOTE = "Q092_1a315d036c3dea9b", "2026-11-28: same -> cannot determine (dosage form visible on only one package)"

r = pd.read_csv(D / "retrieval_qc_2026-11-21_246_images.csv", dtype=str).fillna("")
assert (r.item_id == ITEM).sum() == 1 and r.loc[r.item_id == ITEM, "revised_primary"].item() == "same"
r.loc[r.item_id == ITEM, ["revised_label", "revised_primary"]] = ["unclear", "cannot_determine"]
r.loc[r.item_id == ITEM, "identity_changed"] = "1"
r["relabel_note"] = ""; r.loc[r.item_id == ITEM, "relabel_note"] = NOTE
r.to_csv(D / "retrieval_qc_2026-11-28_246_images.csv", index=False)

c = pd.read_csv(D / "panel_c_reconciled_2026-11-21.csv", dtype=str).fillna("")
assert c.loc[c.item_id == ITEM, "new_group"].item() == "Same, M=0"
c.loc[c.item_id == ITEM, "new_group"] = "Cannot determine"
c.to_csv(D / "panel_c_reconciled_2026-11-28.csv", index=False)

l = pd.read_csv(Q / "labels_final_checked_qc_2026-11-21.csv", dtype=str).fillna("")
assert l.loc[l.item_id == ITEM, "label"].item() == "same"
l.loc[l.item_id == ITEM, ["label", "primary_label"]] = ["unclear", "unclear"]
l.loc[l.item_id == ITEM, "note"] = (l.loc[l.item_id == ITEM, "note"] + " | " + NOTE).str.strip(" |")
l.to_csv(Q / "labels_final_checked_qc_2026-11-28.csv", index=False)

s = r[r.revised_primary == "same"]
print("same:", len(s), "| same M0:", int((s.revised_M == "0").sum()), "| cannot determine:", int((r.revised_primary == "cannot_determine").sum()))
print("same by rank:", s.groupby(s["rank"].astype(int)).size().to_dict())
print("panel c:", c.groupby(["new_group", "outcome"]).size().to_dict())
