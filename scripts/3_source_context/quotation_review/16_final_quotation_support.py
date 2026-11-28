"""
16_final_quotation_support.py
Rebuilds the final quotation-review table (formerly six_signal_final.csv) from the two reviewer exports.

Rules (as stated in data/3_source_context/quotation_review/FINAL_FINDINGS.md):
  * completed A1 answers (complete == 1; 101 quotations) replace the CH answer for the same grouped quotation;
    all other quotations keep the CH answer (508 grouped quotations in total);
  * three analytic resolutions under the agreed P5 rule: C0045 q1, C0049 q1 and C0049 q2 are not supported
    (the raw A1 answer "yes" is kept in the column supported_as_exported);
  * the grouped answers are expanded to the original model-specific quotation instances with
    semantic_group_map.csv; human_only records are excluded (531 instances: 226 GPT-4o, 305 Qwen3 8B).
Cumulative measures per instance:
  grounded  = quotation located in the saved page text
  supported = grounded and the quotation supports the signal (W1: a genuine medicine-quality warning)
  relevant  = supported and the evidence concerns the searched medicine (P1-P5: product_ref searched;
              W1: warning target = this medicine)
Outputs: results/quotation_review_final_instances.csv (one row per instance; replaces six_signal_final.csv)
         results/quotation_review_final_summary.csv (per model and per model x signal)
Usage (repository root): python scripts/3_source_context/quotation_review/16_final_quotation_support.py
"""
from pathlib import Path
import pandas as pd

Q, R = Path("data/3_source_context/quotation_review"), Path("results")
ch = pd.read_csv(Q / "semantic_evidence_CH.csv", dtype=str).fillna("")
a1 = pd.read_csv(Q / "semantic_evidence_A1.csv", dtype=str).fillna("")
mp = pd.read_csv(Q / "semantic_group_map.csv", dtype=str).fillna("")
assert len(ch) == 508 and (ch.complete == "1").all()
done = a1[a1.complete == "1"]
assert len(done) == 101, len(done)

key = ["case_id", "quote_id"]
dec = ch.set_index(key)
dec["source"] = "CH"
d1 = done.set_index(key)
dec.loc[d1.index, d1.columns] = d1
dec.loc[d1.index, "source"] = "A1"
dec = dec.reset_index()
dec["supported_as_exported"] = dec.support
RESOLVED = {("C0045", "q1"), ("C0049", "q1"), ("C0049", "q2")}
mask = [k in RESOLVED for k in zip(dec.case_id, dec.quote_id)]
assert dec.loc[mask, "support"].eq("yes").all() and sum(mask) == 3
dec.loc[mask, "support"] = "no"
dec.loc[mask, "resolution"] = "P5 rule: role or product link not explicit"

x = mp.merge(dec, left_on=["group_case_id", "group_quote_id"], right_on=key, how="left", validate="m:1")
assert x.located.notna().all()
x = x[x.stratum != "human_only"].copy()
w1 = x.signal.eq("W1")
x["grounded"] = x.located.eq("yes")
x["supported"] = x.grounded & x.support.eq("yes")
x["relevant"] = x.supported & ((~w1 & x.product_ref.eq("searched")) | (w1 & x.warning_target.eq("this")))
cols = ["group_case_id", "group_quote_id", "old_case_id", "old_quote_id", "model", "query_id", "rank", "signal", "stratum",
        "source", "located", "support", "supported_as_exported", "product_ref", "warning_quality", "warning_target",
        "warning_official", "grounded", "supported", "relevant", "resolution"]
x[cols].to_csv(R / "quotation_review_final_instances.csv", index=False)

rows = []
for m, g in x.groupby("model"):
    rows.append({"model": m, "signal": "all", "n": len(g), **{c: int(g[c].sum()) for c in ("grounded", "supported", "relevant")}})
    for s, h in g.groupby("signal"):
        rows.append({"model": m, "signal": s, "n": len(h), **{c: int(h[c].sum()) for c in ("grounded", "supported", "relevant")}})
sm = pd.DataFrame(rows)
for c in ("grounded", "supported", "relevant"):
    sm[c + "_pct"] = (100 * sm[c] / sm.n).round(1)
sm.to_csv(R / "quotation_review_final_summary.csv", index=False)
print(sm[sm.signal == "all"].to_string(index=False))
print(sm[sm.signal.isin(["P5", "W1"])].to_string(index=False))
