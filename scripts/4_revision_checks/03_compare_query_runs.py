"""
03_compare_query_runs.py
========================
Paired comparison of two crawl runs on the same products, e.g. plain name
queries vs name + dosage form.

Reads runs/<A>/tables/match_summary.csv and runs/<B>/tables/match_summary.csv
(written by 02_run_pipeline_v2.py).

Reports, over products present in both runs:
  - products with >= 1 image passing the visual filter (A, B)
  - gained (no match in A, match in B) and lost (match in A, none in B)
  - exact McNemar p-value on gained vs lost
  - number of distinct valid pages per product (median, A vs B)

Usage:
    python scripts\\03_compare_query_runs.py --a fresh_oct --b fresh_oct_form
"""
import argparse, math
from pathlib import Path
import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[1]

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    A = pd.read_csv(EXP_ROOT / "runs" / args.a / "tables" / "match_summary.csv")
    B = pd.read_csv(EXP_ROOT / "runs" / args.b / "tables" / "match_summary.csv")
    m = A.merge(B, on="query_id", suffixes=("_a", "_b"))
    m = m[(m.status_a != "no_crawl_folder") & (m.status_b != "no_crawl_folder")]
    a, b = m.has_match_a.astype(bool), m.has_match_b.astype(bool)
    gained, lost = int((~a & b).sum()), int((a & ~b).sum())
    print(f"products compared: {len(m)}")
    print(f"with >=1 match   : {args.a} {int(a.sum())} | {args.b} {int(b.sum())}")
    print(f"gained {gained} | lost {lost} | exact McNemar p = {mcnemar_exact(gained, lost):.4f}")
    print(f"median distinct valid pages: {args.a} {m.n_valid_pages_a.median():.0f} | "
          f"{args.b} {m.n_valid_pages_b.median():.0f}")
    out = EXP_ROOT / "runs" / args.b / "tables" / f"compare_{args.a}_vs_{args.b}.csv"
    m.assign(gained=~a & b, lost=a & ~b).to_csv(out, index=False)
    print(f"per-product table: {out}")

if __name__ == "__main__":
    main()
