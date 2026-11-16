"""
06_save_source_page_text.py
Re-extracts the page text from each saved HTML page with the same steps as 02_match_web_images_to_query.py (fetch_page)
(BeautifulSoup html.parser; script, style, nav, footer, header, aside removed; whitespace joined), without the
8000-character cut of the original pipeline. Only pages that were cut get the re-extracted text; every other page
keeps its original text (re-extraction from the saved HTML can differ in character decoding). The original
rank<k>_page.txt files are not changed.

Output (runs/<run_id>/source_context_v3/page_text/):
  <query folder>/rank<k>_page.txt   same header as the original file + full text (read by sc2 and sc3)
  full_text_check.csv               per page: old and new length, capped, prefix_ok (first 8000 characters of
                                    the new text equal the old text), and the first differing position
Usage:
  python scripts/3_source_context/06_save_source_page_text.py --run_id fresh_oct
"""
import argparse, sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_label_rule import full_text_file, local_file

EXP_ROOT = Path(__file__).resolve().parents[2]   # repository root
CUT = 8000


def extract(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return " ".join(soup.get_text().split())


def split_page_file(p):
    """(header lines incl. the ===== line, text) of an original rank<k>_page.txt."""
    lines = Path(p).read_text(encoding="utf-8").split("\n")
    for k, line in enumerate(lines):
        if line.startswith("=" * 10):
            return lines[:k + 1], "\n".join(lines[k + 1:])
    return [], "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="fresh_oct")
    args = ap.parse_args()
    run = EXP_ROOT / "runs" / args.run_id
    pages = pd.read_csv(run / "tables" / "pipeline_top3_all.csv", dtype=str).fillna("")
    rows = []
    for r in pages.itertuples():
        item = f"{args.run_id}|{r.query_id}|{int(float(r.rank))}"
        hf, pf = local_file(r.html_file, run / "pipeline") if r.html_file else None, local_file(r.page_file, run / "pipeline")
        if hf is None or not hf.exists() or not pf.exists():
            rows.append({"item_id": item, "status": "no saved html" if pf.exists() else "no page file"}); continue
        head, old = split_page_file(pf)
        new = extract(hf.read_text(encoding="utf-8", errors="replace"))
        old = old.strip()
        capped = len(old) >= CUT - 10
        ok = new[:CUT].strip() == old
        diff = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new))) if not ok else ""
        text = new if capped else old                      # uncut pages keep the original text
        out = full_text_file(r.page_file, run)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(head[:-1] + ["FULL_TEXT: sc0_full_text.py (no character cut)", "=" * 60]) + "\n" + text, encoding="utf-8")
        rows.append({"item_id": item, "status": "written", "old_chars": len(old), "new_chars": len(new),
                     "capped": int(capped), "prefix_ok": int(ok), "first_difference_at": diff,
                     "text_used": "full re-extraction" if capped else "original"})
    d = pd.DataFrame(rows)
    out_dir = run / "source_context_v3" / "page_text"
    out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_dir / "full_text_check.csv", index=False)
    w = d[d.status == "written"]
    bad = w[(w.prefix_ok == 0) & (w.capped == 1)]
    print(f"pages: {len(d)} | written {len(w)} | cut before (now full text) {int(w.capped.sum())} | "
          f"re-extraction differs: {int((w.prefix_ok == 0).sum())} (uncut, original text kept) | other: {d[d.status != 'written'].status.value_counts().to_dict()}")
    if len(bad):
        print("[CHECK] cut pages whose re-extraction differs from the original start:", " ".join(bad.item_id))
    print(f"written: {out_dir}")


if __name__ == "__main__":
    main()
