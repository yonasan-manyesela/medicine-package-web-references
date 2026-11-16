"""
07_build_context_annotation_page.py  (guide ctx_v8)
Human evidence annotation for the source context of retrieved reference images.
The SAVED webpage (October HTML snapshot, scripts switched off) is shown as it looks.
The annotator HIGHLIGHTS the evidence; each highlight carries its own attributes, so the
human output has the same shape as the model output (06, pipeline A):
  R   Page can't be read                                  (button)
  P1  highlight where the page names the searched product  or "Not on the page"
  P2-P5 highlight details about it; tick what each shows:  or "No details"
        ingredient / strength / form / manufacturer
  W1  highlight each warning sentence                       or "No warning on this page"
      W2 target: this / other_general / cant_tell
      W3 issued by a named regulator: yes / no
Only text counts. Definitions are the same sentences as in the model prompt.
Rule (computed in 07, never shown): R no -> not evaluable; a warning with target this ->
warning; P1 + any of P2-P5 -> product information; else insufficient.
Each highlight keeps its exact text, start/end in the archived page text (page view: the saved HTML's
body text; text view: the extracted text the model receives), 60 characters before and after,
its vertical position (% of page height), the view and the translation flag. Scroll depth is
recorded per page. sc3 maps every highlight onto the model's input text.

Sets:  --set prescreen | all | second | practice
  prescreen: proposes ok / text / exclude for every page and writes prescreen_viewer.html;
  the first author checks each page and saves prescreen.csv; all and second use it
  (excluded pages are not annotated; text pages are shown as text). Page order is shuffled (seed).
Usage (from the repository root):
    python scripts/3_source_context/07_build_context_annotation_page.py --set practice
    python scripts/3_source_context/07_build_context_annotation_page.py --set second --n_second 57
    python scripts/3_source_context/07_build_context_annotation_page.py --set all
Outputs in runs/<run_id>/source_context_v3/annotation/:
  annotation_practice_<n>_pages.html, annotation_main_<n>_pages.html (second annotator),
  annotation_all_<n>_pages.html (first author); add --review for an _REVIEW copy with Compare.
  Annotators download annotations_<code>_<practice|main|all>_<date>.csv
"""
import argparse, base64, hashlib, io, json, random, re
from html.parser import HTMLParser
from pathlib import Path, PureWindowsPath

import pandas as pd
from PIL import Image

EXP_ROOT = Path(__file__).resolve().parents[2]   # repository root
GUIDE_VERSION = "ctx_v9.3"
MAX_CHARS = 6000                       # same as previous_pipeline_ievc2026.py
MAX_SIDE_PX, JPEG_QUALITY = 640, 80
FETCH_OK = {"ok", "low_content"}
WARNING_LABELS = {"suggests suspicious reference", "warning", "warning or falsification-related information"}

# strong warning terms (case-insensitive); generic words such as "warning" or "alert" are left out
STRONG_TERMS = [
    r"falsifi\w*", r"counterfeit\w*", r"\bfake\b", r"substandard", r"\brecall(?:ed|s)?\b", r"\bseiz(?:ed|ure|ures)\b",
    r"unregistered", r"\bbanned\b", r"\bunapproved\b",
    r"contrefa\w*", r"falsifié\w*", r"\brappel\b",                                   # French
    r"falsificad\w*", r"\bfalsos?\b", r"\bretirad\w*", r"decomisad\w*", r"apreendid\w*",  # Spanish / Portuguese
    r"recolhimento", r"\bprohibid\w*", r"\bproibid\w*",
    r"مغشوش", r"مزيف", r"مزور", r"مقلد",                                                  # Arabic
]
STRONG_RE = re.compile("|".join(STRONG_TERMS), re.IGNORECASE)
EN_STOP = set("the and of to in for is with on this that are it as be by or from at your you an can not "
              "if may take use used it's should have has was were which".split())


def b64_image(path):
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=JPEG_QUALITY)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"  [image missing] {path}: {type(e).__name__}")
        return ""


def locate(path_str, base):
    p = Path(path_str)
    if p.exists():
        return p
    w = PureWindowsPath(path_str)
    return base / w.parent.name / w.name


def read_page_text(page_file):
    """Same reading as previous_pipeline_ievc2026.py: the text after the first '==========' line."""
    try:
        lines = Path(page_file).read_text(encoding="utf-8").split("\n")
    except Exception:
        return ""
    for k, line in enumerate(lines):
        if line.startswith("=" * 10):
            return "\n".join(lines[k + 1:])
    return "\n".join(lines)


def looks_non_english(text):
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 60:
        return False                                   # too short to judge
    if sum(c.isascii() for c in letters) / len(letters) < 0.7:
        return True                                    # mostly non-Latin script
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    return len(toks) >= 20 and sum(t in EN_STOP for t in toks) / len(toks) < 0.05



def saved_html(page_file, url):
    """The saved HTML next to the saved text (rankN_page.html), made safe to display:
    no scripts, iframes, event handlers or meta refresh; base URL so styles and pictures load."""
    p = Path(page_file).with_suffix(".html")
    try:
        h = p.read_bytes().decode("utf-8", errors="replace")
    except Exception:
        return ""
    h = re.sub(r"<script\b.*?</script\s*>", "", h, flags=re.I | re.S)
    h = re.sub(r"<script\b[^>]*/?>", "", h, flags=re.I)
    h = re.sub(r"</?(iframe|object|embed|frame|frameset)\b[^>]*>", "", h, flags=re.I)   # tags only: fast on any page
    h = re.sub(r"<meta[^>]+http-equiv=[\"']?refresh[^>]*>", "", h, flags=re.I)
    h = re.sub(r"\son[a-z]+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", "", h, flags=re.I)
    def lazy(m):
        tag = m.group(0)
        src = re.search(r"\sdata-(?:lazy-)?src\s*=\s*([\"'])(.*?)\1", tag, flags=re.I)
        if src and not re.search(r"\ssrc\s*=\s*[\"'](?!data:)", tag, flags=re.I):
            tag = re.sub(r"\ssrc\s*=\s*([\"']).*?\1", "", tag, flags=re.I)
            tag = tag[:-1].rstrip("/") + f' src="{src.group(2)}">'
        if not re.search(r"\sloading\s*=", tag, flags=re.I):
            tag = tag[:4] + ' loading="lazy"' + tag[4:]
        return tag
    h = re.sub(r"<img\b[^>]*>", lazy, h, flags=re.I)
    safe = ('<meta charset="utf-8"><meta http-equiv="Content-Security-Policy" '
            'content="script-src \'none\'; frame-src \'none\'; object-src \'none\'">'
            f'<base href="{url}" target="_self">')
    if re.search(r"<head(\s[^>]*)?>", h, flags=re.I):
        h = re.sub(r"(<head(?:\s[^>]*)?>)", lambda m: m.group(1) + safe, h, count=1, flags=re.I)
    else:
        h = "<head>" + safe + "</head>" + h
    return h

class _Vis(HTMLParser):
    """Collects the text a reader sees: body text outside style/script/noscript/template. One pass."""
    SKIP = {"style", "script", "noscript", "template"}
    def __init__(self):
        super().__init__(convert_charrefs=True); self.depth = 0; self.body = False; self.out = []
    def handle_starttag(self, tag, attrs):
        if tag == "body": self.body = True
        if tag in self.SKIP: self.depth += 1
    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth: self.depth -= 1
    def handle_data(self, data):
        if not self.depth: self.out.append(data)


def visible_text(html):
    """Text a reader sees in the cleaned saved page (linear time, safe on huge or broken pages)."""
    if not html:
        return ""
    v = _Vis()
    try:
        v.feed(html); v.close()
    except Exception:
        pass
    return " ".join(" ".join(v.out).split())


def show_as_text(html, text, drug):
    """Open the page as text when the saved page cannot show what the model reads:
    no saved page; fewer than 30 visible words; under 20% of the extracted text; or the
    medicine name is in the extracted text but nowhere on the rendered page."""
    if not html:
        return True, "no saved page"
    tw, vis = len(text.split()), visible_text(html)
    v = len(vis.split())
    if tw >= 30 and (v < 30 or v < 0.2 * tw):
        return True, "saved page empty"
    key = re.sub(r"[^a-z0-9]", "", drug.split()[0].lower()) if drug.split() else ""
    squash = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
    if len(key) >= 3 and key in squash(text) and key not in squash(vis):
        return True, "name only in text"
    return False, ""


UNUSABLE_RE = re.compile(r"verify you are (a )?human|checking your browser|just a moment|access denied|forbidden|"
                         r"page not found|\b404\b|\b403\b|enable javascript|captcha|are you a robot|cloudflare", re.I)


def propose(html, text, drug):
    """Proposed pre-screen decision: exclude (no usable content), text (saved page cannot show
    the content), ok. The first author confirms or corrects every proposal in prescreen.csv."""
    w = len(text.split())
    if w <= 10:
        return "exclude", "fewer than 11 words of text"
    if w < 80 and UNUSABLE_RE.search(text):
        return "exclude", "error / verification page"
    as_text, why = show_as_text(html, text, drug)
    return ("text", why) if as_text else ("ok", "")


def load_prescreen(out):
    f = out / "prescreen.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, dtype=str).fillna("")
    if "checked" in d and (d["checked"] != "1").any():
        n = int((d["checked"] != "1").sum())
        raise SystemExit(f"[STOP] prescreen.csv has {n} pages not checked (checked = 0). This looks like a download from a "
                         "fresh viewer (proposals only). Use the prescreen.csv with your decisions.")
    bad = d[~d["decision"].isin(["ok", "text", "exclude"])]
    if len(bad):
        raise SystemExit(f"[STOP] prescreen.csv: decision must be ok / text / exclude (rows: {', '.join(bad.item_id[:5])})")
    return dict(zip(d.item_id, d.decision))


def load_pages(run_id):
    run = EXP_ROOT / "runs" / run_id
    pages = pd.read_csv(run / "tables" / "pipeline_top3_all.csv", dtype=str).fillna("")
    pages["rank"] = pages["rank"].astype(int)
    pages = pages.sort_values(["query_id", "rank"])
    ok = pages[pages["content_quality"].isin(FETCH_OK)].copy()
    # medicine name: once per query from match_summary.csv, the same source the model runner uses
    ms = run / "tables" / "match_summary.csv"
    names = {}
    if ms.exists():
        m = pd.read_csv(ms, dtype=str).fillna("")
        names = dict(zip(m["query_id"], m["drug"]))
    odd = sorted({(r["query_id"], r["drug"], names[r["query_id"]]) for _, r in ok.iterrows()
                  if r["query_id"] in names and r["drug"].strip() != names[r["query_id"]].strip()})
    for q, a, b in odd:
        print(f"  [name] {q}: page rows say '{a}', match_summary says '{b}' -> using '{b}'")
    rows = []
    for _, r in ok.iterrows():
        pf = locate(r["page_file"], run / "pipeline")
        full = read_page_text(pf)
        shown = full[:MAX_CHARS]
        rows.append({"item_id": f"{run_id}|{r['query_id']}|{r['rank']}", "run_id": run_id,
                     "query_id": r["query_id"], "rank": int(r["rank"]), "drug": names.get(r["query_id"], r["drug"]),
                     "url": r["source_page_url"], "image": r["image"], "words": len(full.split()),
                     "full": full, "shown": shown, "page_file": str(pf), "non_english": looks_non_english(shown),
                     "strong_shown": sorted({m.group(0).lower() for m in STRONG_RE.finditer(full)}),   # full text (no cut)
                     "strong_first": sorted({m.group(0).lower() for m in STRONG_RE.finditer(shown)}),  # first 6,000 characters
                     "strong_after": sorted({m.group(0).lower() for m in STRONG_RE.finditer(full[MAX_CHARS:])})})
    return run, pd.DataFrame(rows), len(pages) - len(ok)


def model_warning_items(run, run_id):
    """item_ids labelled warning (or target unclear) by ANY model run: old llm_results_*.csv and v3 context_*.csv."""
    hits, files = set(), sorted((run / "tables").glob("llm_results_*.csv")) + sorted((run / "source_context_v3" / "models").glob("context_[AB]__*.csv"))
    for f in files:
        d = pd.read_csv(f, dtype=str).fillna("")
        flag = d["label"].str.strip().str.lower().isin(WARNING_LABELS)
        if "target_unclear" in d:                      # v3 runs: a warning with an unclear target is a candidate too
            flag |= d["target_unclear"].str.lower().eq("true")
        for _, r in d[flag].iterrows():
            hits.add(f"{run_id}|{r['query_id']}|{int(float(r['rank']))}")
    return hits, [f.name for f in files]


def facts(df, n_failed, model_hits, files, out_path):
    w = df["words"]
    name_after = [r.item_id for r in df.itertuples()
                  if r.drug.split()[0].lower() not in r.shown.lower() and r.drug.split()[0].lower() in r.full.lower()]
    trunc = df[(df.strong_after.str.len() > 0) & (df.strong_first.str.len() == 0)]
    lines = [f"pages to label: {len(df)} ({df.query_id.nunique()} products); fetch failures not labelled: {n_failed}",
             f"words: <=10: {(w <= 10).sum()} | 11-149: {((w > 10) & (w < 150)).sum()} | 150-999: "
             f"{((w >= 150) & (w < 1000)).sum()} | >=1000: {(w >= 1000).sum()} | median {w.median():.0f}",
             f"pages longer than {MAX_CHARS} characters (the old model cut; model and annotators now get the full text): {(df.full.str.len() > MAX_CHARS).sum()}",
             f"non-English pages: {df.non_english.sum()}",
             f"pages with a strong warning term in the text: {(df.strong_shown.str.len() > 0).sum()}",
             f"pages labelled warning by any model run: {len(model_hits & set(df.item_id))}  (files: {', '.join(files) or 'none'})",
             "", f"old 6,000-character cut: pages whose warning terms are ONLY after it: {len(trunc)}"]
    lines += [f"  {r.item_id} {r.drug}: {', '.join(r.strong_after)}" for r in trunc.itertuples()]
    lines += [f"medicine name (first word) ONLY after the cut: {len(name_after)}"] + [f"  {x}" for x in name_after]
    lines += ["", "pages with a strong warning term (full text):"]
    lines += [f"  {r.item_id} {r.drug}: {', '.join(r.strong_shown)}" for r in df[df.strong_shown.str.len() > 0].itertuples()]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="fresh_oct")
    ap.add_argument("--set", choices=["prescreen", "all", "second", "practice"], default="all")
    ap.add_argument("--practice_run", default="fresh_oct_form")
    ap.add_argument("--n_second", type=int, default=57)
    ap.add_argument("--n_practice", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--review", action="store_true",
                    help="reviewer copy for the first author: adds 'Compare with another annotator' (adjudication). "
                         "Annotator copies never have it.")
    args = ap.parse_args()

    run, df, n_failed = load_pages(args.run_id)
    out = run / "source_context_v3" / "annotation"; out.mkdir(parents=True, exist_ok=True)
    model_hits, files = model_warning_items(run, args.run_id)
    df["candidate"] = df.item_id.isin(model_hits) | (df.strong_shown.str.len() > 0)

    pre = None
    if args.set == "prescreen":                         # proposals + a quick viewer for the first author
        rows, items = [], []
        for k, r in enumerate(df.itertuples(), 1):
            print(f"\r  checking page {k}/{len(df)}", end="", flush=True)
            h = saved_html(r.page_file, r.url)
            d, why = propose(h, r.full, r.drug)
            rows.append({"item_id": r.item_id, "query_id": r.query_id, "rank": r.rank, "drug": r.drug, "url": r.url,
                         "words": r.words, "proposed": d, "reason": why, "decision": d, "note": ""})
            items.append({"id": r.item_id, "name": r.drug, "url": r.url, "text": r.full, "html": h,
                          "proposed": d, "reason": why})
        print()
        pcsv = out / "prescreen_proposed.csv"
        pd.DataFrame(rows).to_csv(pcsv, index=False)
        html = PRESCREEN.replace("__DATA__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
        (out / "prescreen_viewer.html").write_text(html, encoding="utf-8")
        c = pd.Series([x["proposed"] for x in rows]).value_counts()
        print(f"proposed: " + " | ".join(f"{k} {v}" for k, v in c.items()))
        print(f"written: {pcsv}\n         {out / 'prescreen_viewer.html'}\n"
              f"Check every page in the viewer, download prescreen.csv into {out}, then build the sets.")
        return
    if args.set in ("all", "second"):
        pre = load_prescreen(out)
        if pre is None:
            raise SystemExit("[STOP] No prescreen.csv yet. Run --set prescreen, check the pages, save prescreen.csv in "
                             f"{out}, then build again.")
        missing = set(df.item_id) - set(pre)
        if missing:
            raise SystemExit(f"[STOP] prescreen.csv lacks {len(missing)} pages, e.g. {sorted(missing)[:3]}")
        n_ex = sum(pre[i] == "exclude" for i in df.item_id)
        df = df[[pre[i] != "exclude" for i in df.item_id]].copy()
        print(f"prescreen: {n_ex} pages excluded (no usable content), {sum(pre[i] == 'text' for i in df.item_id)} shown as text, "
              f"{len(df)} pages to annotate")
    if args.set == "all":
        sel = df
        facts(df, n_failed, model_hits, files, out / "facts_all.txt")
    elif args.set == "second":
        cand = df[df.candidate]
        if len(cand) > args.n_second - 20:             # keep at least 20 random pages
            keep = df[df.item_id.isin(model_hits)]
            extra = cand[~cand.item_id.isin(keep.item_id)]
            k = max(0, args.n_second - 20 - len(keep))
            cand = pd.concat([keep, extra.sample(n=min(k, len(extra)), random_state=args.seed)])
        rest = df[~df.item_id.isin(cand.item_id)]
        fill = rest.sample(n=min(args.n_second - len(cand), len(rest)), random_state=args.seed)
        sel = pd.concat([cand.assign(stratum="candidate"), fill.assign(stratum="random")]) \
                .sort_values(["query_id", "rank"])
        sel[["item_id", "query_id", "rank", "drug", "stratum"]].assign(seed=args.seed) \
           .to_csv(out / "second_set_manifest.csv", index=False)
        print(f"second set: {len(sel)} pages = {len(cand)} candidate warning pages + {len(fill)} random (seed {args.seed})")
    else:
        prun, pdf, _ = load_pages(args.practice_run)
        pdf = pdf[~pdf.url.isin(set(df.url)) & (pdf.words >= 30)].sample(frac=1, random_state=args.seed)
        pick = []                                   # practice covers translation and a page shown as text
        ne = pdf[pdf.non_english]
        if len(ne):
            pick.append(ne.index[0])
        for ix, r in pdf.iterrows():
            if ix not in pick and show_as_text(saved_html(r.page_file, r.url), r.full, r.drug)[0]:
                pick.append(ix); break
        rest = [ix for ix in pdf.index if ix not in pick][:max(0, args.n_practice - len(pick))]
        sel = pdf.loc[pick + rest].sort_values(["query_id", "rank"])
        run = prun
        print(f"practice set: {len(sel)} pages from {args.practice_run}, not in {args.run_id}")

    items = []
    for k, r in enumerate(sel.itertuples(), 1):        # clean the saved HTML only for the pages in this set
        print(f"\r  preparing page {k}/{len(sel)}", end="", flush=True)
        h = saved_html(r.page_file, r.url)
        if pre is not None:
            as_text, why = pre[r.item_id] == "text", ("prescreen" if pre[r.item_id] == "text" else "")
        else:
            as_text, why = show_as_text(h, r.full, r.drug)
        items.append({"id": r.item_id, "qid": r.query_id, "rank": r.rank, "name": r.drug, "url": r.url,
              "text": r.full, "html": "" if as_text else h, "asText": as_text, "why": why,
              "nonEn": bool(r.non_english), "words": int(r.words)})
    print()
    order = random.Random(args.seed).sample(range(len(items)), len(items))   # random order, fixed seed
    items = [items[k] for k in order]
    sel = sel.iloc[order]
    sel = sel.assign(shown_as_text=[x["asText"] for x in items], shown_as_text_reason=[x["why"] for x in items])
    build_id = hashlib.sha1(json.dumps(sorted(i["id"] for i in items)).encode()).hexdigest()[:8]
    sel[["item_id", "run_id", "query_id", "rank", "drug", "url", "words", "non_english", "shown_as_text", "shown_as_text_reason"]] \
        .to_csv(out / f"context_items_{args.set}.csv", index=False)
    html = TEMPLATE.replace("__DATA__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/")) \
                   .replace("__GUIDE__", GUIDE_VERSION).replace("__BUILD__", build_id).replace("__SET__", args.set) \
                   .replace("__REVIEW__", "1" if args.review else "0")
    label = {"practice": "practice", "second": "main", "all": "all"}[args.set]
    page = out / (f"annotation_{label}_{len(items)}_pages" + ("_REVIEW" if args.review else "") + ".html")
    page.write_text(html, encoding="utf-8")
    print(f"pages shown as text: {sum(x['asText'] for x in items)} of {len(items)} | " + ", ".join(f"{k}: {v}" for k, v in pd.Series([x['why'] for x in items if x['asText']]).value_counts().items()))
    print(f"items: {len(items)} | build {build_id} | written: {page} ({len(html) / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Webpage Context Labels</title>
<style>
:root{--bg:#f4f3ef;--fg:#2b2a27;--mut:#6e6c66;--line:#dedbd3;--card:#fffefb;--acc:#3d6c99;--on:#3d6c99;--soft:#eef2f7;--ok:#3d7a4f;--todo:#a2662a;--diff:#f7e1dc;
 --h1:#ffe36b;--h2:#9fe0b4;--h3:#ffb38a}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b19;--fg:#e9e7e1;--mut:#a7a59e;--line:#3a3935;--card:#242422;--acc:#8fb3d9;--on:#4b79a6;--soft:#262c33;--ok:#7fbf90;--todo:#d9a46a;--diff:#4a2e2a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
input,button,textarea{font:inherit;color:inherit}
header{position:sticky;top:0;z-index:3;background:var(--card);border-bottom:1px solid var(--line);padding:8px 14px 11px;display:flex;flex-wrap:wrap;gap:7px;align-items:center}
header input{padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg)}#ann{width:105px}#q{width:190px}
button{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer;font-size:14px}
button:hover{border-color:var(--acc)}button:disabled{opacity:.45;cursor:default}
#prog{color:var(--mut);margin-left:auto;font-size:13px}#mode{font-size:12px;color:#fff;background:var(--todo);border-radius:5px;padding:2px 8px;display:none}
#pbar{position:absolute;left:0;right:0;bottom:0;height:3px;background:var(--line)}#pfill{height:100%;width:0;background:var(--acc);transition:width .2s}
main{display:grid;grid-template-columns:minmax(0,1fr) 390px;gap:14px;padding:12px 14px;align-items:start}
.viewer{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.vbar{display:flex;gap:8px;align-items:center;justify-content:space-between;padding:7px 12px;border-bottom:1px solid var(--line);font-size:13px;color:var(--mut)}
.vbar a{color:var(--acc);cursor:pointer;text-decoration:underline}
#frame{display:block;width:100%;height:calc(100vh - 150px);border:0;background:#fff}
#textv{display:none;height:calc(100vh - 150px);overflow:auto;padding:0 0 20px;background:var(--card);font:16px/1.7 Georgia,"Times New Roman",serif}#textv p{margin:0 26px .55em;max-width:70ch}#textv .tnote{position:sticky;top:0;background:#eef2f7;color:#3d4a5c;font:12.5px system-ui,sans-serif;padding:7px 12px;border-bottom:1px solid var(--line);margin:0 0 14px}
mark.hl{color:#111;border-radius:2px;cursor:pointer}mark.s-name{background:var(--h1)}mark.s-detail{background:var(--h1)}mark.s-warn{background:var(--h3)}
#textv.a-name ::selection{background:var(--h1);color:#111}#textv.a-detail ::selection{background:var(--h1);color:#111}#textv.a-warn ::selection{background:var(--h3);color:#111}
aside{position:sticky;top:62px;background:var(--card);border:2px solid #9fb3c8;border-radius:12px;padding:0;box-shadow:0 2px 6px rgba(0,0,0,.06);max-height:calc(100vh - 76px);overflow:auto}
#status{font-size:13px;color:var(--mut);display:flex;justify-content:space-between;gap:6px;align-items:center;margin-bottom:6px}
#status b.ok{color:var(--ok)}#status b.todo{color:var(--todo)}#clear{font-size:12px;padding:2px 7px}
.lbl{color:var(--mut);font-size:12px}h1{font-size:21px;line-height:1.25;margin:0 0 8px}
#unr{font-size:13px;padding:4px 9px}#unr.on{background:var(--on);border-color:var(--on);color:#fff}
.st{margin:10px 0 0;padding:9px 10px;border:1px solid var(--line);border-left:5px solid var(--line);border-radius:8px;cursor:pointer}
.st.act{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc)}.st.off{opacity:.4;pointer-events:none}
.st[data-st=name]{border-left-color:var(--h1)}.st[data-st=detail]{border-left-color:var(--h1)}.st[data-st=warn]{border-left-color:var(--h3)}
.qh{display:flex;gap:8px;font-weight:600;font-size:14px}.qn{flex:none;background:var(--acc);color:#fff;border-radius:5px;font-size:12px;padding:1px 6px;height:20px}
.qh .ok{margin-left:auto;color:var(--ok)}
.hint{font-size:12px;color:var(--acc);margin:4px 0 0 28px;font-weight:600}
.chips{margin:6px 0 0}.chip{display:flex;gap:6px;align-items:flex-start;font-size:13px;padding:3px 4px 3px 8px;border-radius:5px;margin-top:4px;color:#111}
.chip span{flex:1;overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical}.chip button{font-size:12px;padding:0 6px;line-height:18px;background:rgba(255,255,255,.6)}
.chip.s-name{background:var(--h1)}.chip.s-detail{background:var(--h1)}.chip.s-warn{background:var(--h3)}.chip.miss{outline:1px dashed var(--todo)}
.opts{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px}.opts button{font-size:13px;padding:4px 9px}.opts button.on{background:var(--on);border-color:var(--on);color:#fff}
.tags{display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin:4px 0 2px 6px}.tags button{font-size:12px;padding:2px 7px}.tags button.on{background:var(--on);border-color:var(--on);color:#fff}.tl{font-size:12px;color:var(--mut);margin-right:2px}.need{font-size:12px;color:var(--todo);font-weight:600}.nx{font-size:13px;padding:3px 10px;margin-left:4px;background:var(--acc);border-color:var(--acc);color:#fff}
.act-line{margin-top:8px;padding:7px 9px;border-radius:6px;background:var(--soft);font-size:13px;color:var(--fg);display:flex;flex-wrap:wrap;gap:6px;align-items:center}.act-line.todo{color:var(--todo);font-weight:600}.act-line .or{color:var(--mut)}.act-line button{font-size:13px;padding:3px 10px}
#status{flex-wrap:wrap}.striprow{flex-basis:100%}.strip{display:flex;gap:5px;margin-top:6px}.strip{flex-wrap:wrap}.dot{height:24px;padding:0 9px;border-radius:12px;font-size:12px;font-weight:600}.dot.done{background:var(--ok);border-color:var(--ok);color:#fff}.dot.cur{background:var(--acc);border-color:var(--acc);color:#fff}.dot.todo{background:var(--card);border-color:var(--acc);color:var(--acc)}.dot.locked{background:var(--card);color:var(--mut);opacity:.6}.dot.na{opacity:.35;text-decoration:line-through}
.todo-hint{margin-top:4px;font-style:italic}.fin{margin-top:12px;padding:9px 10px;border-radius:8px;background:var(--soft);color:var(--ok);font-weight:600;display:flex;gap:8px;align-items:center;flex-wrap:wrap}.fin button{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
.sub.off{opacity:.45;pointer-events:none}.sub{margin:8px 0 0 6px;padding:6px 8px;border:1px solid var(--line);border-radius:7px;cursor:pointer}.sub.act{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc)}.sh{display:flex;gap:6px;align-items:center;font-size:13px}.sh .ok{margin-left:auto;color:var(--ok)}.sl{background:var(--h1);color:#111;border-radius:4px;font-size:11px;padding:0 5px}.sub .ex{font-size:12px;color:var(--mut);margin:2px 0 0}.st.grp{cursor:default}.mut{color:var(--mut);font-size:12px;font-weight:400}
.other{margin-top:6px;font-size:12px;padding:3px 7px;border-radius:5px;background:var(--soft)}.other.d{background:var(--diff)}
textarea{width:100%;min-height:38px;margin-top:12px;border:1px solid var(--line);border-radius:6px;background:var(--bg);padding:5px 7px}
details{margin:10px 14px 0}summary{cursor:pointer;font-weight:600}.guide{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-top:8px;font-size:14px}
.guide h3{font-size:14px;margin:10px 0 3px}.guide h3:first-child{margin-top:0}.guide ul{margin:0;padding-left:20px}.guide li{margin:2px 0}
.guide .sw{padding:0 4px;border-radius:3px;color:#111}
#drawer{position:fixed;top:0;right:0;bottom:0;width:min(420px,92vw);background:var(--card);border-left:1px solid var(--line);z-index:5;transform:translateX(105%);transition:transform .2s;display:flex;flex-direction:column}
#drawer.open{transform:none}#dhead{padding:12px 14px;border-bottom:1px solid var(--line)}
#chipsf{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}#chipsf button{font-size:12px;padding:3px 8px}#chipsf button.on{background:var(--on);color:#fff;border-color:var(--on)}
#list{overflow:auto;padding:6px 8px 20px}.row{padding:7px 8px;border-radius:6px;cursor:pointer;font-size:13px;border-bottom:1px solid var(--line)}
.row:hover{background:var(--soft)}.row .s{float:right;font-size:12px;color:var(--mut)}.row.cur{outline:2px solid var(--acc)}.mut{color:var(--mut);font-size:12px}
#toast{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);background:var(--fg);color:var(--bg);padding:8px 16px;border-radius:6px;display:none;z-index:9}
#toast a{color:var(--bg);text-decoration:underline;cursor:pointer;margin-left:10px;font-weight:600}
@media (max-width:950px){main{grid-template-columns:1fr}aside{position:static;max-height:none}#frame,#textv{height:70vh}}
.ah{background:#e9eff6;border-bottom:2px solid #9fb3c8;padding:9px 12px;display:flex;justify-content:space-between;align-items:center;gap:8px;position:sticky;top:0;z-index:1}
.ah b{font-size:15px}#status{font-size:12px;color:var(--mut);margin:0;display:block}
.med{padding:10px 12px 9px;border-bottom:1px solid var(--line)}.ml{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}.med h1{margin:0}
.mrow{display:flex;justify-content:space-between;align-items:center;gap:6px;margin-top:6px}.mrow button{font-size:12px;padding:3px 9px}#unr.on{background:var(--on);border-color:var(--on);color:#fff}
.lnk{border:0;background:none;color:var(--acc);font-weight:600;font-size:12px;padding:0 2px;cursor:pointer}
.pills{padding:8px 12px;border-bottom:1px solid var(--line)}
#qs .sec{border-bottom:1px solid var(--line);padding:9px 12px}
#qs .sh{display:flex;gap:8px;align-items:center;font-weight:600}#qs .nb{flex:none;min-width:26px;height:20px;border-radius:5px;background:var(--acc);color:#fff;font-size:11.5px;text-align:center;line-height:20px;padding:0 4px}
#qs .done .nb{background:var(--ok)}#qs .done{background:#fbfbf8}#qs .ed{margin-left:auto}
#qs .sum{font-size:13px;margin:3px 0 0 34px}#qs .sum i{background:var(--h1);font-style:normal;padding:0 4px;border-radius:3px;color:#111}#qs .sum i.w{background:var(--h3)}#qs .sum .no{color:var(--mut)}
#qs .act{border-left:5px solid var(--acc);background:#f3f7fc;padding-left:9px}#qs .q{font-size:13.5px;margin:4px 0 0 34px}#qs .act .ex{margin:3px 0 0 34px;font-size:12px;line-height:1.4;color:var(--mut)}#qs .act .ex b{font-weight:600;color:var(--fg)}
#qs .act .chips,#qs .act .act-line,#qs .act .opts,#qs .act .other{margin-left:34px}
#qs .lock{color:var(--mut)}#qs .lock .nb{background:#b8c2cc}#qs .lk{margin-left:auto;font-weight:400;font-size:12px}#qs .lock.open{cursor:pointer}#qs .lock.open .lk{color:var(--acc);font-weight:600}#qs .lock.open .nb{background:var(--acc)}
.foot{background:#e9eff6;border-top:2px solid #9fb3c8;padding:9px 12px;position:sticky;bottom:0}.foot textarea{margin-top:6px;background:#fff}
.foot .left{color:var(--todo);font-weight:600;font-size:13px}.fin{margin:0;padding:0;background:none;display:flex;justify-content:space-between;align-items:center;gap:8px}
@media (prefers-color-scheme:dark){.ah,.foot{background:#2a3038}#qs .act{background:#26303a}#qs .done{background:#232320}}
.dl{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}.more{position:relative}#moremenu{position:absolute;top:110%;left:0;z-index:6;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px;display:flex;flex-direction:column;gap:4px;min-width:300px;white-space:nowrap;box-shadow:0 4px 12px rgba(0,0,0,.12)}#moremenu[hidden]{display:none}#moremenu button{text-align:left}
.wq{font-size:12.5px;font-weight:600;margin:8px 0 3px 6px}.tags.col{flex-direction:column;align-items:stretch}.tags.col button{text-align:left}.tags button:disabled{opacity:.4;cursor:not-allowed}
.pic{display:block;margin:6px 0 0 34px;font-size:12.5px;color:var(--fg)}
</style></head><body>
<header translate="no" class="notranslate">
<input id="ann" placeholder="Your code" aria-label="Annotator code">
<button id="prev">&larr; Previous page</button><button id="next">Next page &rarr;</button><button id="todo">Next unfinished page</button>
<input id="q" placeholder="&#128269; Search medicine, Q-ID, site" aria-label="Search">
<button id="save" class="dl">&#11015; Download my annotations</button>
<span class="more"><button id="moreb" title="More options">More &#9662;</button><span id="moremenu" hidden><button id="loadb">Continue from a downloaded file&hellip;</button><button id="cmpb">Compare with another annotator&hellip;</button></span></span>
<input id="load" type="file" accept=".csv" hidden><input id="cmp" type="file" accept=".csv" hidden><button id="cmpx" hidden>Exit compare</button>
<span id="mode"></span><span id="prog"></span><div id="pbar"><div id="pfill"></div></div></header>

<details id="guide" translate="no" class="notranslate"><summary>Guide (read once before you start)</summary><div class="guide">
<h3>How to work</h3><ul>
<li>Highlight with your mouse. The highlight goes to the <b>active step</b> (outlined on the right). You can add more.</li>
<li>Steps open one after another. Each step needs a highlight or a <b>Not on this page</b> answer. The line at the bottom of the step says what to do next; the card shows what is done and what is left.</li>
<li>Click a highlight to remove it.</li>
<li>Text only: names or details shown only in pictures or logos do not count. If the name is only in a picture, tick <b>The name is shown only in a picture</b>.</li>
<li>Mentioned more than once? One highlight is enough, in the main heading if possible.</li>
<li><b>Ctrl+F</b> finds a word. Not in English? Right-click &rarr; <b>Translate to English</b>. Names and numbers often stay in the original ("CETAL SINUS 20TAB", "500 mg"): highlight those before translating where you can.</li>
<li>Only an error, "verify you are human", a cookie notice or menus? Click <b>Page can't be read</b>.</li></ul>
<h3><span class="sw" style="background:var(--h1)">1</span> Name</h3><ul>
<li>Highlight the medicine name. Small differences are fine ("Arh L" = "ARH-L").</li></ul>
<h3><span class="sw" style="background:var(--h1)">2</span> Details: only about this medicine (its heading, description or table)</h3><ul>
<li><b>a Ingredient:</b> the active substance, not the brand. Look after <i>Composition, Contains, Active ingredient, Generic name</i>, or take the word before "x mg". Not sure? Highlight it and add a note.</li>
<li><b>b Strength:</b> a number with a unit: xx mg; xx mg/yy mg; xx mg/zz ml; for creams, gels and ointments xx % or xx mg/g. For combination medicines, numbers like 20/120 or 80/480 count even without units.</li>
<li><b>c Form:</b> e.g. tablet, capsule, syrup, injection, cream, gel, ointment, drops.</li>
<li><b>d Manufacturer:</b> only after <i>Manufactured by, Manufacturer, Marketed by</i>. Highlight the word with the company or address. A seller's profile does not count.</li></ul>
<h3><span class="sw" style="background:var(--h3)">3</span> Warnings</h3><ul>
<li>Highlight the <b>whole sentence</b> saying medicines are fake, falsified, counterfeit, substandard, recalled, seized, unregistered or banned.</li>
<li>Only the page's own content: <b>not related news, related products, ads or menus.</b></li>
<li>Not warnings: side effects, storage, "100% genuine", "prescription required".</li>
<li><b>b. Is this warning about the searched medicine?</b> <i>Yes</i>: it names this medicine or its batch (or clearly all products with its ingredient) &middot; <i>No</i>: another medicine, or medicines in general &middot; <i>Only the ingredient is named</i>: e.g. "fake amoxicillin".</li>
<li><b>c. Issued by a named authority?</b> Yes if a named regulator, ministry or WHO issued it ("MFDA has alerted&hellip;").</li></ul>
<h3>Saving</h3><ul><li>Your answers are kept in this browser. Click <b>&#11015; Download my annotations</b> now and then, and when you finish, and send that file.</li><li>Changed computer or browser? <b>More &#9662; &rarr; Continue from a downloaded file</b>.</li></ul>
</div></details>

<main>
<div class="viewer"><div class="vbar notranslate" translate="no"><span id="meta"></span><span id="tview" class="mut"></span></div>
<iframe id="frame" title="Saved webpage"></iframe><div id="textv"></div></div>
<aside translate="no" class="notranslate acard">
<div class="ah"><b>Annotation card</b><span id="status"></span></div>
<div class="med"><div class="ml">Medicine searched</div><h1 id="name"></h1>
 <div class="mrow"><span id="dom" class="mut"></span><span><button id="unr">Page can't be read</button> <button id="clear" class="lnk">Clear</button></span></div><div id="unrO"></div></div>
<div id="pillrow" class="pills"></div>
<div id="qs"></div>
<div class="foot"><div id="footmsg"></div><textarea id="note" placeholder="Note (optional)"></textarea></div>
</aside></main>
<div id="drawer" translate="no" class="notranslate"><div id="dhead"><b>Search</b> <button id="dclose" style="float:right">Close</button><div id="chipsf"></div></div><div id="list"></div></div>
<div id="toast" translate="no" class="notranslate"></div>
<script>
const items=__DATA__, GUIDE="__GUIDE__", BUILD="__BUILD__", SET="__SET__", REVIEW="__REVIEW__"==="1";
if(!REVIEW){document.getElementById("cmpb").remove()}
const KEY="ctx8_"+SET+"_"+BUILD, KEY_R=KEY+"_resolved", KEY_O=KEY+"_other";
const DET=["ingredient","strength","form","manufacturer"], STEPS=["name"].concat(DET,["warn"]);
const NONE={name:"Not on this page",ingredient:"Not on this page",strength:"Not on this page",form:"Not on this page",manufacturer:"Not on this page",warn:"No warning on this page"};
const DLAB={ingredient:"Ingredient (composition)",strength:"Strength",form:"Form",manufacturer:"Manufacturer \u2014 only where the page says so"};
const DEX={ingredient:"the active substance, not the brand &middot; look after <b>Composition, Contains, Active ingredient, Generic name</b>",
 strength:"a number with a unit: xx mg; xx mg/yy mg; xx mg/zz ml &middot; creams, gels: xx % or xx mg/g &middot; combinations: 20/120, 80/480 (units not needed)",
 form:"tablet, capsule, syrup, injection, cream, gel, ointment, drops&hellip;",
 warn:"highlight the whole sentence &middot; only the page's own content: not related news, related products, ads or menus",
 manufacturer:"only after <b>Manufactured by, Manufacturer, Marketed by</b> &middot; highlight the word <b>with</b> the company"};
const cls=st=>DET.includes(st)?"detail":st;
const GOLBL={name:"1 Name",ingredient:"2a Ingredient",strength:"2b Strength",form:"2c Form",manufacturer:"2d Manufacturer",warn:"3 Warning"};
const NOUN={name:"the name",ingredient:"the ingredient",strength:"the strength",form:"the form",manufacturer:"the manufacturer label",warn:"each warning sentence"};
const SHORT={name:"1",ingredient:"a",strength:"b",form:"c",manufacturer:"d",warn:"3"};
const TGT=[["this","Yes \u2013 it names this medicine or its batch"],["other_general","No \u2013 another medicine, or medicines in general"],["cant_tell","Only the ingredient is named"]];
const OFF=[["yes","Yes"],["no","No"]];
const MAXQ=600;
const $=id=>document.getElementById(id);
function ls(k,v){try{if(v===undefined)return JSON.parse(localStorage.getItem(k)||"{}");localStorage.setItem(k,JSON.stringify(v))}catch(e){return v===undefined?{}:null}}
let cmpOn=false;try{cmpOn=localStorage.getItem(KEY_O+"_on")==="1"}catch(e){}
let OWN=ls(KEY),RES=ls(KEY_R),OTHER=ls(KEY_O),cmp=cmpOn&&Object.keys(OTHER).length>0,L=cmp?RES:OWN,i=0,filt="all",openSig="",origText="",active="name",lastMark=0,undo=null;
let otherName="";try{otherName=localStorage.getItem(KEY_O+"_name")||""}catch(e){}
try{$("ann").value=localStorage.getItem("ctx8_ann")||""}catch(e){}
try{if(!localStorage.getItem("ctx8_guide_seen"))$("guide").open=true}catch(e){$("guide").open=true}
$("guide").addEventListener("toggle",e=>{if(!e.target.open){try{localStorage.setItem("ctx8_guide_seen","1")}catch(_){}}});
function store(){ls(cmp?KEY_R:KEY,L)}
function toast(m,ms,act){const t=$("toast");t.innerHTML=esc(m)+(act?'<a id="tact">'+esc(act[0])+'</a>':"");t.style.display="block";if(act)$("tact").onclick=()=>{t.style.display="none";act[1]()};clearTimeout(toast.h);toast.h=setTimeout(()=>t.style.display="none",ms||1600)}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function cur(){const id=items[i].id;return L[id]=L[id]||{}}
// ---- answers
const marks=(l,st)=>((l&&l.marks)||[]).filter(m=>m.st===st);
const has=(l,st)=>marks(l,st).length>0, no=(l,st)=>!!(l&&l["no_"+st]);
const mOK=m=>m.st!=="warn"||(!!m.tg&&!!m.of);
function stepDone(l,st){if(DET.includes(st)&&no(l,"name"))return true;return no(l,st)||(has(l,st)&&marks(l,st).every(mOK))}
function complete(l){if(!l)return false;if(l.unread)return true;return STEPS.every(st=>stepDone(l,st))}
const LBL={name:"1 name",ingredient:"2a ingredient",strength:"2b strength",form:"2c form",manufacturer:"2d manufacturer",warn:"3 warning"};
function missing(l){l=l||{};if(l.unread)return [];const out=STEPS.filter(st=>!stepDone(l,st)).map(st=>LBL[st]);
 if(marks(l,"warn").some(m=>!mOK(m))&&!out.includes(LBL.warn))out.push(LBL.warn);return out}
const P=(l,k)=>has(l,k);
function vals(l){l=l||{};if(l.unread)return {R:"no"};const w=marks(l,"warn"),pv=k=>no(l,"name")?"n/a":has(l,k)?"yes":stepDone(l,k)?"no":"";
 return {R:"yes",P1:has(l,"name")?"yes":no(l,"name")?"no":"",P2:pv("ingredient"),P3:pv("strength"),P4:pv("form"),P5:pv("manufacturer"),
  Wthis:w.filter(m=>m.tg==="this").length,Wother:w.filter(m=>m.tg==="other_general").length,Wct:w.filter(m=>m.tg==="cant_tell").length,Woff:w.filter(m=>m.of==="yes").length,W:no(l,"warn")?"no":w.length?"yes":""}}
function sig(l){const v=vals(l);if(v.R==="no")return "unreadable";return [v.P1,v.P2,v.P3,v.P4,v.P5,v.W,v.Wthis,v.Wother,v.Wct,v.Woff].join("|")}
function differs(id){return cmp&&OTHER[id]&&sig(OTHER[id])!==sig(OWN[id])}
function nfin(){return items.filter(x=>complete(L[x.id])).length}
function nextOpen(l,from){const k0=from?STEPS.indexOf(from)+1:0;for(let k=k0;k<STEPS.length;k++)if(!stepDone(l,STEPS[k]))return STEPS[k];for(const st of STEPS)if(!stepDone(l,st))return st;return "warn"}
function touch(){const l=cur(),now=new Date().toISOString();if(complete(l)&&!l.first)l.first=now;l.last=now;store()}
function finishedNow(was){const l=cur(),n=nfin();if(!was&&complete(l)&&n===items.length)toast("All pages done! Click \u2b07 Download my annotations and send the file.",8000);else if(!was&&complete(l)&&n%20===0)toast(n+" pages done. Click \u2b07 Download my annotations to keep a copy.",4000)}
// ---- right panel
const qt=m=>m.t.replace(/\s+/g," ").trim();
function summ(o,st){if(!o)return "-";if(o.unread)return "page can't be read";
 if(st==="name")return has(o,"name")?marks(o,"name").map(m=>"\u201c"+qt(m).slice(0,60)+"\u201d").join(" "):no(o,"name")?NONE.name:"-";
 if(st==="detail"){if(no(o,"name"))return "n/a";return DET.map(k=>DLAB[k].split(" ")[0].toLowerCase()+": "+(has(o,k)?marks(o,k).map(m=>"\u201c"+qt(m).slice(0,40)+"\u201d").join(" "):no(o,k)?"no":"-")).join(" · ")}
 const w=marks(o,"warn");return w.length?w.map(m=>"\u201c"+qt(m).slice(0,50)+"\u201d \u2192 "+((TGT.find(x=>x[0]===m.tg)||["","?"])[1])+(m.of==="yes"?", regulator":"")).join("; "):no(o,"warn")?NONE.warn:"-"}
function otherLine(st){if(!cmp||!OTHER[items[i].id])return "";const o=OTHER[items[i].id],m=cur();
 if(st==="unread"){if(!o.unread&&!m.unread)return "";return '<div class="other'+(!!o.unread!==!!m.unread?" d":"")+'">'+esc(otherName||"Other")+': '+(o.unread?"page can't be read":"page readable")+'</div>'}
 const a=summ(o,st),b=summ(m,st);return '<div class="other'+(a!==b?" d":"")+'">'+esc(otherName||"Other")+': '+esc(a)+'</div>'}
function chip(m,st){let h='<div class="chip s-'+cls(st)+(m.miss?" miss":"")+'" title="'+(m.miss?"Not found on the page any more":"")+'"><span>'+esc(qt(m))+'</span><button data-rm="'+m.id+'" title="Remove">&times;</button></div>';
 if(st==="warn"){const nn=no(cur(),"name");h+='<div class="wq">b. Is this warning about <i>'+esc(items[i].name)+'</i>?</div><div class="tags col">'+TGT.map(([k,t])=>'<button data-tg="'+k+'" data-mid="'+m.id+'" class="'+(m.tg===k?"on":"")+'"'+(k==="this"&&nn?' disabled title="The name is not on this page"':'')+'>'+t+'</button>').join("")+'</div><div class="wq">c. Is it issued by a named authority (e.g. MFDA, NAFDAC, WHO)?</div><div class="tags">'+OFF.map(([k,t])=>'<button data-of="'+k+'" data-mid="'+m.id+'" class="'+(m.of===k?"on":"")+'">'+t+'</button>').join("")+(mOK(m)?'':'<span class="need">answer b and c</span>')+'</div>'}
 return h}
function nextLabel(l,st){const nx=STEPS[STEPS.indexOf(st)+1];if(!nx)return null;if(DET.includes(nx)&&!has(l,"name"))return "warn";return nx}
function ACTION(l,st){const nx=nextLabel(l,st),go=nx?'<button class="nx" data-next="'+nx+'">Next: '+GOLBL[nx]+' &rarr;</button>':'';
 if(no(l,st))return '<div class="act-line">Answered: '+NONE[st].toLowerCase()+'. <button data-none="'+st+'">Undo</button> '+go+'</div>'+(st==="name"?picFlag(l):'');
 if(!has(l,st))return '<div class="act-line">Highlight it on the page <span class="or">&mdash; or &mdash;</span> <button data-none="'+st+'">'+NONE[st]+'</button></div>';
 if(st==="warn"&&marks(l,"warn").some(m=>!mOK(m)))return '<div class="act-line todo">Answer b and c for each warning.</div>';
 const n=marks(l,st).length;return '<div class="act-line">&#10003; '+n+' highlighted. '+(st==="warn"?'Add more if there are other warnings.':'Add more, or '+go)+'</div>'}
function isOpen(l,st){if(l.unread)return false;if(DET.includes(st)&&no(l,"name"))return false;
 if(stepDone(l,st))return true;const k=STEPS.indexOf(st);return STEPS.slice(0,k).every(x=>stepDone(l,x))}
function picFlag(l){return no(l,"name")?'<label class="pic"><input type="checkbox" id="picbox"'+(l.pic?" checked":"")+'> The name is shown only in a picture</label>':''}
function noneTag(l,st){if(no(l,st))return '<div class="opts"><button class="on" data-none="'+st+'" title="Click to undo">'+NONE[st]+' &#10003;</button></div>';return isOpen(l,st)&&!stepDone(l,st)?'<div class="mut todo-hint">Click here to answer</div>':''}
const PILL={name:"1 Name",ingredient:"2a Ingredient",strength:"2b Strength",form:"2c Form",manufacturer:"2d Manufacturer",warn:"3 Warning"};
function strip(l){if(l.unread)return '<span class="strip"><span class="dot locked">Page can\'t be read</span></span>';
 return '<span class="strip">'+STEPS.map(st=>{const skip=DET.includes(st)&&no(l,"name"),open=isOpen(l,st),
  c=skip?"na":st===active&&open?"cur":stepDone(l,st)?"done":open?"todo":"locked";
  return '<button class="dot '+c+'" data-jump="'+st+'" title="'+(skip?"skipped: name not on this page":open?PILL[st]:"opens after the step before")+'"'+(open?"":" disabled")+'>'+PILL[st]+'</button>'}).join("")+'</span>'}
const QTXT={name:N=>"Highlight the name "+N+" (main heading if possible)",ingredient:N=>"Highlight the ingredient of "+N,strength:N=>"Highlight the strength of "+N,
 form:N=>"Highlight the form of "+N,manufacturer:N=>"Highlight the manufacturer of "+N+" (only where the page says so)",warn:()=>"Highlight any warning about fake, substandard, recalled or seized medicines"};
function doneSum(l,st){if(no(l,st))return '<span class="no">'+NONE[st]+'</span>';
 return marks(l,st).map(m=>'<i class="'+(st==="warn"?"w":"")+'">'+esc(qt(m).slice(0,70))+(qt(m).length>70?"&hellip;":"")+'</i>'+(st==="warn"?' <span class="no">&rarr; '+((TGT.find(x=>x[0]===m.tg)||["","?"])[1])+(m.of==="yes"?", regulator":"")+'</span>':'')).join(" ")}
function row(l,st,N){const un=!!l.unread,skip=DET.includes(st)&&no(l,"name"),open=isOpen(l,st),act=!un&&open&&active===st,done=stepDone(l,st)&&!skip,lab=PILL[st].split(" "),nb=lab[0],nm=lab.slice(1).join(" ");
 if(act){let h='<div class="sec act" data-st="'+st+'"><div class="sh"><span class="nb">'+nb+'</span>'+nm+'</div><div class="q">'+QTXT[st](N)+'</div>'+(DEX[st]?'<div class="ex">'+DEX[st]+'</div>':'')+
   '<div class="chips">'+marks(l,st).map(m=>chip(m,st)).join("")+'</div>'+ACTION(l,st);
  return h+otherLine(DET.includes(st)?"detail":st)+'</div>'}
 if(done)return '<div class="sec done" data-st="'+st+'"><div class="sh"><span class="nb">'+nb+'</span>'+nm+'<button class="lnk ed" data-jump="'+st+'">edit</button></div><div class="sum">'+doneSum(l,st)+'</div>'+(st==="name"?picFlag(l):'')+otherLine(DET.includes(st)?"detail":st)+'</div>';
 const k=STEPS.indexOf(st),firstLocked=!skip&&!open&&STEPS.slice(0,k).filter(x=>!(DET.includes(x)&&no(l,"name"))).every((x,j,a)=>j<a.length-1?stepDone(l,x):true);
 const tag=skip&&!un?"skipped":"";
 return '<div class="sec lock'+(open?" open":"")+'" data-st="'+st+'"><div class="sh"><span class="nb">'+nb+'</span>'+nm+'<span class="lk">'+tag+'</span></div></div>'}
function render(){const it=items[i],l=L[it.id]||{},N="<i>"+esc(it.name)+"</i>",un=!!l.unread;
 if(!un&&!isOpen(l,active))active=nextOpen(l);
 $("unr").className=un?"on":"";$("unrO").innerHTML=otherLine("unread");$("dom").textContent=it.url.split("/")[2]||"";
 $("pillrow").innerHTML=strip(l);
 $("qs").innerHTML=un?'<div class="sec lock"><div class="sh">Page can\'t be read. Nothing else to answer.</div></div>':STEPS.map(st=>row(l,st,N)).join("");
 const ms=missing(l);
 $("footmsg").innerHTML=complete(l)?(nfin()===items.length?'<div class="fin">&#10003; All pages are done <button id="finish2" class="dl">&#11015; Download my annotations</button></div>':'<div class="fin">&#10003; This page is done <button id="finish">Go to next page &rarr;</button></div>'):'<div class="left">Still to answer: '+esc(ms.map(x=>x.split(" ")[0]).join(", "))+'</div>';
 $("unr").onclick=()=>{const l=cur(),was=complete(l);l.unread=l.unread?0:1;touch();finishedNow(was);render()};
 $("qs").querySelectorAll(".sec.lock.open").forEach(c=>c.onclick=()=>{active=c.dataset.st;render();paintSel()});
 $("qs").querySelectorAll("[data-none]").forEach(b=>b.onclick=()=>{const l=cur(),was=complete(l),st=b.dataset.none;l["no_"+st]=l["no_"+st]?0:1;if(!l["no_"+st])delete l["no_"+st];active=l["no_"+st]?nextOpen(l,st):st;touch();finishedNow(was);render();paintSel()});
 if($("rest"))$("rest").onclick=()=>{const l=cur(),was=complete(l);DET.forEach(k=>{if(!has(l,k))l["no_"+k]=1});active=nextOpen(l,"manufacturer");touch();finishedNow(was);render();paintSel()};
 if($("finish"))$("finish").onclick=()=>$("todo").click();if($("finish2"))$("finish2").onclick=()=>$("save").click();
 const setM=(b,f)=>{const l=cur(),was=complete(l),m=(l.marks||[]).find(x=>x.id===b.dataset.mid);if(!m)return;f(m);touch();finishedNow(was);render();paintSel()};
 if($("picbox"))$("picbox").onchange=e=>{const l=cur();l.pic=e.target.checked?1:0;touch()};
 $("qs").querySelectorAll("[data-tg]").forEach(b=>b.onclick=()=>setM(b,m=>{m.tg=b.dataset.tg}));
 $("qs").querySelectorAll("[data-of]").forEach(b=>b.onclick=()=>setM(b,m=>{m.of=b.dataset.of}));
 $("qs").querySelectorAll("[data-rm]").forEach(b=>b.onclick=()=>removeMark(b.dataset.rm));
 $("qs").querySelectorAll("[data-next]").forEach(b=>b.onclick=e=>{e.stopPropagation();active=b.dataset.next;render();paintSel()});
 document.querySelectorAll("#qs [data-jump],#pillrow [data-jump]").forEach(b=>b.onclick=e=>{e.stopPropagation();active=b.dataset.jump;render();paintSel()});
 const n=nfin();$("prog").textContent=n+" / "+items.length+" done";$("pfill").style.width=(100*n/items.length)+"%";
 $("status").innerHTML='Page '+(i+1)+' of '+items.length+' &middot; '+(complete(l)?'<b class="ok">&#10003; done</b>'+(l.edits?" (edited "+l.edits+"&times;)":""):'<b class="todo">not done yet</b>')+(differs(it.id)?' &middot; <b class="todo">answers differ</b>':'');
 $("mode").style.display=cmp?"inline":"none";$("mode").textContent=cmp?"Compare mode: "+(otherName||"other file"):"";$("cmpx").hidden=!cmp;$("loadb").disabled=cmp;
 if($("drawer").classList.contains("open"))list()}
$("clear").onclick=()=>{const id=items[i].id,o=L[id]||{};L[id]={sec:o.sec,tv:o.tv,tr:o.tr,edits:o.edits,first:o.first,scp:o.scp,sct:o.sct};store();active="name";show(true)};
// ---- marking: works the same on the saved page (iframe) and on the saved text
const HCSS="mark.hl{color:#111!important;border-radius:2px;cursor:pointer}mark.s-name{background:#ffe36b!important}mark.s-detail{background:#ffe36b!important}mark.s-warn{background:#ffb38a!important}";
const SELC={name:"#ffe36b",detail:"#ffe36b",warn:"#ffb38a"};
function frameDoc(){try{const d=$("frame").contentDocument;return d&&d.body?d:null}catch(e){return null}}
function paintSel(){const st=cur().unread?"":cls(active);$("textv").className=st?"a-"+st:"";try{const d=$("frame").contentDocument;let s=d.getElementById("hlsel");if(!s){s=d.createElement("style");s.id="hlsel";(d.head||d.documentElement).appendChild(s)}s.textContent=st?"::selection{background:"+SELC[st]+";color:#111}":""}catch(e){}}
function offsetOf(doc,root,node,off){const r=doc.createRange();r.setStart(root,0);r.setEnd(node,off);return r.toString().length}
function rangeAt(doc,root,s,e){const w=doc.createTreeWalker(root,NodeFilter.SHOW_TEXT);let pos=0,n,r=doc.createRange(),st=false;
 while((n=w.nextNode())){const len=n.data.length;if(!st&&s<pos+len){r.setStart(n,s-pos);st=true}if(st&&e<=pos+len){r.setEnd(n,e-pos);return r}pos+=len}return null}
function wrap(doc,range,st,id){let root=range.commonAncestorContainer;if(root.nodeType===3)root=root.parentNode;
 const w=doc.createTreeWalker(root,NodeFilter.SHOW_TEXT),parts=[];let n;
 while((n=w.nextNode())){if(!range.intersectsNode(n))continue;const a=n===range.startContainer?range.startOffset:0,b=n===range.endContainer?range.endOffset:n.data.length;
  if(b>a&&n.data.slice(a,b).trim()&&!/^(STYLE|SCRIPT|NOSCRIPT|TITLE|TEXTAREA|OPTION)$/.test(n.parentNode.nodeName))parts.push([n,a,b])}
 for(const [node,a,b] of parts){let mid=node;if(a>0)mid=node.splitText(a);if(b-a<mid.data.length)mid.splitText(b-a);
  const m=doc.createElement("mark");m.className="hl s-"+cls(st);m.dataset.mid=id;m.title=LBL[st]+" (click to remove)";mid.parentNode.insertBefore(m,mid);m.appendChild(mid)}}
function unwrap(doc,id){doc.querySelectorAll('mark.hl[data-mid="'+id+'"]').forEach(m=>{const p=m.parentNode;while(m.firstChild)p.insertBefore(m.firstChild,m);p.removeChild(m);p.normalize()})}
function restore(vd){const l=L[items[i].id]||{};let txt=null;
 for(const m of (l.marks||[])){if(m.v!==vd.v)continue;if(txt===null)txt=vd.root.textContent;let s=m.s,e=m.e;
  if(txt.slice(s,e)!==m.t){const k=txt.indexOf(m.t);if(k<0){m.miss=1;continue}s=k;e=k+m.t.length}delete m.miss;
  const r=rangeAt(vd.doc,vd.root,s,e);if(r)wrap(vd.doc,r,m.st,m.id)}}
function onSel(doc,root,v){const l=cur();const sel=doc.getSelection();if(!sel||sel.isCollapsed||!sel.rangeCount)return;
 const r=sel.getRangeAt(0);if(!root.contains(r.commonAncestorContainer))return;const t=r.toString();
 if(t.replace(/\s+/g,"").length<2)return;
 if(l.unread){toast("Page marked as can't be read");sel.removeAllRanges();return}
 if(t.length>MAXQ){toast("Highlight a shorter part: one sentence is enough",2500);sel.removeAllRanges();return}
 let st=active;if(!isOpen(l,st))st=nextOpen(l);
 const was=complete(l),id=Date.now().toString(36)+Math.random().toString(36).slice(2,6);
 const s=offsetOf(doc,root,r.startContainer,r.startOffset),all=root.textContent;
 let y=null;try{const b=r.getBoundingClientRect(),win=doc.defaultView,sc=v==="page"?doc.documentElement:root,top=v==="page"?b.top+win.scrollY:b.top-root.getBoundingClientRect().top+root.scrollTop;y=Math.round(1000*top/Math.max(1,sc.scrollHeight))/10}catch(e){}
 wrap(doc,r,st,id);sel.removeAllRanges();lastMark=Date.now();
 (l.marks=l.marks||[]).push({id:id,st:st,t:t,s:s,e:s+t.length,pre:all.slice(Math.max(0,s-60),s),post:all.slice(s+t.length,s+t.length+60),y:y,v:v,tr:l.tr?1:0});delete l["no_"+st];
 active=st;touch();finishedNow(was);render();paintSel();toast("Added to "+GOLBL[st],1200)}
function removeMark(id){const l=cur(),k=(l.marks||[]).findIndex(m=>m.id===id);if(k<0)return;const m=l.marks.splice(k,1)[0];
const un2=x=>{unwrap(document,x);const d=frameDoc();if(d)unwrap(d,x)};un2(id);
 if(m.st==="name"&&!has(l,"name")){const gone=l.marks.filter(x=>DET.includes(x.st));gone.forEach(x=>un2(x.id));l.marks=l.marks.filter(x=>!DET.includes(x.st));DET.forEach(k=>delete l["no_"+k])}
 touch();render();undo={item:items[i].id,m:m};toast("Mark removed",3500,["Undo",()=>{if(undo&&undo.item===items[i].id){const l=cur();(l.marks=l.marks||[]).push(undo.m);delete l["no_"+undo.m.st];touch();show(true)}}])}
function onMarkClick(e){const m=e.target.closest&&e.target.closest("mark.hl");if(!m)return;e.preventDefault();e.stopPropagation();if(Date.now()-lastMark<500)return;removeMark(m.dataset.mid)}
$("textv").addEventListener("mouseup",()=>setTimeout(()=>{if($("tvb"))onSel(document,$("tvb"),"text")},0));
$("textv").addEventListener("click",onMarkClick);
function scrolled(el,bottom,k){const l=cur(),pct=Math.min(100,Math.round(100*bottom/Math.max(1,el.scrollHeight)));if(!(l[k]>=pct)){l[k]=pct}}
$("textv").addEventListener("scroll",()=>{const t=$("textv");scrolled(t,t.scrollTop+t.clientHeight,"sct")},{passive:true});
// ---- page viewer: the saved page (scripts off), or the saved text
let textMode=false;
function paras(t){
 // display only: line breaks and spaces are added, nothing is removed (highlight mapping ignores spacing)
 t=String(t).replace(/\r/g,"")
  .replace(/(?<=[a-z]{3}|[0-9)])(?=[A-Z][a-z]{3})/g,"\n")     // glued pieces: "AlamyHi", "customersCreate"
  .replace(/(?<=[a-z]{2}[.!?])(?=[A-Z])/g,"\n")                // "childbirth.Investigations"
  .replace(/(?<=[a-z]):(?=[A-Z])/g,": ");                      // "clearly:Poor" -> "clearly: Poor"
 const out=[];
 for(const line of t.split(/\n+/)){const x=line.trim();if(!x)continue;
  const sents=x.split(/(?<=[.!?])\s+(?=["“(]?[A-Z0-9])/);let c="";
  for(const z of sents){if(c&&(c.length+z.length>320)){out.push(c);c=""}c+=(c?" ":"")+z}
  if(c)out.push(c)}
 return out}
function showView(){const it=items[i];const useText=!it.html;$("frame").style.display=useText?"none":"block";$("textv").style.display=useText?"block":"none";
 $("tview").textContent="";
 if(useText){$("textv").innerHTML='<div class="tnote" translate="no">The saved page could not be displayed, so its text is shown. Highlight in the text as usual.</div><div id="tvb">'+(it.text.trim()?paras(it.text).map(x=>'<p dir="auto">'+esc(x)+'</p>').join(""):"<p><i>(no text saved)</i></p>")+'</div>';cur().tv=1;restore({doc:document,root:$("tvb"),v:"text"});paintSel()}}
let frameInit=null;
function clearOverlays(d){let n=0;try{const w=d.defaultView,vw=w.innerWidth,vh=w.innerHeight;
 for(const el of d.body.querySelectorAll("*")){const cs=w.getComputedStyle(el);if(cs.position!=="fixed"&&cs.position!=="absolute")continue;
  if(cs.position==="absolute"&&el.parentElement!==d.body)continue;const r=el.getBoundingClientRect();
  if(r.width*r.height<0.5*vw*vh)continue;if((el.innerText||"").replace(/\s+/g,"").length>40)continue;el.style.setProperty("display","none","important");n++}
 for(const x of [d.documentElement,d.body]){const cs=w.getComputedStyle(x);if(cs.overflow==="hidden"||cs.overflowY==="hidden")x.style.setProperty("overflow","auto","important");
  if(cs.opacity==="0")x.style.setProperty("opacity","1","important");if(cs.visibility==="hidden")x.style.setProperty("visibility","visible","important")}}catch(e){}return n}
function initFrame(){try{const d=$("frame").contentDocument;if(!d||!d.body||d.__hlInit)return;d.__hlInit=1;
 const s=d.createElement("style");s.textContent=HCSS;(d.head||d.documentElement).appendChild(s);
 d.addEventListener("click",e=>{if(e.target.closest("mark.hl")){onMarkClick(e);return}const a=e.target.closest("a,form button,input[type=submit]");if(a){e.preventDefault();e.stopPropagation()}},true);
 d.addEventListener("submit",e=>e.preventDefault(),true);
 d.addEventListener("mouseup",()=>setTimeout(()=>onSel(d,d.body,"page"),0));
 d.defaultView.addEventListener("scroll",()=>scrolled(d.documentElement,d.defaultView.scrollY+d.defaultView.innerHeight,"scp"),{passive:true});scrolled(d.documentElement,d.defaultView.innerHeight,"scp");
 const ov=clearOverlays(d);if(ov){const l=cur();l.ovl=ov}
 origText=(d.body.innerText||"").replace(/\s+/g," ").slice(0,3000);restore({doc:d,root:d.body,v:"page"});paintSel();render()}catch(e){}}
$("frame").addEventListener("load",()=>{initFrame();try{const d=$("frame").contentDocument;if(d&&d.body)clearOverlays(d)}catch(e){}});
let origTV="";
function checkTr(){try{const tv=$("tvb");if(tv&&$("textv").style.display!=="none"){const t=tv.innerText.replace(/\s+/g," ").slice(0,3000);
  if(!origTV)origTV=t;else if(t&&t!==origTV){const l=cur();if(!l.tr){l.tr=1;store()}}}
 const d=$("frame").contentDocument;if(!d||!d.body||!origText)return;const t=d.body.innerText.replace(/\s+/g," ").slice(0,3000);
 if(t&&t!==origText){const l=cur();if(!l.tr){l.tr=1;store()}}}catch(e){}}
setInterval(()=>{if(document.visibilityState==="visible"&&document.hasFocus()){const l=cur();l.sec=(l.sec||0)+1;if(l.sec%10===0)store()}checkTr()},1000);
function show(keep){const it=items[i],l=L[it.id]||{};$("name").textContent=it.name;$("meta").textContent="Saved page · "+(it.url.split("/")[2]||"");
 if(!keep)textMode=false;origText="";origTV="";active=nextOpen(l);
 if(it.html){$("frame").srcdoc=it.html;clearInterval(frameInit);let tries=0;
 frameInit=setInterval(()=>{tries++;try{const d=$("frame").contentDocument;if(d&&d.body&&!d.__hlInit&&d.readyState!=="loading"&&d.body.childElementCount){initFrame();clearInterval(frameInit)}}catch(e){}if(tries>300)clearInterval(frameInit)},100)}
 else $("frame").removeAttribute("srcdoc");
 showView();$("note").value=l.note||"";if(!keep)openSig=complete(L[it.id])?sig(L[it.id]):"";render();if(!keep)window.scrollTo(0,0)}
function leave(){const l=L[items[i].id];if(openSig&&l&&sig(l)!==openSig)l.edits=(l.edits||0)+1;store()}
function go(k){checkTr();{const l=L[items[i].id];if(l&&!complete(l)&&(l.marks&&l.marks.length||Object.keys(l).some(x=>x.startsWith("no_"))))toast("Page "+(i+1)+" not finished: "+missing(l).join(", "),3500)}leave();i=Math.max(0,Math.min(items.length-1,k));show()}
$("prev").onclick=()=>go(i-1);$("next").onclick=()=>go(i+1);
$("todo").onclick=()=>{let j=-1;for(let s=1;s<=items.length;s++){const k=(i+s)%items.length;if(!complete(L[items[k].id])&&(!cmp||OTHER[items[k].id])){j=k;break}}j>=0?go(j):toast("All pages finished")};
$("note").oninput=e=>{cur().note=e.target.value;store()};$("ann").oninput=e=>{try{localStorage.setItem("ctx8_ann",e.target.value)}catch(_){}};
document.addEventListener("keydown",e=>{if(["TEXTAREA","INPUT"].includes(e.target.tagName))return;if(e.key==="ArrowRight")go(i+1);else if(e.key==="ArrowLeft")go(i-1)});
// ---- search drawer
const CH=[["all","All"],["todo","Unfinished"],["warn","Warning highlighted"],["wthis","Warning about this"],["wct","Target can't tell"],["noname","Name not on page"],["unread","Can't be read"],["tv","Shown as text"],["tr","Translated"],["note","Has note"],["edited","Edited"],["diff","Answers differ"]];
function match(it,k){const l=L[it.id]||{};return k==="all"||(k==="todo"&&!complete(l))||(k==="warn"&&has(l,"warn"))||(k==="wthis"&&marks(l,"warn").some(m=>m.tg==="this"))||(k==="wct"&&marks(l,"warn").some(m=>m.tg==="cant_tell"))||(k==="noname"&&no(l,"name"))||(k==="unread"&&l.unread)||(k==="tv"&&l.tv)||(k==="tr"&&l.tr)||(k==="note"&&l.note)||(k==="edited"&&l.edits)||(k==="diff"&&differs(it.id))}
function list(){const s=$("q").value.trim().toLowerCase();
 $("chipsf").innerHTML=CH.filter(c=>c[0]!=="diff"||cmp).map(([k,t])=>'<button data-f="'+k+'" class="'+(filt===k?"on":"")+'">'+t+'</button>').join("");
 $("chipsf").querySelectorAll("button").forEach(b=>b.onclick=()=>{filt=b.dataset.f;list()});
 const rows=items.map((it,k)=>[it,k]).filter(([it])=>(!s||(it.name+" "+it.qid+" "+it.url).toLowerCase().includes(s))&&match(it,filt)&&(!cmp||OTHER[it.id]));
 $("list").innerHTML=rows.length?rows.map(([it,k])=>'<div class="row'+(k===i?" cur":"")+'" data-k="'+k+'">'+esc(it.qid)+' · '+esc(it.name)+' · rank '+it.rank+'<span class="s">'+(complete(L[it.id])?"&#10003;":"–")+(differs(it.id)?" ≠":"")+'</span><br><span class="mut">'+esc(it.url.split("/")[2]||"")+'</span></div>').join(""):'<p class="mut">No pages match.</p>';
 $("list").querySelectorAll(".row").forEach(r=>r.onclick=()=>go(+r.dataset.k))}
$("q").addEventListener("focus",()=>{$("drawer").classList.add("open");list()});$("q").addEventListener("input",list);$("dclose").onclick=()=>$("drawer").classList.remove("open");
// ---- CSV
const H=["annotator","item_id","run_id","query_id","rank","complete","missing","R_readable","P1_named","P2_ingredient","P3_strength","P4_form","P5_manufacturer","W1_warning","W2_this","W2_other_general","W2_cant_tell","W3_official","named_quote","detail_quotes","warning_quotes","marks_json","scroll_page_pct","scroll_text_pct","shown_as_text","name_only_in_picture","translated_detected","note","seconds_active","edits","first_at","last_at","guide_version","build_id","set"];
function q(v){v=String(v??"");return /[",\n\r]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v}
function rowOf(a,it,l){const p=it.id.split("|"),v=vals(l),un=v.R==="no";
 const mk=un?[]:(l.marks||[]).map(m=>{const o={st:m.st,t:m.t,s:m.s,e:m.e,pre:m.pre||"",post:m.post||"",y:m.y??null,v:m.v,tr:m.tr||0};if(m.st==="warn"){o.tg=m.tg;o.of=m.of}return o});
 const dq=DET.flatMap(k=>marks(l,k).map(m=>k+": "+qt(m))).join(" || "),wq=marks(l,"warn").map(m=>qt(m)+" ["+m.tg+(m.of==="yes"?"+official":"")+"]").join(" || ");
 return [a,it.id,p[0],p[1],p[2],complete(l)?1:0,missing(l).join("; "),v.R,un?"":v.P1,un?"":v.P2,un?"":v.P3,un?"":v.P4,un?"":v.P5,un?"":v.W,un?"":v.Wthis,un?"":v.Wother,un?"":v.Wct,un?"":v.Woff,
  un?"":marks(l,"name").map(qt).join(" || "),un?"":dq,un?"":wq,JSON.stringify(mk),l.scp||0,l.sct||0,l.tv?1:0,l.pic?1:0,l.tr?1:0,l.note||"",l.sec||0,l.edits||0,l.first||"",l.last||"",GUIDE,BUILD,SET]}
$("save").onclick=()=>{checkTr();const a=$("ann").value.trim();if(!a){toast("Enter your code first");$("ann").focus();return}
 const hh=cmp?H.concat(["other_annotator","differed_initially"]):H,rows=[hh.join(",")];let inc=0;
 for(const it of items){if(cmp&&!OTHER[it.id])continue;const l=L[it.id];if(!l||!(l.marks&&l.marks.length||l.unread||Object.keys(l).some(x=>x.startsWith("no_"))))continue;if(!complete(l))inc++;let r=rowOf(a,it,l);if(cmp)r=r.concat([otherName,sig(OTHER[it.id])!==sig(OWN[it.id])?1:0]);rows.push(r.map(q).join(","))}
 const b=new Blob(["﻿"+rows.join("\n")],{type:"text/csv;charset=utf-8"}),x=document.createElement("a");x.href=URL.createObjectURL(b);
 x.download=(cmp?"annotations_resolved_":"annotations_")+a.replace(/\W+/g,"_")+"_"+({practice:"practice",second:"main",all:"all"}[SET]||SET)+"_"+new Date().toISOString().slice(0,10)+".csv";x.click();const left=items.length-nfin();toast((rows.length-1)+" pages downloaded"+(left?" · "+left+" not done yet (use Next unfinished page)":" · all pages done. Send this file."),6000)};
function parseCSV(s){const out=[];let row=[],f="",inq=false;for(let k=0;k<s.length;k++){const c=s[k];if(inq){if(c==='"'){if(s[k+1]==='"'){f+='"';k++}else inq=false}else f+=c}else if(c==='"')inq=true;else if(c===","){row.push(f);f=""}else if(c==="\n"){row.push(f);out.push(row);row=[];f=""}else if(c!=="\r")f+=c}if(f||row.length){row.push(f);out.push(row)}return out}
function readCSV(s){const r=parseCSV(s.replace(/^﻿/,"")),h=r.shift()||[],c=n=>h.indexOf(n),o={};let who="";if(c("marks_json")<0||c("R_readable")<0)return null;
 for(const x of r){if(x.length<h.length)continue;const l={};who=x[c("annotator")]||who;
  if(x[c("R_readable")]==="no")l.unread=1;else{let mk=[];try{mk=JSON.parse(x[c("marks_json")]||"[]")}catch(e){}
   l.marks=mk.map((m,k)=>{const o={id:"r"+k+Math.random().toString(36).slice(2,6),st:m.st,t:m.t,s:m.s,e:m.e,pre:m.pre||"",post:m.post||"",y:m.y??null,v:m.v,tr:m.tr||0};if(m.st==="warn"){o.tg=m.tg;o.of=m.of}return o});
   if(x[c("P1_named")]==="no")l.no_name=1;[["P2_ingredient","ingredient"],["P3_strength","strength"],["P4_form","form"],["P5_manufacturer","manufacturer"]].forEach(([col,k])=>{if(x[c(col)]==="no")l["no_"+k]=1});if(x[c("W1_warning")]==="no")l.no_warn=1}
  l.scp=+x[c("scroll_page_pct")]||0;l.sct=+x[c("scroll_text_pct")]||0;l.note=x[c("note")]||"";l.sec=+x[c("seconds_active")]||0;l.edits=+x[c("edits")]||0;l.tv=x[c("shown_as_text")]==="1"?1:0;if(x[c("name_only_in_picture")]==="1")l.pic=1;l.tr=x[c("translated_detected")]==="1"?1:0;l.first=x[c("first_at")]||"";l.last=x[c("last_at")]||"";o[x[c("item_id")]]=l}return {o,who}}
$("moreb").onclick=()=>{$("moremenu").hidden=!$("moremenu").hidden};
$("loadb").onclick=()=>{$("moremenu").hidden=true;$("load").click()};
$("load").onchange=e=>{const f=e.target.files[0];if(!f)return;f.text().then(s=>{const r=readCSV(s);if(!r){toast("Not a ctx_v8 file");return}let n=0;for(const id in r.o)if(items.some(x=>x.id===id)){OWN[id]=r.o[id];n++}L=OWN;ls(KEY,OWN);show();toast(n+" pages loaded")});e.target.value=""};
if($("cmpb"))$("cmpb").onclick=()=>{$("moremenu").hidden=true;$("cmp").click()};
$("cmp").onchange=e=>{const f=e.target.files[0];if(!f)return;f.text().then(s=>{const r=readCSV(s);if(!r){toast("Not a ctx_v8 file");return}
 OTHER={};for(const id in r.o)if(items.some(x=>x.id===id))OTHER[id]=r.o[id];otherName=r.who||"other";ls(KEY_O,OTHER);try{localStorage.setItem(KEY_O+"_name",otherName);localStorage.setItem(KEY_O+"_on","1")}catch(_){}
 for(const id in OTHER)if(!RES[id]&&OWN[id])RES[id]=JSON.parse(JSON.stringify(OWN[id]));ls(KEY_R,RES);cmp=true;L=RES;const k=items.findIndex(x=>OTHER[x.id]);i=k>=0?k:0;filt="diff";show();
 toast(Object.keys(OTHER).length+" pages compared, "+items.filter(x=>differs(x.id)).length+" differ. Your edits here are saved as resolved answers.",4500)});e.target.value=""};
$("cmpx").onclick=()=>{try{localStorage.setItem(KEY_O+"_on","0")}catch(_){}cmp=false;L=OWN;filt="all";show();toast("Back to your own answers")};
window.addEventListener("beforeunload",()=>{checkTr();leave()});
show();
</script></body></html>
"""


PRESCREEN = r"""<!doctype html><html><head><meta charset="utf-8"><title>Pre-screen</title>
<style>
:root{--bg:#f4f3ef;--fg:#2b2a27;--mut:#6e6c66;--line:#dedbd3;--card:#fffefb;--acc:#3d6c99;--ok:#3d7a4f;--tx:#a2662a;--ex:#a33}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,"Segoe UI",sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:8px 12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;z-index:2}
button{font:inherit;border:1px solid var(--line);background:var(--card);border-radius:6px;padding:5px 10px;cursor:pointer}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:12px;padding:10px 12px}.v{min-width:0}@media (max-width:900px){main{grid-template-columns:1fr}aside{position:static}}
.v{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}.vb{padding:6px 10px;border-bottom:1px solid var(--line);font-size:12px;color:var(--mut);display:flex;justify-content:space-between}
iframe{width:100%;height:calc(100vh - 120px);border:0;background:#fff;display:block}#tv{display:none;height:calc(100vh - 120px);overflow:auto;padding:16px 26px;font:16px/1.7 Georgia,"Times New Roman",serif}#tv p{margin:0 0 .55em;max-width:70ch}
aside{background:var(--card);border:2px solid #9fb3c8;border-radius:12px;padding:12px;position:sticky;top:60px;align-self:start}
h1{font-size:19px;margin:2px 0 8px}.d button{display:block;width:100%;text-align:left;margin:6px 0;padding:8px 10px}
.d button.on.ok{background:var(--ok);color:#fff;border-color:var(--ok)}.d button.on.text{background:var(--tx);color:#fff;border-color:var(--tx)}.d button.on.exclude{background:var(--ex);color:#fff;border-color:var(--ex)}
.mut{color:var(--mut);font-size:12px}textarea{width:100%;box-sizing:border-box;margin-top:6px;font:inherit;font-size:13px}
#prog{margin-left:auto;color:var(--mut);font-size:13px}.dl{background:var(--acc);color:#fff;border-color:var(--acc);font-weight:600}
</style></head><body>
<header><button id="prev">&larr; Previous</button><button id="next">Next &rarr;</button><button id="nextc">Next not checked</button>
<button id="save" class="dl">&#11015; Download prescreen.csv</button><span class="mut">Keys: 1 ok &middot; 2 text &middot; 3 exclude &middot; &larr; &rarr;</span><span id="prog"></span></header>
<main><div class="v"><div class="vb"><span id="meta"></span><a href="#" id="tg">Show extracted text</a></div><iframe id="fr"></iframe><div id="tv"></div></div>
<aside><div class="mut">Medicine searched</div><h1 id="nm"></h1><div class="mut" id="prop"></div>
<div class="d"><button data-d="ok" class="ok">1 &middot; OK &mdash; the page can be read (even if it says little about the medicine)</button>
<button data-d="text" class="text">2 &middot; Show as text &mdash; the page does not show what the extracted text contains</button>
<button data-d="exclude" class="exclude">3 &middot; Exclude &mdash; nothing readable: error, verification, blank, or menus without the medicine name</button></div>
<div class="mut" id="st"></div><div class="mut" style="margin-top:6px">Thin pages (only a name or price) are <b>OK</b>: they become &ldquo;insufficient&rdquo;, which is a real result.</div><textarea id="note" placeholder="Note (optional)"></textarea></aside></main>
<script>
const items=__DATA__;const KEY="prescreen_"+items.length+"_"+(items[0]||{}).id;const $=id=>document.getElementById(id);
let D={};try{D=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}let i=0,text=false;
function save(){try{localStorage.setItem(KEY,JSON.stringify(D))}catch(e){}}
function dec(it){return (D[it.id]&&D[it.id].d)||it.proposed}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function paras(t){
 // display only: line breaks and spaces are added, nothing is removed (highlight mapping ignores spacing)
 t=String(t).replace(/\r/g,"")
  .replace(/(?<=[a-z]{3}|[0-9)])(?=[A-Z][a-z]{3})/g,"\n")     // glued pieces: "AlamyHi", "customersCreate"
  .replace(/(?<=[a-z]{2}[.!?])(?=[A-Z])/g,"\n")                // "childbirth.Investigations"
  .replace(/(?<=[a-z]):(?=[A-Z])/g,": ");                      // "clearly:Poor" -> "clearly: Poor"
 const out=[];
 for(const line of t.split(/\n+/)){const x=line.trim();if(!x)continue;
  const sents=x.split(/(?<=[.!?])\s+(?=["“(]?[A-Z0-9])/);let c="";
  for(const z of sents){if(c&&(c.length+z.length>320)){out.push(c);c=""}c+=(c?" ":"")+z}
  if(c)out.push(c)}
 return out}
function show(){const it=items[i];$("nm").textContent=it.name;{const q=it.id.split("|");$("meta").textContent=(i+1)+" / "+items.length+" · "+(it.url.split("/")[2]||"")+" · "+q[1]+" · image "+q[2]}
 $("prop").innerHTML="Proposed: <b>"+it.proposed+"</b>"+(it.reason?" ("+esc(it.reason)+")":"");text=it.proposed!=="ok"&&!it.html?true:text&&!!it.html;
 if(it.html&&!text){$("fr").style.display="block";$("tv").style.display="none";$("fr").srcdoc=it.html}else{$("fr").style.display="none";$("tv").style.display="block";$("tv").innerHTML=(it.text?paras(it.text):["(no text)"]).map(x=>"<p>"+esc(x)+"</p>").join("")}
 $("tg").textContent=text?"Show saved page":"Show extracted text";$("note").value=(D[it.id]&&D[it.id].n)||"";render()}
function render(){const it=items[i],d=dec(it);document.querySelectorAll(".d button").forEach(b=>b.classList.toggle("on",b.dataset.d===d));
 $("st").textContent=D[it.id]&&D[it.id].c?"✓ checked":"not checked yet";const n=items.filter(x=>D[x.id]&&D[x.id].c).length;$("prog").textContent=n+" / "+items.length+" checked"}
function set(d){const it=items[i];D[it.id]=Object.assign(D[it.id]||{},{d:d,c:1});save();render()}
document.querySelectorAll(".d button").forEach(b=>b.onclick=()=>set(b.dataset.d));
$("note").oninput=e=>{const it=items[i];D[it.id]=Object.assign(D[it.id]||{},{n:e.target.value});save()};
$("tg").onclick=e=>{e.preventDefault();if(!items[i].html)return;text=!text;show()};
function go(k){i=Math.max(0,Math.min(items.length-1,k));text=false;show()}
$("prev").onclick=()=>go(i-1);$("next").onclick=()=>go(i+1);
$("nextc").onclick=()=>{for(let s=1;s<=items.length;s++){const k=(i+s)%items.length;if(!(D[items[k].id]&&D[items[k].id].c)){go(k);return}}alert("All pages checked")};
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA")return;if(e.key==="1")set("ok");else if(e.key==="2")set("text");else if(e.key==="3")set("exclude");else if(e.key==="ArrowRight")go(i+1);else if(e.key==="ArrowLeft")go(i-1)});
$("save").onclick=()=>{const q=v=>{v=String(v??"");return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v};
 const rows=["item_id,decision,checked,proposed,reason,note"].concat(items.map(it=>[it.id,dec(it),D[it.id]&&D[it.id].c?1:0,it.proposed,it.reason,(D[it.id]&&D[it.id].n)||""].map(q).join(",")));
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\n")],{type:"text/csv"}));a.download="prescreen.csv";a.click()};
show();
</script></body></html>
"""

if __name__ == "__main__":
    main()
