"""
05_recheck_single_labelled_images.py
Consistency check of the first author's product-identity labels against the written
guide (v7) and the adjudication conventions, for ACCEPTED images that were NOT
double-labelled. Blind to system output: no inliers, ranks or LLM labels are shown.

Step 1 - build the check page (written next to the labelling images):
    python scripts/2_product_identity/05_recheck_single_labelled_images.py
  -> runs/labeling/check_page.html   (open it, check each image, press Save CSV)

Step 2 - apply the saved check:
    python scripts/2_product_identity/05_recheck_single_labelled_images.py --apply runs\\labeling\\label_check_<name>_<date>.csv
  -> runs/labeling_agreement/labels_final_checked.csv  (labels_final.csv + checked labels)
  -> runs/labeling_agreement/label_check_summary.txt   (changes, by reason - for Methods)

Inputs: runs/labeling_items.csv, runs/labeling_second_sample_manifest.csv,
        runs/labeling_agreement/labels_final.csv
"""
import argparse, io, json
from datetime import datetime
from pathlib import Path
import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[2]  # repository root
RUNS = ["fresh_oct", "fresh_oct_form"]
ITEMS = EXP_ROOT / "runs" / "labeling_items.csv"
MANIFEST = EXP_ROOT / "runs" / "labeling_second_sample_manifest.csv"
FINAL = EXP_ROOT / "runs" / "labeling_agreement" / "labels_final.csv"
PAGE = EXP_ROOT / "runs" / "labeling" / "check_page.html"
REASONS = {"pack": "only pack size, unit count, language, logo or design differ",
           "missing": "a criterion is printed on only one package, or is unreadable",
           "name": "printed names differ / same base name (step 1)",
           "visible": "a visible difference in ingredient, strength or dosage form",
           "other": "other (see note)"}
ADJ_M = ["Q092_b442ef988dcd5ca3"]  # same + M after re-resolution

def read_labels(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    body = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
    return pd.read_csv(io.StringIO(body), dtype=str).fillna("")

def build(runs):
    it = pd.read_csv(ITEMS, dtype=str).fillna("")
    double = set(pd.read_csv(MANIFEST, dtype=str).item_id)
    lab = read_labels(FINAL).set_index("item_id")
    sel = it[(it.kind == "selected") & it.run.isin(runs) & ~it.item_id.isin(double)]
    sel = sel.drop_duplicates("item_id").sort_values(["query_id", "item_id"])
    items = [{"id": r.item_id, "qid": r.query_id, "name": r.query_product_name,
              "q": f"img/query_{r.query_id}.jpg", "c": f"img/{r.item_id}.jpg",
              "old": lab.at[r.item_id, "label"] if r.item_id in lab.index else "",
              "m": lab.at[r.item_id, "packaging_differs"] if r.item_id in lab.index else ""}
             for r in sel.itertuples()]
    missing = [i["id"] for i in items if not i["old"]]
    if missing:
        raise SystemExit(f"{len(missing)} items have no label in {FINAL}: {missing[:3]}")
    html = TEMPLATE.replace("__DATA__", json.dumps(items, ensure_ascii=False)) \
                   .replace("__REASONS__", json.dumps(REASONS))
    PAGE.write_text(html, encoding="utf-8")
    print(f"items to check: {len(items)} (accepted images in {', '.join(runs)}, not double-labelled)")
    print(pd.Series([i["old"] for i in items]).value_counts().to_string())
    print(f"written: {PAGE}")

def apply(check_csv):
    chk = pd.read_csv(check_csv, dtype=str).fillna("")
    lab = read_labels(FINAL)
    lab["checked"] = 0
    idx = lab.set_index("item_id").index
    ch = chk[chk.changed == "1"]
    for r in ch.itertuples():
        k = idx.get_loc(r.item_id)
        lab.iat[k, lab.columns.get_loc("label")] = r.new_label
        lab.iat[k, lab.columns.get_loc("packaging_differs")] = r.m_flag
    lab.loc[lab.item_id.isin(chk.item_id), "checked"] = 1
    # Adjudicated items re-resolved under the "printed on neither package" convention (2026-10-24);
    # disagreements_resolved.csv sets only the label, so their M flag is set here.
    for item_id in ADJ_M:
        if item_id in idx:
            lab.iat[idx.get_loc(item_id), lab.columns.get_loc("packaging_differs")] = "1"
    fold = {"unclear": "cannot_determine", "brand_only": "cannot_determine"}
    lab["primary_label"] = lab.label.map(lambda v: fold.get(v, v))
    out = FINAL.parent / "labels_final_checked.csv"
    lab.to_csv(out, index=False, encoding="utf-8-sig")
    lines = [f"Consistency check: {len(chk)} accepted images checked, {len(ch)} labels changed "
             f"({len(ch) / max(len(chk), 1):.1%})", "",
             "changes (old -> new):",
             (ch.old_label + " -> " + ch.new_label).value_counts().to_string() if len(ch) else "none", "",
             "by reason:",
             ch.reason.map(lambda r: f"{r}: {REASONS.get(r, r)}").value_counts().to_string() if len(ch) else "none"]
    (FINAL.parent / "label_check_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    ch.to_csv(FINAL.parent / "label_check_changes.csv", index=False, encoding="utf-8-sig")
    print("\n".join(lines)); print(f"\nwritten: {out}")

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Label Consistency Check</title>
<style>
:root{--bg:#f4f3ef;--fg:#2b2a27;--mut:#6e6c66;--line:#dedbd3;--card:#fffefb;--acc:#2f6f73;--warn:#a2662a}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b19;--fg:#e9e7e1;--mut:#a7a59e;--line:#3a3935;--card:#242422;--acc:#7fb8bb;--warn:#d9a46a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--card);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button,input,textarea{font:inherit;color:inherit}button{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 11px;cursor:pointer}
button.on{background:var(--acc);border-color:var(--acc);color:#fff}#prog{margin-left:auto;color:var(--mut);font-size:14px}
main{max-width:1320px;margin:0 auto;padding:14px 16px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:12px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pair img{width:100%;max-height:40vh;object-fit:contain;background:var(--bg);border-radius:6px}
.cap{font-size:13px;color:var(--mut);text-align:center}h1{font-size:20px;margin:0 0 6px}
.old{display:inline-block;padding:2px 10px;border-radius:6px;background:var(--bg);border:1px solid var(--line);font-weight:600}
.row{margin-top:12px}.row b{display:block;margin-bottom:6px}.opts{display:flex;flex-wrap:wrap;gap:6px}
textarea{width:100%;min-height:40px;border:1px solid var(--line);border-radius:6px;background:var(--bg);padding:6px}
details .card{font-size:15px}details p,details li{margin:0 0 .4em}#status b{color:var(--warn)}#status b.ok{color:var(--acc)}
@media (max-width:800px){.pair{grid-template-columns:1fr}}
</style></head><body>
<header><input id="who" placeholder="Your code" style="width:110px">
<button id="prev">&larr; Prev</button><button id="next">Next &rarr;</button><button id="todo">Next unchecked</button>
<button id="save">Save CSV</button><span id="prog"></span></header>
<main>
<details id="guide" open><summary><b>Rules (guide v7 + adjudication conventions)</b></summary><div class="card">
<p>Check your earlier label against the written rules, using only what is visible in both images. Four criteria: printed product name, active ingredient(s), strength per unit, dosage form.</p>
<ul><li><b>Same</b>: all criteria printed on the packages match. A criterion printed on <i>neither</i> package is not compared.</li>
<li><b>Variant</b>: same printed (or base) name; ingredient, strength or dosage form <i>visibly</i> differs.</li>
<li><b>Different</b>: printed names differ, or no medicine package is shown.</li>
<li><b>Cannot determine</b>: no difference found, but a criterion is printed on only one package, or is not readable.</li>
<li>Do not translate. Text in a language or script you cannot read counts as unreadable.</li>
<li>Pack size, unit count, language, logo and design &rarr; flag <b>M</b>, never variant. Manufacturer is not a criterion.</li>
<li>A suffix (-40, -H, DS) does not decide by itself; read step 2. Printed form words (film-coated, sugar-coated, dispersible, gastro-resistant, injectable) count as dosage form.</li></ul>
<p>Keys: K keep &middot; Enter next &middot; &larr; previous.</p></div></details>
<div class="card"><div id="status"></div><h1 id="name"></h1>
<div class="pair"><div><img id="qi"><div class="cap">Query package</div></div><div><img id="ci"><div class="cap">Web image</div></div></div>
<div class="row">Your earlier label: <span class="old" id="old"></span></div>
<div class="row"><b>Decision</b><div class="opts" id="dec"></div></div>
<div class="row" id="rrow"><b>Reason for the change</b><div class="opts" id="rea"></div></div>
<div class="row"><label><input type="checkbox" id="m"> M &mdash; packaging differs (only with same or cannot determine)</label></div>
<div class="row"><textarea id="note" placeholder="Note (optional)"></textarea></div></div>
</main>
<script>
const D=__DATA__, R=__REASONS__, KEY="labcheck_v1";
const LAB={same:"Same",variant:"Variant",different:"Different",unclear:"Cannot determine",brand_only:"Cannot determine (name unreadable)"};
let S={},i=0;try{S=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
try{document.getElementById("who").value=localStorage.getItem(KEY+"_who")||""}catch(e){}
const $=x=>document.getElementById(x);
function store(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}}
function done(s){return !!(s&&s.new&&(s.new===s.old||s.reason))}
function render(){const it=D[i],s=S[it.id]||{old:it.old,m:it.m==="1"};
 $("name").textContent=it.name;$("qi").src=it.q;$("ci").src=it.c;$("old").textContent=LAB[it.old]||it.old;
 const opts=[["keep","Keep: "+(LAB[it.old]||it.old)],["same","Same"],["variant","Variant"],["different","Different"],["unclear","Cannot determine"]];
 $("dec").innerHTML=opts.filter(([v])=>v==="keep"||v!==it.old).map(([v,t])=>`<button data-v="${v}" class="${(v==="keep"&&s.new===it.old)||(v!=="keep"&&s.new===v)?"on":""}">${t}</button>`).join("");
 $("dec").querySelectorAll("button").forEach(b=>b.onclick=()=>{const v=b.dataset.v,nv=v==="keep"?it.old:v;set({new:nv,reason:nv===s.new?s.reason:""})});
 const ch=s.new&&s.new!==it.old;$("rrow").style.display=ch?"":"none";
 $("rea").innerHTML=Object.entries(R).map(([k,t])=>`<button data-r="${k}" class="${s.reason===k?"on":""}">${t}</button>`).join("");
 $("rea").querySelectorAll("button").forEach(b=>b.onclick=()=>set({reason:b.dataset.r}));
 $("m").checked=!!s.m;$("note").value=s.note||"";
 $("status").innerHTML=`Image ${i+1} of ${D.length} &middot; `+(done(s)?'<b class="ok">checked</b>':"<b>not checked</b>");
 $("prog").textContent=D.filter(x=>done(S[x.id])).length+" / "+D.length+" checked"}
function set(o){const it=D[i];S[it.id]=Object.assign(S[it.id]||{old:it.old,m:it.m==="1"},o,{at:new Date().toISOString()});store();render()}
function go(d){i=Math.max(0,Math.min(D.length-1,i+d));render();window.scrollTo(0,0)}
$("m").onchange=e=>set({m:e.target.checked});$("note").oninput=e=>{const it=D[i];S[it.id]=Object.assign(S[it.id]||{old:it.old,m:it.m==="1"},{note:e.target.value});store()};
$("who").oninput=e=>{try{localStorage.setItem(KEY+"_who",e.target.value)}catch(_){}};
$("prev").onclick=()=>go(-1);$("next").onclick=()=>go(1);
$("todo").onclick=()=>{const k=D.findIndex((x,j)=>j>i&&!done(S[x.id]));const j=k>=0?k:D.findIndex(x=>!done(S[x.id]));if(j>=0){i=j;render();window.scrollTo(0,0)}};
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA"||(e.target.tagName==="INPUT"&&e.target.type==="text"))return;
 if(e.key==="Enter"||e.key==="ArrowRight"){e.preventDefault();go(1)}else if(e.key==="ArrowLeft")go(-1);
 else if(e.key==="k"||e.key==="K"){set({new:D[i].old,reason:""})}});
function q(v){v=String(v??"");return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v}
$("save").onclick=()=>{const w=$("who").value.trim()||"first_author";const rows=[["item_id","query_id","old_label","new_label","changed","reason","m_flag","note","checked_at","checker"].join(",")];
 let n=0;for(const it of D){const s=S[it.id];if(!done(s))continue;n++;
  rows.push([it.id,it.qid,it.old,s.new,s.new!==it.old?1:0,s.new!==it.old?s.reason:"",s.m?1:0,s.note||"",s.at||"",w].map(q).join(","))}
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\n")],{type:"text/csv"}));
 a.download="label_check_"+w.replace(/\W+/g,"_")+"_"+new Date().toISOString().slice(0,10)+".csv";a.click();$("prog").textContent=n+" checked images saved to CSV"};
render();
</script></body></html>"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", default="", help="CSV saved from check_page.html")
    ap.add_argument("--runs", default=",".join(RUNS))
    a = ap.parse_args()
    apply(a.apply) if a.apply else build([r.strip() for r in a.runs.split(",") if r.strip()])

if __name__ == "__main__":
    main()
