#!/usr/bin/env python3
"""Combine duplicate evidence in a completed, independently built Comment 6 assignment.

Produces ONE blinded review page. Cases with the same (query_id, rank, signal) are
merged; quotations with identical text are reviewed only once, but every original
model quote and sampling stratum is retained in a private mapping. This cannot
add or omit source evidence, and expansion restores the original row format.

Usage:
  python 14_group_semantic_assignment.py --build --html semantic_evidence.html --manifest semantic_manifest.csv --out-dir grouped
  python 14_group_semantic_assignment.py --expand --answers grouped/semantic_evidence_HM.csv --mapping grouped/semantic_group_map.csv --original-manifest semantic_manifest.csv --out expanded_HM.csv
  python 13_semantic_evidence_audit.py --apply expanded_HM.csv --manifest semantic_manifest.csv
"""
from __future__ import annotations
import argparse,csv,hashlib,json,re,unicodedata
from collections import OrderedDict
from pathlib import Path
GROUP_VERSION='semantic_group_v4_2_no_prior_knowledge_2026-11-19'
FIELDS=['case_id','quote_id','reviewer','build_id','version','complete','located','occurrence','product_ref','support','reason','warning_quality','warning_target','warning_official','note','evidence_mode','evidence_text','evidence_start','evidence_end']
MAP_FIELDS=['group_case_id','group_quote_id','old_case_id','old_quote_id','model','query_id','rank','signal','stratum','old_build_id','old_version']

def fail(x):raise SystemExit('STOP: '+str(x))
def sha(x):return hashlib.sha256(x).hexdigest()
def read(p,fields):
    p=Path(p)
    if not p.is_file():fail('Missing file '+str(p))
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or not set(fields).issubset(reader.fieldnames):fail('Missing columns in '+str(p)+': '+str(set(fields)-set(reader.fieldnames or [])))
        return list(reader)
def write(p,fields,rows):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def load_html(path):
    src=Path(path).read_text(encoding='utf-8')
    m=re.search(r'const DATA=(\[.*?\]), BUILD="([^"\\]*)", VERSION="([^"\\]*)";',src,re.S)
    if not m:fail('Original HTML format not recognized: DATA/BUILD/VERSION declaration not found')
    try:data=json.loads(m.group(1))
    except ValueError as e:fail('Invalid embedded assignment JSON: '+str(e))
    return src,m,data,m.group(2),m.group(3)
def quote_key(text):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text).casefold()).strip()

def build(a):
    src,matcher,cards,old_build,old_version=load_html(a.html)
    mf=read(a.manifest,['case_id','model','query_id','rank','signal','stratum','quote_count','build_id','version'])
    byid={m['case_id']:m for m in mf}
    if len(byid)!=len(mf) or len(cards)!=len(mf):fail('Manifest/card row counts or unique case IDs differ')
    groups=OrderedDict(); cross=[]; reused=0; quote_total=0
    for card in cards:
        cid=card['id']
        if cid not in byid:fail('Card missing from manifest: '+cid)
        m=byid[cid]
        if m['build_id']!=old_build or m['version']!=old_version:fail('Build/version mismatch with original HTML')
        if int(m['quote_count'])!=len(card['quotes']):fail('Quote count mismatch: '+cid)
        key=(m['query_id'],m['rank'],m['signal'])
        if key not in groups:
            c={k:v for k,v in card.items() if k!='quotes'}
            c['id']=f'C{len(groups)+1:04d}';c['quotes']=[]
            groups[key]={'card':c,'seen':{}}
        dest=groups[key]; dc=dest['card']
        if dc['searched']!=card['searched'] or dc['signal']!=card['signal'] or dc['source_text']!=card['source_text']:
            fail('Inconsistent page or product content within merged case '+str(key))
        for j,q in enumerate(card['quotes'],start=1):
            quote_total+=1
            k=quote_key(q['quote'])
            # W3 evidence may differ even when W1 warning text is identical; keep a separate row.
            if dc['signal']=='W1':k+='\u241f'+quote_key(q.get('issuer_quote',''))
            # Identical short quotations can refer to different occurrences. If the
            # model supplied different context on a repeated quote, keep separate.
            if len(q.get('occ',[]))>1:k+='\u241f'+quote_key(q.get('model_context',''))
            if k in dest['seen']:
                nq=dest['seen'][k];reused+=1
            else:
                nq=f'q{len(dc["quotes"])+1}'
                nrec=dict(q);nrec['id']=nq
                dc['quotes'].append(nrec)
                dest['seen'][k]=nq
            cross.append({'group_case_id':dc['id'],'group_quote_id':nq,'old_case_id':cid,'old_quote_id':f'q{j}',
                          'model':m['model'],'query_id':m['query_id'],'rank':m['rank'],'signal':m['signal'],
                          'stratum':m['stratum'],'old_build_id':old_build,'old_version':old_version})
    grouped=[r['card'] for r in groups.values()]
    mapbytes=json.dumps(cross,sort_keys=True,ensure_ascii=False).encode('utf-8')
    new_build=sha(json.dumps({'cards':grouped,'original':old_build,'mapping_hash':sha(mapbytes)},sort_keys=True,ensure_ascii=False).encode('utf-8'))[:20]
    data_json=json.dumps(grouped,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    replacement=f'const DATA={data_json}, BUILD="{new_build}", VERSION="{GROUP_VERSION}";'
    rewritten=src[:matcher.start()]+replacement+src[matcher.end():]
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'semantic_evidence_HM_grouped.html').write_text(rewritten,encoding='utf-8')
    write(out/'semantic_group_map.csv',MAP_FIELDS,cross)
    write(out/'semantic_group_manifest.csv',['group_case_id','signal','quote_count','build_id','version'],[
         {'group_case_id':g['id'],'signal':g['signal'],'quote_count':len(g['quotes']),'build_id':new_build,'version':GROUP_VERSION} for g in grouped])
    (out/'semantic_group_build.json').write_text(json.dumps({'version':GROUP_VERSION,'build_id':new_build,'original_build':old_build,
        'original_cards':len(cards),'grouped_cards':len(grouped),'original_quotes':quote_total,
        'grouped_quotes':sum(len(c['quotes']) for c in grouped),'duplicates_removed':reused,
        'original_page_signal_count':len(groups),'mapping_sha256':sha(mapbytes)},indent=2),encoding='utf-8')
    print('Grouped:',len(cards),'original cards ->',len(grouped),'cards;',quote_total,'original quotations ->',sum(len(c['quotes']) for c in grouped),'quotations')
    print('H.M. file:',out/'semantic_evidence_HM_grouped.html')
    print('Keep semantic_group_map.csv and semantic_group_manifest.csv PRIVATE.')

def expand(a):
    mapping=read(a.mapping,MAP_FIELDS)
    responses=read(a.answers,FIELDS)
    originals=read(a.original_manifest,['case_id','quote_count','build_id','version'])
    old={r['case_id']:r for r in originals}
    mx={ (r['group_case_id'],r['group_quote_id']) for r in mapping}
    if len(mx)==0:fail('Empty map')
    seen=set();vals={};identity=set()
    for r in responses:
        k=(r['case_id'],r['quote_id'])
        if k in seen:fail('Duplicate group evidence: '+str(k))
        seen.add(k)
        if k not in mx:fail('Unexpected grouped evidence '+str(k))
        identity.add((r['build_id'],r['version'],r['reviewer']))
        vals[k]=r
    if set(vals)!=mx:fail('Missing grouped responses: '+str(len(mx-set(vals))))
    if len(identity)!=1:fail('Inconsistent reviewer/build/version in grouped CSV')
    build,ver,reviewer=next(iter(identity))
    if not reviewer or ver!=GROUP_VERSION:fail('Invalid group CSV reviewer/version')
    for r in mapping:
        if r['old_case_id'] not in old:fail('Original case missing from manifest: '+r['old_case_id'])
        if r['old_build_id']!=old[r['old_case_id']]['build_id'] or r['old_version']!=old[r['old_case_id']]['version']:
            fail('Original build mismatch in map')
    output=[]
    for r in mapping:
        ans=dict(vals[(r['group_case_id'],r['group_quote_id'])])
        ans['case_id']=r['old_case_id'];ans['quote_id']=r['old_quote_id']
        ans['build_id']=r['old_build_id'];ans['version']=r['old_version']
        output.append(ans)
    exp={(r['case_id'],f'q{i+1}') for r in originals for i in range(int(r['quote_count']))}
    obs={(r['case_id'],r['quote_id']) for r in output}
    if obs!=exp or len(output)!=len(exp):fail('Expanded responses do not exactly cover original manifest')
    write(a.out,FIELDS,output)
    print('Expanded',len(responses),'unique evidence answers into',len(output),'original evidence rows:',a.out)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    x=ap.add_mutually_exclusive_group(required=True);x.add_argument('--build',action='store_true');x.add_argument('--expand',action='store_true')
    ap.add_argument('--html',type=Path);ap.add_argument('--manifest',type=Path)
    ap.add_argument('--out-dir',type=Path,default=Path('grouped'))
    ap.add_argument('--answers',type=Path);ap.add_argument('--mapping',type=Path)
    ap.add_argument('--original-manifest',type=Path);ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    if a.build:
        if not a.html or not a.manifest:fail('--build requires --html and --manifest')
        build(a)
    else:
        if not all((a.answers,a.mapping,a.original_manifest,a.out)):fail('--expand requires --answers, --mapping, --original-manifest, --out')
        expand(a)
if __name__=='__main__':main()
