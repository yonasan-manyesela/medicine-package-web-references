#!/usr/bin/env python3
"""Comment 6: one-pass, blinded semantic-evidence audit (signal-rule runs).

This script consumes explicitly normalized, reproducible inputs rather than guessing
how sc2_run_models.py serialized the LLM output. The researcher's preparation is
separate from Annotator 4's independent assessment.

  python 13_semantic_evidence_audit.py --templates
  python 13_semantic_evidence_audit.py --demo
  python 13_semantic_evidence_audit.py --build --model-quotes model_quotes.csv \
     --model-pages model_pages.csv --human-evidence human_evidence.csv \
     --page-dir runs/fresh_oct/source_context_v3/page_text
  python 13_semantic_evidence_audit.py --apply path/to/HM_export.csv \
     --manifest path/to/semantic_manifest.csv

The model quote rows contain all quoted POSITIVE P1-P5 signals and W1 warning evidence.
For W1, include target/official values of the model warning, when available.
Status rows must include ALL annotated pages for each model, including negative pages;
failed and not-fetched pages are not treated as model omissions.
"""
from __future__ import annotations
import argparse, csv, hashlib, html, json, random, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

VERSION = 'semantic_v4_2_no_prior_knowledge_2026-11-19'
SIGNALS = ['P1','P2','P3','P4','P5','W1']
SIGNAL_LABELS = {
'P1':'Product name',
'P2':'Active ingredient (medicine substance)',
'P3':'Strength or concentration (amount of active substance)',
'P4':'Dosage form (tablet, capsule, syrup, etc.)',
'P5':'Manufacturer or marketer (who makes or markets it)',
'W1':'Medicine-quality warning (not normal medical advice)'}
# Short, plain-English descriptions of the SAME frozen signal definitions.
RULES = {
'P1':'The saved page names this searched medicine. Small spelling, spacing or punctuation differences are okay. A different medicine with a similar name does not count.',
'P2':'An active ingredient is a medicine substance. The page must clearly link the quoted ingredient information to the searched medicine (e.g., in its Composition or Contains field). You do not need to know what the medicine really contains. An ingredient mentioned for another medicine does not count.',
'P3':'Strength or concentration means the amount of active substance, for example 500 mg, 250 mg/5 mL, 1%, or 20/120. It must be stated for the searched medicine. Pack quantity, bottle volume alone, and dosing instructions do NOT count.',
'P4':'Dosage form means tablet, capsule, syrup, injection, cream, gel, drops, etc. The page must associate the stated form with the searched medicine. A form belonging to another medicine, or a bottle size (such as 60 mL), does not count.',
'P5':'The saved page must explicitly say who MANUFACTURES or MARKETS this medicine (for example “Manufactured by” or “Marketed by”). Just listing a brand, seller, pharmacy, distributor or company name does NOT count.',
'W1':'A medicine-QUALITY warning (fake/counterfeit, substandard, recalled, seized, unregistered or banned). Side effects, dosage advice, “100% genuine” and prescription cautions do NOT count. For warnings, also record the target: THIS medicine or its batch (or explicitly ALL medicines containing its ingredient), OTHER/GENERAL medicine(s), or UNCLEAR if only an ingredient is mentioned. Count an authority only when the warning is attributed to a named regulator or public-health body.'}
MODEL_FIELDS = ['model','query_id','rank','signal','quote_idx','quote','model_value','warning_target','warning_official','issuer_quote','model_context']
STATUS_FIELDS = ['model','query_id','rank','status','chars_sent','cut']
HUMAN_FIELDS = ['query_id','rank','signal','positive','quote']
AUDIT_FIELDS = ['case_id','quote_id','reviewer','build_id','version','complete','located','occurrence','product_ref','support',
                'reason','warning_quality','warning_target','warning_official','note','evidence_mode','evidence_text','evidence_start','evidence_end']
VALID_STATUS = {'ok','empty','not_fetched','failed'}
VALID_LOCATED = {'yes','no','unclear'}
VALID_REF = {'searched','other_or_general','unclear'}
VALID_SUPPORT = {'yes','no','unclear'}
VALID_REASON = {'','seller_distributor_pharmacy','other_product','wrong_attribute_or_value','clinical_caution','other'}
VALID_EVIDENCE = {'','selected','quote_itself','not_available'}
VALID_QUALITY = {'','yes','no','unclear'}
VALID_TARGET = {'','this','other_general','unclear'}
VALID_OFFICIAL = {'','yes','no','unclear'}

def fail(msg): raise SystemExit('STOP: '+str(msg))
def read_csv(path, required):
    path=Path(path)
    if not path.exists(): fail('Input not found: '+str(path))
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f); names=r.fieldnames or []; absent=set(required)-set(names)
        if absent: fail(f'{path.name} missing columns: {sorted(absent)}; has {names}')
        return [{k:(v or '').strip() for k,v in row.items() if k is not None} for row in r]

def names_from_existing_items(items_path):
    """Reuse the verified query_product_name column from the existing labeling_items.csv.

    This is a name lookup only; no identity labels or annotation results are changed.
    """
    rows=read_csv(items_path,['query_id','query_product_name'])
    names={}
    for r in rows:
        qid=r['query_id'].strip()
        name=r['query_product_name'].strip()
        if not qid or not name:
            continue
        if qid in names and names[qid].casefold()!=name.casefold():
            fail(f'Conflicting query product names in {items_path} for {qid}: {names[qid]!r} vs {name!r}')
        names[qid]=name
    if not names:fail(f'No non-empty query product names in {items_path}')
    print(f'Product names: loaded {len(names)} query IDs from {items_path}')
    return names


def find_existing_items():
    """Locate project/runs/labeling_items.csv from script location or current workdir."""
    for root in [Path.cwd(),*Path(__file__).resolve().parents]:
        candidate=root/'runs'/'labeling_items.csv'
        if candidate.is_file():
            return candidate
    return None


def load_product_names(args):
    """Prefer provided lookup; fall back to existing labeling_items.csv if not present."""
    if args.names_csv is not None and args.names_csv.is_file():
        rows=read_csv(args.names_csv,['query_id','product_name'])
        names={}
        for r in rows:
            qid,name=r['query_id'].strip(),r['product_name'].strip()
            if not qid or not name:continue
            if qid in names and names[qid].casefold()!=name.casefold():
                fail(f'Conflicting searched product names for {qid} in {args.names_csv}')
            names[qid]=name
        print(f'Product names: loaded {len(names)} query IDs from {args.names_csv}')
        return names
    if args.names_csv is not None:
        print(f'NOTE: prepared name file not found: {args.names_csv}; trying existing labeling_items.csv')
    candidate=args.items_csv if args.items_csv is not None else find_existing_items()
    if candidate is None or not candidate.is_file():
        fail('Product-name input not found. Expected either --names-csv with product_names.csv '
             'OR the existing runs/labeling_items.csv (columns query_id, query_product_name). '
             'Pass --items-csv <path> if the original file is elsewhere.')
    return names_from_existing_items(candidate)


def write_csv(path, fields, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def sha(b):return hashlib.sha256(b).hexdigest()
def norm_rank(r):
    try: return str(int(float(r)))
    except (TypeError,ValueError): fail(f'invalid rank {r!r}')

def clean(s):
    # Maps normalized characters back to the source code-point offset, even for accented text.
    out=[]; idx=[]
    for pos,ch in enumerate(s):
        val=unicodedata.normalize('NFKC',ch).casefold()
        for c in val:
            if c in '“”„‟«»':c='"'
            elif c in '‘’‚‛':c="'"
            elif c in '‐‑‒–—―−':c='-'
            elif c.isspace():c=' '
            out.append(c); idx.append(pos)
    condensed=[]; newidx=[]; prevspace=False
    for c,i in zip(out,idx):
        if c==' ':
            if prevspace:continue
            prevspace=True
        else:prevspace=False
        condensed.append(c);newidx.append(i)
    return ''.join(condensed),newidx

def _boundary_ok(page, begin, end, q):
    """Avoid short fragments such as 'tab' inside 'About' or '500mg' inside '1500mg'.

    The source text can join HTML fields; spacing-only candidates are clearly
    marked as uncertain and manually confirmed, rather than rejected here.
    """
    left = q[:1].isalnum()
    right = q[-1:].isalnum()
    return not ((left and begin > 0 and page[begin-1].isalnum()) or
                (right and end < len(page) and page[end].isalnum()))


def occurrences(page, quotation):
    text, offsets = clean(page)
    q, _ = clean(quotation)
    q = q.strip()
    if len(q) < 2:
        return []
    found = []
    start = 0
    while len(found) < 25:
        j = text.find(q, start)
        if j < 0:
            break
        a, b = offsets[j], offsets[j + len(q) - 1] + 1
        if _boundary_ok(page, a, b, q):
            found.append({'start': a, 'end': b})
        start = j + 1
    return found


def quote_locations(page, quotation):
    """Normalized word/number-boundary matching; optional spacing-only navigation.

    Location is never a semantic judgment. For joined HTML text, a spacing-only
    candidate must be explicitly checked by the reviewer before a Yes judgment.
    """
    strict = occurrences(page, quotation)
    if strict:
        return strict, 'normalized'
    page_norm, page_idx = clean(page)
    quote_norm, _ = clean(quotation)
    compact_q = quote_norm.replace(' ', '').strip()
    if len(compact_q) < 8:
        return [], 'not_located'
    flat = []
    idx = []
    for ch, pos in zip(page_norm, page_idx):
        if ch != ' ':
            flat.append(ch)
            idx.append(pos)
    squeezed = ''.join(flat)
    found = []
    at = 0
    while len(found) < 25:
        j = squeezed.find(compact_q, at)
        if j < 0:
            break
        a, b = idx[j], idx[j + len(compact_q) - 1] + 1
        candidate, _ = clean(page[a:b])
        if candidate.replace(' ', '') == compact_q:
            found.append({'start': a, 'end': b})
        at = j + 1
    return (found, 'spacing_only') if found else ([], 'not_located')


def page_body(raw):
    """Page body after the first sc0 metadata separator; same rule as sc2.

    Work in LF line endings, as Python's read_text and browser views do.
    Do not strip or reformat the returned body: model chars_sent and source
    quotation offsets must refer to precisely this text.
    """
    lines=raw.replace('\r\n','\n').replace('\r','\n').split('\n')
    for i,line in enumerate(lines):
        if line.startswith('=' * 10):
            return '\n'.join(lines[i+1:])
    return '\n'.join(lines)

def read_model_text(path):
    return page_body(Path(path).read_text(encoding='utf-8-sig'))

def find_page(base,qid,rank):
    """Use the working Comment 3 folder/rank mapping, never live pages.

    If both rankN_page.txt and legacy rankN.txt exist in one folder, use
    the sc0 full-page file. Two matching folders are a genuine ambiguity.
    """
    base=Path(base)
    folders=[base/qid]+sorted(base.glob(qid+'_*'))
    matches=[]
    for folder in folders:
        candidates=[folder/f'rank{rank}_page.txt',folder/f'rank{rank}.txt']
        available=[p for p in candidates if p.is_file()]
        if available: matches.append(available[0])
    if not matches:
        fail(f'No saved page text for {qid} rank {rank} under {base}. '
             'Check the saved-page directory; do not substitute a live webpage.')
    if len(matches)>1:
        fail(f'Several page-text folders match {qid} rank {rank}: {matches}')
    return matches[0]

def normalize_model(rows, statuses):
    validpages={}; names=set(); output=defaultdict(list); checked=set()
    for row in statuses:
        mdl=row['model'];qid=row['query_id'];rank=norm_rank(row['rank']);key=(mdl,qid,rank)
        if not mdl or not qid:fail('missing model or query_id in model_pages')
        if key in validpages:fail(f'duplicate model page {key}')
        st=row['status'].lower();
        if st not in VALID_STATUS:fail(f'unknown status {st!r} for {key}')
        validpages[key]=st;names.add(mdl)
    if len(names)!=2:fail(f'Expected exactly TWO signal-rule runs; found {sorted(names)}')
    for row in rows:
        mdl=row['model'];qid=row['query_id'];rank=norm_rank(row['rank']);sig=row['signal'].upper();idx=row['quote_idx'];quote=row['quote']
        key=(mdl,qid,rank)
        if sig not in SIGNALS:fail(f'invalid signal {sig!r} (use P1-P5 or W1; attach W2/W3 to W1)')
        if key not in validpages:fail(f'quotation without status row: {key}')
        if validpages[key] not in {'ok','empty'}:fail(f'quotation on failed/not_fetched page: {key}')
        if not quote.strip():fail(f'empty quoted evidence for {key} {sig}')
        qk=(*key,sig,idx)
        if qk in checked:fail(f'duplicate quotation key {qk}')
        checked.add(qk);r=dict(row);r['rank']=rank;r['signal']=sig
        output[(*key,sig)].append(r)
    for k,v in output.items():
        if len(v)>20:fail(f'too many quotations in a page-signal case {k}: {len(v)}')
    return output,validpages,sorted(names)

def normalize_human(rows):
    human={};evidence=defaultdict(list)
    for r in rows:
        qid=r['query_id'];rank=norm_rank(r['rank']);sig=r['signal'].upper();key=(qid,rank,sig)
        if sig not in SIGNALS:fail('unexpected human signal '+sig)
        yes=r['positive'].lower() in {'1','yes','true','y'}
        if r['positive'].lower() not in {'1','0','yes','no','true','false','y','n'}:fail(f'invalid human positive in {key}: {r["positive"]!r}')
        if key in human and human[key]!=yes:fail(f'conflicting human flags for {key}')
        human[key]=yes
        if yes and r['quote']:evidence[key].append(r['quote'])
    return human,evidence

def choose_units(output,validpages,human,names,seed,agreed_n,include_human_only):
    rng=random.Random(seed);units=[];pool=defaultdict(list);class_counts=Counter();omitted=set()
    for (mdl,qid,rank),status in sorted(validpages.items()):
        if status != 'ok':continue
        for sig in SIGNALS:
            hkey=(qid,rank,sig)
            if hkey not in human:
                omitted.add((mdl,qid,rank))
                continue
            modelrows=output.get((mdl,qid,rank,sig),[])
            h=human[hkey]
            if modelrows:
                stratum='both' if h else 'model_only'
                rule='census' if sig in {'P5','W1'} or stratum=='model_only' else 'sample'
                u={'model':mdl,'query_id':qid,'rank':rank,'signal':sig,'stratum':stratum,'rule':rule,'quotes':modelrows}
                pool[(mdl,sig,stratum)].append(u)
            elif include_human_only and h:
                # These reference-only checks are reported separately, not as a model-quotation accuracy denominator.
                u={'model':mdl,'query_id':qid,'rank':rank,'signal':sig,'stratum':'human_only','rule':'census','quotes':[]}
                pool[(mdl,sig,'human_only')].append(u)
    for key,group in sorted(pool.items()):
        n=len(group)
        sample=group if group[0]['rule']=='census' else rng.sample(group,min(agreed_n,len(group)))
        class_counts[key]=(n,len(sample))
        units.extend(sample)
    rng.shuffle(units)
    return units,class_counts,omitted

def build(args):
    modelrows=read_csv(args.model_quotes,MODEL_FIELDS)
    statuses=read_csv(args.model_pages,STATUS_FIELDS)
    humanrows=read_csv(args.human_evidence,HUMAN_FIELDS)
    out,valid,names=normalize_model(modelrows,statuses)
    sent_lengths={}
    for r in statuses:
        k=(r['model'],r['query_id'],norm_rank(r['rank']))
        length=(r.get('chars_sent') or '').strip()
        if r.get('cut','').lower() in {'1','true','yes'}:
            fail(f'Model text was truncated for {k}; use the exact input text, not the complete page.')
        if length:
            try:
                v=int(float(length))
            except (ValueError,TypeError):
                fail(f'Invalid chars_sent in model-pages for {k}: {length!r}')
            if v < 0:fail(f'Negative chars_sent for {k}')
            sent_lengths[k]=v
    h,hquotes=normalize_human(humanrows)
    units,counts,missing_human=choose_units(out,valid,h,names,args.seed,args.agreed_n,not args.no_human_only)
    if not args.names:fail('Product names are required for blinded product-attribution review. Use --names-csv or --items-csv.')
    missing_names={u['query_id'] for u in units if u['query_id'] not in args.names or not args.names[u['query_id']]}
    if missing_names:fail(f'Product names missing for query IDs: {sorted(missing_names)[:12]}')
    cards=[];manifest=[];snaps={};rejected=[]
    for seq,u in enumerate(units):
        mdl,qid,rank,sig=u['model'],u['query_id'],u['rank'],u['signal']
        page=find_page(args.page_dir,qid,rank)
        raw=read_model_text(page)
        if not raw.strip():fail(f'empty page {page}')
        expected_len=sent_lengths.get((mdl,qid,rank))
        if expected_len is not None and len(raw)!=expected_len:
            fail(f'Saved text length differs from model chars_sent on {mdl}/{qid}/{rank}: '
                 f'{len(raw)} vs {expected_len}. Recheck the page file rather than silently auditing a different text.')
        digest=sha(raw.encode('utf-8'));snaps[str(page.resolve())]=digest
        qu=u['quotes'] or [{'quote':v,'quote_idx':str(i+1),'model_value':'','warning_target':'','warning_official':''} for i,v in enumerate(hquotes.get((qid,rank,sig),[]))]
        if not qu:
            rejected.append((mdl,qid,rank,sig,'missing human-only quotation'))
            continue
        qout=[]
        for qi,r in enumerate(qu):
            quote=r['quote'];occ,location_kind=quote_locations(raw,quote)
            # Quote contexts taken from raw saved text, not fetched web or independent LLM.
            contexts=[{'idx':i,'snippet':raw[max(0,v['start']-280):min(len(raw),v['end']+280)],
                       'start':v['start'],'end':v['end'],
                       'before':raw[max(0,v['start']-280):v['start']],
                       'focus':raw[v['start']:v['end']],
                       'after':raw[v['end']:min(len(raw),v['end']+280)]} for i,v in enumerate(occ)]
            qout.append({'id':f'q{qi+1}','quote':quote,'occ':contexts,'location_kind':location_kind,'model_value':r.get('model_value',''),
                         'model_context':r.get('model_context',''),'issuer_quote':r.get('issuer_quote',''),
                         'warning_target':r.get('warning_target',''),'warning_official':r.get('warning_official','')})
        cid=f'C{len(cards)+1:04d}'
        card={'id':cid,'searched':u['query_id'], 'signal':sig,'title':SIGNAL_LABELS[sig], 'rule':RULES[sig],
              'quotes':qout,'source_text':raw}
        # Query ID may not be a medicine name: add a product name map when available.
        if args.names:
            card['searched']=args.names.get(qid,qid)
        cards.append(card)
        manifest.append({'case_id':cid,'model':mdl,'query_id':qid,'rank':rank,'signal':sig,'stratum':u['stratum'],
            'sampling_rule':u['rule'],'source_file':str(page.resolve()),'source_sha256':digest,
            'quote_count':len(qout),'quote_located':json.dumps([bool(q['occ']) for q in qout]),'quote_occurrences':json.dumps([len(q['occ']) for q in qout]),'quote_location_kinds':json.dumps([q['location_kind'] for q in qout]),'source_model_values':json.dumps([{'model_value':v.get('model_value',''),
               'warning_target':v.get('warning_target',''),'warning_official':v.get('warning_official',''),'issuer_quote':v.get('issuer_quote','')} for v in qu],ensure_ascii=False)})
    if not cards:fail('No cases prepared; check data and selection rule')
    outdir=Path(args.out_dir);outdir.mkdir(parents=True,exist_ok=True)
    build_id=sha(json.dumps({'version':VERSION,'cards':cards,'manifest':manifest,'seed':args.seed},sort_keys=True,ensure_ascii=False).encode('utf-8'))[:20]
    (outdir/'semantic_evidence.html').write_text(HTML.replace('__DATA__',js_safe(cards)).replace('__BUILD__',build_id).replace('__VERSION__',VERSION),encoding='utf-8')
    for row in manifest:row.update(build_id=build_id,version=VERSION)
    write_csv(outdir/'semantic_manifest.csv',['case_id','model','query_id','rank','signal','stratum','sampling_rule','source_file','source_sha256','quote_count','quote_located','quote_occurrences','quote_location_kinds','source_model_values','build_id','version'],manifest)
    write_csv(outdir/'semantic_sampling.csv',['model','signal','stratum','population','selected'],[
        dict(model=k[0],signal=k[1],stratum=k[2],population=v[0],selected=v[1]) for k,v in sorted(counts.items())])
    # Location method is only a navigation result, never the semantic verdict.
    loc_counts=Counter((m['model'],m['signal'],q['location_kind']) for c,m in zip(cards,manifest) for q in c['quotes'])
    write_csv(outdir/'semantic_quote_location_methods.csv',['model','signal','location_method','quotes'],[
        {'model':key[0],'signal':key[1],'location_method':key[2],'quotes':v}
        for key,v in sorted(loc_counts.items())])
    totals=Counter({kind:sum(v for (_,_,k),v in loc_counts.items() if k==kind) for kind in ('normalized','spacing_only','not_located')})
    print('Quotation locations (not semantic verdicts): '+', '.join(f'{kind}={totals[kind]}' for kind in ('normalized','spacing_only','not_located')))
    if rejected:
        write_csv(outdir/'semantic_exclusions.csv',['model','query_id','rank','signal','reason'],[
            dict(zip(['model','query_id','rank','signal','reason'],v)) for v in rejected])
    (outdir/'semantic_build.json').write_text(json.dumps({'build_id':build_id,'version':VERSION,'cases':len(cards),
        'quote_count':sum(len(c['quotes']) for c in cards),'names':names,'excluded_human_only':len(rejected),'excluded_no_human_reference_pages':len(missing_human)},indent=2),encoding='utf-8')
    print(f'Built {len(cards)} cases, {sum(len(c["quotes"]) for c in cards)} quotations. Build {build_id}')
    print('All selected groups are in ONE assignment. Files:',outdir)
    if rejected:print(f'NOTE: {len(rejected)} reference-only cases lacked human quotations; see exclusions CSV')
    if missing_human:print(f'NOTE: {len(missing_human)} model-page entries lacked a human reference and were excluded (not counted as model errors)')

def js_safe(v):
    return json.dumps(v,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')

def apply(args):
    """Verify a one-pass H.M. export without asking for correction quotations.

    The existing CSV schema is intentionally preserved for grouping/expansion.
    The occurrence column is now only a light confirmation flag ('checked') for
    formatting-only/unlocated quotations marked Yes; it is NOT an occurrence ID.
    """
    mf=read_csv(args.manifest,['case_id','model','signal','stratum','quote_count',
        'quote_located','quote_occurrences','quote_location_kinds','build_id',
        'version','source_file','source_sha256'])
    a=read_csv(args.apply,AUDIT_FIELDS)
    lookup={m['case_id']:m for m in mf}
    if len(lookup)!=len(mf):fail('duplicate case ids in manifest')
    expected={(m['case_id'],f'q{i+1}') for m in mf for i in range(int(m['quote_count']))}
    seen=set();done=[]
    for r in a:
        key=(r['case_id'],r['quote_id'])
        if key in seen:fail(f'duplicate evidence row {key}')
        seen.add(key)
        if r['case_id'] not in lookup:fail(f'unknown case {r["case_id"]}')
        m=lookup[r['case_id']]
        if r['version']!=VERSION or r['build_id']!=m['build_id']:
            fail(f'{key}: version/build mismatch; use matching new HTML and manifest')
        if key not in expected:fail(f'invalid quote ID {key}')
        if r['complete'] not in {'0','1'}:fail(f'{key}: invalid complete flag')
        if r['complete']=='0':continue
        n=int(r['quote_id'][1:])-1
        locs=json.loads(m['quote_located'])
        kinds=json.loads(m['quote_location_kinds'])
        present=bool(locs[n]); kind=kinds[n]
        if kind not in {'normalized','spacing_only','not_located'} or present!=(kind!='not_located'):
            fail(f'{key}: inconsistent location metadata')
        if r['located']!=('yes' if present else 'no'):
            fail(f'{key}: automatic text-location metadata changed')
        if r['support'] not in VALID_SUPPORT:fail(f'{key}: answer must be Yes / No / Cannot determine')
        if m['signal']=='W1':
            if r['warning_quality']!=r['support']:
                fail(f'{key}: warning judgment inconsistent')
            if r['warning_quality']=='yes':
                if r['warning_target'] not in {'this','other_general','unclear'} or r['warning_official'] not in {'yes','no','unclear'}:
                    fail(f'{key}: W1=Yes also needs target and named-authority judgments')
                expected_ref={'this':'searched','other_general':'other_or_general','unclear':'unclear'}[r['warning_target']]
            else:
                if r['warning_target'] or r['warning_official']:
                    fail(f'{key}: only W1=Yes requires target and named-authority judgments')
                expected_ref='' if r['support']=='no' else 'unclear'
        else:
            if r['warning_quality'] or r['warning_target'] or r['warning_official']:
                fail(f'{key}: unexpected warning judgments')
            expected_ref='searched' if r['support']=='yes' else 'unclear' if r['support']=='unclear' else ''
        if r['product_ref']!=expected_ref:fail(f'{key}: product-link derived column changed')
        if any(r[x] for x in ('reason','evidence_mode','evidence_text','evidence_start','evidence_end')):
            fail(f'{key}: obsolete mandatory-evidence fields must be blank')
        # A short, explicit manual verification is requested ONLY when automatic
        # text-location is uncertain and the reviewer makes a positive judgment.
        needs_confirmation=r['support']=='yes' and kind in {'spacing_only','not_located'}
        if needs_confirmation and r['occurrence']!='checked':
            fail(f'{key}: Yes on an uncertain text location must be manually confirmed')
        if not needs_confirmation and r['occurrence']:
            fail(f'{key}: unexpected manual-location flag')
        done.append(r)
    if seen!=expected:fail(f'Export rows missing {len(expected-seen)} / extra {len(seen-expected)}')
    if not args.partial and len(done)!=len(expected):
        fail(f'{len(expected)-len(done)} decisions unfinished; use --partial for a partial CSV')
    reviewer={r['reviewer'] for r in a}
    if len(reviewer)!=1 or not next(iter(reviewer)):
        fail('one nonempty reviewer code required')
    for m in mf:
        page=Path(m['source_file'])
        if not page.is_file() or sha(read_model_text(page).encode('utf-8'))!=m['source_sha256']:
            fail(f'saved page altered or missing: {page}')
    rows=defaultdict(list);pages=defaultdict(list)
    for r in done:
        m=lookup[r['case_id']]
        rows[(m['model'],m['signal'],m['stratum'])].append(r)
        pages[r['case_id']].append(r)
    def valid(r):return r['support']=='yes' and (r['located']=='yes' or r['occurrence']=='checked')
    totals=[]
    for (model,sig,stratum),rr in sorted(rows.items()):
        totals.append({'model':model,'signal':sig,'stratum':stratum,'audited_quotes':len(rr),
            'valid_quotes':sum(valid(r) for r in rr),
            'wrong_product':'',  # not inferred from a bare No answer
            'unsupported':sum(r['support']=='no' for r in rr),
            'ambiguous':sum(r['support']=='unclear' for r in rr),
            'not_located':sum(r['located']=='no' for r in rr),
            'manual_location_checks':sum(r['occurrence']=='checked' for r in rr),
            'warning_target_this':sum(r['warning_target']=='this' for r in rr),
            'warning_target_other':sum(r['warning_target']=='other_general' for r in rr),
            'warning_target_unclear':sum(r['warning_target']=='unclear' for r in rr),
            'warning_official_yes':sum(r['warning_official']=='yes' for r in rr)})
    dest=Path(args.apply).with_suffix('')
    write_csv(str(dest)+'_summary.csv',
        ['model','signal','stratum','audited_quotes','valid_quotes','wrong_product','unsupported','ambiguous','not_located','manual_location_checks','warning_target_this','warning_target_other','warning_target_unclear','warning_official_yes'],totals)
    case_rows=[]
    for cid,rr in pages.items():
        m=lookup[cid]
        case_rows.append({**{k:m[k] for k in ('case_id','model','query_id','rank','signal','stratum')},
            'n_quotations':len(rr),'any_valid':int(any(valid(r) for r in rr)),
            'all_valid':int(all(valid(r) for r in rr))})
    write_csv(str(dest)+'_case_summary.csv',
        ['case_id','model','query_id','rank','signal','stratum','n_quotations','any_valid','all_valid'],case_rows)
    print(f'Validated {len(done)}/{len(expected)} quotation judgments in {len(pages)} cases.')
    print('Summaries:',str(dest)+'_summary.csv,',str(dest)+'_case_summary.csv')
    print('Manual-location confirmations are reported separately; a No judgment does not identify an error cause.')
    print('Counts are stratum-specific. Do not treat disagreement-enriched samples as overall accuracy.')


def templates(outdir):
    out=Path(outdir);out.mkdir(exist_ok=True,parents=True)
    for f,header in [('model_quotes_TEMPLATE.csv',MODEL_FIELDS),('model_pages_TEMPLATE.csv',STATUS_FIELDS),('human_evidence_TEMPLATE.csv',HUMAN_FIELDS)]:
        write_csv(out/f,header,[])
    print('Templates saved to',out)

def demo(args):
    out=Path(args.out_dir);pd=out/'fake_pages'/'Q001_example';pd.mkdir(parents=True,exist_ok=True)
    (pd/'rank1_page.txt').write_text('MEDICINE A manufactured by Example Pharma Ltd. Product A available from Example Pharmacy. General caution: may cause headache. MEDICINE A 250 mg tablets.',encoding='utf-8')
    models=[];status=[];human=[]
    for mdl in ('gpt4o_sr','qwen_sr'):
        status.append(dict(model=mdl,query_id='Q001',rank='1',status='ok'))
        for signal,txt in [('P1','MEDICINE A'),('P3','250 mg tablets'),('P5','Example Pharmacy'),('W1','General caution: may cause headache.')]:
            models.append(dict(model=mdl,query_id='Q001',rank='1',signal=signal,quote_idx='1',quote=txt,model_value='',warning_target='this',warning_official='no'))
    for sig in SIGNALS:
        human.append(dict(query_id='Q001',rank='1',signal=sig,positive='1' if sig in {'P1','P3','P5'} else '0',quote='MEDICINE A manufactured by Example Pharma Ltd.' if sig=='P5' else ''))
    write_csv(out/'demo_model_quotes.csv',MODEL_FIELDS,models);write_csv(out/'demo_model_pages.csv',STATUS_FIELDS,status)
    write_csv(out/'demo_human.csv',HUMAN_FIELDS,human)
    args.model_quotes=out/'demo_model_quotes.csv';args.model_pages=out/'demo_model_pages.csv';args.human_evidence=out/'demo_human.csv';args.page_dir=out/'fake_pages'
    args.no_human_only=True;args.names={'Q001':'MEDICINE A'};build(args)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    g=p.add_mutually_exclusive_group(required=True)
    for flag in ('build','demo','templates'):
        g.add_argument('--'+flag,action='store_true')
    g.add_argument('--apply',type=Path)
    p.add_argument('--partial',action='store_true',help='Allow partial exports with --apply')
    p.add_argument('--model-quotes',type=Path);p.add_argument('--model-pages',type=Path)
    p.add_argument('--human-evidence',type=Path);p.add_argument('--page-dir',type=Path)
    p.add_argument('--out-dir',type=Path,default=Path('runs/fresh_oct/source_context_v3/semantic_audit'))
    p.add_argument('--manifest',type=Path);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--agreed-n',type=int,default=15);p.add_argument('--no-human-only',action='store_true')
    p.add_argument('--names-csv',type=Path,help='Prepared query_id,product_name lookup (optional if labeling_items.csv exists)')
    p.add_argument('--items-csv',type=Path,help='Existing labeling_items.csv with query_id,query_product_name (auto-detected by default)')
    a=p.parse_args()
    # Demo uses a synthetic name; --apply validates an existing assignment.
    if a.templates:templates(a.out_dir);return
    if a.demo:demo(a);return
    if a.build:
        if any(not v for v in (a.model_quotes,a.model_pages,a.human_evidence,a.page_dir)):
            fail('--build requires --model-quotes, --model-pages, --human-evidence, --page-dir')
        if not(0<a.agreed_n<=10000):fail('--agreed-n must be positive')
        a.names=load_product_names(a)
        build(a);return
    if a.apply:
        if not a.manifest:fail('--manifest must be supplied with --apply')
        apply(a)

HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Medicine page evidence review</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f9;color:#243448;font:14px system-ui,Segoe UI,Arial,sans-serif}button,input,textarea{font:inherit}button{cursor:pointer}button:disabled{opacity:.5}header{position:sticky;top:0;z-index:10;background:white;border-bottom:1px solid #ced9e3;padding:9px 13px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}header b{font-size:15px}.grow{flex:1}header button{background:#fff;border:1px solid #becbd8;border-radius:6px;padding:6px 9px}#reviewer{width:80px;border-radius:5px;border:1px solid #afc0cd;padding:5px}main{max-width:1120px;margin:12px auto;padding:0 10px}.panel{background:#fff;border:1px solid #d7e0e9;border-radius:8px;margin-bottom:10px}.top{padding:15px 19px;border-bottom:1px solid #e4ebf2}.top h2{margin:0 0 4px;font-size:19px}.sub{font-weight:600;color:#536b82}.rule{font-size:13px;background:#f2f6fa;border-left:3px solid #6685a5;border-radius:3px;padding:9px 10px;margin-top:9px;line-height:1.5}.helper{font-size:12px;color:#65778b;margin-top:8px}.quote{padding:15px 19px;border-top:1px solid #e5ecf2}.quote:first-child{border-top:0}.n{color:#55718d;font-size:12px;font-weight:650}.quotation{background:#fff0bb;padding:10px 12px;margin:7px 0;border-radius:6px;font-size:14px;overflow-wrap:anywhere;white-space:pre-wrap}.excerpt{background:#f6f9fc;border:1px solid #d9e3ec;border-radius:6px;margin:8px 0;padding:11px;font-size:13px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere}mark.model{background:#ffdc80}mark.spacing{background:#ffebb6;border-bottom:2px dashed #b9842d}.muted{font-size:12px;color:#65788c}.question{font-size:14px;font-weight:700;margin:16px 0 8px}.choices{display:flex;flex-wrap:wrap;gap:8px}.choice{color:#2c4761;background:#fff;border:1px solid #b9cadb;border-radius:6px;padding:9px 13px}.choice.on{background:#245f8d;color:white;border-color:#245f8d}.foot{border-top:1px solid #e5ecf2;padding:13px 19px;display:flex;align-items:center;gap:12px}.primary{background:#245f8d;color:#fff;border:0;border-radius:6px;padding:9px 16px}.error{color:#a12f26;font-weight:650;font-size:13px}.error.ok{color:#267246}.note{width:100%;min-height:52px;resize:vertical;border:1px solid #c0d0df;border-radius:6px;padding:9px;margin-top:8px}.smallwarn{background:#fff7e2;border:1px solid #ead49c;border-radius:6px;padding:10px;margin:8px 0;font-size:12px;line-height:1.5}.checkline{display:flex;align-items:flex-start;gap:8px;cursor:pointer}.checkline input{margin-top:3px}.sourcepane{padding:11px 19px}.sourcepane summary{cursor:pointer;font-weight:600}.source{white-space:pre-wrap;line-height:1.7;font-size:13px;overflow:auto;max-height:70vh;margin-top:10px;border:1px solid #d0dfea;padding:13px;border-radius:6px;overflow-wrap:anywhere}.more{padding:8px 10px;margin:5px 0;border:1px solid #d9e3ec;border-radius:6px}.more summary{cursor:pointer;font-size:12px;color:#536b82}.status{min-height:20px;font-size:12px;color:#526c82}.badge{font-size:12px;color:#4a6178;margin:5px 0}.wrap{display:flex;align-items:center;gap:8px;flex-wrap:wrap}@media(max-width:650px){main{padding:0 5px;margin:6px auto}.top,.quote{padding:11px}header{padding:6px;gap:6px}}
</style></head><body>
<header><b>Medicine page evidence review</b><span class="muted" id="pos"></span><span class="muted" id="done"></span><span class="grow"></span><label class="muted">Reviewer <input id="reviewer" placeholder="HM"></label><button id="prev">← Previous</button><button id="next">Next →</button><button id="save">Save CSV</button></header>
<main>
<section class="panel"><div class="top"><h2 id="searched"></h2><div class="sub" id="signal"></div><div class="rule" id="rule"></div><div class="helper" id="guide"></div></div><div id="quotes"></div><div class="foot"><button class="primary" id="finish">Complete case</button><span class="error" id="error"></span></div></section>
<section class="panel sourcepane"><details id="full"><summary>Full saved webpage text (only if the nearby passage is insufficient)</summary><div class="muted" style="margin-top:9px">Original saved text; some webpages have joined words. Highlighting is a navigation aid, not a correctness judgment.</div><div class="source" id="source"></div></details></section>
</main>
<script>
const DATA=__DATA__, BUILD="__BUILD__", VERSION="__VERSION__";
const $=id=>document.getElementById(id);
const F=['located','occurrence','product_ref','support','reason','warning_quality','warning_target','warning_official','note','evidence_mode','evidence_text','evidence_start','evidence_end'];
const ANS=[['yes','Yes — supported'],['no','No — not supported'],['unclear','Cannot determine']];
const YY=[['yes','Yes'],['no','No'],['unclear','Cannot determine']];
const TARGET=[['this','This searched medicine (or its batch)'],['other_general','Other medicine or general warning'],['unclear','Cannot determine']];
// The quoted evidence is shown in full above the question. Repeat a short
// quotation in the question so the product-to-attribute relationship is explicit.
// Very long quotations use a short reference to avoid unreadable questions.
function quotedTerm(q){
 const t=String(q?.quote??'').replace(/\s+/g,' ').trim();
 return t.length<=100 ? '“'+t+'”' : 'the quotation above';
}
const QUESTIONS={
 P1:(n,q)=>'Does the saved webpage use '+quotedTerm(q)+' as the name of '+n+', rather than another medicine?',
 P2:(n,q)=>'Does the saved webpage clearly identify '+quotedTerm(q)+' as an active ingredient of '+n+'?',
 P3:(n,q)=>'Does the saved webpage clearly state that '+quotedTerm(q)+' describes the active-ingredient strength or concentration of '+n+'?',
 P4:(n,q)=>'Does the saved webpage clearly state that '+quotedTerm(q)+' describes the dosage form of '+n+'?',
 P5:(n,q)=>'Does the saved webpage explicitly identify '+quotedTerm(q)+' as the manufacturer or marketer of '+n+'?',
 W1:(n,q)=>'Does the quoted passage describe a medicine-quality warning on this webpage?'
};
let i=0, S={}, reviewer='';
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function key(){return VERSION+'_'+BUILD+'_'+reviewer}
function state(cid){return S[cid]||(S[cid]={})}
function answer(cid,qid){return state(cid)[qid]||(state(cid)[qid]={})}
function saveLocal(){if(reviewer)try{localStorage.setItem(key(),JSON.stringify(S))}catch(e){alert('Local storage is unavailable. Export CSV now.')}}
function ensureReviewer(){const r=$('reviewer').value.trim().toUpperCase();if(!/^[A-Z0-9._-]{1,24}$/.test(r)){alert('Enter reviewer code (e.g., HM).');return false}if(r!==reviewer){reviewer=r;try{S=JSON.parse(localStorage.getItem(key())||'{}')}catch(e){S={}}render()}return true}
function opts(v,qid,field,options){return '<div class="choices">'+options.map(([k,t])=>'<button type="button" class="choice'+(v[field]===k?' on':'')+'" data-q="'+qid+'" data-field="'+field+'" data-val="'+k+'">'+esc(t)+'</button>').join('')+'</div>'}
function fullSource(d){const t=Array.from(d.source_text), ranges=[];d.quotes.forEach(q=>q.occ.forEach(o=>ranges.push({a:o.start,b:o.end,k:q.location_kind==='spacing_only'?'spacing':'model'})));
 const pts=[...new Set([0,t.length,...ranges.flatMap(z=>[z.a,z.b])])].filter(x=>x>=0&&x<=t.length).sort((a,b)=>a-b);let s='';for(let j=0;j<pts.length-1;j++){const a=pts[j],b=pts[j+1];const k=ranges.find(x=>x.a<=a&&x.b>=b);const part=esc(t.slice(a,b).join(''));s+=k?'<mark class="'+k.k+'">'+part+'</mark>':part}return s}
function verdictError(d,q,v){const val=d.signal==='W1'?v.warning_quality:v.support;
 if(!val)return 'Choose Yes, No, or Cannot determine.';
 if(d.signal==='W1'&&val==='yes'&&(!v.warning_target||!v.warning_official))return 'For a quality warning, answer target and named-authority questions.';
 if(val==='yes'&&q.location_kind!=='normalized'&&!v.manualChecked)return 'Confirm you inspected the saved page before answering Yes (automatic location is uncertain).';
 return ''}
function render(){const d=DATA[i];if(!d)return;const st=state(d.id);$('pos').textContent='Case '+(i+1)+' / '+DATA.length;$('done').textContent=DATA.filter(c=>!!state(c.id)._done).length+' completed';$('searched').textContent='Searched medicine: '+d.searched;$('signal').textContent=d.title;$('rule').textContent=d.rule;$('guide').textContent='No prior knowledge of this medicine is needed. Use only this saved webpage to check whether the quoted information concerns '+d.searched+' and supports the stated attribute. Do not guess from the brand name or look up medicine details elsewhere. Yellow highlights help locate text; they are not automatic Yes answers.';
 $('quotes').innerHTML=d.quotes.map((q,k)=>{const v=st[q.id]||{},count=q.occ.length;let s='<article class="quote"><div class="n">Quotation '+(k+1)+' / '+d.quotes.length+'</div><div class="quotation">'+esc(q.quote)+'</div>';
 s+='<div class="badge">'+(q.location_kind==='normalized'?'Found in the saved page (normalized text match)':q.location_kind==='spacing_only'?'Possible match after ignoring spaces — please inspect':'No reliable automatic quotation match — check the full saved page')+(count>1?' · '+count+' occurrences':'')+'</div>';
 if(count){const excerpt=(o,j)=>'<div class="excerpt">'+(count>1?'<b>Occurrence '+(j+1)+'</b> · ':'')+esc(o.before)+'<mark class="'+(q.location_kind==='spacing_only'?'spacing':'model')+'">'+esc(o.focus)+'</mark>'+esc(o.after)+'</div>';s+=q.occ.slice(0,3).map(excerpt).join('');if(count>3)s+='<details class="more"><summary>Show '+(count-3)+' other occurrences</summary>'+q.occ.slice(3).map((x,j)=>excerpt(x,j+3)).join('')+'</details>'}
 else s+='<div class="excerpt muted"><b>Quotation not located automatically.</b> Open <b>Full saved webpage text</b> below and check whether the quoted words appear in a passage about <b>'+esc(d.searched)+'</b>. Saved text sometimes joins words or changes spacing. <b>Do not select No only because the automatic search failed.</b></div>';
 if(d.signal==='W1'){
   s+='<div class="question">'+esc(QUESTIONS.W1(d.searched,q))+'</div>'+opts(v,q.id,'warning_quality',YY);
   if(v.warning_quality==='yes'){
     s+='<div class="question">Does this warning concern '+esc(d.searched)+', another medicine, or medicines generally?</div>'+opts(v,q.id,'warning_target',TARGET);
     s+='<div class="question">Does the saved webpage attribute this warning to a named regulator or public-health body?</div>'+opts(v,q.id,'warning_official',YY);
   }
   if(q.issuer_quote)s+='<div class="muted" style="margin-top:10px">Model quotation naming a possible authority:</div><div class="quotation">'+esc(q.issuer_quote)+'</div>';
 }else{s+='<div class="question">'+esc(QUESTIONS[d.signal](d.searched,q))+'</div>'+opts(v,q.id,'support',ANS)}
 const value=d.signal==='W1'?v.warning_quality:v.support;
 if(value==='yes'&&q.location_kind!=='normalized')s+='<div class="smallwarn"><label class="checkline"><input type="checkbox" data-check="'+q.id+'"'+(v.manualChecked?' checked':'')+'>I checked the saved page and confirmed that this quotation occurs in the relevant passage.</label></div>';
 s+='<div class="question" style="font-weight:500;font-size:12px;color:#687d90">Optional note (not required)</div><textarea class="note" data-note="'+q.id+'" placeholder="Only if clarification is useful">'+esc(v.note||'')+'</textarea>';
 return s+'</article>'}).join('');
 $('source').innerHTML=fullSource(d);$('error').textContent=st._done?'Case complete':'';$('error').classList.toggle('ok',!!st._done);
 document.querySelectorAll('[data-q]').forEach(b=>b.onclick=()=>{if(!ensureReviewer())return;const v=answer(d.id,b.dataset.q);v[b.dataset.field]=b.dataset.val;if(b.dataset.field==='warning_quality'&&v.warning_quality!=='yes'){v.warning_target='';v.warning_official=''}if(b.dataset.field==='support'||b.dataset.field==='warning_quality')v.manualChecked=false;state(d.id)._done=false;saveLocal();render()});
 document.querySelectorAll('[data-check]').forEach(b=>b.onchange=()=>{if(!ensureReviewer())return;answer(d.id,b.dataset.check).manualChecked=b.checked;state(d.id)._done=false;saveLocal();render()});
 document.querySelectorAll('[data-note]').forEach(el=>el.oninput=()=>{if(!ensureReviewer())return;answer(d.id,el.dataset.note).note=el.value;state(d.id)._done=false;saveLocal()});
}
function finish(){if(!ensureReviewer())return;const d=DATA[i],r=state(d.id);for(const q of d.quotes){const error=verdictError(d,q,r[q.id]||{});if(error){$('error').classList.remove('ok');$('error').textContent='Quotation '+q.id.slice(1)+': '+error;return}}r._done=true;saveLocal();render()}
function go(step){if(!ensureReviewer())return;i=Math.min(DATA.length-1,Math.max(0,i+step));$('full').open=false;render()}
$('reviewer').onchange=ensureReviewer;$('finish').onclick=finish;$('prev').onclick=()=>go(-1);$('next').onclick=()=>go(1);
$('save').onclick=()=>{if(!ensureReviewer())return;const rows=[['case_id','quote_id','reviewer','build_id','version','complete',...F]];
 for(const d of DATA){const st=state(d.id);for(const q of d.quotes){const v=st[q.id]||{},w=d.signal==='W1',support=w?(v.warning_quality||''):(v.support||'');const prod=w?(support==='yes'?({this:'searched',other_general:'other_or_general',unclear:'unclear'}[v.warning_target]||''):(support==='unclear'?'unclear':'')):(support==='yes'?'searched':support==='unclear'?'unclear':'');const loc=q.occ.length?'yes':'no';const checked=(support==='yes'&&q.location_kind!=='normalized'&&v.manualChecked)?'checked':'';rows.push([d.id,q.id,reviewer,BUILD,VERSION,st._done?'1':'0',loc,checked,prod,support,'',w?(v.warning_quality||''):'',w&&support==='yes'?(v.warning_target||''):'',w&&support==='yes'?(v.warning_official||''):'',v.note||'','','','',''])}}
 const csv=x=>'"'+String(x??'').replace(/"/g,'""')+'"';const text='\ufeff'+rows.map(row=>row.map(csv).join(',')).join('\r\n')+'\r\n';const blob=new Blob([text],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='semantic_evidence_'+reviewer+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000)};
render();
</script></body></html>'''

if __name__=='__main__':main()
