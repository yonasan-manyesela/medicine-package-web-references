#!/usr/bin/env python3
"""Reproducible reconciliation of CH and A1 grouped semantic quotation audit.
No sampling-weighted generalizations; no manuscript edits. Standard-library CSV parsing.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,math

ROOT=Path('/mnt/data'); OUT=ROOT/'CH_A1_adjudication_2026-11-22'

def load(n):
    with (ROOT/n).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write(n, rows, fields):
    with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader();w.writerows(rows)

ch=load('semantic_evidence_CH (2).csv'); a1=load('semantic_evidence_A1.csv');mp=load('semantic_group_map.csv');manifest=load('semantic_group_manifest.csv')
keys=lambda row:(row['case_id'],row['quote_id'])
ch_by={keys(x):x for x in ch};a1_by={keys(x):x for x in a1};mani={x['group_case_id']:x for x in manifest}
assert len(ch)==len(a1)==len(ch_by)==len(a1_by)==508
assert set(ch_by)==set(a1_by)
assert {x['group_case_id'] for x in mp}==set(mani)
assert len({(x['old_case_id'],x['old_quote_id']) for x in mp})==len(mp)==572
assert len({(x['group_case_id'],x['group_quote_id']) for x in mp})==len(ch)
assert all(ch_by[k]['build_id']==a1_by[k]['build_id'] and ch_by[k]['version']==a1_by[k]['version'] for k in ch_by)
assert all(mani[k[0]]['build_id']==ch_by[k]['build_id'] for k in ch_by)
assert all(ch_by[k]['complete']=='1' for k in ch_by)
for cid in set(x['case_id'] for x in a1):
    assert len({x['complete'] for x in a1 if x['case_id']==cid})==1, f'Partial case {cid}'

completed={x['case_id'] for x in a1 if x['complete']=='1'}
corrections=[]; merged=[]
for orig in ch:
    z=dict(orig); r=a1_by[keys(orig)]
    if r['complete']=='1':
        for f in ['located','support','warning_quality','note']:
            if r[f] and r[f]!=orig[f]:
                corrections.append({'case_id':r['case_id'],'quote_id':r['quote_id'],'signal':mani[r['case_id']]['signal'],
                                    'field':f,'CH_value':orig[f],'A1_value':r[f], 'reason_a1':r['note']})
                z[f]=r[f]
        # Preserve CH's target and authority fields: simplified A1 interface did NOT reassess W2 or W3.
        if z['support']=='no' and not z['reason']:
            if mani[z['case_id']]['signal']=='W1':z['reason']='clinical_caution'
            elif z['case_id'] in {'C0017','C0052'}:z['reason']='other_product'
        z['reviewer']='CH+A1'
    merged.append(z)
write('audit_merged_CH_with_completed_A1.csv',merged,list(ch[0]))
write('field_changes_CH_to_A1.csv',corrections,['case_id','quote_id','signal','field','CH_value','A1_value','reason_a1'])

# row and case audits
rows_by_case=defaultdict(list)
for r in merged:rows_by_case[r['case_id']].append(r)
for cid in rows_by_case: assert len(rows_by_case[cid])==int(mani[cid]['quote_count'])

# Completeness: CH and A1 outside changed cases identical
for r in merged:
    if r['case_id'] not in completed:
        assert r==ch_by[keys(r)]
# scoring: non-warning evidence needs quoted text + signal + searched-product link;
# warning signal W1 valid regardless of target, and product-specific requires W2=this.
def score(row,signal,strict_target=False):
    loc=row['located']=='yes';supp=row['support']=='yes'
    if signal=='W1':
        return int(loc and supp and (not strict_target or row['warning_target']=='this'))
    return int(loc and supp and row['product_ref']=='searched')
def all3(row):return int(row['located']=='yes' and row['support']=='yes' and row['product_ref']=='searched')

mapped=defaultdict(list)
for r in mp:
    mapped[(r['group_case_id'],r['group_quote_id'])].append(r)
expanded=[]
for rec in merged:
    for x in mapped[keys(rec)]:
        expanded.append({**{k:v for k,v in x.items()},'CH_valid_signal':score(ch_by[keys(rec)],x['signal']),
                         'A1_valid_signal':score(rec,x['signal']),
                         'CH_valid_product_specific':score(ch_by[keys(rec)],x['signal'],strict_target=True),
                         'A1_valid_product_specific':score(rec,x['signal'],strict_target=True),
                         'CH_three_checks':all3(ch_by[keys(rec)]), 'A1_three_checks':all3(rec),
                         'corrected_case':int(rec['case_id'] in completed),
                         'original_support':ch_by[keys(rec)]['support'], 'reconciled_support':rec['support'],
                         'original_target':ch_by[keys(rec)]['warning_target'], 'reconciled_target':rec['warning_target']})
assert len(expanded)==572
write('expanded_model_quote_results.csv',expanded,list(expanded[0]))

# summarize grouped unique quotations; expanded model-specific with model-only/both positive quotes
summ=[]
def summarize(rows, grp_name, label, metric):
    N=len(rows);s=sum(int(x[metric]) for x in rows)
    return {'scope':grp_name,'label':label,'metric':metric,'valid':s,'audited':N,'percentage': f'{100*s/N:.1f}' if N else ''}
for dataset,name in [(ch,'CH'),(merged,'CH+A1')]:
    summ.append({'scope':'grouped_unique','label':name,'metric':'all3','valid':sum(all3(r) for r in dataset),'audited':len(dataset),'percentage':f'{100*sum(all3(r) for r in dataset)/len(dataset):.1f}'})
    summ.append({'scope':'grouped_unique','label':name,'metric':'located','valid':sum(r['located']=='yes' for r in dataset),'audited':len(dataset),'percentage':f'{100*sum(r["located"]=="yes" for r in dataset)/len(dataset):.1f}'})
    summ.append({'scope':'grouped_unique','label':name,'metric':'located_support','valid':sum(r['located']=='yes' and r['support']=='yes' for r in dataset),'audited':len(dataset),'percentage':f'{100*sum(r["located"]=="yes" and r["support"]=="yes" for r in dataset)/len(dataset):.1f}'})
for model in sorted({x['model'] for x in expanded}):
    for sig in ['P5','W1']:
        subset=[r for r in expanded if r['model']==model and r['signal']==sig and r['stratum'] in {'both','model_only'}]
        for metric in ['CH_valid_signal','A1_valid_signal','CH_valid_product_specific','A1_valid_product_specific']:
            summ.append(summarize(subset,'model_positive_'+sig,model,metric))
    # per signal and stratum; non-W1 sampled both, model-only oversampled
    for sig in ['P1','P2','P3','P4','P5','W1']:
        for stratum in ['both','model_only','human_only']:
            sub=[r for r in expanded if r['model']==model and r['signal']==sig and r['stratum']==stratum]
            if not sub:continue
            summ.append(summarize(sub,f'{sig}/{stratum}',model,'A1_valid_signal'))
write('semantic_result_summary.csv',summ,['scope','label','metric','valid','audited','percentage'])

# show the specific pending groups requiring product-target adjudication
pending=['C0048','C0073','C0254','C0275','C0299']
needs=[]
for cid in pending:
    for r in rows_by_case[cid]:
        needs.append({'case_id':cid,'quote_id':r['quote_id'],'signal':mani[cid]['signal'],
                      'models':';'.join(sorted({x['model'] for x in mapped[keys(r)]})),
                      'CH_warning_quality':ch_by[keys(r)]['warning_quality'],
                      'CH_warning_target':ch_by[keys(r)]['warning_target'],
                      'A1_case_complete':int(cid in completed),
                      'action':'Recheck W2 target from saved webpage; W1 if needed'})
write('remaining_target_cases.csv',needs,['case_id','quote_id','signal','models','CH_warning_quality','CH_warning_target','A1_case_complete','action'])

# integrity: CH untouched original; optional reason-coded fields added to merged from adjudicated interpretation only
counters={
'CH_cases':len(rows_by_case),'CH_grouped_quotes':len(ch),
'A1_complete_cases':len(completed),'A1_complete_quotes':sum(x['complete']=='1' for x in a1),
'A1_changed_quotes':len({(x['case_id'],x['quote_id']) for x in corrections}),
'A1_changed_fields':len(corrections),'A1_W1_rejected':sum(x['case_id'] in completed and mani[x['case_id']]['signal']=='W1' and x['support']=='no' for x in merged),
'CH_valid_all3':sum(all3(r) for r in ch),'merged_valid_all3':sum(all3(r) for r in merged),
'CH_located_and_support':sum(r['located']=='yes' and r['support']=='yes' for r in ch),
'merged_located_and_support':sum(r['located']=='yes' and r['support']=='yes' for r in merged),
'mapped_original_quotes':len(mp),'model_positive_W1_quotes':sum(x['signal']=='W1' and x['stratum'] in {'both','model_only'} for x in expanded),
'model_positive_P5_quotes':sum(x['signal']=='P5' and x['stratum'] in {'both','model_only'} for x in expanded),
'pending_W2_cases':pending
}
(OUT/'audit_validation.json').write_text(json.dumps(counters,indent=2))
print(json.dumps(counters,indent=2))
for r in summ:
 if r['scope'] in ['model_positive_P5','model_positive_W1'] and r['metric'] in ('CH_valid_signal','A1_valid_signal','A1_valid_product_specific'):
  print(r)
