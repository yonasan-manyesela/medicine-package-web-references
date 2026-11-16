"""
previous_pipeline_ievc2026.py
Runs LLM signal extraction + deterministic labelling on the fetched source pages.

Changes from the June version (rules and aggregation are UNCHANGED):
  1. The model also receives the website domain of the hosting page
     ("Website domain: ..."), the same in all three prompt conditions.
  2. The prompt no longer names websites or products from the evaluation data;
     the few-shot examples use invented names (Product A, Company X, *.example).
  3. Signals are numbered S1-S6 in the prompt, as in the paper.
  4. Pages are read from tables/pipeline_top3_all.csv (the pages actually
     selected), not from whatever rank files exist in the pipeline folders.
  5. Failed API calls are retried twice, then recorded as status=api_error and
     left out of aggregation (they are NOT counted as "insufficient").
  6. Every product gets an aggregated row; products with no selected page have
     n_pages = 0.
  7. The exact prompt, model settings and returned model version are saved.

Label codes in the CSV files are kept for compatibility. Names used in the paper:
  "suggests suspicious reference" -> warning or falsification-related information
  "supports normal reference"     -> routine product information
  "insufficient evidence"         -> insufficient webpage evidence

Usage (from the repository root):
    python scripts/3_source_context/previous_pipeline_ievc2026.py --run_id fresh_oct --condition cot
    python scripts/3_source_context/previous_pipeline_ievc2026.py --run_id fresh_oct --condition cot --model llama-3.1-8b-instant
  GPT-4o uses the key in OPENAI_API_KEY2_lab; Llama runs on Groq with the key in llama_api_key,
  with the SAME prompts, inputs, rules and error handling. Llama files get the model in the name,
  e.g. tables/llm_results_cot__llama-3.1-8b-instant.csv.
  --resume  reuses pages already labelled by the same model/condition/prompt and calls the model only
            for missing or failed pages (useful after a quota stop).
  The run stops at once on a wrong/expired key or a used-up daily quota (no CSV is written then).
"""
import os, re, json, argparse, time, urllib.parse
from datetime import datetime
from pathlib import Path, PureWindowsPath
import pandas as pd
from openai import OpenAI

EXP_ROOT = Path(__file__).resolve().parents[2]  # repository root
L_SUSPICIOUS   = "suggests suspicious reference"
L_NORMAL       = "supports normal reference"
L_INSUFFICIENT = "insufficient evidence"

PROMPT_VERSION = "v2_2026-10-21"
MODEL, TEMPERATURE, MAX_TOKENS = "gpt-4o", 0, 800       # MODEL can be changed with --model
GROQ_URL = "https://api.groq.com/openai/v1"
MAX_CHARS, MIN_WORDS = 6000, 150

ALLOWED = {
    'regulatory_alert':    {'yes', 'no', 'unclear'},
    'named_manufacturer':  {'yes', 'no', 'unclear'},
    'warning_language':    {'present', 'absent', 'unclear'},
    'commercial_language': {'strong', 'moderate', 'absent', 'unclear'},
    'source_authority':    {'high', 'medium', 'low', 'unclear'},
    'product_specificity': {'high', 'medium', 'low', 'unrelated', 'unclear'},
}

def clean_signals(signals):
    """Lower-case the six values; anything missing or outside the allowed set
    becomes 'unclear' and is listed in invalid_signals."""
    out, bad = {}, []
    for k, allowed in ALLOWED.items():
        v = str(signals.get(k, '')).strip().lower()
        if v not in allowed:
            bad.append(f"{k}={v or 'missing'}"); v = 'unclear'
        out[k] = v
    return out, ';'.join(bad)

def assign_label(signals):
    """
    Deterministic label assignment from the six extracted signals (UNCHANGED).

    Signals:
      S1: regulatory_alert     — official alert/recall/falsification warning
      S2: named_manufacturer   — manufacturer explicitly named
      S3: warning_language     — falsified/counterfeit/recall/seized terms
      S4: commercial_language  — price/discount/buy now/free shipping
      S5: source_authority     — high/medium/low/unclear
      S6: product_specificity  — high/medium/low/unrelated/unclear
    """
    s1 = signals.get('regulatory_alert',   'unclear').lower()
    s2 = signals.get('named_manufacturer', 'unclear').lower()
    s3 = signals.get('warning_language',   'unclear').lower()
    s4 = signals.get('commercial_language','unclear').lower()
    s5 = signals.get('source_authority',   'unclear').lower()
    s6 = signals.get('product_specificity','unclear').lower()

    # ── Gate: page is not about this product ─────────────────────────
    if s6 == 'unrelated':
        return L_INSUFFICIENT, 0.15
    if s6 == 'unclear' and s5 == 'unclear' and s2 == 'unclear':
        return L_INSUFFICIENT, 0.20

    # ── WARNING OR FALSIFICATION-RELATED ─────────────────────────────
    if s1 == 'yes' and s5 in ['high', 'medium']:
        return L_SUSPICIOUS, 0.90
    if s1 == 'yes' and s5 == 'low':
        return L_SUSPICIOUS, 0.65
    if s1 == 'yes':  # s5 = unclear but alert present
        return L_SUSPICIOUS, 0.70
    if s3 == 'present' and s5 in ['high', 'medium']:
        return L_SUSPICIOUS, 0.80
    if s3 == 'present' and s5 == 'low':
        return L_SUSPICIOUS, 0.55

    # ── ROUTINE PRODUCT INFORMATION ──────────────────────────────────
    # N1: named manufacturer + high/medium authority + no alert/warning
    if (s2 == 'yes' and s5 in ['high', 'medium']
            and s1 == 'no' and s3 in ['absent', 'unclear']):
        return L_NORMAL, 0.85
    # N2: named manufacturer + high/medium specificity + no alert/warning
    if (s2 == 'yes' and s6 in ['high', 'medium']
            and s1 == 'no' and s3 in ['absent', 'unclear']):
        return L_NORMAL, 0.65
    # N3: high/medium specificity + high/medium authority + no alert/warning
    if (s6 in ['high', 'medium'] and s5 in ['high', 'medium']
            and s1 == 'no' and s3 in ['absent', 'unclear']):
        return L_NORMAL, 0.75
    # N4: high/medium specificity + low authority + no alert/warning + not strongly commercial
    if (s6 in ['high', 'medium'] and s5 == 'low'
            and s1 == 'no' and s3 in ['absent', 'unclear']
            and s4 != 'strong'):
        return L_NORMAL, 0.50
    # N5: named manufacturer + high authority + low specificity + no alert/warning
    if (s2 == 'yes' and s5 == 'high'
            and s6 == 'low'
            and s1 == 'no' and s3 in ['absent', 'unclear']):
        return L_NORMAL, 0.60

    return L_INSUFFICIENT, 0.25

def aggregate(page_results):
    """
    Aggregate the labels of the top-k pages of one product (UNCHANGED rules):
      1. any warning page                        -> warning
      2. else at least one routine page          -> routine
      3. else                                    -> insufficient
    Pages with status api_error or missing_file are left out. If every page failed -> 'error'.
    The returned number is the mean rule weight of the winning label
    (a fixed weight from assign_label, not a calibrated probability).
    """
    ok = [r for r in page_results if r['status'] not in ('api_error', 'missing_file')]
    if not ok:
        return 'error', None
    labels = [r['label'] for r in ok]
    confs = [r['confidence_adj'] for r in ok]
    for lab in (L_SUSPICIOUS, L_NORMAL):
        if lab in labels:
            c = [x for l, x in zip(labels, confs) if l == lab]
            return lab, round(sum(c) / len(c), 3)
    return L_INSUFFICIENT, round(sum(confs) / len(confs), 3)

SYSTEM = ("You are a pharmaceutical packaging analyst. "
          "You are NOT authenticating the medicine. "
          "Assess what this webpage implies about a retrieved packaging image "
          "as an inspection reference. Report only what is in the text.")

SIGNALS_DEF = """
S1. regulatory_alert: Does the page contain an official alert, recall notice,
    seizure notice, or falsification warning from a government body, WHO, or a
    medicines regulatory authority that refers specifically to this product or
    its active ingredient?
    Values: yes / no / unclear

S2. named_manufacturer: Is a manufacturer, brand owner, or official distributor
    explicitly named in the text?
    Values: yes / no / unclear

S3. warning_language: Does the text contain terms such as: falsified, counterfeit,
    fake, substandard, recall, alert, seized, unregistered, banned?
    Values: present / absent / unclear

S4. commercial_language: Does the text primarily promote sale using terms such as:
    price, discount, buy now, order, free shipping, wholesale, cart?
    Values: strong / moderate / absent / unclear

S5. source_authority: What type of source is this? Use the page text and the
    website domain.
    high    = medicines regulatory authority, WHO, public health body, or the
              manufacturer's own website
    medium  = a website whose main business is medicines: online or retail
              pharmacy, distributor, hospital formulary, drug information
              database. Rate medium even if the pharmacy is small or unknown.
              If the domain contains words such as pharmacy, pharma, med, drug
              or health, rate medium unless the site is a general marketplace.
    low     = general marketplace selling many kinds of goods, personal blog,
              forum, news article, social media, or a site not about medicines
    unclear = cannot be inferred from the text and the domain
    This signal describes the type of source only. It does not verify that a
    seller is licensed.

S6. product_specificity: How specifically does the page describe the queried product?
    high      = product name + dosage/form + at least one of: batch, registration, manufacturer
    medium    = product name + dosage/form OR product name + manufacturer
    low       = product name only mentioned vaguely or briefly
    unrelated = page is about a different product
    unclear   = insufficient text to determine

IMPORTANT LABELING GUIDANCE:
- A pharmacy, manufacturer, drug information, or product catalogue page that
  clearly describes the queried product should be treated as a usable normal
  reference. You do NOT need regulatory confirmation to assign product specificity.
- Only assign regulatory_alert=yes if the page explicitly warns that THIS product
  (or its exact active ingredient combination) is falsified, recalled, or unsafe.
- If the product is clearly described and listed as a commercial product with no
  warning language, assign S6=medium or high and S3=absent.
"""

# Invented products, companies and domains only (none from the evaluation data).
FEW_SHOT_EX = """
[Example 1 Regulatory alert]
Website domain: medicines-authority.example.gov
Text: "Public alert: falsified Product A 500 mg tablets found in circulation. The registered manufacturer, Company X, confirmed that the batch is not genuine."
S1. regulatory_alert: yes | Official alert about this product.
S2. named_manufacturer: yes | Company X named.
S3. warning_language: present | Uses falsified, alert.
S4. commercial_language: absent | No sale language.
S5. source_authority: high | Regulatory authority.
S6. product_specificity: high | Name, strength, form and manufacturer.

[Example 2 Pharmacy]
Website domain: citypharmacy.example
Text: "Product B 250 mg capsules. Manufacturer: Company Y. Price: 850. Add to cart."
S1. regulatory_alert: no | No alert.
S2. named_manufacturer: yes | Company Y named.
S3. warning_language: absent | No warning terms.
S4. commercial_language: moderate | Price and cart.
S5. source_authority: medium | Pharmacy website.
S6. product_specificity: high | Name, strength, form and manufacturer.

[Example 3 Marketplace]
Website domain: shop-everything.example
Text: "Product C tablets 1700. Treats malaria. Free shipping on orders over 5000."
S1. regulatory_alert: no | No alert.
S2. named_manufacturer: no | No manufacturer.
S3. warning_language: absent | No warning.
S4. commercial_language: strong | Price and free shipping.
S5. source_authority: low | General marketplace.
S6. product_specificity: low | Name only.
"""

JSON_SIGNALS = ('{"regulatory_alert":"yes/no/unclear","named_manufacturer":"yes/no/unclear",'
                '"warning_language":"present/absent/unclear","commercial_language":"strong/moderate/absent/unclear",'
                '"source_authority":"high/medium/low/unclear","product_specificity":"high/medium/low/unrelated/unclear"}')

def header(drug, domain):
    return f"Product: {drug}\nWebsite domain: {domain or 'unknown'}\n\nYou are NOT authenticating the medicine."

def prompt_zero_shot(drug, domain, text):
    return (f"{header(drug, domain)}\n{SIGNALS_DEF}\n"
            f'Return JSON: {{"summary":"...","signals":{JSON_SIGNALS},"confidence_raw":0.0}}\n\nWebpage:\n{text}')

def prompt_few_shot(drug, domain, text):
    return (f"{header(drug, domain)}\nAssess what this webpage implies about the retrieved "
            f"packaging image as an inspection reference.\n{FEW_SHOT_EX}\n"
            f"Now extract the signals for this webpage and return a JSON object with EXACTLY this structure:\n"
            f'{{"summary":"one sentence description","signals":{JSON_SIGNALS},"confidence_raw":0.0}}\n\nWebpage:\n{text}')

def prompt_cot(drug, domain, text):
    return (f"{header(drug, domain)}\nStep 1: Source type? (regulatory/pharmacy/ecommerce/news/blog/other)\n"
            f"Step 2: List evidence per signal.\nStep 3: Assign values.\n{SIGNALS_DEF}\n"
            f'Return JSON: {{"step1_source_type":"...","step2_evidence":"...","summary":"...",'
            f'"signals":{JSON_SIGNALS},"confidence_raw":0.0}}\n\nWebpage:\n{text}')

PROMPTS = {'zero_shot': prompt_zero_shot, 'few_shot': prompt_few_shot, 'cot': prompt_cot}

def parse_json(text):
    """JSON object from the reply; if the model added text around it, take the outer {...}."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text or '', re.DOTALL)
        if not m:
            raise ValueError("no JSON in reply")
        return json.loads(m.group())

JSON_MODE = [True]

class StopRun(Exception):
    """Errors that retrying cannot fix (wrong/expired key, daily quota used up)."""

def is_fatal(e):
    m = str(e).lower()
    return (getattr(e, 'status_code', None) == 401 or 'authentication' in type(e).__name__.lower()
            or 'per day' in m or '(tpd)' in m or '(rpd)' in m)

def call_llm(client, prompt, tries=6):
    """Returns (parsed JSON, model version). Same request for every model (JSON mode, temperature 0).
    Rate-limit errors (HTTP 429) wait longer; other errors are retried briefly. Raises after `tries`."""
    for i in range(tries):
        try:
            extra = {"response_format": {"type": "json_object"}} if JSON_MODE[0] else {}
            resp = client.chat.completions.create(
                model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], **extra)
            return parse_json(resp.choices[0].message.content), getattr(resp, 'model', MODEL)
        except Exception as e:
            if is_fatal(e):
                raise StopRun(f"{type(e).__name__}: {str(e)[:200]}")
            if JSON_MODE[0] and 'response_format' in str(e):
                JSON_MODE[0] = False       # endpoint without JSON mode: ask again, parse the reply
                print("  [note] JSON mode not supported by this model; parsing the JSON from the reply")
                continue
            if i == tries - 1:
                raise
            limited = getattr(e, 'status_code', None) == 429 or 'rate' in str(e).lower()
            time.sleep(min(60, 10 * 2 ** i) if limited else 2 * (i + 1))

def load_previous(path):
    """A page result saved by an earlier run of the same model/condition/prompt, if it was labelled."""
    try:
        prev = json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return None
    if prev.get('status') == 'ok' and prev.get('prompt_version') == PROMPT_VERSION and prev.get('model', MODEL) == MODEL:
        return prev
    return None

def read_page_file(page_file):
    try:
        with open(page_file, encoding='utf-8') as f: raw = f.read()
        lines = raw.split('\n'); meta = {}; text_start = 0
        for i, line in enumerate(lines):
            if line.startswith('=' * 10): text_start = i + 1; break
            if ':' in line:
                k, v = line.split(':', 1); meta[k.strip()] = v.strip()
        return meta, '\n'.join(lines[text_start:])
    except Exception:
        return {}, ''

def locate(page_file, pipe_dir):
    """Path stored by 02 (absolute); fall back to <pipe_dir>/<folder>/<file>."""
    p = Path(page_file)
    if p.exists():
        return p
    w = PureWindowsPath(page_file)
    return pipe_dir / w.parent.name / w.name

def domain_of(url):
    return urllib.parse.urlsplit(str(url).strip()).netloc.lower().removeprefix('www.')

def save_prompt(tables_dir, cond, tag=''):
    p = tables_dir / f'prompt_{cond}{tag}_{PROMPT_VERSION}.txt'
    p.write_text(f"PROMPT_VERSION: {PROMPT_VERSION}\nMODEL: {MODEL}\nTEMPERATURE: {TEMPERATURE}\n"
                 f"MAX_TOKENS: {MAX_TOKENS}\nPAGE TEXT: first {MAX_CHARS} characters; pages under "
                 f"{MIN_WORDS} words are not sent\n\n=== SYSTEM ===\n{SYSTEM}\n\n=== USER (template) ===\n"
                 + PROMPTS[cond]('<PRODUCT NAME>', '<website domain>', '<PAGE TEXT>'), encoding='utf-8')

def main():
    global MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', default='fresh_oct')
    parser.add_argument('--condition', default='cot', choices=['zero_shot', 'few_shot', 'cot'])
    parser.add_argument('--model', default=MODEL,
                        help="gpt-4o (OpenAI, default) or e.g. llama-3.1-8b-instant (Groq)")
    parser.add_argument('--sleep', type=float, default=1.0, help="seconds between calls")
    parser.add_argument('--resume', action='store_true',
                        help="reuse pages already labelled (status ok) by this model, condition and prompt "
                             "version; call the model only for pages that are missing or failed")
    args = parser.parse_args()

    # Same OpenAI-style client for both providers; only the endpoint and key differ.
    MODEL = args.model
    groq = not MODEL.startswith('gpt')
    key_env = 'llama_api_key' if groq else 'OPENAI_API_KEY2_lab'
    api_key = (os.environ.get(key_env) or '').strip().strip('"').strip("'")
    if not api_key:
        raise SystemExit(f"Set the {key_env} environment variable.")
    client = OpenAI(api_key=api_key, base_url=GROQ_URL) if groq else OpenAI(api_key=api_key)
    # GPT-4o keeps the original file names; other models get the model in the name
    tag = '' if MODEL == 'gpt-4o' else '__' + re.sub(r'[^A-Za-z0-9.-]+', '-', MODEL)
    print(f"model: {MODEL} | condition: {args.condition} | output tag: '{tag or '(none)'}'")

    run_dir = EXP_ROOT / 'runs' / args.run_id
    pipe_dir, llm_dir, tables_dir = run_dir / 'pipeline', run_dir / 'llm_outputs', run_dir / 'tables'
    llm_dir.mkdir(parents=True, exist_ok=True)
    save_prompt(tables_dir, args.condition, tag)

    summary = pd.read_csv(tables_dir / 'match_summary.csv', dtype=str).fillna('')
    pages = pd.read_csv(tables_dir / 'pipeline_top3_all.csv', dtype=str).fillna('')
    pages['rank'] = pages['rank'].astype(int)

    build_prompt = PROMPTS[args.condition]
    all_results, all_agg = [], []

    for _, prod in summary.iterrows():
        qid, drug = prod['query_id'], prod['drug']
        sel = pages[pages['query_id'] == qid].sort_values('rank')
        print(f"\n{'=' * 60}\n[{qid}] {drug} | {args.condition} | {len(sel)} pages")
        page_results = []

        for _, pg in sel.iterrows():
            rank, url = int(pg['rank']), pg['source_page_url']
            page_path = locate(pg['page_file'], pipe_dir)
            meta, text = read_page_file(page_path)
            quality = meta.get('QUALITY', pg.get('content_quality', ''))
            words = len(text.split())          # counted from the text actually sent, not from the CSV
            r = {'query_id': qid, 'drug': drug, 'rank': rank, 'condition': args.condition + tag, 'model': MODEL,
                 'url': url, 'domain': domain_of(url), 'content_quality': quality, 'words': words,
                 'fetch_failed': quality not in ('ok', 'low_content'),
                 'prompt_version': PROMPT_VERSION, 'llm_called': False, 'status': '',
                 'label': '', 'confidence_adj': None, 'invalid_signals': '', 'error': ''}

            if not page_path.exists():
                r.update(status='missing_file', error='page file not found')
                print(f"  rank{rank}: page file missing -> recorded, not labelled")
            elif r['fetch_failed'] or words < MIN_WORDS:
                r.update(status='skipped_fetch' if r['fetch_failed'] else 'skipped_short',
                         label=L_INSUFFICIENT, confidence_adj=0.20)
                print(f"  rank{rank}: skip ({quality}, {words}w)")
            elif args.resume and (prev := load_previous(llm_dir / qid / f'rank{rank}_{args.condition}{tag}.json')):
                r = prev
                print(f"  rank{rank}: reused ({prev['label']})")
            else:
                try:
                    out, model_version = call_llm(client, build_prompt(drug, r['domain'], text[:MAX_CHARS]))
                    signals, bad = clean_signals(out.get('signals', {}))
                    label, conf = assign_label(signals)
                    r.update(status='ok', llm_called=True, label=label, confidence_adj=round(conf, 3),
                             invalid_signals=bad, model_version=model_version,
                             summary=out.get('summary', ''),
                             confidence_raw=out.get('confidence_raw', ''),
                             step1_source=out.get('step1_source_type', ''),
                             step2_evidence=out.get('step2_evidence', ''))
                    for k, v in signals.items():
                        r[f'sig_{k}'] = v
                    print(f"  rank{rank}: {label} | S1={signals['regulatory_alert']} "
                          f"S5={signals['source_authority']} S6={signals['product_specificity']}"
                          + (f" | invalid: {bad}" if bad else ''))
                    time.sleep(args.sleep)
                except StopRun as e:
                    print(f"\n[STOPPED] {e}\nNo CSV written. Pages labelled so far are saved; "
                          f"rerun the same command with --resume to continue.")
                    raise SystemExit(1)
                except Exception as e:
                    r.update(status='api_error', llm_called=True, error=type(e).__name__)
                    print(f"  rank{rank}: API error ({type(e).__name__}) -> recorded, not labelled")
            (llm_dir / f"{qid}").mkdir(exist_ok=True)
            with open(llm_dir / qid / f'rank{rank}_{args.condition}{tag}.json', 'w', encoding='utf-8') as f:
                json.dump(r, f, ensure_ascii=False, indent=2, default=str)
            page_results.append(r); all_results.append(r)

        agg_label, agg_conf = aggregate(page_results) if page_results else (L_INSUFFICIENT, None)
        all_agg.append({'query_id': qid, 'drug': drug, 'condition': args.condition + tag, 'model': MODEL,
                        'aggregated_label': agg_label, 'agg_confidence': agg_conf,
                        'n_pages': len(page_results),
                        'n_api_error': sum(r['status'] in ('api_error', 'missing_file') for r in page_results),
                        'labels': '|'.join(r['label'] or r['status'] for r in page_results),
                        'n_suspicious': sum(r['label'] == L_SUSPICIOUS for r in page_results),
                        'n_normal': sum(r['label'] == L_NORMAL for r in page_results),
                        'n_insufficient': sum(r['label'] == L_INSUFFICIENT for r in page_results),
                        'prompt_version': PROMPT_VERSION})
        print(f"  AGGREGATE: {agg_label}")

    res = pd.DataFrame(all_results)
    res.to_csv(tables_dir / f'llm_results_{args.condition}{tag}.csv', index=False)
    agg = pd.DataFrame(all_agg)
    agg.to_csv(tables_dir / f'aggregated_{args.condition}{tag}.csv', index=False)
    print(f"\n[DONE] {datetime.now():%Y-%m-%d %H:%M}  pages: {len(res)} | "
          + ' | '.join(f"{k}: {v}" for k, v in res['status'].value_counts().items()))
    print(agg[agg.n_pages > 0]['aggregated_label'].value_counts().to_string())
    if res['status'].isin(['api_error', 'missing_file']).any():
        print("[WARN] some pages were not labelled (api_error / missing_file). They are left out of "
              "aggregation. Re-running the condition re-calls ALL pages; report the count instead.")

if __name__ == '__main__':
    main()
