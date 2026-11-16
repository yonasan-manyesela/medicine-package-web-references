"""
08_label_pages_with_language_models.py
Source-context labelling of the selected reference pages with quote-backed evidence.

Pipeline A  signals + rule:  the model returns R, P1-P5 (a verbatim quote or null) and a list
            of warnings (quote, target, official, issuer quote); the label comes from the
            shared rule in context_label_rule.py (the same rule is applied to the human signals).
Pipeline B  decision model:  the model reads the page and returns the label, the
            target_unclear flag and the quotes it relied on. Same definitions, rule given in words.
Both:       the FULL extracted page text is sent (no 6,000-character cut, no word gate).
            Pages with no text are not sent (R = no). Pages that were not fetched are
            not_evaluable. Every quote is checked against the page text:
            grounded / ungrounded / not_asserted (context_label_rule.ground).

Usage (from the repository root; keys in OPENAI_API_KEY2_lab or llama_api_key):
    python scripts/3_source_context/08_label_pages_with_language_models.py --list_models --provider openai
    python scripts/3_source_context/08_label_pages_with_language_models.py --pipeline A --model gpt-4o
    python scripts/3_source_context/08_label_pages_with_language_models.py --pipeline A --model <groq model id>
    python scripts/3_source_context/08_label_pages_with_language_models.py --pipeline B --model <reasoning model id> --reasoning
    add --resume to continue after a stop; --max_chars N only if a provider refuses long pages
    --provider local (Ollama) needs only the Python standard library (no pandas, no openai package)
Inputs (read only): runs/<run_id>/tables/match_summary.csv, pipeline_top3_all.csv and pipeline/
Outputs (runs/<run_id>/source_context_v3/models/):
    context_<A|B>__<model>.csv            one row per page
    context_products_<A|B>__<model>.csv   one row per product
    prompt_context_<A|B>__<model>_<version>.txt  exact prompt and settings used
    raw/<A|B>__<model>/<query>_rank<k>.json   raw reply + parsed result
"""
import argparse, csv, json, os, re, sys, time, urllib.error, urllib.parse, urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_label_rule import full_text_file, local_file
from context_label_rule import (PROMPT_VERSION, P_KEYS, TARGETS, DEFINITIONS, LABEL_RULE,
                           norm, ground, page_label, sensitivity_label, product_label)

EXP_ROOT = Path(__file__).resolve().parents[2]   # repository root
GROQ_URL = "https://api.groq.com/openai/v1"
FETCH_OK = {"ok", "low_content"}
LABELS = {"not_evaluable", "warning", "product_information", "insufficient"}

SYSTEM = ("You extract evidence from webpages about medicines. You are NOT authenticating the "
          "medicine. Report only what the page text states and copy quotes exactly.")

SCHEMA_A = """Return a JSON object with exactly these keys:
{"readable": "yes" or "no",
 "product_named": [ {"quote": quote, "context": line} ],
 "ingredient":    [ {"quote": quote, "context": line} ],
 "strength":      [ {"quote": quote, "context": line} ],
 "form":          [ {"quote": quote, "context": line} ],
 "manufacturer":  [ {"quote": quote, "context": line} ],
 "warnings": [ {"quote": quote, "target": "this" or "other_general" or "cant_tell",
                "official": "yes" or "no", "issuer_quote": quote or null} ]}
Each signal list has one to three items (heading first, otherwise first mention), or is
empty when the page does not state it. "quote" is only the words that show the signal;
"context" is the whole line or sentence it comes from, copied exactly. "warnings" has one item per warning
sentence and is empty when there is no warning."""

SCHEMA_B = """Decide the page label using the definitions and the rule.
Return a JSON object with exactly these keys:
{"label": "not_evaluable" or "warning" or "product_information" or "insufficient",
 "target_unclear": true or false,
 "evidence": [ {"quote": quote, "context": line, "role": "named" or "identity" or "warning"} ]}
"target_unclear" is true when the page has a warning whose target is cant_tell.
"evidence" lists the quotes your decision relies on (empty for not_evaluable or when
nothing supports the product)."""


def build_prompt(pipeline, drug, domain, text):
    head = f"Searched product: {drug}\nWebsite domain: {domain or 'unknown'}\n\n{DEFINITIONS}\n"
    tail = SCHEMA_A if pipeline == "A" else f"{LABEL_RULE}\n{SCHEMA_B}"
    return f"{head}{tail}\n\nWebpage text:\n{text}"


# ---------------------------------------------------------------- model call
class StopRun(Exception):
    """Errors that retrying cannot fix (wrong key, daily quota)."""


def is_fatal(e):
    m = str(e).lower()
    return (getattr(e, "status_code", None) == 401 or "authentication" in type(e).__name__.lower()
            or "per day" in m or "(tpd)" in m or "(rpd)" in m or "insufficient_quota" in m)


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("no JSON in reply")
        return json.loads(m.group())


class Caller:
    """One OpenAI-style client for OpenAI and Groq. Parameters a model refuses (JSON mode,
    temperature, max_tokens, reasoning_effort) are dropped or renamed once and remembered."""

    def __init__(self, client, model, temperature, max_tokens, reasoning, effort):
        self.client, self.model = client, model
        self.opts = {"response_format": {"type": "json_object"}}
        if reasoning:
            self.opts["max_completion_tokens"] = max_tokens
            if effort:
                self.opts["reasoning_effort"] = effort
        else:
            self.opts["temperature"] = temperature
            self.opts["max_tokens"] = max_tokens
        self.notes = []

    def _adapt(self, e):
        m = str(e).lower()
        for k in ("response_format", "temperature", "reasoning_effort"):
            if k in self.opts and k in m:
                self.opts.pop(k); self.notes.append(f"dropped {k}"); return True
        if "max_tokens" in self.opts and "max_tokens" in m and "max_completion_tokens" in m:
            self.opts["max_completion_tokens"] = self.opts.pop("max_tokens")
            self.notes.append("max_tokens -> max_completion_tokens"); return True
        return False

    def __call__(self, prompt, tries=6):
        for i in range(tries + 4):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "system", "content": SYSTEM},
                                                {"role": "user", "content": prompt}], **self.opts)
                content = resp.choices[0].message.content or ""
                u = getattr(resp, "usage", None)
                meta = {"model_version": getattr(resp, "model", self.model), "latency_s": round(time.time() - t0, 2),
                        "prompt_tokens": getattr(u, "prompt_tokens", None), "completion_tokens": getattr(u, "completion_tokens", None),
                        "finish_reason": resp.choices[0].finish_reason}
                return content, meta
            except Exception as e:
                if is_fatal(e):
                    raise StopRun(f"{type(e).__name__}: {str(e)[:200]}")
                if getattr(e, "status_code", None) == 400 and self._adapt(e):
                    print(f"  [note] {self.notes[-1]} for {self.model}")
                    continue
                if getattr(e, "status_code", None) == 413 or "too large" in str(e).lower():
                    raise ValueError(f"too_large: {str(e)[:300]}")
                if i >= tries - 1:
                    raise
                limited = getattr(e, "status_code", None) == 429 or "rate" in str(e).lower()
                time.sleep(min(60, 10 * 2 ** i) if limited else 2 * (i + 1))


def ns(x):
    return SimpleNamespace(**{k: ns(v) for k, v in x.items()}) if isinstance(x, dict) else [ns(v) for v in x] if isinstance(x, list) else x


class LocalClient:
    """OpenAI-style client for a local Ollama server, standard library only (no compiled packages needed)."""
    def __init__(self, base_url, timeout=900):
        self.base, self.timeout = base_url.rstrip("/"), timeout
        self.open = urllib.request.build_opener(urllib.request.ProxyHandler({})).open     # localhost: no proxy
        self.chat = ns({"completions": {}}); self.chat.completions.create = lambda **b: ns(self.req("/chat/completions", b))
        self.models = SimpleNamespace(list=lambda: ns(self.req("/models")))

    def req(self, path, body=None):
        r = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body else None,
                                   headers={"Content-Type": "application/json"})
        try:
            with self.open(r, timeout=self.timeout) as f:
                return json.loads(f.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err = RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"); err.status_code = e.code
            raise err


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, p):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval=""); w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------- parse + ground
def yn(v):
    v = str(v).strip().lower() if v is not None else ""
    return v if v in ("yes", "no") else ""


def quote_of(v):
    if isinstance(v, dict):
        v = v.get("quote")
    return v if isinstance(v, str) and v.strip() else None


def quotes_of(v, cap=3):
    """A signal value as a list of (quote, context) pairs (list, string, dict or null accepted); at most `cap`."""
    if v is None:
        return []
    if not isinstance(v, list):
        v = [v]
    out = []
    for x in v:
        q = quote_of(x)
        if q:
            c = x.get("context") if isinstance(x, dict) else None
            out.append((q, c if isinstance(c, str) and c.strip() else ""))
    return out[:cap]


def result_A(out, tnorm):
    r, invalid = {}, []
    r["R"] = yn(out.get("readable")) or "yes"
    if not yn(out.get("readable")):
        invalid.append("readable")
    asserted, grounded = {}, {}
    n_q = n_u = 0
    for k in P_KEYS:
        if not (out.get(k) is None or isinstance(out.get(k), (list, str, dict))):
            invalid.append(f"{k}_type")
        qc = quotes_of(out.get(k))
        qs = [q for q, _ in qc]
        gs = [ground(q, tnorm) for q in qs]
        states = [g[0] for g in gs]
        state = "not_asserted" if not qs else ("grounded" if "grounded" in states else "ungrounded")
        r[f"{k}_quote"] = " || ".join(qs)
        r[f"{k}_quotes_json"] = json.dumps([{"quote": q, "context": c, "state": g[0], "offset": g[1],
                                             "context_state": ground(c, tnorm)[0] if c else "none"}
                                            for (q, c), g in zip(qc, gs)], ensure_ascii=False)
        r[f"{k}_state"], r[f"{k}_n"] = state, len(qs)
        asserted[k], grounded[k] = bool(qs), state == "grounded"
        n_q += len(qs); n_u += states.count("ungrounded")
    warns = []
    W = out.get("warnings")
    if W is not None and not isinstance(W, list):
        invalid.append("warnings_type")
    for w in W if isinstance(W, list) else []:
        if not isinstance(w, dict):
            invalid.append("warning_item"); continue
        q = quote_of(w.get("quote"))
        if not q:
            invalid.append("warning_without_quote"); continue        # format error, not an ungrounded quote
        tg = str(w.get("target", "")).strip().lower()
        if tg not in TARGETS:
            invalid.append(f"target={tg or 'missing'}"); tg = "invalid"
        st, off = ground(q, tnorm)
        iq = quote_of(w.get("issuer_quote"))
        ist, _ = ground(iq, tnorm)
        warns.append({"quote": q or "", "target": tg, "official": yn(w.get("official")) or "invalid",
                      "issuer_quote": iq or "", "state": st, "offset": off, "issuer_state": ist})
    r["warnings_json"] = json.dumps(warns, ensure_ascii=False)
    r["W1"] = "yes" if warns else "no"
    for t in ("this", "other_general", "cant_tell"):
        r[f"W2_{t}"] = sum(w["target"] == t for w in warns)
    r["W3_official"] = sum(w["official"] == "yes" for w in warns)
    r["label"], r["target_unclear"] = page_label(r["R"], asserted, [w["target"] for w in warns])
    r["label_grounded"], r["target_unclear_grounded"] = page_label(
        r["R"], grounded, [w["target"] for w in warns if w["state"] == "grounded"])
    r["label_sensitivity"] = sensitivity_label(r["label"], r["target_unclear"])
    r["n_quotes"] = n_q + sum(w["state"] != "not_asserted" for w in warns)          # per quote
    r["n_ungrounded"] = n_u + sum(w["state"] == "ungrounded" for w in warns)
    r["invalid"] = ";".join(invalid)
    return r


def result_B(out, tnorm):
    r, invalid = {}, []
    lab = str(out.get("label", "")).strip().lower()
    if lab not in LABELS:
        invalid.append(f"label={lab or 'missing'}"); lab = "invalid"
    tu = out.get("target_unclear")
    tu = tu if isinstance(tu, bool) else str(tu).strip().lower() == "true"
    ev = []
    for e in out.get("evidence") or []:
        q = quote_of(e) if isinstance(e, dict) else (e if isinstance(e, str) else None)
        c = e.get("context", "") if isinstance(e, dict) and isinstance(e.get("context"), str) else ""
        st, off = ground(q, tnorm)
        ev.append({"quote": q or "", "context": c, "role": (e.get("role", "") if isinstance(e, dict) else ""), "state": st,
                   "offset": off, "context_state": ground(c, tnorm)[0] if c else "none"})
    if lab in ("warning", "product_information") and not any(e["quote"] for e in ev):
        invalid.append("no_evidence")
    r.update(label=lab, target_unclear=bool(tu), label_sensitivity=sensitivity_label(lab, bool(tu)),
             evidence_json=json.dumps(ev, ensure_ascii=False), n_quotes=sum(e["state"] != "not_asserted" for e in ev),
             n_ungrounded=sum(e["state"] == "ungrounded" for e in ev), invalid=";".join(invalid))
    return r


# ---------------------------------------------------------------- pages
def read_page_file(p):
    try:
        lines = Path(p).read_text(encoding="utf-8").split("\n")
    except Exception:
        return None
    for k, line in enumerate(lines):
        if line.startswith("=" * 10):
            return "\n".join(lines[k + 1:])
    return "\n".join(lines)


def locate(page_file, pipe_dir):
    p = Path(page_file)
    if p.exists():
        return p
    w = PureWindowsPath(page_file)
    return pipe_dir / w.parent.name / w.name


def domain_of(url):
    return urllib.parse.urlsplit(str(url).strip()).netloc.lower().removeprefix("www.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="fresh_oct")
    ap.add_argument("--pipeline", choices=["A", "B"], default="A")
    ap.add_argument("--allow_capped", action="store_true", help="use the original 8000-character text where sc0 output is missing")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--base_url", default="http://localhost:11434/v1", help="local OpenAI-compatible server (Ollama)")
    ap.add_argument("--num_ctx", type=int, default=0, help="context window of the local model (Ollama num_ctx): replies whose prompt fills it are flagged input_cut")
    ap.add_argument("--provider", choices=["openai", "groq", "local"], help="default: groq if the model id has a '/' "
                    "or does not start with gpt-/o<digit>, else openai")
    ap.add_argument("--reasoning", action="store_true", help="reasoning model: no temperature, max_completion_tokens")
    ap.add_argument("--effort", default="", help="reasoning_effort (low/medium/high) if the model accepts it")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=0, help="default 1500, or 12000 with --reasoning")
    ap.add_argument("--max_chars", type=int, default=0, help="0 = full text (default)")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--repeat", default="", help="tag for a repeated run (stability check), e.g. r2")
    ap.add_argument("--only", default="", help="CSV with item_id column: run only these pages (e.g. the second set)")
    ap.add_argument("--list_models", action="store_true")
    args = ap.parse_args()

    prov = args.provider or ("groq" if ("/" in args.model or not re.match(r"^(gpt-|o\d)", args.model)) else "openai")
    key_env = {"groq": "llama_api_key", "openai": "OPENAI_API_KEY2_lab"}.get(prov)
    key = "local" if prov == "local" else (os.environ.get(key_env) or "").strip().strip('"').strip("'")
    if not key:
        raise SystemExit(f"Set the {key_env} environment variable.")
    if prov == "local":
        client = LocalClient(args.base_url)
    else:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=GROQ_URL if prov == "groq" else None, timeout=900)
    if args.list_models:
        for m in sorted(x.id for x in client.models.list().data):
            print(m)
        return

    max_tokens = args.max_tokens or (12000 if args.reasoning else 1500)
    CFG = {"pipeline": args.pipeline, "model": args.model, "provider": args.provider or "", "prompt_version": PROMPT_VERSION,
           "max_chars": args.max_chars, "max_tokens": max_tokens, "reasoning": args.reasoning, "effort": args.effort,
           "temperature": None if args.reasoning else args.temperature, "allow_capped": args.allow_capped, "num_ctx": args.num_ctx}
    call = Caller(client, args.model, args.temperature, max_tokens, args.reasoning, args.effort)
    tag = f"{args.pipeline}__{re.sub(r'[^A-Za-z0-9.-]+', '-', args.model)}" + (f"__{args.repeat}" if args.repeat else "")
    run = EXP_ROOT / "runs" / args.run_id
    pipe_dir, tables = run / "pipeline", run / "tables"
    models = run / "source_context_v3" / "models"
    out_dir = models / "raw" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    def write_prompt():   # written at the start and again at the end with the settings actually used
        (models / f"prompt_context_{tag}_{PROMPT_VERSION}.txt").write_text(
            f"PROMPT_VERSION: {PROMPT_VERSION}\nPIPELINE: {args.pipeline}\nMODEL: {args.model} ({prov})\n"
            f"SETTINGS: {json.dumps(call.opts)}\nPAGE TEXT: full extracted text"
            f"{f' (cut at {args.max_chars} characters)' if args.max_chars else ''}; pages with no text are not sent\n\n"
            f"=== SYSTEM ===\n{SYSTEM}\n\n=== USER (template) ===\n"
            + build_prompt(args.pipeline, "<PRODUCT NAME>", "<website domain>", "<PAGE TEXT>"), encoding="utf-8")
    write_prompt()

    summary = read_csv(tables / "match_summary.csv")
    pages = read_csv(tables / "pipeline_top3_all.csv")
    only = {r["item_id"] for r in read_csv(args.only)} if args.only else None
    print(f"pipeline {args.pipeline} | model {args.model} ({prov}) | output tag {tag}")

    rows, prods = [], []
    for prod in summary:
        qid, drug = prod["query_id"], prod["drug"]
        sel = sorted((p for p in pages if p["query_id"] == qid), key=lambda p: int(float(p["rank"])))
        labs = []
        for pg in sel:
            rank, url = int(float(pg["rank"])), pg["source_page_url"]
            item = f"{args.run_id}|{qid}|{rank}"
            if only is not None and item not in only:
                continue
            path = full_text_file(pg["page_file"], run)            # full text from sc0 (no 8000-character cut)
            has_html = bool(pg.get("html_file", "")) and locate(pg["html_file"], pipe_dir).exists()
            if not path.exists() and locate(pg["page_file"], pipe_dir).exists():
                if has_html and not args.allow_capped:          # sc0 should have written this page
                    raise SystemExit(f"[STOP] no full text for {item}: run 06_save_source_page_text.py first (or --allow_capped)")
                path = locate(pg["page_file"], pipe_dir)
            src = "full_text_sc0" if path == full_text_file(pg["page_file"], run) else ("original_capped" if has_html else "original")
            text = read_page_file(path) if path.exists() else None
            quality = pg.get("content_quality", "")
            r = {"item_id": item, "query_id": qid, "drug": drug, "rank": rank, "url": url, "domain": domain_of(url),
                 "pipeline": args.pipeline, "model": args.model, "prompt_version": PROMPT_VERSION,
                 "content_quality": quality, "words": len((text or "").split()), "chars": len(text or ""),
                 "chars_sent": 0, "cut": 0, "text_source": src, "config": CFG, "status": "", "label": "", "error": ""}
            jf = out_dir / f"{qid}_rank{rank}.json"
            prev = None
            if args.resume and jf.exists():
                try:
                    prev = json.loads(jf.read_text(encoding="utf-8"))
                except Exception:
                    prev = None
            if prev and prev.get("status") in ("ok", "empty", "not_fetched") and prev.get("config") == CFG:
                r = prev
                print(f"  {qid} rank{rank}: reused ({r['label']})")
            elif text is None or quality not in FETCH_OK:
                r.update(status="not_fetched" if text is not None or quality not in FETCH_OK else "missing_file",
                         label="not_evaluable", R="no")
                print(f"  {qid} rank{rank}: {r['status']}")
            elif not text.strip():
                r.update(status="empty", label="not_evaluable", R="no")
                print(f"  {qid} rank{rank}: no text -> not_evaluable")
            else:
                sent = text[:args.max_chars] if args.max_chars else text
                r.update(chars_sent=len(sent), cut=int(len(sent) < len(text)))
                try:
                    content, meta = call(build_prompt(args.pipeline, drug, r["domain"], sent))
                    r.update(meta); r["raw_reply"] = content
                    out = parse_json(content)
                    r.update(result_A(out, norm(sent)) if args.pipeline == "A" else result_B(out, norm(sent)))
                    full_ctx = args.num_ctx and (meta.get("prompt_tokens") or 0) >= args.num_ctx - 16     # local server cut the input
                    r["status"] = ("input_cut" if full_ctx else "truncated_output" if meta.get("finish_reason") == "length"
                                   else "invalid_output" if r["invalid"] or r["label"] == "invalid" else "ok")
                    print(f"  {qid} rank{rank}: {r['label'] if r['status'] == 'ok' else r['status'].upper()}" + (" (target unclear)" if r.get("target_unclear") else "")
                          + f" | quotes {r['n_quotes']}, ungrounded {r['n_ungrounded']}" + (f" | invalid {r['invalid']}" if r["invalid"] else ""))
                    time.sleep(args.sleep)
                except StopRun as e:
                    print(f"\n[STOPPED] {e}\nNo CSV written. Pages done so far are saved; rerun with --resume.")
                    raise SystemExit(1)
                except Exception as e:
                    r.update(status="too_large" if str(e).startswith("too_large") else "truncated_output" if r.get("finish_reason") == "length" else "api_error",
                             error=f"{type(e).__name__}: {str(e)[:300]}")
                    print(f"  {qid} rank{rank}: {r['status']} ({r['error']})")
            jf.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            rows.append(r)
            if r["status"] in ("ok", "empty", "not_fetched", "missing_file"):
                labs.append(r["label"])
        n_err = sum(x["query_id"] == qid and x["status"] not in ("ok", "empty", "not_fetched", "missing_file") for x in rows)
        prods.append({"query_id": qid, "drug": drug, "pipeline": args.pipeline, "model": args.model,
                      "product_label": product_label(labs) if not n_err else "error",
                      "n_pages": len(sel), "n_error": n_err, "page_labels": "|".join(labs),
                      "target_unclear": int(any(x["query_id"] == qid and x.get("target_unclear") for x in rows)),
                      "product_label_sensitivity": product_label([x.get("label_sensitivity") or x["label"] for x in rows
                                                                  if x["query_id"] == qid and x["label"]]) if not n_err else "error"})

    write_csv([{k: v for k, v in r.items() if k != "raw_reply"} for r in rows],
              models / (f"context_{tag}.csv" if only is None else f"context_{tag}__subset.csv"))
    if only is None:
        write_csv(prods, models / f"context_products_{tag}.csv")
    st = Counter(r["status"] for r in rows)
    print(f"\n[DONE] {datetime.now():%Y-%m-%d %H:%M} pages {len(rows)} | " + " | ".join(f"{k}: {v}" for k, v in st.most_common()))
    print("\n".join(f"{k:<20} {v}" for k, v in Counter(r["label"] for r in rows).most_common()))
    q, u = sum(r.get("n_quotes") or 0 for r in rows), sum(r.get("n_ungrounded") or 0 for r in rows)
    print(f"quotes {int(q)} | ungrounded {int(u)} ({100 * u / max(q, 1):.1f}%)")
    write_prompt()
    if call.notes:
        print("settings changed for this model:", "; ".join(dict.fromkeys(call.notes)))
    bad = sum(st[k] for k in ("api_error", "too_large", "invalid_output", "truncated_output", "input_cut"))
    if bad:
        print(f"[WARN] {bad} pages not labelled; rerun with --resume (too_large: add --max_chars and report it).")


if __name__ == "__main__":
    main()
