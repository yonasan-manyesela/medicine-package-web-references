"""
context_label_rule.py
Shared definitions, quote grounding and decision rules for the source-context step.
Imported by 08_label_pages_with_language_models.py (models) and 09_evaluate_against_human_reference.py (human and model), so the human
signals and the model signals are turned into labels by the SAME code.

Page labels:    not_evaluable | warning | product_information | insufficient
Flag:           target_unclear (a warning whose target is cant_tell); counted as a warning
                only in the sensitivity analysis.
Product labels: warning > product_information > insufficient (>=1 readable page)
                > not_evaluable (all pages unreadable or not fetched); no_page if none selected.
"""
import re
import unicodedata
from pathlib import Path, PureWindowsPath

PROMPT_VERSION = "v3.9_2026-10-28"
P_KEYS = ["product_named", "ingredient", "strength", "form", "manufacturer"]       # P1..P5
TARGETS = {"this", "other_general", "cant_tell"}

DEFINITIONS = """\
You are not authenticating the medicine. Report only what the page text states.
Copy every quote exactly from the page text, without adding or changing words.
Quote only the words that show the signal. Give one to three quotes per signal:
the product's main heading first when it shows it, otherwise the first mention.
With each quote give its context: the whole line or sentence it comes from, copied exactly.

R  Readable: "no" if the text has only error, access or "verify you are human"
   messages, cookie notices or menus; otherwise "yes".

P1 Product named: the page names the searched product. Extra words, a strength,
   or spacing and punctuation differences are allowed ("Arh L" = "ARH-L").
   A different product with a similar name does not count.

P2-P5 are facts about the searched product only. Each must be stated in the
product's own title, description, table or sentence, not in related products,
advertisements or other listings.
P2 Ingredient (composition): the active substance(s) of the searched product,
   e.g. paracetamol (acetaminophen), ibuprofen, amoxicillin, artemether + lumefantrine.
   It usually follows "Composition", "Contains", "Active ingredient", "Salt",
   "Generic name" or "Each tablet contains", or stands just before a strength
   ("Ibuprofen 400 mg"). The brand name is not an ingredient.
P3 Strength: a number with a unit for the amount of active substance ("500 mg",
   "20 mg/120 mg", "250 mg/5 ml"; for creams, gels and ointments "1%", "2% w/w",
   "10 mg/g"). For combination medicines, a pair of numbers such as "20/120" or
   "80/480" counts even without units. A single number without a unit, pack sizes
   ("x 6 tablets", "30 ml", "20 g") and dosing instructions ("take two tablets twice
   daily") are not strength.
P4 Form: the dosage form (tablet, capsule, syrup, dry syrup, injection, cream, gel,
   ointment, drops), including form words such as film-coated or dispersible.
P5 Manufacturer: a company the page LABELS as the manufacturer or marketer of the
   searched product ("Manufactured by", "Manufacturer", "Mfd. by", "Marketed by",
   "Marketer"). Quote the label together with the company name, or the address if
   only an address follows it. A company name without such a label does not count,
   and neither does a seller's profile ("Nature of business: Manufacturer").

W1 Warning: a sentence stating that medicines are or were falsified, counterfeit,
   substandard, recalled, seized, unregistered or banned, including general
   cautions about fake medicines. Quote the whole sentence. Only the page's own
   content counts: not related news, related products, advertisements or menus. Not warnings: side effects, contraindications,
   storage, dose limits, "100% genuine", "prescription required".
W2 Target of each warning:
   this          - names the searched product or its batch, or explicitly covers
                   all products containing its active ingredient(s);
   other_general - names another product, or no product;
   cant_tell     - names only the active ingredient, or the link is unclear.
W3 Official: "yes" if the warning is issued by, or attributed to, a named
   regulator, ministry, WHO or public-health body ("NAFDAC has alerted ...");
   quote the name. A regulator named elsewhere on the page does not count.
"""

LABEL_RULE = """\
Page label:
  not_evaluable       - the page is not readable (R = no);
  warning             - a warning whose target is this product (W2 = this);
  product_information - the page names the product (P1) and states at least one of
                        its ingredient, strength, form or manufacturer (P2-P5);
  insufficient        - otherwise.
A warning about this product decides the label even if product details are also present.
"""


# ---------------------------------------------------------------- grounding
_TRANS = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
                        "“": '"', "”": '"', "„": '"', "″": '"',
                        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
                        " ": " ", "​": ""})


def norm(s):
    """NFKC, case-folded, unified quotes/dashes, single spaces."""
    s = unicodedata.normalize("NFKC", str(s or "")).translate(_TRANS).casefold()
    return re.sub(r"\s+", " ", s).strip()


def ground(quote, text_norm):
    """Is the quote in the (normalized) page text?
    Returns (state, offset): state 'grounded' | 'ungrounded' | 'not_asserted'.
    Surrounding quote marks and a leading/trailing ellipsis are ignored; a quote with an
    inner ellipsis is grounded when all its parts occur in order."""
    if quote is None or not str(quote).strip():
        return "not_asserted", None
    q = norm(quote).strip(" \"'")
    q = re.sub(r"^(\.\.\.|…)\s*|\s*(\.\.\.|…)$", "", q)
    parts = [p.strip(" \"'") for p in re.split(r"\.\.\.|…", q) if p.strip(" \"'")]
    if not parts:
        return "not_asserted", None
    pos, first = 0, None
    for p in parts:
        k = text_norm.find(p, pos)
        if k < 0:
            return "ungrounded", None
        first = k if first is None else first
        pos = k + len(p)
    return "grounded", first


# ---------------------------------------------------------------- rules
def page_label(readable, p, targets):
    """readable: 'yes'/'no'; p: dict P_KEYS -> bool (asserted); targets: list of warning targets.
    Returns (label, target_unclear)."""
    unclear = "cant_tell" in targets
    if readable == "no":
        return "not_evaluable", False
    if "this" in targets:
        return "warning", unclear
    if p.get("product_named") and any(p.get(k) for k in P_KEYS[1:]):
        return "product_information", unclear
    return "insufficient", unclear


def sensitivity_label(label, target_unclear):
    """Sensitivity analysis: a warning with an unclear target counts as a warning."""
    return "warning" if (target_unclear and label in ("product_information", "insufficient")) else label


def product_label(page_labels):
    """page_labels: list of page labels of one product (not_fetched pages given as 'not_evaluable')."""
    if not page_labels:
        return "no_page"
    for lab in ("warning", "product_information"):
        if lab in page_labels:
            return lab
    if "insufficient" in page_labels:
        return "insufficient"
    return "not_evaluable"


# ---------------------------------------------------------------- page files
def local_file(path, pipe_dir):
    """A file listed with a path from another PC: the same name under <run>/pipeline/<query folder>/."""
    p = Path(str(path))
    if p.exists():
        return p
    w = PureWindowsPath(str(path))
    return Path(pipe_dir) / w.parent.name / w.name


def full_text_file(page_file, run_dir):
    """Full page text written by 06_save_source_page_text.py: source_context_v3/page_text/<query folder>/rank<k>_page.txt."""
    w = PureWindowsPath(str(page_file))
    return Path(run_dir) / "source_context_v3" / "page_text" / w.parent.name / w.name
