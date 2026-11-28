# Finalized six-signal quotation-evidence audit (22 November 2026)

## Scope and preservation

Inputs were the original complete CH independent quotation audit (508 grouped quotations), the latest A1 export (69 completed cases, 101 completed quotations), the 370-case manifest, the 572-record de-duplication mapping, and the previously identified 53 distinct model-positive/human-negative P5 pages. No source annotations, model predictions, human page-level reference labels, or the manuscript file were overwritten.

The analytic reconciliation uses completed A1 entries in preference to the corresponding CH entries, otherwise retains CH entries. The mapping expands 508 de-duplicated quotations into 531 selected model-generated quotation instances, excluding `human_only` stratum records. Repeated quotations are counted once per original model instance, not once globally.

## Final support proportions in the selected model-positive audit

| Signal | GPT-4o | Qwen3 8B |
|:--|--:|--:|
| P1 product name | 28/30 (93.3%) | 29/34 (85.3%) |
| P2 ingredient | 48/54 (88.9%) | 53/63 (84.1%) |
| P3 strength | 29/38 (76.3%) | 43/57 (75.4%) |
| P4 dosage form | 34/37 (91.9%) | 35/45 (77.8%) |
| P5 manufacturer/marketer | **43/60 (71.7%)** | **45/77 (58.4%)** |
| W1 genuine medicine-quality warning | 6/7 (85.7%) | 8/29 (27.6%) |

**Definitions:** P1–P5 require quotation found, support of the attribute, and evidence about the searched product. W1 requires quotation found and a genuine quality warning; it is *not* a product-specific W2 target measure. The audit is targeted; pooled comparisons between models cannot be interpreted as general model accuracies or formal statistical superiority. The separate existing page-label results (κ=0.39–0.53) are unchanged.

## Manufacturer disagreement

- 53 distinct pages with at least one P5 model-positive/human-negative finding (80 model-level instances).
- 18/53 pages have ≥1 valid P5 model quotation under the strict criterion; 35/53 do not have a valid audited quotation.
- Of those 18: 14 already had human `product_information` page labels; 4 were human `insufficient` without human-confirmed P1. None of these results alone establishes a need to change the original human page labels.
- P5 among human-negative cases: GPT-4o 16/31 and Qwen3 8B 18/49 supporting quotation instances (model instances, not distinct pages).
- Being unsupported in the audited model quotations does **not** demonstrate the full webpage lacks all possible manufacturer evidence; the inference is about the quotation submitted by the model.

## Three explicit rule-consistency resolutions

The A1 export saved `Yes` for all three quotations below. The final *analytic* classification treats them as unsupported under the previously agreed P5 annotation rule. They are documented and reversible. CH and A1 files remain unchanged.

| Case | Quote | Raw A1 | Final analysis | Reason |
|:--|:--|:--|:--|:--|
| C0045 | q1 | Yes | Not supported | Quoted company evidence belongs to the Histazine-1 page, whereas the searched medicine is Cataflam |
| C0049 | q1 | Yes | Not supported | Company listed alongside Zoladex but its manufacturer/marketer role is not explicitly stated |
| C0049 | q2 | Yes | Not supported | Same company name in a separate product-listing context; no explicit role establishing P5 for Zoladex |

Without these analytic resolutions, P5 would be GPT-4o 44/60 (73.3%) and Qwen3 8B 47/77 (61.0%), with 20/53 supported disagreement pages. The transparent alternative is preserved in `six_signal_final.csv` under `supported_as_exported`.

## Practical manuscript update

Insert the three passages provided in `manuscript_additions.tex` in the paper's Evaluation Measures, Automated Source-Context Annotation Results, and Automated Annotation and Evidence Traceability Discussion. Copy `figure_six_signal_quotation_support_FINAL.pdf` into Overleaf at `img/fig_semantic_quotation_support_FINAL.pdf`.

Avoid changing the original page-level κ statistics: CH/A1 reviewed quotation support, not adjudicated a new set of all 195 page labels.
