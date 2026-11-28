# Web images as references for medicine package inspection

Code and data records for the article *Evaluating Web Images as References for Medicine Package Visual Inspection: Product Identity and Source Context* (MDPI *Applied Sciences*, under review).

For each query package, the study searches Google Images with the printed product name, matches the web images to the query photograph, and keeps the three best matches from different pages. It then evaluates (1) whether each matched image shows the same medicine and (2) what its source page states, using human annotation and two language models.

**Every number reported in the paper is listed in [`RESULTS_EVIDENCE.md`](RESULTS_EVIDENCE.md)** with the file it comes from. `scripts/4_revision_checks/11_reproduce_reported_numbers.py` recomputes them from the files in this repository (`results/reported_numbers.csv`).

## Scripts

Run every script from the repository root. The number gives the run order.

| Script | What it does | Paper |
|---|---|---|
| `scripts/1_retrieval/01_search_google_images.py` | Searches Google Images (Selenium + Chrome) with the product name, or name + dosage form, and saves up to 10 images with their page URLs; checks whether each page's HTML contains the image | 3.2 |
| `scripts/1_retrieval/02_match_web_images_to_query.py` | Matches each web image to the query photograph (DISK + LightGlue), keeps images with at least 50 matches, takes the top 3 from different pages and saves each page | 3.2 |
| `scripts/2_product_identity/03_build_identity_annotation_page.py` | Builds the offline page for labelling same medicine / variant / different / cannot determine, and draws the shared 100-pair sample | 3.3 |
| `scripts/2_product_identity/04_compare_identity_annotators.py` | Agreement between annotators; disagreement sheet for adjudication | 3.3, 4.1 |
| `scripts/2_product_identity/05_recheck_single_labelled_images.py` | Re-check of the single-labelled accepted images against the final guide | 3.3 |
| `scripts/2_product_identity/05b_targeted_identity_review_41.py` | Page for the targeted review of 41 images (earlier disagreements and flagged variant labels) | 3.3 |
| `scripts/3_source_context/06_save_source_page_text.py` | Extracts the full readable text of each saved page | 3.4 |
| `scripts/3_source_context/07_build_context_annotation_page.py` | Builds the offline page for marking the nine context signals with quotations | 3.5 |
| `scripts/3_source_context/pharmacist_W1_verification_page.html` | Page used by the pharmacist annotator to re-assess their warning highlights against W1 | 3.5 |
| `scripts/3_source_context/08_label_pages_with_language_models.py` | Runs the signal–rule (`--pipeline A`) and direct-rule (`--pipeline B`) pipelines with GPT-4o or Qwen3 8B | 3.6 |
| `scripts/3_source_context/09_evaluate_against_human_reference.py` | Agreement (κ, product-level bootstrap), identity × context tables, quotation checks | 3.7, 4 |
| `scripts/3_source_context/10_plot_result_figures.py` | Figure of model agreement (Fig. 10) from the outputs of step 09 | 4.3 |
| `scripts/3_source_context/quotation_review/13–15_*.py` | Builds the quotation review page, groups repeated quotations, and reconciles the two review passes | 3.7, 4.3 |
| `scripts/3_source_context/quotation_review/16_final_quotation_support.py` | Final quotation-review table from the CH and A1 exports (grounding, support, relevance per model and signal) | 4.3 |
| `scripts/3_source_context/quotation_review/17_plot_quotation_evidence.py` | Figure of grounding, support and relevance (Fig. 11) | 4.3 |
| `scripts/3_source_context/context_label_rule.py` | The fixed rule that maps the nine signals to a page label; used for humans and models alike | 3.4 |
| `scripts/3_source_context/previous_pipeline_ievc2026.py` | The pipeline of our IEVC 2026 paper (comparison row in Table 4) | 4.3, App. A |
| `scripts/4_revision_checks/00_export_retrieval_records.py` | Exports the retrieval records (candidates, match counts, top three) into `data/1_retrieval/` | — |
| `scripts/4_revision_checks/01_check_query_photo_duplicates.py` | Compares every downloaded candidate with its query photograph (hash, thumbnail correlation, matches) | 3.2, 4.1 |
| `scripts/4_revision_checks/02_match_count_roc_and_cutoffs.py` | Precision-recall of the match count and product coverage by threshold (Fig. 6) | 4.1 |
| `scripts/4_revision_checks/03_compare_query_runs.py` | Paired comparison of two searches (name vs name + form; US vs Tanzania region), exact McNemar | 4.1, 5.5 |
| `scripts/4_revision_checks/04_relabel_Q092_2026-11-28.py` | Consistency correction of one identity label (dated copies; originals kept) | 3.3 |
| `scripts/4_revision_checks/05_plot_figure7_identity_by_rank.py`, `06_plot_figure8_identity_and_source_context.py` | Figures 7 and 8 from the final labels | 4.1, 4.2 |
| `scripts/4_revision_checks/11_reproduce_reported_numbers.py` | Recomputes every reported number from this repository | all |
| `scripts/4_revision_checks/12_write_results_evidence.py` | Writes `RESULTS_EVIDENCE.md` | — |

`scripts/1_retrieval/google_images_helpers.py` holds the browser settings, CAPTCHA handling and metadata writer used by step 01. Steps 00–02 and 04–06 of `4_revision_checks` were run in the original working folder (they need the images); their outputs are in `data/` and `results/`.

## Running

```
pip install -r requirements.txt
python scripts/1_retrieval/01_search_google_images.py --run_id fresh_oct
python scripts/1_retrieval/01_search_google_images.py --run_id fresh_oct_form --expand form
python scripts/1_retrieval/01_search_google_images.py --run_id fresh_oct_tz --country tz
python scripts/1_retrieval/02_match_web_images_to_query.py --run_id fresh_oct
python scripts/3_source_context/06_save_source_page_text.py --run_id fresh_oct
python scripts/3_source_context/08_label_pages_with_language_models.py --pipeline A --model gpt-4o
python scripts/3_source_context/08_label_pages_with_language_models.py --pipeline A --provider local --model qwen3:8b
python scripts/3_source_context/09_evaluate_against_human_reference.py --human1 annotations_FirstAuthor_all_2026-11-14_final.csv --human2 annotations_PH01_main_2026-11-01_completed_2026-11-26.csv
python scripts/4_revision_checks/11_reproduce_reported_numbers.py
```

Steps 04, 05, 05b and 07 are interactive (annotation and adjudication); see the header of each script. Repeat step 08 with `--pipeline B` for the direct-rule runs. Step 09 reads the annotation files from `source_context_v3/annotation/`, the model files from `runs/fresh_oct/source_context_v3/models/` and `llm_results_cot.csv` from `runs/fresh_oct/tables/`; copy them there to rerun it. Its outputs used in the paper are in `results/from_step09/`.

- **Search:** Chrome and a matching chromedriver in `tools/chromedriver`. Google may show a CAPTCHA; solve it in the browser window and press Enter. Results change over time, so a new search will not return the same images.
- **API key:** GPT-4o reads the key from the environment variable `OPENAI_API_KEY2_lab`. No key is stored in this repository.
- **Local model:** Qwen3 8B (Q4_K_M) runs through [Ollama](https://ollama.com) 0.40.16 (`ollama pull qwen3:8b`) at `http://localhost:11434`.

## Data

`config/queries.csv` lists the 100 query products (ID, printed product name, dosage form, source). The query images are not included: the 49 public images are available from Mendeley Data (https://doi.org/10.17632/bjy2svvmn8.1), and the field photographs are available from the corresponding author on reasonable request. **Web images and page contents are owned by third parties and are not redistributed**; the files below contain their URLs, match counts and short quotations only.

Annotator codes: `Yona` and `FirstAuthor` = Annotator 1 (first author); `EI` = Annotator 2 (co-author, product identity); `RM` = Annotator 3 (ADDO dispenser, product identity); `PH01` = the pharmacist annotator (source context); `CH` = quotation reviewer (co-author); `A1` = Annotator 1 in the attribute check and quotation reconciliation.

| Folder / file | Content | Paper |
|---|---|---|
| `data/1_retrieval/candidates_<run>.csv` | Every downloaded candidate: download position (search order among downloadable images), image and page URL, number of matches | 3.2, 4.1 |
| `data/1_retrieval/match_summary_<run>.csv`, `top3_<run>.csv` | One row per product; the selected top three | 4.1 |
| `data/1_retrieval/duplicate_check_*.csv` | Similarity of every candidate to its query photograph; verdicts of the visually inspected pairs | 4.1 |
| `data/2_product_identity/labeling_items.csv` | The 570 image pairs: query, run, rank, match count, image and page URL | 3.3 |
| `data/2_product_identity/labels_primary_Yona_2026-10-22.csv`, `labels_second_EI_2026-10-22.csv`, `labels_second_RM_2026-11-03.csv` | Independent identity labels before reconciliation (100 shared pairs) | 3.3, 4.1 |
| `data/2_product_identity/disagreements_resolved.csv`, `label_check_Yona_2026-10-24.csv` | Adjudication of disagreements; re-check of single-labelled images | 3.3 |
| `data/2_product_identity/identity_review_manifest_41.csv` | The 41 images of the targeted review and why each was selected | 3.3 |
| `data/2_product_identity/labels_final_checked_qc_2026-11-28.csv` | **Final identity labels** (570 pairs) | 4.1 |
| `data/2_product_identity/retrieval_qc_2026-11-28_246_images.csv` | Final labels of the 246 name-only top-three images, with the original label and QC evidence | 4.1, 4.2 |
| `data/2_product_identity/attribute_check_A1_2026-11-18.csv` | Visibility and agreement of ingredient, strength and form for same-medicine images (strict sensitivity) | 4.1 |
| `data/2_product_identity/*_2026-11-21.csv` | Labels before the 28 Nov consistency correction (kept for comparison) | — |
| `data/3_source_context/annotations/annotations_FirstAuthor_all_2026-11-14_final.csv` | Annotator 1, nine signals and quotations for the 195 pages (human reference) | 3.5, 4.2 |
| `data/3_source_context/annotations/annotations_PH01_main_2026-11-01_completed_2026-11-26.csv` | Pharmacist annotator, 57 pages (the original file is also included) | 3.5, 4.2 |
| `data/3_source_context/annotations/PH01_W1_verified_2026-11-26.csv` | Pharmacist's re-assessment of 44 warning highlights against W1 | 3.5 |
| `data/3_source_context/annotations/warning_coding.csv`, `second_set_manifest.csv`, `prescreen.csv` | Warning coding; the 57-page sample; pages excluded before annotation | 3.5 |
| `data/3_source_context/attribute_consistency/panel_c_reconciled_2026-11-28.csv` | Page attributes vs query package for 100 image–page cases | 3.4, 4.2 |
| `data/3_source_context/models/context_<A\|B>__<model>.csv` | Page-level output of each model run with quotations, grounding and status | 3.6, 4.3 |
| `data/3_source_context/models/context_products_<A\|B>__<model>.csv`, `llm_results_cot.csv` | Product labels of each run; page labels of the previous pipeline | 4.3, App. A |
| `data/3_source_context/quotation_review/semantic_evidence_CH.csv`, `semantic_evidence_A1.csv` | The two reviewer exports (508 grouped quotations; A1 completed 101) | 3.7, 4.3 |
| `data/3_source_context/quotation_review/semantic_group_map.csv` | Grouped quotation → original model-specific instance (572 records; 531 model instances) | 3.7 |
| `data/3_source_context/quotation_review/` (other files) | Earlier reconciliation outputs and the written summary `FINAL_FINDINGS.md` | 4.3 |
| `results/from_step09/` | Outputs of step 09 (agreement tables, human page labels, identity × context) | 4.2, 4.3 |
| `results/reported_numbers.csv`, `results/figures/` | Recomputed numbers with sources; figure files | all |

`gpt-4o-2024-08-06` is GPT-4o; `webref-qwen3-8b` is Qwen3 8B run locally in Ollama.


## License

Code: MIT License (see `LICENSE`). The data files are shared for research use; web page quotations remain the property of their owners.
