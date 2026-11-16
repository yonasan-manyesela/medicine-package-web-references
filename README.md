# Web images as references for medicine package inspection

Code for the article *Evaluating Web Images as References for Medicine Package Visual Inspection: Product Identity and Source Context* (MDPI *Applied Sciences*, under review).

For each query package, the study searches Google Images with the printed product name, matches the web images to the query photograph, and keeps the three best matches from different pages. It then evaluates (1) whether each matched image shows the same medicine and (2) what its source page states, using human annotation and two language models.

## Scripts

Run every script from the repository root. The number gives the run order.

| Script | What it does | Paper |
|---|---|---|
| `scripts/1_retrieval/01_search_google_images.py` | Searches Google Images with the product name (or name + dosage form) and saves up to 10 images with their page URLs; checks whether each page's HTML contains the image | 3.2 |
| `scripts/1_retrieval/02_match_web_images_to_query.py` | Matches each web image to the query photograph (DISK + LightGlue), keeps images with at least 50 matches, takes the top 3 from different pages and saves each page | 3.2 |
| `scripts/2_product_identity/03_build_identity_annotation_page.py` | Builds the offline page for labelling same medicine / variant / different / cannot determine, and draws the second annotator's sample | 3.3 |
| `scripts/2_product_identity/04_compare_identity_annotators.py` | Agreement between Annotators 1 and 2; disagreement sheet for adjudication | 3.3, 4.1 |
| `scripts/2_product_identity/05_recheck_single_labelled_images.py` | Re-check of the single-labelled accepted images against the final guide | 3.3 |
| `scripts/3_source_context/06_save_source_page_text.py` | Extracts the full readable text of each saved page | 3.4 |
| `scripts/3_source_context/07_build_context_annotation_page.py` | Builds the offline page for marking the nine context signals with quotations | 3.5 |
| `scripts/3_source_context/08_label_pages_with_language_models.py` | Runs the signal–rule (`--pipeline A`) and direct-rule (`--pipeline B`) pipelines with GPT-4o or Qwen3 8B | 3.6 |
| `scripts/3_source_context/09_evaluate_against_human_reference.py` | Agreement (κ, product-level bootstrap), identity × context tables, quotation checks | 3.7, 4 |
| `scripts/3_source_context/10_plot_result_figures.py` | Figures from the outputs of step 09 | 4 |
| `scripts/3_source_context/context_label_rule.py` | The fixed rule that maps the nine signals to a page label; used by steps 07–09 for humans and models alike | 3.4 |
| `scripts/3_source_context/previous_pipeline_ievc2026.py` | The pipeline of our IEVC 2026 paper, used for the comparison row in Table 3 | 4.3, App. A |

`scripts/1_retrieval/google_images_helpers.py` holds the browser settings, CAPTCHA handling and metadata writer used by step 01.

## Running

```
pip install -r requirements.txt
python scripts/1_retrieval/01_search_google_images.py --run_id fresh_oct
python scripts/1_retrieval/01_search_google_images.py --run_id fresh_oct_form --expand form
python scripts/1_retrieval/02_match_web_images_to_query.py --run_id fresh_oct
python scripts/1_retrieval/02_match_web_images_to_query.py --run_id fresh_oct_form
python scripts/2_product_identity/03_build_identity_annotation_page.py --runs fresh_oct,fresh_oct_form
python scripts/3_source_context/06_save_source_page_text.py --run_id fresh_oct
python scripts/3_source_context/08_label_pages_with_language_models.py --pipeline A --model gpt-4o
python scripts/3_source_context/08_label_pages_with_language_models.py --pipeline A --provider local --model qwen3:8b
python scripts/3_source_context/09_evaluate_against_human_reference.py --human1 <Annotator 1 file> --human2 <Annotator 3 file>
python scripts/3_source_context/10_plot_result_figures.py
```

Steps 04, 05 and 07 are interactive (annotation and adjudication); see the header of each script. Repeat step 08 with `--pipeline B` for the direct-rule runs.

- **Search:** Chrome and a matching chromedriver in `tools/chromedriver`. Google may show a CAPTCHA; solve it in the browser window and press Enter.
- **API key:** GPT-4o reads the key from the environment variable `OPENAI_API_KEY2_lab`. No key is stored in this repository.
- **Local model:** Qwen3 8B runs through [Ollama](https://ollama.com) (`ollama pull qwen3:8b`) at `http://localhost:11434`.

## Data

`config/queries.csv` lists the 100 query products (ID, printed product name, dosage form, source). The query images are not included: the 49 public images are available from Mendeley Data (https://doi.org/10.17632/bjy2svvmn8.1), and the field photographs are available from the corresponding author on reasonable request. Web images and page contents are owned by third parties and are not redistributed; the files below contain their URLs and short quotations only.

Annotator codes: `Yona` and `FirstAuthor` = Annotator 1, `EI` = Annotator 2, `PH01` = Annotator 3 (licensed pharmacist).

| File | Content | Paper |
|---|---|---|
| `data/2_product_identity/labeling_items.csv` | The 570 image pairs: query, run, rank, match count, image and page URL | 3.3 |
| `data/2_product_identity/labels_primary_Yona_2026-10-22.csv` | Annotator 1, identity labels before adjudication | 3.3 |
| `data/2_product_identity/labels_second_EI_2026-10-22.csv` | Annotator 2, identity labels (100 images) | 3.3, 4.1 |
| `data/2_product_identity/label_check_Yona_2026-10-24.csv` | Re-check of 303 single-labelled images (62 changed) | 3.3 |
| `data/2_product_identity/labels_final_checked.csv` | Final identity labels used in the results | 4.1, 4.2 |
| `data/3_source_context/annotations/annotations_FirstAuthor_all_2026-11-14_final.csv` | Annotator 1, nine signals and quotations for the 195 pages (human reference) | 3.5, 4.2 |
| `data/3_source_context/annotations/annotations_PH01_main_2026-11-01.csv` | Annotator 3, the 57-page second set | 3.5, 4.2 |
| `data/3_source_context/annotations/second_set_manifest.csv` | The 57 pages of the second set and how they were drawn (stratum, seed) | 3.5 |
| `data/3_source_context/annotations/warning_coding.csv` | Coding of each highlighted warning as quality/regulatory or clinical caution | 3.5, 4.2 |
| `data/3_source_context/models/context_<A\|B>__<model>.csv` | Page-level output of each model run with quotations and their grounding (A = signal–rule, B = direct-rule) | 3.6, 4.3 |
| `data/3_source_context/models/context_products_<A\|B>__<model>.csv` | Product-level labels of each run | 4.3 |
| `data/3_source_context/models/llm_results_cot.csv` | Page labels of the previous (IEVC 2026) pipeline | 4.3, App. A |

`gpt-4o-2024-08-06` is GPT-4o; `webref-qwen3-8b` is Qwen3 8B run locally in Ollama. Step 09 reads the annotation files from `source_context_v3/annotation/` the model files from `runs/fresh_oct/source_context_v3/models/` and `llm_results_cot.csv` from `runs/fresh_oct/tables/`; copy them there to rerun it.

## License

Code: MIT License (see `LICENSE`). The data files are shared for research use; web page quotations remain the property of their owners.
