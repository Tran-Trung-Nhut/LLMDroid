# LLM Detector

Multimodal detection of LLM integration in Android apps using:
- **Text branch**: SBERT + keyword/meta features + SLM reasoning score
- **Image branch**: CLIP embeddings + zero-shot scores + OCR keyword features
- **Fusion**: Early Fusion and Late Fusion (Stacking / Soft Voting / Max Voting)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Train (full pipeline)

```bash
python src/train_pipeline.py
```

Pipeline steps:
1. Preprocess app text + deduplicate screenshots
2. Create stratified 5-fold splits
3. Run OCR on screenshots
4. Extract text, image, and SLM features
5. Train/evaluate text-only, image-only, and fusion models

---

### 2. Inference on new apps (from raw JSONL)

```bash
python src/run_inference.py --input-raw data/apps_inference_raw.jsonl --output predictions
```

This runs preprocessing → OCR → feature extraction → inference automatically.

---

### 3. Inference from pre-extracted features

```bash
python src/run_inference.py --input data/features --output predictions
```

---

## Fetch metadata for new apps

If you start from package names in CSV (`pkg_name` column):

```bash
python src/fetch_app_metadata.py --input data/apps.csv --output data/apps_inference_raw.jsonl
```

Outputs:
- `data/apps_inference_raw.jsonl`
- `data/images/{pkg_name}/...`
- `data/failed_apps.txt` (if any app fails)

---

## Training/Evaluation Protocol

The training code uses strict train/validation/test separation per outer fold:

1. **Outer CV (5 folds)** for final performance reporting.
2. **Inner validation split** inside each outer-train fold for:
   - feature selection fitting
   - LightGBM early stopping
3. **Late-fusion stacking** uses out-of-fold (OOF) base predictions from outer-train.
4. **Threshold selection** is based on validation predictions (`validation_predictions.csv`).

Primary metrics are reported at `classification_threshold` (default `0.5`), and validation-selected thresholds are saved for inference-time usage.

---

## Key Outputs

Under `runs/<run_name>/...` each experiment folder includes:
- `metrics_per_fold.json`
- `metrics_aggregated.json`
- `predictions.csv` (outer-test predictions)
- `validation_predictions.csv` (inner validation predictions)
- `best_threshold_metrics.json`
- `saved_models/` (selectors/models used by inference)

Inference outputs (`--output`) include:
- `early_fusion_inference.csv`
- `stacking_inference.csv`
- `soft_voting_inference.csv`
- `max_voting_inference.csv`

---

## Configuration

Edit `src/config.py`:

- **Models**: `text_model`, `clip_model`, `slm_model`
- **CV/evaluation**: `n_folds`, `inner_val_ratio`, `stacking_inner_cv_folds`
- **Classifier**: `lgbm_params`, `feature_selection_k`, `classification_threshold`
- **Threshold search**: `threshold_search_min`, `threshold_search_max`, `threshold_search_step`
- **Paths**: dataset/features/runs/inference locations

---

## Project Structure

```text
LLM_Detector/
├── data/
│   ├── apps_raw.jsonl
│   ├── apps.jsonl
│   ├── images/
│   ├── splits/
│   └── features/
├── src/
│   ├── config.py
│   ├── train_pipeline.py
│   ├── run_inference.py
│   ├── fetch_app_metadata.py
│   ├── steps/
│   │   ├── preprocessing.py
│   │   ├── make_splits.py
│   │   ├── run_ocr.py
│   │   ├── extract_text_features.py
│   │   ├── extract_image_features.py
│   │   ├── extract_slm_features.py
│   │   └── train_evaluate.py
│   └── utils/
│       ├── io.py
│       ├── metrics.py
│       ├── seed.py
│       └── inference_helper.py
├── runs/
└── inference_results/
```
