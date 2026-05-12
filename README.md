# LLM Detector

Multimodal detection of LLM integration in Android apps using:
- **Text branch**: SBERT embeddings + keyword/meta handcrafted features
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
python src/train_pipeline.py --train-only
```

Pipeline steps (when running `--train-only`):
1. **k Sensitivity Analysis** — grid search k ∈ {20, 50, 100, 200, 500, all} on text features; auto-updates `config.py` with optimal k
2. **Train & Evaluate** — text-only, image-only, and fusion classifiers with 5-fold CV
3. **Ablation Study** — 7 text-branch configurations including SLM comparison
4. **α Grid Search** — finds optimal soft-voting weight per fold automatically
5. **Statistical Tests** — McNemar's test + Bootstrap AUC CI vs text-only baseline

Full pipeline (includes data download, OCR, feature extraction):
```bash
python src/train_pipeline.py
```

Useful flags:
```bash
# Skip image download step
python src/train_pipeline.py --skip-image-download

# Skip k sensitivity (already ran, k known)
python src/train_pipeline.py --train-only --skip-k-sensitivity

# Keep downloaded images after pipeline completes
python src/train_pipeline.py --keep-images
```

---

### 2. Inference on new apps (from raw JSONL)

```bash
python src/run_inference.py --input-raw data/apps_inference_raw.jsonl --output runs/inference_results/
```

Runs preprocessing → OCR → feature extraction → inference automatically.

---

### 3. Inference from pre-extracted features

```bash
python src/run_inference.py --input data/inference_features --output runs/inference_results/
```

---

### 4. Fetch metadata for new apps

If starting from package names in a CSV (`pkg_name` column):

```bash
python src/fetch_app_metadata.py --input data/apps.csv --output data/apps_inference_raw.jsonl
```

Outputs:
- `data/apps_inference_raw.jsonl`
- `data/images/{pkg_name}/...`
- `data/failed_apps.txt` (if any app fails)

---

### 5. Analyze inference results against manual labels

```bash
python src/analyze_inference_results.py \
    --inference-dir runs/inference_results/ \
    --manual-file data/test_set_manual.csv \
    --out-dir runs/inference_analyzed/
```

---

## Training/Evaluation Protocol

The training code uses strict train/validation/test separation per outer fold:

1. **Outer CV (5 folds)** for final performance reporting.
2. **Inner validation split** inside each outer-train fold for:
   - SelectKBest feature selection fitting
   - LightGBM early stopping
   - Soft-voting α search
3. **Late-fusion stacking** uses out-of-fold (OOF) base predictions from outer-train.
4. **Threshold selection** is based on inner validation predictions.
5. **Soft-voting α** is selected per fold via grid search on `soft_voting_alpha_candidates`.

Primary metrics are reported at `classification_threshold` (default `0.5`). Validation-selected thresholds are saved for inference-time use.

---

## Key Outputs

Under `runs/<run_name>/`:

| Path | Description |
|------|-------------|
| `text_only/` | Text-only classifier results |
| `image_only/` | Image-only classifier results |
| `fusion/early_fusion/` | Early fusion results |
| `fusion/late_fusion_stacking/` | Stacking results |
| `fusion/late_fusion_soft_voting/` | Soft voting results + `alpha_grid_search.json` |
| `fusion/late_fusion_max_voting/` | Max voting results |
| `ablation/ablation_summary.json` | Ablation study table (all text-branch configs) |
| `k_sensitivity/summary.json` | k sensitivity analysis table |
| `statistical_tests/results.json` | McNemar p-values + Bootstrap AUC CI |

Each experiment folder includes:
- `metrics_per_fold.json`, `metrics_aggregated.json`
- `predictions.csv`, `validation_predictions.csv`
- `best_threshold_metrics.json`
- `saved_models/`

---

## Configuration

Edit `src/config.py`:

| Parameter | Description |
|-----------|-------------|
| `text_model` | SBERT model (default: `BAAI/bge-large-en-v1.5`) |
| `clip_model` | CLIP model (default: `openai/clip-vit-large-patch14-336`) |
| `feature_selection_k` | SelectKBest k (auto-updated by k sensitivity analysis) |
| `k_sensitivity_values` | k candidates for sensitivity analysis |
| `soft_voting_alpha_candidates` | α values to grid-search for soft voting |
| `n_folds` | Number of CV folds (default: 5) |
| `lgbm_params` | LightGBM hyperparameters |
| `classification_threshold` | Binary threshold for evaluation (default: 0.5) |

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
│       ├── text/        ← SBERT + keyword + meta features
│       ├── image/       ← CLIP + zero-shot + OCR features
│       └── slm/         ← SLM scores (ablation use only)
├── src/
│   ├── config.py
│   ├── train_pipeline.py
│   ├── run_inference.py
│   ├── fetch_app_metadata.py
│   ├── analyze_inference_results.py
│   ├── steps/
│   │   ├── preprocessing.py
│   │   ├── make_splits.py
│   │   ├── run_ocr.py
│   │   ├── extract_text_features.py
│   │   ├── extract_image_features.py
│   │   ├── extract_slm_features.py   ← ablation only
│   │   ├── train_evaluate.py
│   │   ├── k_sensitivity.py          ← new
│   │   └── statistical_tests.py      ← new
│   └── utils/
│       ├── io.py
│       ├── metrics.py
│       ├── seed.py
│       └── inference_helper.py
└── runs/
```
