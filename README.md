# LLM Detector

Multimodal LLM integration detection in Android apps using text (SBERT + keywords + SLM reasoning) and image (CLIP + OCR) features with Early/Late Fusion strategies.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Fetch Metadata for New Apps

Before running predictions on new apps, you need to fetch metadata from Google Play Store:

```bash
python src/fetch_app_metadata.py --input data/apps.csv --output data/apps_raw.jsonl --api-key YOUR_API_KEY
```

**Input**: CSV file with `pkg_name` column
**Output**: 
- `data/apps_raw.jsonl` - App metadata in JSONL format
- `data/images/{pkg_name}/` - Downloaded screenshots
- `data/failed_apps.txt` - List of apps that failed to fetch

After fetching metadata, you can run the full inference pipeline manually:
```bash
# Extract features
python src/steps/extract_text_features.py
python src/steps/extract_image_features.py
python src/steps/extract_slm_features.py

# Run inference
python src/run_inference.py --input data/features --output predictions
```

### 2. Train Pipeline (Full CV)
```bash
# Run from project root directory
python src/train_pipeline.py
```

Runs complete pipeline:
- Preprocessing (text cleaning, image deduplication)
- Create 5-fold stratified splits
- OCR extraction from screenshots
- Extract text features (SBERT embeddings, keywords, metadata)
- Extract image features (CLIP embeddings, zero-shot classification, OCR)
- Extract SLM reasoning scores (Qwen2.5-1.5B)
- Train & evaluate Early Fusion + Late Fusion (Stacking, Max Voting, Soft Voting)

### 3. Inference on New Apps

**Option A: Run Full Pipeline from Raw JSONL** (Recommended - Automated)

After fetching metadata with `fetch_app_metadata.py`, run the complete pipeline automatically:

```bash
# Full pipeline: preprocessing → OCR → feature extraction → inference
python src/run_inference.py --input-raw data/apps_inference_raw.jsonl --output predictions

# With custom output directory
python src/run_inference.py --input-raw data/apps_inference_raw.jsonl --output my_predictions
```

This will automatically:
1. Preprocess text and deduplicate images
2. Run OCR on screenshots
3. Extract text features (SBERT, keywords, metadata)
4. Extract image features (CLIP, zero-shot, OCR)
5. Extract SLM reasoning scores
6. Run inference with trained models using 4 fusion strategies:
   - **Early Fusion**: All features combined (optimized for recall)
   - **Stacking**: Meta-learner combines text/image predictions (optimized for precision)
   - **Soft Voting**: Average of text and image predictions (balanced)
   - **Max Voting**: Maximum of text and image predictions (recall-focused)

**Option B: Use Pre-extracted Features** (Manual)

If you already have extracted features:

```bash
# Use default features directory
python src/run_inference.py

# Or specify custom features directory
python src/run_inference.py --input data/features --output predictions
```

**Complete Workflow Example:**

```bash
# Step 1: Fetch metadata from Google Play
python src/fetch_app_metadata.py --input data/apps.csv --api-key YOUR_KEY

# Step 2: Run full inference pipeline
python src/run_inference.py --input-raw data/apps_inference_raw.jsonl --output predictions

# Output: predictions/early_fusion_inference.csv
#         predictions/stacking_inference.csv
#         predictions/soft_voting_inference.csv
#         predictions/max_voting_inference.csv
```

## Configuration

Edit `src/config.py` to customize:
- Models: Text encoder (SBERT), Image encoder (CLIP), SLM (Qwen/Gemma)
- Hyperparameters: LightGBM params, feature selection, thresholds
- Paths: Data, features, runs, inference directories

## Project Structure

```
LLM_Detector/
├── data/
│   ├── apps_raw.jsonl          # Raw dataset
│   ├── apps.jsonl              # Preprocessed dataset
│   ├── images/                 # App screenshots
│   ├── splits/                 # CV fold splits
│   └── features_v2/            # Cached features
├── src/
│   ├── config.py               # Central configuration
│   ├── train_pipeline.py       # Full training pipeline
│   ├── run_inference.py        # Inference script
│   ├── steps/                  # Pipeline steps
│   │   ├── preprocessing.py
│   │   ├── make_splits.py
│   │   ├── run_ocr.py
│   │   ├── extract_text_features.py
│   │   ├── extract_image_features.py
│   │   ├── extract_slm_features.py
│   │   └── train_evaluate.py
│   └── utils/                  # Utilities
│       ├── io.py
│       ├── metrics.py
│       └── seed.py
├── runs/                       # Training outputs
└── inference_results/          # Inference predictions
```

## Features

**Text Features:**
- SBERT embeddings (BAAI/bge-large-en-v1.5)
- Keyword matching (LLM-related terms)
- Metadata (category, ratings, installs)
- SLM reasoning scores (Qwen2.5-1.5B)

**Image Features:**
- CLIP embeddings (mean, max pooling)
- Zero-shot classification (LLM vs non-LLM UI)
- OCR text extraction

**Fusion Strategies:**
- Early Fusion: Concatenate all features → LightGBM
- Late Fusion (Stacking): Text/Image branches → Meta-learner
- Late Fusion (Max/Soft Voting): Ensemble predictions

## Results

Output CSV includes:
- `app_id`: Application identifier
- `y_prob`: Prediction probability
- `prediction_label`: Binary label (0/1)
- `y_true`: Ground truth (if available)
- `correct`: Prediction correctness (if labels available)