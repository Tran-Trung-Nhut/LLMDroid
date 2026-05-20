# LLMDroid

**Multimodal screening for LLM-integrated Android apps — no APK required.**

LLMDroid flags likely LLM-powered apps (ChatGPT, Claude, Gemini, …) using only public app-store metadata: text descriptions and promotional screenshots.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-under%20review-orange)]()

---

## How it works

```
App listing (title + description + screenshots)
        │
        ├─ Text branch ──── BGE-large-en-v1.5 (1024-d)
        │                   + LLM keyword features (13-d)
        │                   + metadata features (21-d)
        │
        └─ Image branch ─── CLIP ViT-L/14 mean+max (1536-d)
                            + zero-shot chat-UI score (1-d)
                            + OCR keyword features (15-d)
                                    │
                                    ▼
                    Fusion → LightGBM (SelectKBest k=200)
                    Score-Max / Soft Voting / Stacking / Early Fusion
```

**5-fold CV on 300 apps · Independent test N=110 · ROC-AUC up to 0.938**

---

## Setup

```bash
git clone https://github.com/Tran-Trung-Nhut/LLMDroid
cd LLMDroid

conda create -n llmdroid python=3.10 -y && conda activate llmdroid
pip install -r requirements.txt

# System deps (Ubuntu / Lightning AI)
sudo apt-get install -y tesseract-ocr default-jre
```

---

## Run

### 1 — Training pipeline

```bash
python src/train_pipeline.py
```

| Step | What happens | Skip flag |
|------|-------------|-----------|
| 0 | Download screenshots from Google Play | `--skip-image-download` |
| 1 | Preprocess text (clean HTML, dedup images) | — |
| 2 | OCR screenshots via Tesseract | `--skip-ocr` |
| 3 | Create 5-fold stratified splits | — |
| 4a | Extract text features (BGE + keywords) | `--skip-features` |
| 4b | Extract image features (CLIP + OCR) | `--skip-features` |
| 5 | Train & evaluate all fusion strategies | — |
| 6 | k-sensitivity analysis | `--skip-k-sensitivity` |
| 7 | Statistical significance tests | — |

If features are already cached (`data/features/*.npz`), skip straight to training:

```bash
python src/train_pipeline.py --train-only
```

### 2 — Post-training analysis

```bash
python src/run_analysis.py
```

### 3 — Baselines

```bash
export OPENAI_API_KEY=sk-...
python src/run_baselines.py
```

### 4 — Inference on new apps

```bash
# From raw JSONL
python src/run_inference.py --input-raw data/apps_inference_raw.jsonl --output results/

# From pre-extracted features
python src/run_inference.py --input data/features_test --output results/
```

---

## Annotation pipelines (independent)

### Inter-annotator agreement

```bash
# Requires: data/inter_annotator.csv
python src/run_iaa.py
# → runs/cohen_kappa/iaa.txt
```

### Code-level validation

```bash
# One-time: download Androzoo metadata (~3 GB)
wget https://androzoo.uni.lu/static/lists/latest.csv.gz
gunzip latest.csv.gz && mv latest.csv data/androzoo_latest.csv

export ANDROZOO_API_KEY=your_key
python src/run_code_validation.py
# → runs/cohen_kappa/validation.txt
# Checkpoint auto-saved — safe to interrupt and resume
```

---

## Data files required

| File | Description |
|------|-------------|
| `data/apps_raw.jsonl` | Training apps with labels and image paths |
| `data/apps_inference_raw.jsonl` | Apps to run inference on |
| `data/inference_manual.csv` | Ground-truth labels for 110-app test set |
| `data/images/{app_id}/*.png` | Screenshots (downloaded by Step 0 if missing) |

---

## Output structure

```
runs/feature_fusion/
├── fusion/base_models_saved/     ← LightGBM models (×5 folds)
├── fusion/early_fusion/
├── fusion/late_fusion_stacking/
├── fusion/late_fusion_soft_voting/
├── independent_test/
├── statistical_tests/
├── branch_complementarity/
├── temporal_split/
└── robustness/

runs/cohen_kappa/
├── iaa.txt
└── validation.txt
```

---

## Reproduce from cached features

If `data/features/` and `data/features_test/` are shared (e.g., via cloud):

```bash
python src/train_pipeline.py --train-only
python src/run_analysis.py
python src/run_baselines.py
```

---

## Citation

```bibtex
@article{llmdroid2026,
  title   = {LLMDroid: A Multimodal Framework for LLM Integration Detection
             in Mobile Apps Using Textual and Visual App Store Data},
  year    = {2026},
  note    = {Under review}
}
```
