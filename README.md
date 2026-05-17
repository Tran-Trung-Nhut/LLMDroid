# LLMDroid — Multimodal Framework for LLM Integration Detection in Android Apps

LLMDroid phát hiện ứng dụng Android tích hợp LLM (ChatGPT, Claude, Gemini, v.v.) chỉ từ **metadata công khai trên app store** (mô tả văn bản + ảnh chụp màn hình), không cần APK hay quyền truy cập code.

**Kiến trúc:**
- **Text branch**: BGE-large-en-v1.5 embeddings (1024-d) + keyword features (13-d) + metadata features (21-d) = **1058-d**
- **Image branch**: CLIP ViT-L/14 mean+max pooled (768+768-d) + zero-shot chatbot-UI score (1-d) + OCR keyword features (15-d) = **1552-d**
- **Fusion**: Early Fusion + Late Fusion (Score-Max / Soft Voting / Stacking)
- **Classifier**: LightGBM + SelectKBest (mutual information, k=200)
- **Evaluation**: 5-fold stratified CV + independent test set (N=110)

---

## Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|-----------|-------------------|
| Python | 3.10+ |
| RAM | 16 GB (32 GB khuyến nghị) |
| GPU | NVIDIA T4 hoặc tương đương (cho BGE + CLIP encoding) |
| Disk | ~10 GB (models + data + features) |
| Tesseract OCR | Cài riêng — xem bên dưới |

---

## Cài đặt

### Bước 1 — Clone & cài Python dependencies

```bash
pip install -r requirements.txt
```

### Bước 2 — Cài Tesseract OCR (bắt buộc)

**Windows:**
```
Tải installer tại: https://github.com/UB-Mannheim/tesseract/wiki
Cài vào C:\Program Files\Tesseract-OCR\
Thêm vào PATH: C:\Program Files\Tesseract-OCR\
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

Kiểm tra: `tesseract --version`

### Bước 3 — HuggingFace token (nếu cần)

Nếu gặp lỗi khi tải model BGE hoặc CLIP:
```bash
# Windows PowerShell
$env:HF_TOKEN = "hf_xxx..."

# Linux/macOS
export HF_TOKEN="hf_xxx..."
```

---

## Cấu trúc dữ liệu cần có trước khi chạy

```
data/
├── apps_raw.jsonl              ← Dataset gốc 300 apps training (bắt buộc)
├── apps.jsonl                  ← Preprocessed (tự tạo khi chạy pipeline)
├── inference_manual.csv        ← Ground truth labels 110 test apps (bắt buộc)
├── apps_inference_raw.jsonl    ← Raw metadata 110 test apps (bắt buộc cho test)
├── images/                     ← Screenshots (tự download hoặc đã có sẵn)
└── splits/                     ← 5-fold splits (tự tạo khi chạy pipeline)
```

**Format `apps_raw.jsonl`** — mỗi dòng là một JSON:
```json
{
  "app_id": "com.example.app",
  "title": "Example App",
  "description": "Full description...",
  "short_description": "Short desc",
  "category": "Productivity",
  "recent_changes_text": "...",
  "image_paths": ["data/images/com.example.app/0.png"],
  "last_updated": "2025-03-15",
  "label_binary": 1
}
```

---

## Chạy pipeline chính (Training + Evaluation)

### Option A — Chạy toàn bộ từ đầu (khuyến nghị lần đầu)

```bash
python src/train_pipeline.py
```

Thực hiện theo thứ tự:
1. Download ảnh còn thiếu từ Google Play
2. Preprocessing (làm sạch text, dedup ảnh)
3. OCR (trích xuất text từ ảnh)
4. Tạo 5-fold CV splits
5. Extract text features (BGE + keyword + meta)
6. Extract image features (CLIP + zero-shot + OCR keywords)
7. k-sensitivity analysis (tìm k tối ưu cho SelectKBest)
8. Train + Evaluate (text-only, image-only, fusion, ablation)
9. Statistical tests

**Thời gian ước tính** (trên GPU T4):
- Feature extraction text: ~10 phút
- Feature extraction image: ~20-30 phút
- Training: ~30 phút

### Option B — Chỉ train (features đã có sẵn)

```bash
python src/train_pipeline.py --train-only
```

### Option C — Bỏ qua một số bước

```bash
# Bỏ qua download ảnh (đã có trong data/images/)
python src/train_pipeline.py --skip-image-download

# Bỏ qua OCR (đã có ocr_by_image trong apps.jsonl)
python src/train_pipeline.py --skip-ocr

# Bỏ qua extract features (đã có data/features/*.npz)
python src/train_pipeline.py --skip-features

# Bỏ qua k-sensitivity (đã biết k=200 là tối ưu)
python src/train_pipeline.py --train-only --skip-k-sensitivity

# Giữ lại ảnh sau khi train xong
python src/train_pipeline.py --keep-images
```

### Option D — Chạy từng bước riêng lẻ

```bash
# Preprocessing
python src/steps/preprocessing.py

# OCR
python src/steps/run_ocr.py

# Tạo splits
python src/steps/make_splits.py

# Extract features
python src/steps/extract_text_features.py
python src/steps/extract_image_features.py
python src/steps/extract_slm_features.py   # Chỉ cần cho ablation study

# k-sensitivity analysis
python src/steps/k_sensitivity.py

# Train + Evaluate
python src/steps/train_evaluate.py

# Statistical tests
python src/steps/statistical_tests.py
```

---

## Chạy các phân tích bổ sung (sau khi train xong)

> Tất cả scripts này đọc từ `runs/feature_fusion/` — phải train xong trước.

### Image branch ablation — Table 5

```bash
python src/steps/image_ablation.py
# Output: runs/feature_fusion/image_ablation/image_ablation_summary.json
```

### Evaluate trên independent test set — Table 20

> Yêu cầu: `data/features_test/text/features.npz` và `data/features_test/image/features.npz`

```bash
# Bước 1: Extract features cho 110 test apps
python src/steps/extract_text_features.py   # Sửa path trong script để trỏ vào apps_inference_raw.jsonl
python src/steps/extract_image_features.py  # Tương tự

# Bước 2: Evaluate
python src/steps/independent_test_eval.py
# Output: runs/feature_fusion/independent_test/table20_independent_test.json
#         runs/feature_fusion/independent_test/predictions_*.csv
```

### Branch complementarity — Table 10, Figure 4

```bash
python src/steps/branch_complementarity.py
# Output: runs/feature_fusion/branch_complementarity/table10_correlations.json
#         runs/feature_fusion/branch_complementarity/figure4_branch_scatter.png
```

### Disagreement-restricted accuracy — Table 11

```bash
python src/steps/disagreement_accuracy.py
# Output: runs/feature_fusion/disagreement_accuracy/table11_disagree_accuracy.json
```

### Per-category performance — Table 13

> Yêu cầu: independent_test_eval.py đã chạy (cần predictions_soft_voting.csv)

```bash
python src/steps/per_category.py
# Output: runs/feature_fusion/per_category/table13_per_category.json
```

### Prior-corrected precision — Table 14, Figure 5

> Yêu cầu: independent_test_eval.py đã chạy

```bash
python src/steps/prior_correction.py
# Output: runs/feature_fusion/prior_correction/table14_prior_corrected.json
#         runs/feature_fusion/prior_correction/figure5_prior_corrected_precision.png
```

### Temporal split — Table 15

> Yêu cầu: trường `last_updated` trong `apps.jsonl`

```bash
python src/steps/temporal_split.py
# Output: runs/feature_fusion/temporal_split/table15_temporal.json
```

### Inference latency — Table 16

> Yêu cầu: independent test features có sẵn

```bash
python src/steps/latency_benchmark.py
# Output: runs/feature_fusion/latency/table16_latency.json
```

### Probability calibration — Table 17

```bash
python src/steps/calibration.py
# Output: runs/feature_fusion/calibration/table17_calibration.json
```

### Robustness to missing modalities — Table 18

> Yêu cầu: independent_test_eval.py đã chạy

```bash
# (Tùy chọn) Extract truncated text features cho condition 3
python src/steps/extract_text_features_trunc50.py
# Output: data/features_test_trunc50/text/features.npz

python src/steps/robustness.py
# Output: runs/feature_fusion/robustness/table18_robustness.json
```

---

## External Baselines — Table 12

> Các baselines này tốn kém (API hoặc GPU). Kết quả từ paper có sẵn tại:
> `runs/feature_fusion/paper_reported_results/table12_external_baselines.json`
> (chạy `python src/steps/report_paper_results.py`)

### Baseline 1 — Qwen2.5-7B (description-only, zero-shot)

> Yêu cầu: GPU ~20GB VRAM, model tự download (~15GB)

```bash
python src/steps/baselines/baseline_qwen.py
# Output: runs/feature_fusion/independent_test/baseline_qwen.json
```

### Baseline 2 — GPT-4o-mini + Gemini-1.5-Flash (multimodal, zero-shot)

> Yêu cầu: API keys trong environment variables

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
$env:GOOGLE_API_KEY = "AIza..."

python src/steps/baselines/baseline_mllm_zeroshot.py
# Output: runs/feature_fusion/independent_test/baseline_mllm_zeroshot_gpt4o_mini.json
#         runs/feature_fusion/independent_test/baseline_mllm_zeroshot_gemini_1_5_flash.json
# Chi phí ước tính: ~$0.19 (GPT-4o-mini) + ~$0.14 (Gemini) cho 110 apps
```

### Baseline 3 — GPT-4o (multimodal, 6-shot)

> Yêu cầu: OPENAI_API_KEY, chi phí ~$4.62 cho 110 apps

```bash
$env:OPENAI_API_KEY = "sk-..."
python src/steps/baselines/baseline_mllm_fewshot.py
# Output: runs/feature_fusion/independent_test/baseline_mllm_fewshot_gpt4o.json
```

### Baseline 4 — E2E fine-tuned multimodal transformer

> Yêu cầu: GPU, thời gian train ~2-4 giờ
> Kiến trúc: BGE(1024) + CLIP_mean(768) → Linear(512) → 2×TransformerEncoder → binary head

```bash
python src/steps/baselines/baseline_e2e_transformer.py
# Output: runs/feature_fusion/independent_test/baseline_e2e_transformer.json
```

### Lấy kết quả từ paper (không cần chạy lại)

```bash
python src/steps/report_paper_results.py
# Output: runs/feature_fusion/paper_reported_results/*.json
```

---

## Inference trên apps mới

### Từ package names

```bash
# Bước 1: Fetch metadata từ Google Play
python src/fetch_app_metadata.py \
    --input data/inference_apps.csv \
    --output data/apps_inference_raw.jsonl

# Bước 2: Run inference
python src/run_inference.py \
    --input-raw data/apps_inference_raw.jsonl \
    --output runs/inference_results/
```

### Từ features đã extract sẵn

```bash
python src/run_inference.py \
    --input data/inference_features \
    --output runs/inference_results/
```

### Phân tích kết quả inference

```bash
python src/analyze_inference_results.py \
    --inference-dir runs/inference_results/ \
    --manual-file data/inference_manual.csv \
    --out-dir runs/inference_analyzed/
```

---

## Tất cả outputs sau khi chạy đầy đủ

```
runs/feature_fusion/
├── text_only/
│   ├── metrics_aggregated.json        ← ROC-AUC, F1, v.v. (mean ± std)
│   ├── metrics_per_fold.json          ← Kết quả từng fold
│   ├── predictions.csv                ← y_true, y_prob cho mọi app
│   ├── validation_predictions.csv     ← Inner-val predictions
│   ├── best_threshold_metrics.json    ← Threshold tối ưu từ validation
│   └── saved_models/                  ← lgbm_fold_N.txt, selector_fold_N.joblib
│
├── image_only/                        ← Tương tự text_only
│
├── fusion/
│   ├── early_fusion/                  ← Tương tự text_only
│   ├── late_fusion_score_max/         ← + không có saved_models
│   ├── late_fusion_soft_voting/       ← + alpha_grid_search.json
│   ├── late_fusion_stacking/          ← + saved_models/meta_clf_fold_N.joblib
│   └── base_models_saved/             ← text_lgbm_fold_N.txt, img_lgbm_fold_N.txt
│
├── ablation/
│   ├── ablation_summary.json          ← Table 3: 7 text-branch configs
│   ├── sbert_only/, handcrafted_only/, ...
│
├── image_ablation/
│   └── image_ablation_summary.json    ← Table 5: leave-one-out image ablation
│
├── k_sensitivity/
│   └── summary.json                   ← Table 4: k ∈ {20,50,100,200,500,all}
│
├── statistical_tests/
│   └── summary.csv                    ← Table 9: DeLong + Holm-BH + Cliff's δ
│
├── independent_test/
│   ├── table20_independent_test.json  ← Table 20: N=110
│   ├── predictions_*.csv              ← Per-strategy predictions on test set
│   └── baseline_*.json                ← External baseline results
│
├── branch_complementarity/
│   ├── table10_correlations.json      ← Table 10: ρ_Pearson/Spearman
│   └── figure4_branch_scatter.png     ← Figure 4
│
├── disagreement_accuracy/
│   └── table11_disagree_accuracy.json ← Table 11
│
├── per_category/
│   └── table13_per_category.json      ← Table 13
│
├── prior_correction/
│   ├── table14_prior_corrected.json   ← Table 14
│   └── figure5_prior_corrected_precision.png ← Figure 5
│
├── temporal_split/
│   └── table15_temporal.json          ← Table 15
│
├── latency/
│   └── table16_latency.json           ← Table 16
│
├── calibration/
│   └── table17_calibration.json       ← Table 17
│
├── robustness/
│   └── table18_robustness.json        ← Table 18
│
└── paper_reported_results/            ← Kết quả từ paper (không cần chạy lại)
    ├── table3_text_ablation.json
    ├── table5_image_ablation.json
    ├── table7_cv_results.json
    ├── table8_threshold_optimized.json
    ├── table12_external_baselines.json
    ├── table15_temporal.json
    ├── table16_latency.json
    ├── table17_calibration_paper.json
    ├── table18_robustness_paper.json
    └── table19_llmaid_validation.json
```

---

## Cấu hình (`src/config.py`)

| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `text_model` | `BAAI/bge-large-en-v1.5` | Text encoder |
| `clip_model` | `openai/clip-vit-large-patch14-336` | Image encoder |
| `feature_selection_k` | `200` | Số features giữ lại (SelectKBest) |
| `n_folds` | `5` | Số fold CV |
| `classification_threshold` | `0.5` | Ngưỡng phân loại mặc định |
| `threshold_search_min/max` | `0.01 / 0.99` | Khoảng tìm threshold tối ưu |
| `soft_voting_alpha_candidates` | `(0.3, 0.4, 0.5, 0.6, 0.7)` | α candidates cho Soft Voting |
| `inner_val_ratio` | `0.2` | Tỷ lệ inner-validation split |
| `seed` | `42` | Random seed |

---

## Kết quả mong đợi (5-fold CV, τ=0.5)

| Strategy | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|----------|---------|-----------|--------|-----|---------|--------|
| Text-Only | 0.809 | 0.816 | 0.704 | 0.753 | 0.872 | 0.869 |
| Image-Only | 0.742 | 0.741 | 0.592 | 0.654 | 0.825 | 0.776 |
| Early Fusion | 0.792 | 0.793 | 0.688 | 0.732 | 0.873 | 0.867 |
| Score-Max | 0.792 | 0.733 | 0.808 | 0.766 | 0.894 | 0.872 |
| **Soft Voting** | **0.809** | **0.881** | 0.632 | 0.727 | **0.904** | **0.886** |
| Stacking | 0.802 | 0.809 | 0.696 | 0.742 | 0.902 | 0.885 |

*Số liệu từ paper main-v2.pdf Table 7*

---

## Troubleshooting

**Lỗi Tesseract not found:**
```
pytesseract.pytesseract.TesseractNotFoundError
→ Cài Tesseract và thêm vào PATH (xem mục Cài đặt)
```

**Lỗi CUDA out of memory:**
```
→ Giảm batch size trong config.py: text_batch_size, clip_batch_size
```

**Lỗi features.npz app order mismatch:**
```
→ Xóa data/features/ và chạy lại extract features
```

**Model download chậm hoặc lỗi:**
```
→ Set HF_TOKEN trong environment variable
→ Hoặc tải model thủ công vào ~/.cache/huggingface/
```

**Lỗi statsmodels not found:**
```bash
pip install statsmodels
```

---

## Tài liệu tham khảo

- Paper: *LLMDroid: A Multimodal Framework for LLM Integration Detection in Mobile Apps Using Textual and Visual App Store Data* (Under Review, 2026)
- Repository: https://github.com/Tran-Trung-Nhut/LLMDroid
- BGE model: [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)
- CLIP model: [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)
