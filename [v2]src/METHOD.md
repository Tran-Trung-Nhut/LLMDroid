# V2 — Multi-Modal Feature Fusion for LLM-Integrated App Detection

## 1. Tổng quan

Phương pháp V2 phát hiện ứng dụng Android có tích hợp LLM (Large Language Model) hay không, dựa trên **metadata text** (tiêu đề, mô tả, ...) và **screenshots** từ Google Play Store.

Thay vì fine-tune một Vision-Language Model trên tập dữ liệu nhỏ (298 mẫu) như V1, V2 sử dụng chiến lược **trích xuất đặc trưng từ các pre-trained model mạnh (zero-shot)** rồi kết hợp qua **classifier truyền thống (LightGBM)**. Cách tiếp cận này phù hợp hơn với dataset nhỏ, tránh overfitting, và cho phép phân tích feature importance.

### So sánh V1 vs V2

| | V1 | V2 |
|---|---|---|
| Phương pháp | Fine-tune PaliGemma 3B (LoRA) | Feature extraction → LightGBM |
| Vấn đề | 298 mẫu không đủ fine-tune VLM | Dùng pre-trained model zero-shot, không cần fine-tune |
| Image resolution | 224 px | 336 px (CLIP ViT-L/14) |
| Explainability | Black box | Feature importance, SHAP-ready |
| Thời gian train | Hàng giờ (GPU) | ~10 phút extract features + vài giây train |
| Kết quả V1 | < 50% accuracy | Mục tiêu ≥ 80% |

---

## 2. Kiến trúc Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                          DATA PREPARATION                            │
│  apps_raw.jsonl → [preprocessing] → apps.jsonl → [OCR] → apps.jsonl │
│                                     [make_splits] → fold_0..4.json   │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
┌─────────────────────────┐    ┌─────────────────────────────┐
│     TEXT STREAM          │    │       IMAGE STREAM           │
│                          │    │                              │
│ ① BGE-large embedding   │    │ ④ CLIP ViT-L/14 embedding   │
│    (1024-d)              │    │    mean-pool (768-d)         │
│                          │    │    max-pool  (768-d)         │
│ ② Keyword features      │    │                              │
│    (13-d)                │    │ ⑤ Zero-shot similarity      │
│                          │    │    vs LLM prompts (~12-d)    │
│ ③ Metadata features     │    │                              │
│    (21-d)                │    │ ⑥ OCR keyword features       │
│                          │    │    (15-d)                     │
│ Total: 1058-d            │    │ Total: 1563-d                │
└────────────┬─────────────┘    └──────────────┬──────────────┘
             │                                  │
             ▼                                  ▼
   ┌──────────────────┐              ┌──────────────────┐
   │ [A] Text-only    │              │ [B] Image-only   │
   │     LightGBM     │              │     LightGBM     │
   └────────┬─────────┘              └────────┬─────────┘
            │                                  │
            │         ┌────────────────────────┤
            │         │                        │
            ▼         ▼                        ▼
   ┌──────────────────────┐        ┌──────────────────────┐
   │ [C1] Early Fusion    │        │ [C2] Late Fusion     │
   │ concat all → LightGBM│        │ stack probs → LogReg │
   └──────────────────────┘        └──────────────────────┘
```

---

## 3. Chi tiết từng thành phần

### 3.1 Text Stream

**File:** `extract_text_features.py`

#### ① Sentence Embedding — BGE-large-en-v1.5

- Model: `BAAI/bge-large-en-v1.5` (pre-trained text encoder, 1024-d output)
- Input: Cleaned app text (title + category + description + recent changes)
- Method: Mean pooling over non-padding tokens
- Output: Vector 1024 chiều capture ngữ nghĩa tổng thể của mô tả app
- Tại sao BGE-large: Là state-of-the-art embedding model, mạnh hơn Sentence-BERT gốc, vừa đủ cho GPU L4

#### ② Keyword Features (13-d)

Sử dụng danh sách 38 keyword LLM-related (file `keywords.py`), chia thành 5 category:

| Category | Ví dụ keywords |
|----------|----------------|
| `model_name` | chatgpt, gpt-4, claude, gemini, llama |
| `core_llm` | llm, large language model, chatbot, ai assistant |
| `generation` | text generation, ai writer, paraphrase, summarize |
| `interaction` | ask ai, chat with ai, prompt, conversational ai |
| `content` | essay generator, ai copywriting, ai content |

Features trích xuất:
- Tổng số keyword hit (1)
- log(1 + count) (1)
- Per-category: binary presence + count (5 × 2 = 10)
- Max keyword match length (1) — proxy cho mức độ cụ thể

#### ③ Metadata Features (21-d)

Các đặc trưng handcrafted từ metadata:
- Độ dài description, short_description, title (3)
- Có recent_changes hay không (1)
- Số lượng screenshots (1)
- Category one-hot encoding (16 = 15 top categories + 1 other)

### 3.2 Image Stream

**File:** `extract_image_features.py`

#### ④ CLIP Image Embedding (768-d × 2)

- Model: `openai/clip-vit-large-patch14-336` (Vision Transformer, 336px input)
- Input: Tất cả screenshots của app
- Method: Encode từng ảnh → pooling qua tất cả ảnh
  - **Mean pooling**: Đặc trưng trung bình (768-d)
  - **Max pooling**: Giữ lại tín hiệu mạnh nhất (768-d)
- 336px resolution cho phép CLIP đọc được text trên screenshots (so với 224px của V1)

#### ⑤ Zero-shot Similarity Scores (~12-d)

Đây là feature discriminative nhất — dùng khả năng zero-shot của CLIP:

**Positive prompts** (mô tả app có LLM):
- "a screenshot of an AI chatbot conversation"
- "a mobile app with AI chat assistant interface"
- "a screenshot showing AI-generated text responses"
- "an app with a large language model powered chat"
- "a conversational AI interface on a phone screen"

**Negative prompts** (mô tả app bình thường):
- "a mobile app screenshot with no AI features"
- "a standard mobile application interface"
- "a photo editing or camera app screenshot"
- "a settings or profile page of a mobile app"

Features:
- max/mean cosine similarity với positive prompts (2)
- Per-positive-prompt max similarity (5)
- max/mean cosine similarity với negative prompts (2)
- Discriminative gap: max_pos - max_neg (1)

**Ý tưởng**: Nếu screenshot nào đó giống "AI chatbot conversation" hơn là "standard app interface", app đó rất có khả năng tích hợp LLM.

#### ⑥ OCR Keyword Features (15-d)

- Concatenate OCR text từ tất cả screenshots
- Áp dụng cùng keyword matching logic như text stream
- Thêm 2 features: tổng dộ dài OCR text + tỷ lệ ảnh có text
- Tổng: 13 (keyword) + 2 (extra) = 15 features

### 3.3 Classifiers

**File:** `train_evaluate.py`

#### [A] Text-Only Classifier

- Input: 1058-d (SBERT 1024 + keywords 13 + meta 21)
- Model: LightGBM với early stopping
- Đo lường khả năng dự đoán chỉ từ metadata text

#### [B] Image-Only Classifier

- Input: 1563-d (CLIP mean 768 + CLIP max 768 + zero-shot 12 + OCR 15)
- Model: LightGBM
- Đo lường khả năng dự đoán chỉ từ visual evidence

#### [C1] Early Fusion

- Input: 2621-d (text 1058 + image 1563 — nối tất cả features)
- Model: LightGBM
- Kết hợp mọi tín hiệu vào một mô hình duy nhất

#### [C2] Late Fusion (Stacking)

- Train Text-only LightGBM → output probability
- Train Image-only LightGBM → output probability
- Meta-learner: Logistic Regression trên 2 probabilities
- Cho phép mỗi stream "chuyên gia" ở domain riêng, meta-learner quyết định trọng số

### 3.4 Evaluation

- **5-fold Stratified Cross-Validation** (cùng splits với V1 để so sánh công bằng)
- Metrics per fold: Accuracy, Precision, Recall, F1, Macro-F1, PR-AUC, ROC-AUC
- Aggregated: mean ± std qua 5 folds
- **Threshold search**: Grid search threshold 0.30–0.70 trên aggregated predictions để tìm optimal F1

---

## 4. Cấu trúc file

```
[v2]src/
├── config.py                    # Cấu hình tập trung (models, paths, prompts, hyperparams)
├── keywords.py                  # Danh sách 38 LLM keywords (5 categories)
├── preprocessing.py             # Làm sạch text, dedup ảnh, build unified text field
├── make_splits.py               # Tạo 5-fold stratified CV splits
├── run_ocr.py                   # Chạy Tesseract OCR trên tất cả screenshots
├── extract_text_features.py     # Trích xuất features từ text (BGE + keywords + meta)
├── extract_image_features.py    # Trích xuất features từ ảnh (CLIP + zero-shot + OCR)
├── train_evaluate.py            # Train & evaluate 4 classifiers, threshold search
├── run.py                       # Orchestrator — chạy toàn bộ pipeline 1 lệnh
├── requirements.txt             # Dependencies
└── utils/
    ├── __init__.py
    ├── io.py                    # Đọc/ghi JSONL, JSON, CSV
    ├── metrics.py               # Binary classification metrics
    └── seed.py                  # Reproducibility (random, numpy, torch seeds)
```

### Chi tiết từng file

| File | Chức năng | Reuse từ V1? |
|------|-----------|-------------|
| `config.py` | Frozen dataclass chứa mọi config: model names, embed dims, batch sizes, CLIP zero-shot prompts, paths, hyperparams | Cấu trúc từ V1, nội dung mới |
| `keywords.py` | List 38 keywords + 5 categories dùng cho keyword matching | ✅ Giữ nguyên |
| `preprocessing.py` | HTML cleaning, URL/email removal, footer truncation, whitespace normalization, image dedup (perceptual hash), build unified text | ✅ Giữ nguyên logic |
| `make_splits.py` | StratifiedKFold(5) trên label_binary, save fold JSON | ✅ Giữ nguyên |
| `run_ocr.py` | Tesseract OCR trên từng screenshot, cache kết quả vào apps.jsonl | ✅ Giữ nguyên |
| `extract_text_features.py` | Load BGE-large, encode texts, compute keyword & meta features, save .npz | 🆕 Mới hoàn toàn |
| `extract_image_features.py` | Load CLIP ViT-L/14, encode images, compute zero-shot scores & OCR features, save .npz | 🆕 Mới hoàn toàn |
| `train_evaluate.py` | Load features, train LightGBM per fold, evaluate 4 experiments, threshold search | 🆕 Mới hoàn toàn |
| `run.py` | Orchestrator gọi tuần tự: preprocess → OCR → splits → features → train | Cấu trúc từ V1, nội dung mới |
| `utils/*` | IO, metrics (thêm accuracy + ROC-AUC), seed | ✅ Mở rộng từ V1 |

---

## 5. Cách chạy

```bash
# Cài dependencies
pip install -r "[v2]src/requirements.txt"

# Chạy toàn bộ pipeline
python "[v2]src/run.py"

# Bỏ qua OCR nếu đã chạy trước đó (dữ liệu V1)
python "[v2]src/run.py" --skip-ocr

# Bỏ qua feature extraction (nếu đã cache)
python "[v2]src/run.py" --skip-features

# Chỉ train & evaluate (features đã có)
python "[v2]src/run.py" --train-only
```

### Pipeline steps

```
Step 1: Preprocessing     — apps_raw.jsonl → apps.jsonl
Step 2: Make Splits        — apps.jsonl → data/splits/fold_0..4.json
Step 3: OCR               — apps.jsonl += ocr_by_image (Tesseract)
Step 4a: Text Features     — apps.jsonl → data/features_v2/text/features.npz
Step 4b: Image Features    — apps.jsonl → data/features_v2/image/features.npz
Step 5: Train & Evaluate   — features + splits → runs/v2_feature_fusion/
```

### Output structure

```
runs/v2_feature_fusion/
├── text_only/
│   ├── predictions.csv
│   ├── metrics_per_fold.json
│   ├── metrics_aggregated.json
│   ├── best_threshold_metrics.json
│   └── feature_importance_fold0.json
├── image_only/
│   └── (same structure)
├── fusion/
│   ├── early_fusion/
│   │   └── (same structure)
│   └── late_fusion/
│       ├── predictions.csv          # includes text_prob, image_prob columns
│       ├── metrics_per_fold.json
│       ├── metrics_aggregated.json
│       └── best_threshold_metrics.json
```

---

## 6. Tại sao phương pháp này phù hợp hơn V1

### 6.1 Dataset nhỏ (298 mẫu)

Fine-tuning model 3B parameters với 298 mẫu (V1) gần như chắc chắn overfit hoặc underfit. LightGBM với ~2600 features và regularization mạnh (L1/L2, bagging, early stopping) xử lý tốt hơn nhiều.

### 6.2 Zero-shot knowledge

BGE-large và CLIP đã được train trên hàng tỷ text-image pairs. Thay vì phá hỏng knowledge này bằng fine-tuning trên 298 mẫu, ta giữ nguyên và chỉ trích xuất features — tận dụng toàn bộ pre-trained knowledge.

### 6.3 Explainability

LightGBM cung cấp feature importance → biết được feature nào quan trọng nhất (ví dụ: CLIP zero-shot score? keyword count? BGE embedding dimension nào?). Điều này quan trọng cho bài nghiên cứu.

### 6.4 Ablation study tự nhiên

4 experiments (text-only, image-only, early fusion, late fusion) tạo thành ablation study hoàn chỉnh — cho thấy đóng góp của từng modality.

### 6.5 Reproducibility

- Features được cache (.npz) → train lại classifier trong vài giây
- Fixed seeds ở mọi component
- Cùng splits với V1 → so sánh công bằng
