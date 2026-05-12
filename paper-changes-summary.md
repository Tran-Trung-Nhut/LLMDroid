# Tóm tắt thay đổi cho teammate — Cập nhật bài báo LLMDroid

> Tất cả thay đổi bên dưới xuất phát từ việc thực hiện 4 vấn đề trong ablation/evaluation protocol theo yêu cầu reviewer. Mỗi mục có **section cần sửa** và **nội dung cụ thể**.

---

## 1. Thay đổi kiến trúc: Bỏ SLM khỏi Text Branch

### Phát hiện
Ablation study cho thấy SLM (Qwen2.5-1.5B-Instruct) **không đóng góp** vào hiệu suất phân loại:

| Config | Dims | ROC-AUC | F1 |
|--------|------|---------|----|
| SBERT + Handcrafted (main) | 1058 | **0.8723 ± 0.0343** | **0.7531 ± 0.0499** |
| SBERT + Handcrafted + SLM | 1059 | 0.8652 ± 0.0347 | 0.7337 ± 0.0477 |
| SLM only | 1 | 0.5227 ± 0.0421 | 0.0000 ± 0.0000 |

**Lý do kỹ thuật:** SLM feature có F-statistic = 0.18, trong khi ngưỡng top-200 của SBERT là 8.25. SelectKBest luôn loại SLM trước khi LightGBM nhìn thấy nó. Ngay cả khi force-include, SLM thay thế một SBERT dimension hữu ích bằng noise → giảm AUC.

### Cần sửa trong paper

**§3 (System Architecture / Text Branch):**
- Xóa mô tả về SLM (Qwen2.5-1.5B, prompting, score extraction)
- Text branch hiện tại: **SBERT embeddings (1024-d) + Keyword features (13-d) + Metadata features (21-d) = 1058-d**

**§4.1 (Ablation Study) — viết mới:**
Thêm bảng ablation đầy đủ:

| Config | Dims | ROC-AUC | F1 |
|--------|------|---------|----|
| SBERT only | 1024 | 0.8079 ± 0.0695 | 0.6820 ± 0.0842 |
| Handcrafted only | 34 | 0.7827 ± 0.0494 | 0.7020 ± 0.0532 |
| SLM only | 1 | 0.5227 ± 0.0421 | 0.0000 ± 0.0000 |
| SBERT + Handcrafted | 1058 | **0.8723 ± 0.0343** | **0.7531 ± 0.0499** |
| SBERT + SLM | 1025 | 0.8131 ± 0.0711 | 0.6930 ± 0.1019 |
| Handcrafted + SLM | 35 | 0.7762 ± 0.0589 | 0.6832 ± 0.0883 |
| SBERT + Handcrafted + SLM | 1059 | 0.8652 ± 0.0347 | 0.7337 ± 0.0477 |

**Narrative đề xuất:** *"Despite integrating SLM reasoning scores (Qwen2.5-1.5B-Instruct), our ablation reveals no performance improvement over SBERT+Handcrafted features. Analysis shows the SLM score has a univariate F-statistic of 0.18, far below the top-200 selection threshold of 8.25, indicating that app descriptions do not contain sufficient explicit LLM terminology for zero-shot SLM prompting to be discriminative. This finding suggests that semantic embeddings already capture the relevant signal more effectively than explicit LLM reasoning for this task."*

---

## 2. Hyperparameter thay đổi: k = 50 → k = 200

### Phát hiện
k sensitivity analysis trên text features (TextOnly classifier):

| k | ROC-AUC | F1 |
|---|---------|----|
| 20 | 0.8339 ± 0.0539 | 0.7192 ± 0.0687 |
| 50 | 0.8388 ± 0.0432 | 0.7155 ± 0.0683 |
| 100 | 0.8579 ± 0.0352 | 0.7301 ± 0.0539 |
| **200** | **0.8723 ± 0.0343** | **0.7531 ± 0.0499** |
| 500 | 0.8692 ± 0.0421 | 0.7339 ± 0.0289 |
| all (1058) | 0.8706 ± 0.0180 | 0.7299 ± 0.0444 |

k=200 cho kết quả tốt nhất. Đường AUC gần phẳng từ k=200 trở đi → không phải cherry-pick.

### Cần sửa trong paper

**§4.2 (k Sensitivity Analysis) — viết mới:**
- Thêm hình line chart AUC và F1 theo k
- **k = 200** là giá trị cuối cùng được dùng cho tất cả experiments
- Justify: *"Performance plateaus beyond k=200, confirming this is not a cherry-picked value but a stable operating point."*

**§3 hoặc §4 (Implementation Details):**
- Đổi `k = 50` → `k = 200` ở mọi nơi đề cập

---

## 3. Soft Voting: α cố định → α tối ưu theo validation

### Thay đổi
Thay vì α = 0.5 hardcoded:

```
p_soft = α * p_text + (1 − α) * p_image
```

α được grid-search trên inner validation set mỗi fold với candidates {0.3, 0.4, 0.5, 0.6, 0.7}.

### Cần sửa trong paper

**§3 (Soft Voting definition):**
- Cập nhật: *"α is selected per fold via grid search on the inner validation set, with candidates α ∈ {0.3, 0.4, 0.5, 0.6, 0.7}."*
- Báo cáo mean α và distribution qua 5 folds trong kết quả

**§4.4 (Hyperparameter Sensitivity):**
- Thêm bảng/figure α grid search results
- Justify lý do không dùng α = 0.5 cố định

---

## 4. Thêm mới: Statistical Significance Tests

### Tests được thực hiện
- **McNemar's Test**: so sánh binary predictions (threshold = 0.5) giữa mỗi fusion strategy và text-only baseline
- **Bootstrap AUC CI (95%)**: ước tính confidence interval cho ΔAUC (n_bootstrap = 2000)

### Cần sửa trong paper

**§4 hoặc §5 (Results) — thêm bảng mới:**

| Comparison vs Text-Only | ΔAUC | 95% CI | McNemar p | Bootstrap p |
|------------------------|------|--------|-----------|-------------|
| Early Fusion | — | — | — | — |
| Late Fusion (Stacking) | — | — | — | — |
| Late Fusion (Soft Voting) | — | — | — | — |
| Late Fusion (Max Voting) | — | — | — | — |

*(Điền số từ `runs/feature_fusion/statistical_tests/results.json` sau khi chạy xong)*

**Xóa/sửa:**
- Bất kỳ chỗ nào dùng từ **"substantially outperforms"** mà không có p-value → thêm "(p < 0.05, McNemar's test)" hoặc đổi thành "significantly outperforms" nếu p < 0.05, hoặc "outperforms" nếu không significant

---

## 5. Bug fix: stacking base models k=100 → k=200

### Vấn đề cũ
Stacking base models (text branch và image branch) dùng `k_features=100` hardcoded, trong khi text-only và image-only dùng `CFG.feature_selection_k = 50`. Inconsistency này làm kết quả stacking không comparable với các baselines.

### Đã fix
Tất cả experiments giờ dùng cùng `CFG.feature_selection_k = 200`.

### Cần sửa trong paper
**§3 (Implementation Details):**
- Đảm bảo chỉ đề cập MỘT giá trị k duy nhất (k = 200) cho tất cả experiments

---

## 6. Tóm tắt các số liệu cần cập nhật

Tất cả kết quả dưới đây cần được lấy từ lần chạy mới nhất (sau khi pipeline hoàn tất) từ `runs/feature_fusion/`:

| Metric | Lấy từ file |
|--------|-------------|
| Text-Only AUC, F1 | `text_only/metrics_aggregated.json` |
| Image-Only AUC, F1 | `image_only/metrics_aggregated.json` |
| Early Fusion AUC, F1 | `fusion/early_fusion/metrics_aggregated.json` |
| Stacking AUC, F1 | `fusion/late_fusion_stacking/metrics_aggregated.json` |
| Soft Voting AUC, F1 + mean α | `fusion/late_fusion_soft_voting/metrics_aggregated.json` + `alpha_grid_search.json` |
| Max Voting AUC, F1 | `fusion/late_fusion_max_voting/metrics_aggregated.json` |
| Ablation table | `ablation/ablation_summary.json` |
| k sensitivity table | `k_sensitivity/summary.json` |
| Statistical tests | `statistical_tests/results.json` + `summary.csv` |

---

## 7. Sections KHÔNG cần thay đổi

- §1 Introduction (task motivation vẫn giữ nguyên)
- §2 Related Work (có thể thêm 1–2 câu về negative SLM result)
- §3 Dataset collection
- §3 Image branch architecture (CLIP + OCR + zero-shot)
- §3 Fusion strategies architecture (chỉ update soft voting α)
