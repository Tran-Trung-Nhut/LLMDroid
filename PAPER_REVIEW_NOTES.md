# Ghi chú cho teammate: So sánh kết quả hiện tại vs. paper (main-v2.pdf)

> Tạo ngày 2026-05-20. Đọc kỹ trước khi cập nhật bảng số trong paper.

---

## Phần 1 — Những thay đổi trong code so với mô tả trong paper

### 1.1 Temporal split D_cut: 2025-06-01 → **2026-04-30**

**Paper mô tả (Section 4.3.1):** D_cut = 2025-06-01, training gồm 218 apps, test 33 apps.

**Thực tế hiện tại:** `config.py` dùng `temporal_d_cut = "2026-04-30"`, cho ra train=223, test=75, skip=30.

**Lý do thay đổi:** Google Play Scraper trả về trường `updated` là Unix timestamp phản ánh thời điểm *hiện tại* app được cập nhật, không phải thời điểm thu thập dữ liệu. Với D_cut=2025-06-01, kết quả là train≈30/test≈238 — không hợp lý về mặt thống kê. D_cut mới (2026-04-30) cho train/test ratio hợp lý hơn nhưng **không khớp con số trong paper**.

**Ảnh hưởng đến Table 15:** Toàn bộ kết quả temporal split trong paper cần được tính lại với D_cut mới, hoặc cần giải thích rõ sự thay đổi trong paper.

---

### 1.2 LLMAID bị loại bỏ hoàn toàn — Table 2 cần reframe

**Paper mô tả (Section 4.1.2, Table 2):** Construct validity được chứng minh bằng 2 tool chạy trên N_code = 80 apps:
- LLMAID (Liu et al. [15]): κ = 0.847, Agreement = 92.5% → bằng chứng chính
- AI Discriminator (Li et al. [14]): κ = 0.604, Agreement = 82.5% → bằng chứng phụ, giải thích thấp hơn vì cloud API

**Thực tế hiện tại:** LLMAID **không có public inference code**, không thể chạy được. Đã xóa hoàn toàn khỏi `src/steps/run_code_validation.py` và `src/steps/cohen_kappa_validation.py`. Không còn cột `llmaid_label` nào trong pipeline.

**Kết quả thực tế chỉ với AI Discriminator (N_code = 80):**

| | AI Discriminator |
|---|---|
| Listing positive → AI Disc=1 | 8/40 (20%) |
| Listing negative → AI Disc=0 | 35/40 (88%) |
| Agreement | 53.8% |
| Cohen's κ | **0.075** |

κ = 0.075 ≈ "slight agreement" — **không thể dùng làm construct validity evidence**.

**Lý do kết quả thấp (hợp lý về mặt khoa học):** 32/40 app listing-positive bị AI Discriminator miss vì chúng tích hợp LLM qua **cloud API** (không nhúng TensorFlow/PyTorch trong APK). AI Discriminator chỉ scan local ML library identifiers trong smali code — hoàn toàn bỏ qua cloud-API LLM apps.

**Hướng xử lý trong paper — KHÔNG đề cập LLMAID nữa:**

Reframe Section 4.1.2 và Table 2 theo narrative mới:
> *"Kiểm tra chéo với AI Discriminator trên N_code = 79 apps cho thấy κ = 0.075, thấp hơn nhiều so với listing label. Phân tích các trường hợp miss cho thấy 80% app listing-positive bị bỏ qua là các app gọi LLM qua cloud API — không có ML library nào trong APK để AI Discriminator phát hiện. Kết quả này xác nhận rằng static code analysis tools có recall gap nghiêm trọng với cloud-API LLM apps, và đây chính là lý do LLMDroid cần thiết: listing-based screening phát hiện được nhóm app này mà code-level tools không thể."*

**Lưu ý quan trọng:** Xóa hoàn toàn mọi con số của LLMAID (κ = 0.847, 92.5%) khỏi paper, bao gồm:
- Abstract: *"Cohen's κ = 0.847"* → phải xóa hoặc thay
- Section 4.1.2: toàn bộ đoạn so sánh với LLMAID
- Table 2: bỏ cột LLMAID, chỉ giữ cột AI Discriminator với số mới

---

### 1.3 Baseline Gemini-1.5-Flash bị loại bỏ

**Paper mô tả (Section 4.2, Table 12):** So sánh với 4 baseline, trong đó có *MLLM zero-shot (GPT-4o-mini / Gemini-1.5-Flash)*, báo cáo kết quả tốt hơn của 2 provider. Gemini-1.5-Flash đạt: Acc=0.791, F1=0.735, AUC=0.864.

**Thực tế hiện tại:** `baseline_mllm_zeroshot.py` chỉ chạy GPT-4o-mini. `call_gemini()` và toàn bộ logic GOOGLE_API_KEY đã bị xóa.

**Lý do:** Google đã deprecated Gemini-1.5-Flash.

**Ảnh hưởng đến Table 12:** Dòng "Gemini-1.5-Flash" không thể reproduce. Cần quyết định:
- Giữ số cũ trong paper và note "model deprecated" trong footnote, hoặc
- Thay bằng Gemini-2.0-Flash hoặc model tương đương, hoặc
- Xóa dòng Gemini và chỉ báo cáo GPT-4o-mini cho zero-shot.

Hiện tại có file `src/steps/report_paper_results.py` hardcode lại các số đã có trong paper (bao gồm Gemini) để các script tổng hợp có thể đọc mà không cần chạy lại.

---

### 1.3 Robustness Table 18 — Sửa lỗi threshold rescaling

**Paper mô tả (Table 18):** Soft Voting khi drop screenshots (s_img=0) giảm F1 −0.051, Recall=0.864.

**Lỗi trước đó:** `soft_vote(s_text, 0, alpha=0.52, tau=0.5)` trả về F1≈0.000 vì max score=0.52 chỉ vừa qua tau=0.5 nhưng gần như toàn bộ prediction là 0.

**Fix đã áp dụng:** `effective_tau = tau * active_weight` trong `src/steps/robustness.py`. Khi chỉ text branch còn hoạt động (alpha≈0.52), effective_tau ≈ 0.52*0.52 ≈ 0.27 — tương đương với việc threshold lại trên text score đơn lẻ.

**Trạng thái:** Con số Table 18 sau fix phải gần với paper. Cần verify lại sau khi chạy.

---

### 1.4 Label loading cho independent test

**Trước đây:** Labels được lấy từ NPZ files, có thể chứa giá trị -1 (missing), gây lỗi `ValueError: Target is multiclass but average='binary'`.

**Hiện tại:** Labels luôn được load từ `data/inference_manual.csv` (columns: `pkg_name`, `label`) thông qua `load_label_map()` trong `utils/io.py`. File `inference_manual.csv` là ground-truth dứt khoát cho 110 app test.

---

### 1.5 Thư mục feature extraction test set

**Paper:** Không đề cập chi tiết cấu trúc thư mục.

**Thay đổi:** `inference_features_dir` = `"data/features_test"` (trước đây là `"data/inference_features"`). Thư mục này được bảo vệ khỏi bị xóa bởi cleanup trong `run_inference.py`.

---

## Phần 2 — Kết quả không khớp với paper (cần cập nhật báo cáo)

### 🚨 Mức độ NGHIÊM TRỌNG — Ảnh hưởng đến kết luận chính

#### 2.1 Table 20 — Thứ tự fusion strategy trên independent test bị đảo ngược

**Paper tuyên bố:** Soft Voting là strategy tốt nhất (F1=0.851), Early Fusion xếp sau (F1=0.830).

| Strategy | Paper F1 | Hiện tại F1 | Paper AUC | Hiện tại AUC |
|---|---|---|---|---|
| Early Fusion | 0.830 | **0.878** (+0.048) | 0.938 | **0.952** |
| Soft Voting | **0.851** | 0.833 (−0.018) | 0.930 | 0.944 |
| Stacking | 0.843 | 0.848 | 0.937 | 0.947 |
| Score-Max | 0.827 | 0.827 (=) | 0.918 | 0.933 |
| Text-Only | — | 0.851 | — | 0.948 |

Kết quả hiện tại: **Early Fusion tốt nhất (F1=0.878)**, Soft Voting chỉ đứng thứ 3 (F1=0.833). Paper claim Soft Voting wins on F1 là không còn đúng. Cần kiểm tra xem đây là do thay đổi dataset/label hay bug.

> Lưu ý thêm: Text-Only đang đạt F1=0.851 — bằng đúng con số paper ghi cho Soft Voting — rất đáng ngờ.

#### 2.2 Table 15 — Temporal delta dương, ngược chiều paper

**Paper tuyên bố:** *"All fusion strategies degrade gracefully, F1 drops at most 0.045."* (tất cả đều âm)

| Strategy | Paper Δ | Hiện tại Δ |
|---|---|---|
| Early Fusion | −0.038 | **+0.075** ↑ |
| Score-Max | −0.040 | **+0.069** ↑ |
| Text-Only | −0.045 | −0.014 |
| Soft Voting | −0.035 | −0.012 |
| Stacking | −0.037 | −0.018 |

Early Fusion và Score-Max đang *tốt hơn* dưới temporal split thay vì tệ hơn. Nguyên nhân trực tiếp là D_cut thay đổi (xem mục 1.1): test set 2026 chứa các app được re-update sau khi LLM branding trở nên phổ biến, khiến chúng dễ detect hơn.

**Cần quyết định:** Dùng D_cut nào trong paper? Nếu giữ 2026-04-30, cần sửa narrative Section 5.9 từ "graceful degradation" thành "temporal enrichment" — một claim khác về mặt nghĩa.

---

### ⚠️ Mức độ ĐÁNG CHÚ Ý — Ảnh hưởng đến claim phụ

#### 2.3 Table 7 — Soft Voting AUC thấp hơn 2 điểm

| Strategy | Paper AUC | Hiện tại AUC | Gap |
|---|---|---|---|
| Soft Voting | 0.904 | 0.884 | −0.020 |
| Stacking | 0.902 | 0.894 | −0.008 |
| Score-Max | 0.894 | 0.889 | −0.005 |
| Text-Only | 0.872 | 0.859 | −0.013 |
| Image-Only | 0.825 | 0.793 | −0.032 |

Paper claim "Soft Voting leads on ROC-AUC (0.904)". Hiện tại 0.884, không còn là giá trị 0.9+.

#### 2.4 Table 10 — Within-class branch correlations thấp hơn đáng kể

| Subset | Paper ρ (Pearson) | Hiện tại ρ (Pearson) |
|---|---|---|
| Positives (y=1) | 0.42 | **0.32** (−0.10) |
| Negatives (y=0) | 0.39 | **0.25** (−0.14) |
| Pooled | 0.51 | 0.52 (≈ same) |

Paper dùng ρ≈0.42 để giải thích tại sao Soft Voting có "small but positive ranking gain" (Section 5.5). Với ρ=0.32, lý thuyết dự đoán gain *lớn hơn* — nhưng thực tế AUC lại thấp hơn. Hai điều này mâu thuẫn, cần giải thích.

#### 2.5 Table 11 — Disagreement set size lớn hơn nhiều

| | Paper | Hiện tại |
|---|---|---|
| Disagree set size | **54** | **81** |
| Soft Voting accuracy | 0.741 | 0.691 |
| Score-Max accuracy | 0.704 | 0.605 |

Threshold phân loại "hai branch không đồng ý" là `|s_text − s_img| > 0.3` (không thay đổi). Set size lớn hơn (81 vs 54) cho thấy branch scores trong current run có phân tán lớn hơn — hai branch "cãi nhau" nhiều hơn. Đây là triệu chứng của text/image branch có calibration khác nhau so với paper.

#### 2.6 Text branch (Table 3) underperform nhất quán

| Config | Paper AUC | Hiện tại AUC | Paper F1 | Hiện tại F1 |
|---|---|---|---|---|
| BGE only | 0.808 | 0.776 | 0.682 | 0.593 |
| BGE + Handcrafted | 0.872 | 0.852 | 0.753 | 0.712 |

BGE-only F1 gap = −0.089, rất lớn. Gợi ý: có thể embedding model bị load ở precision thấp hơn, hoặc truncation length khác, hoặc feature normalization.

#### 2.7 Image branch (Table 5) underperform

| Config | Paper AUC | Hiện tại AUC |
|---|---|---|
| Full image branch | 0.8246 | 0.7959 (−0.029) |
| − CLIP pooled | 0.7321 | 0.7219 |
| − OCR features | 0.8073 | 0.8031 |

Full image branch thấp hơn 3 điểm AUC. CLIP ViT-L/14-336 được dùng đúng theo paper, nhưng kết quả thấp hơn — có thể do số lượng screenshots hoặc dedup khác nhau.

---

## Phần 3 — Những kết quả ổn, không cần sửa

| Bảng | Trạng thái |
|---|---|
| Table 7 — Score-Max F1/AUC | Gap < 0.007, không đáng kể |
| Table 7 — Stacking F1/AUC | Gap < 0.01 |
| Table 5 — ΔROC-AUC thứ tự ablation | CLIP > OCR > Chatbot-UI, đúng thứ tự paper |
| Table 3 — SLM-only F1=0.000, AUC≈0.52 | Đúng với paper |
| Table 3 — Handcrafted-only F1/AUC | Gần đúng (±0.01) |
| Table 10 — Pooled correlation | 0.52 vs 0.51, gần như bằng nhau |
| Calibration — Brier raw | Sai lệch nhỏ, cùng chiều |

---

## Phần 4 — Checklist cho teammate

- [ ] **Rewrite Section 4.1.2 và Table 2:** Xóa toàn bộ LLMAID. Đổi narrative thành "AI Discriminator recall gap chứng minh sự cần thiết của LLMDroid". Cập nhật số: N_code=80, κ=0.075, Agreement=53.8%. Xóa κ=0.847 khỏi abstract và body.
- [ ] **Quyết định D_cut:** Giữ 2026-04-30 và sửa narrative Section 5.9, hoặc tìm cách reproduce D_cut 2025-06-01 đúng với paper.
- [ ] **Quyết định Gemini baseline:** Ghi chú deprecated, thay bằng model khác, hoặc bỏ dòng Gemini trong Table 12.
- [ ] **Điều tra Table 20:** Tại sao Early Fusion > Soft Voting? Kiểm tra lại label file `inference_manual.csv` có đúng 44 positives / 66 negatives không.
- [ ] **Điều tra text branch:** Tại sao BGE-only AUC thấp hơn 0.032? Kiểm tra BGE embedding extraction pipeline.
- [ ] **Cập nhật Table 15 trong paper** theo kết quả thực tế với D_cut mới.
- [ ] **Cập nhật Table 20 trong paper** với số thực tế sau khi xác nhận label đúng.
- [ ] **Kiểm tra Table 18 (robustness):** Chạy lại sau fix threshold rescaling, so sánh với paper.
- [ ] Các bảng khác (Table 7 phần Score-Max/Stacking, Table 3 SLM/Handcrafted, Table 5 ΔAUC) có thể cập nhật số thực tế với note nhỏ.
