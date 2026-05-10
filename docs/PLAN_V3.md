# PLAN v3: Smart Vietnamese Summarization — Improved

> Bản kế hoạch đã cải thiện 4 vấn đề: dataset, controlled training, confidence naming, web app quality.

---

## 1. Tóm Tắt Thay Đổi So Với PLAN Cũ

| Vấn đề | PLAN cũ | PLAN v3 |
|---|---|---|
| Dataset | WikiLingua (dịch máy, domain how-to) | **VietNews chính** + 50-100 synthetic meeting/lecture samples |
| Controlled generation | Prefix + post-processing, không train riêng | Prefix + post-processing + **synthetic fine-tune thêm cho bullet/study_notes** |
| Confidence | Gọi là "confidence" | Đổi thành **"Quality Estimate"**, document rõ là heuristic |
| Web app | Streamlit basic | **FastAPI backend + Streamlit frontend**, custom CSS premium |

---

## 2. Bài Toán, Input, Output (Giữ Nguyên)

**Tên:** Vietnamese Controllable Abstractive Summarization

**Input:**
- Văn bản tiếng Việt dài (meeting notes, lecture, article, tin tức)
- Mode: `concise` | `bullet` | `action_items` | `study_notes`
- Length: `short` | `medium` | `long`

**Output:**
- Tóm tắt tiếng Việt theo mode và length
- Keywords, Quality Estimate (0-100), latency, token count

---

## 3. Dataset Strategy (CẢI THIỆN #1)

### 3.1 Dataset chính: VietNews

```
Nguồn:   harouzie/vietnews hoặc ithieund/VietNews-Abs-Sum
Domain:  Tin tức tiếng Việt (chính trị, kinh tế, xã hội, thể thao...)
Ưu điểm: Tiếng Việt tự nhiên, văn bản dài dạng tường thuật
         Gần với meeting notes/lecture hơn WikiLingua
Split:   80% train / 10% val / 10% test, seed=42
```

**Tại sao đổi từ WikiLingua?**
- WikiLingua = how-to dịch máy → tiếng Việt không tự nhiên → model học output kém
- VietNews = tin tức gốc tiếng Việt → model học tóm tắt tiếng Việt chuẩn
- Domain tin tức (tường thuật sự kiện) gần meeting notes hơn domain how-to (hướng dẫn bước)

### 3.2 Augmented data: Synthetic Meeting/Lecture Samples

```
Cách tạo:
  1. Dùng ChatGPT/Gemini generate 50-100 cặp:
     - 25 meeting notes → meeting summary
     - 25 lecture notes → lecture summary
     - 25 meeting notes → bullet points
     - 25 lecture notes → study notes

  2. Format:
     {"document": "...", "summary": "...", "mode": "bullet"}

  3. Review thủ công: đọc lại, sửa lỗi, đảm bảo chất lượng

  4. Thêm vào cuối training (fine-tune thêm 1 epoch trên synthetic data)
```

**Tại sao chỉ 50-100?**
- Không cần nhiều — mục đích là domain adaptation nhẹ, không phải train từ đầu
- Model đã học summarization tốt từ VietNews → chỉ cần "hint" thêm cho meeting/lecture domain
- 50-100 samples đủ để model quen với format bullet/study_notes
- Quá nhiều synthetic → risk overfitting trên data giả

### 3.3 Báo cáo phải note rõ

```
Section "Dataset" trong báo cáo cần viết:

"Training data chính là VietNews (XX,XXX samples), thuộc domain tin tức.
Use case mục tiêu là meeting notes và lecture notes, có domain gap.
Để giảm gap, chúng tôi augment thêm 100 synthetic samples
cho meeting/lecture domain. Đây là limitation cần lưu ý:
model hoạt động tốt nhất trên văn bản tường thuật,
có thể kém hơn trên văn bản technical hoặc informal."
```

---

## 4. Model & Training (CẢI THIỆN #2 — Controlled Training)

### 4.1 Phase 1: Fine-tune chính trên VietNews

```
Model:     VietAI/vit5-base
Dataset:   VietNews processed (JSONL)
Prefix:    "tom tat: {document}"
Target:    summary gốc từ dataset
Epochs:    3
Config:    lr=2e-5, batch=2, grad_accum=8, fp16=true
           max_source=512, max_target=128
Output:    Checkpoint v1 — biết tóm tắt tiếng Việt
```

### 4.2 Phase 2: Fine-tune thêm trên Synthetic Multi-Mode Data

```
Model:     Checkpoint v1 từ Phase 1
Dataset:   50-100 synthetic samples với prefix khác nhau:
           "tom tat ngan gon: {doc}"       → concise summary
           "tom tat thanh cac y chinh: {doc}" → bullet summary
           "trich xuat viec can lam: {doc}"  → action items
           "tao ghi chu hoc tap: {doc}"      → study notes
Epochs:    1-2 (nhẹ, chỉ adapt thêm)
Config:    lr=1e-5 (nhỏ hơn Phase 1 để không phá knowledge)
Output:    Checkpoint v2 — biết tóm tắt + hiểu prefix modes
```

**Tại sao 2 phases?**
- Phase 1 (VietNews lớn): model học kỹ năng summarization cốt lõi
- Phase 2 (synthetic nhỏ): model học hiểu prefix → output style khác nhau
- Nếu train cả 2 cùng lúc: synthetic chỉ 100 samples vs VietNews vạn samples → bị overwhelm

### 4.3 Trong báo cáo giải thích rõ

```
Section "Controllable Generation" cần viết:

"Hệ thống sử dụng instruction-style prompting trên fine-tuned seq2seq model.
Model được fine-tune 2 phases:
  Phase 1: Summarization cốt lõi trên VietNews (XX,XXX samples, 3 epochs)
  Phase 2: Multi-mode adaptation trên synthetic data (100 samples, 1 epoch)

Output modes (concise, bullet, action_items, study_notes) được điều khiển bởi:
  1. Prefix instruction: thay đổi prompt prefix cho mỗi mode
  2. Generation parameters: max_new_tokens khác nhau cho mỗi length
  3. Rule-based post-processing: format output theo mode (bullet → thêm '- ')

Đây KHÔNG phải multi-task supervised training với dataset riêng cho mỗi task.
Action items và study notes chất lượng phụ thuộc vào khả năng generalize
của model từ summarization sang extraction/restructuring."
```

---

## 5. Evaluation

### 5.1 Quantitative

| Metric | Dùng cho | Cách tính |
|---|---|---|
| ROUGE-1 | Word overlap | `evaluate` library |
| ROUGE-2 | Bigram overlap | `evaluate` library |
| ROUGE-L | Subsequence | `evaluate` library |
| BERTScore (optional) | Semantic similarity | `bert-score` library |
| Latency | UX | Measure inference time |

### 5.2 Qualitative

- 5 good examples + phân tích tại sao tốt
- 5 bad examples + phân loại lỗi:
  - Hallucination
  - Missing key points
  - Entity/date errors
  - Repetition
  - Truncation artifacts
- So sánh baseline (chưa fine-tune) vs fine-tuned
- So sánh 4 modes trên cùng 1 input

### 5.3 Baseline Comparison

```
Baseline 1: ViT5-base chưa fine-tune (zero-shot)
Baseline 2: VietAI/vit5-base-vietnews-summarization (pretrained summarizer)
Our model:  Fine-tuned ViT5 (Phase 1 + Phase 2)

Bảng so sánh:
| Model     | ROUGE-1 | ROUGE-2 | ROUGE-L | Controllable? |
|-----------|---------|---------|---------|---------------|
| Zero-shot |   ???   |   ???   |   ???   |      No       |
| Pretrained|   ???   |   ???   |   ???   |      No       |
| Ours      |   ???   |   ???   |   ???   |     Yes       |
```

---

## 6. Quality Estimate (CẢI THIỆN #3 — Đổi Tên + Document)

### Thay đổi

```
CŨ:  "Confidence" → gây hiểu lầm là calibrated probability
MỚI: "Quality Estimate" hoặc "Summary Quality Score"
```

### Implementation

```python
def compute_quality_estimate(
    input_text: str,
    output_text: str,
    generation_scores: list[float] | None,
) -> int:
    """
    Heuristic quality score from 0-100.
    NOT a calibrated probability.
    """
    scores = []

    # Factor 1: Generation probability (if available)
    if generation_scores:
        avg_log_prob = mean(generation_scores)
        scores.append(normalize(avg_log_prob, min=-5, max=0) * 30)

    # Factor 2: Length sanity (output not too short/long vs input)
    ratio = len(output_text) / max(len(input_text), 1)
    if 0.05 < ratio < 0.5:
        scores.append(25)
    else:
        scores.append(5)

    # Factor 3: No repetition
    bigrams = get_bigrams(output_text)
    unique_ratio = len(set(bigrams)) / max(len(bigrams), 1)
    scores.append(unique_ratio * 25)

    # Factor 4: Keyword coverage
    input_keywords = extract_keywords(input_text)
    covered = sum(1 for kw in input_keywords if kw in output_text)
    coverage = covered / max(len(input_keywords), 1)
    scores.append(coverage * 20)

    return clamp(sum(scores), 0, 100)
```

### Trong báo cáo

```
"Quality Estimate là một heuristic proxy, KHÔNG phải xác suất.
Nó kết hợp 4 yếu tố: generation probability, length sanity,
repetition check, và keyword coverage.
Mục đích: cho user tín hiệu rough về chất lượng output.
Giới hạn: không correlate tuyến tính với human judgment,
chưa được calibrate trên labeled quality dataset."
```

---

## 7. Web App Architecture (CẢI THIỆN #4 — FastAPI + Styled Streamlit)

### 7.1 Architecture

```
┌─────────────────────┐     HTTP/REST      ┌─────────────────────┐
│   STREAMLIT APP     │ ◄────────────────► │   FastAPI Backend   │
│   (Frontend)        │    /api/summarize   │                     │
│                     │    /api/health      │   - Load model 1 lần│
│   - Custom CSS      │                     │   - generate_summary│
│   - Premium UI      │                     │   - Return JSON     │
│   - Responsive      │                     │                     │
└─────────────────────┘                     └─────────────────────┘
```

### 7.2 FastAPI Backend

```python
# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Vietnamese Summarizer API")

class SummarizeRequest(BaseModel):
    text: str
    mode: str = "concise"
    length: str = "medium"

class SummarizeResponse(BaseModel):
    summary: str
    keywords: list[str]
    quality_estimate: int
    latency_ms: int
    input_tokens: int
    mode: str
    length: str

@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    result = generate_summary(
        text=request.text,
        mode=request.mode,
        length=request.length,
    )
    return SummarizeResponse(**result)

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": True}
```

**Tại sao thêm FastAPI?**
- Đề bài yêu cầu "Web Application" → có backend API = chuyên nghiệp hơn
- Tách model serving khỏi UI → scalable
- Có thể demo API bằng Swagger docs tự động
- Streamlit gọi API thay vì import trực tiếp → clean architecture

### 7.3 Streamlit Premium UI

```
Custom CSS:
  - Dark theme với gradient background
  - Ẩn Streamlit default footer/header
  - Custom font (Google Fonts: Inter hoặc Roboto)
  - Card-style layout cho kết quả
  - Progress bar cho Quality Estimate
  - Color-coded keywords (highlight)
  - Smooth transitions

Layout:
  ┌──────────────────────────────────────────────┐
  │  🧠 Smart Vietnamese Summarizer              │
  │  ─────────────────────────────────────────── │
  │                                              │
  │  ┌─── Input ──────┐  ┌─── Output ─────────┐ │
  │  │                 │  │                    │ │
  │  │  [Text Area]    │  │  Summary text      │ │
  │  │                 │  │  with highlighted   │ │
  │  │                 │  │  keywords           │ │
  │  │  Mode: [____▼]  │  │                    │ │
  │  │  Length: ○S ●M ○L│  │  ┌──────────────┐ │ │
  │  │                 │  │  │ Quality: 78   │ │ │
  │  │  [▶ Summarize]  │  │  │ ████████░░░   │ │ │
  │  │                 │  │  │ Latency: 2.1s │ │ │
  │  │  Samples:       │  │  │ Tokens: 342   │ │ │
  │  │  [Meeting][Lect]│  │  └──────────────┘ │ │
  │  └─────────────────┘  └────────────────────┘ │
  │                                              │
  │  ═══ Compare All Modes ══════════════════    │
  │  ┌Concise┐ ┌Bullet┐ ┌Actions┐ ┌Study┐       │
  │  │ ...   │ │ ...  │ │ ...   │ │ ... │       │
  │  └───────┘ └──────┘ └───────┘ └─────┘       │
  └──────────────────────────────────────────────┘
```

---

## 8. Cấu Trúc Codebase Cập Nhật

```
smart-vietnamese-summarizer/
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── configs/
│   ├── train_phase1.yaml        # Fine-tune trên VietNews
│   ├── train_phase2.yaml        # Fine-tune thêm synthetic
│   ├── eval.yaml
│   └── app.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── synthetic/               # NEW: synthetic meeting/lecture data
│   │   ├── meeting_samples.jsonl
│   │   ├── lecture_samples.jsonl
│   │   └── README.md            # Giải thích cách tạo
│   └── samples/
│       ├── meeting_note_vi.txt
│       ├── lecture_note_vi.txt
│       └── article_vi.txt
├── models/
│   └── .gitkeep
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── examples/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_training_colab.ipynb
│   ├── 03_error_analysis.ipynb
│   └── 04_synthetic_data_gen.ipynb  # NEW
├── scripts/
│   ├── prepare_data.py
│   ├── generate_synthetic.py        # NEW
│   ├── train.py
│   ├── train_phase2.py              # NEW
│   ├── evaluate.py
│   ├── predict.py
│   └── export_model.py
├── src/
│   └── smart_summarizer/
│       ├── __init__.py
│       ├── config.py
│       ├── constants.py
│       ├── data/
│       ├── modeling/
│       ├── evaluation/
│       ├── product/
│       │   ├── summarizer.py
│       │   ├── keyword_extractor.py
│       │   ├── quality_estimate.py  # RENAMED from confidence.py
│       │   └── postprocess.py
│       └── utils/
├── api/                             # NEW: FastAPI backend
│   ├── main.py
│   ├── schemas.py
│   └── dependencies.py
├── app/
│   ├── streamlit_app.py
│   ├── components.py
│   ├── style.py                     # Premium CSS
│   └── assets/                      # NEW: fonts, icons
├── tests/
│   ├── test_preprocessing.py
│   ├── test_generation.py
│   ├── test_postprocess.py
│   ├── test_quality_estimate.py     # RENAMED
│   └── test_api.py                  # NEW
└── docs/
    ├── SYSTEM_OVERVIEW.md
    ├── DEEP_DIVE_PART1_TRANSFORMER.md
    ├── DEEP_DIVE_PART2_SYSTEM.md
    └── PLAN_V3.md                   # This file
```

---

## 9. Timeline 4 Tuần (Chi Tiết Theo Ngày)

### Tuần 1: Data + Baseline (7 ngày)

| Ngày | Member A (NLP) | Member B (Product) |
|---|---|---|
| 1 | Setup env, pip install | Setup env, test streamlit chạy |
| 2 | Load VietNews, inspect 50 samples | Viết script generate synthetic data |
| 3 | Run prepare_data.py, tạo splits | Generate 50 synthetic meeting samples (ChatGPT) |
| 4 | Ghi dataset statistics | Generate 50 synthetic lecture/study samples |
| 5 | Run baseline inference (zero-shot) | Review + clean synthetic samples |
| 6 | So sánh baseline vs pretrained model | Save synthetic → data/synthetic/ |
| 7 | Document dataset notes | Bắt đầu FastAPI skeleton |

**Deliverables tuần 1:**
- [ ] data/processed/*.jsonl (VietNews splits)
- [ ] data/synthetic/*.jsonl (100 samples)
- [ ] reports/examples/baseline_outputs.md
- [ ] reports/examples/dataset_notes.md

### Tuần 2: Fine-tuning (7 ngày)

| Ngày | Member A (NLP) | Member B (Product) |
|---|---|---|
| 1 | Upload lên Colab, smoke test | Build FastAPI endpoints |
| 2 | Phase 1: Fine-tune trên VietNews | FastAPI: /api/summarize, /api/health |
| 3 | Phase 1: monitoring training loss | Streamlit: custom CSS, dark theme |
| 4 | Phase 1: complete, save checkpoint v1 | Streamlit: layout 2 cột |
| 5 | Phase 2: Fine-tune thêm synthetic | Streamlit: compare modes view |
| 6 | Phase 2: complete, save checkpoint v2 | Streamlit: sample picker |
| 7 | Download checkpoints, test predict.py | Connect Streamlit → FastAPI |

**Deliverables tuần 2:**
- [ ] models/vit5-summarizer-v1/ (Phase 1)
- [ ] models/vit5-summarizer-v2/ (Phase 2)
- [ ] reports/metrics/training_log.json
- [ ] API chạy được (localhost:8000)
- [ ] App chạy được (localhost:8501)

### Tuần 3: Evaluation + Polish (7 ngày)

| Ngày | Member A (NLP) | Member B (Product) |
|---|---|---|
| 1 | Run evaluate.py trên test set | Polish app UI |
| 2 | ROUGE scores: baseline vs v1 vs v2 | Thêm Quality Estimate bar |
| 3 | Select 5 good + 5 bad examples | Thêm keyword highlighting |
| 4 | Write error_analysis.md | Take screenshots for report |
| 5 | Optional: run BERTScore | Test tất cả modes + lengths |
| 6 | Verify ROUGE numbers | Fix edge cases (empty input, quá dài) |
| 7 | Finalize evaluation section | Finalize app |

**Deliverables tuần 3:**
- [ ] reports/metrics/eval_results.json
- [ ] reports/examples/test_predictions.jsonl
- [ ] reports/examples/error_analysis.md
- [ ] App hoàn chỉnh + screenshots

### Tuần 4: Report + Demo (7 ngày)

| Ngày | Member A (NLP) | Member B (Product) |
|---|---|---|
| 1 | Review báo cáo outline | Viết Introduction + Problem |
| 2 | Viết Dataset + Model sections | Viết Web App section |
| 3 | Viết Evaluation + Error Analysis | Viết Limitations + Future Work |
| 4 | Review toàn bộ báo cáo | Format, add figures/tables |
| 5 | Làm slides (8-12 slides) | Làm slides |
| 6 | Rehearse demo | Rehearse demo |
| 7 | Final check: code, weights, report | Submit |

**Deliverables tuần 4:**
- [ ] Final report PDF
- [ ] Slides (8-12)
- [ ] Source code cleaned
- [ ] Model weights on Drive
- [ ] README updated

---

## 10. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| VietNews load fail | Low | High | Fallback: WikiLingua + heavy filtering |
| Colab T4 OOM | Medium | High | batch=1, grad_accum=16, max_source=384 |
| Colab disconnect mid-training | Medium | Medium | Save checkpoint every epoch, resume |
| Synthetic data quality low | Medium | Medium | Manual review, fix 2-3 lần trước khi train |
| Phase 2 degrades Phase 1 quality | Low | High | Keep v1 checkpoint, compare v1 vs v2 ROUGE |
| FastAPI too slow | Low | Medium | Load model once at startup, cache |
| ROUGE scores too low | Medium | Medium | Document honestly, focus on qualitative demo |

---

## 11. Acceptance Criteria (Cập Nhật)

Project hoàn thành khi:
- [ ] Fine-tune ViT5 trên VietNews (Phase 1) + synthetic (Phase 2)
- [ ] ROUGE evaluation trên test set, có bảng so sánh baseline vs model
- [ ] Error analysis với >=5 good + >=5 bad examples
- [ ] FastAPI backend chạy được với /api/summarize
- [ ] Streamlit app có custom CSS, 4 modes, compare view
- [ ] Quality Estimate hiển thị đúng, document rõ là heuristic
- [ ] Báo cáo giải thích rõ: dataset choice, domain gap, controllable generation approach, limitations
- [ ] Demo 5-7 phút smooth

---

## 12. Final Submission Checklist

- [ ] Source code (GitHub hoặc zip)
- [ ] Fine-tuned model weights (Google Drive link)
- [ ] Dataset files hoặc link
- [ ] Synthetic data + script tạo
- [ ] Final report PDF
- [ ] README với running instructions
- [ ] Evaluation results (JSON + examples)
- [ ] Web app screenshots
- [ ] API docs (Swagger auto-generated)
- [ ] Presentation slides
