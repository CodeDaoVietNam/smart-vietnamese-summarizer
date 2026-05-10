# Tuần 1: Data + Baseline — Hướng Dẫn Chi Tiết

---

## Ngày 1: Setup Environment (Cả 2 members)

### Member A: Setup Python + dependencies

**File liên quan:** `pyproject.toml`, `.env.example`

```bash
# Bước 1: Tạo virtual environment
cd /home/ductien/Documents/Transformer
python3 -m venv .venv
source .venv/bin/activate

# Bước 2: Install project
pip install -e ".[dev]"

# Bước 3: Verify
python -m compileall src scripts app tests
pytest --co  # list tests, chưa chạy thật
```

**Tại sao venv?** Isolate dependencies — tránh conflict với system Python.
**Tại sao `pip install -e`?** Editable mode — sửa code trong `src/` tự động reflect, không cần reinstall.

### Member B: Test Streamlit + FastAPI chạy

```bash
# Test Streamlit
streamlit run app/streamlit_app.py
# → Mở browser localhost:8501, xem có lỗi syntax không

# Test FastAPI (nếu đã có skeleton)
uvicorn api.main:app --reload --port 8000
# → Mở localhost:8000/docs xem Swagger UI
```

**Deliverable ngày 1:** Cả 2 members chạy được `pytest` và `streamlit` không lỗi.

---

## Ngày 2: Load Dataset + Inspect

### Member A: Load VietNews, inspect 50 samples

**File:** `notebooks/01_dataset_exploration.ipynb`

```python
# Cell 1: Load dataset
from datasets import load_dataset

ds = load_dataset("ithieund/VietNews-Abs-Sum")
print(ds)  # Xem structure: train/test? bao nhiêu samples?

# Cell 2: Xem 1 sample
sample = ds["train"][0]
print("Document:", sample["document"][:500])
print("Summary:", sample["summary"])
print("Doc length:", len(sample["document"]))
print("Sum length:", len(sample["summary"]))

# Cell 3: Inspect 50 random samples
import random
random.seed(42)
indices = random.sample(range(len(ds["train"])), 50)
for i in indices[:10]:  # in 10 cái đầu
    s = ds["train"][i]
    print(f"--- Sample {i} ---")
    print(f"Doc ({len(s['document'])} chars): {s['document'][:200]}...")
    print(f"Sum ({len(s['summary'])} chars): {s['summary'][:200]}")
    print()
```

**Câu hỏi cần trả lời sau khi inspect:**
1. Dataset có bao nhiêu samples? Có sẵn train/val/test split chưa?
2. Document trung bình dài bao nhiêu ký tự? Summary?
3. Tiếng Việt có tự nhiên không? Có sample nào bị lỗi encoding?
4. Có sample rỗng hoặc quá ngắn không?
5. → Quyết định: dùng VietNews hay cần fallback WikiLingua?

**Tại sao inspect trước khi train?** Garbage in = garbage out. Nếu data kém mà train luôn → mất 4 giờ GPU + model kém → phí thời gian.

### Member B: Viết script generate synthetic data

**File:** `scripts/generate_synthetic.py`

Script này KHÔNG chạy ngay — chỉ chuẩn bị structure:

```python
"""
Script hỗ trợ tạo synthetic meeting/lecture samples.
Workflow:
  1. Chuẩn bị prompt template cho ChatGPT/Gemini
  2. Copy output vào file JSON
  3. Script này validate + format thành JSONL
"""
import json
import argparse
from pathlib import Path

def validate_sample(sample: dict) -> bool:
    """Kiểm tra sample có đủ fields và không rỗng."""
    required = ["document", "summary", "mode"]
    for field in required:
        if field not in sample or not sample[field].strip():
            return False
    if sample["mode"] not in ["concise", "bullet", "action_items", "study_notes"]:
        return False
    if len(sample["document"]) < 100:
        return False
    if len(sample["summary"]) < 20:
        return False
    return True

def process_raw_synthetic(input_path: str, output_dir: str):
    """Đọc raw JSON, validate, split 80/20, save JSONL."""
    with open(input_path) as f:
        samples = json.load(f)

    valid = [s for s in samples if validate_sample(s)]
    print(f"Valid: {len(valid)}/{len(samples)}")

    # Split 80% train, 20% validation
    split_idx = int(len(valid) * 0.8)
    train = valid[:split_idx]
    val = valid[split_idx:]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("validation", val)]:
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} samples to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw JSON file")
    parser.add_argument("--output-dir", default="data/synthetic")
    args = parser.parse_args()
    process_raw_synthetic(args.input, args.output_dir)
```

---

## Ngày 3: Preprocessing + Tạo Splits

### Member A: Run prepare_data.py

**File:** `scripts/prepare_data.py`

```bash
python scripts/prepare_data.py --config configs/train_phase1.yaml
```

Script cần làm những gì (logic):

```
INPUT:  Raw dataset từ HuggingFace (ithieund/VietNews-Abs-Sum)
PROCESS:
  1. Load dataset
  2. Lọc empty: document rỗng hoặc summary rỗng → bỏ
  3. Lọc quá ngắn: document < 50 chars → bỏ (không đủ nội dung để tóm tắt)
  4. Lọc quá dài: document > 10000 chars → bỏ (outlier, gây noise)
  5. Normalize whitespace: multiple spaces → single space
  6. Normalize Unicode: NFC (chuẩn hóa dấu tiếng Việt)
  7. Nếu dataset chỉ có train split → tự chia 80/10/10 seed=42
  8. Save mỗi split thành JSONL
OUTPUT:
  data/processed/train.jsonl
  data/processed/validation.jsonl
  data/processed/test.jsonl
```

**Verify sau khi chạy:**
```bash
wc -l data/processed/*.jsonl
# Xem số dòng mỗi file = số samples

head -1 data/processed/train.jsonl | python -m json.tool
# Xem 1 sample có đúng format không
```

### Member B: Generate 50 synthetic meeting samples

**Cách làm:** Dùng ChatGPT/Gemini với prompt sau:

```
Bạn là trợ lý tạo dữ liệu training cho AI tóm tắt văn bản.
Hãy tạo 10 cặp (meeting notes, summary) bằng tiếng Việt.

Mỗi cặp gồm:
- "document": Nội dung cuộc họp dài 200-500 từ, gồm: chủ đề, người tham gia, 
  nội dung thảo luận, quyết định, deadline.
- "summary": Bản tóm tắt ngắn gọn 50-100 từ.
- "mode": "concise"

Format: JSON array.
Chủ đề đa dạng: doanh thu, tuyển dụng, marketing, dự án, sản phẩm...

Ví dụ:
{
  "document": "Cuộc họp phòng Marketing ngày 5/5/2026. Tham dự: Anh Minh (trưởng phòng), 
  chị Lan, anh Tuấn. Nội dung: Thảo luận kế hoạch Q3...",
  "summary": "Phòng Marketing họp bàn kế hoạch Q3, quyết định tăng ngân sách quảng cáo 
  Facebook 20% và ra mắt campaign mới vào tháng 7.",
  "mode": "concise"
}
```

Chạy prompt 5 lần (mỗi lần 10 samples) → 50 meeting samples.
Lưu vào `data/synthetic/raw_meeting.json`.

**Tại sao dùng AI generate?** Không có meeting notes dataset tiếng Việt công khai. Tự viết 50 cái = rất lâu. AI generate + human review = nhanh + chất lượng đủ tốt.

---

## Ngày 4: Dataset Statistics + Synthetic Lecture Samples

### Member A: Ghi dataset statistics

**File:** `notebooks/01_dataset_exploration.ipynb` (tiếp tục)

```python
import json
import statistics

def compute_stats(jsonl_path):
    docs, sums = [], []
    with open(jsonl_path) as f:
        for line in f:
            s = json.loads(line)
            docs.append(len(s["document"]))
            sums.append(len(s["summary"]))
    return {
        "count": len(docs),
        "doc_avg": round(statistics.mean(docs)),
        "doc_min": min(docs),
        "doc_max": max(docs),
        "doc_median": round(statistics.median(docs)),
        "sum_avg": round(statistics.mean(sums)),
        "sum_min": min(sums),
        "sum_max": max(sums),
        "sum_median": round(statistics.median(sums)),
    }

for split in ["train", "validation", "test"]:
    stats = compute_stats(f"data/processed/{split}.jsonl")
    print(f"\n=== {split.upper()} ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
```

**Output mẫu cần ghi lại cho báo cáo:**

| Split | Samples | Doc Avg (chars) | Doc Median | Sum Avg | Sum Median |
|---|---|---|---|---|---|
| Train | ~XX,XXX | ~??? | ~??? | ~??? | ~??? |
| Validation | ~X,XXX | | | | |
| Test | ~X,XXX | | | | |

### Member B: Generate 50 synthetic lecture/study samples

Prompt cho **bullet mode** (25 samples):
```
Tạo 10 cặp (lecture notes, bullet summary) tiếng Việt.
- "document": Nội dung bài giảng 200-500 từ (Toán, Lý, Hóa, CNTT, Kinh tế...)
- "summary": 5-8 bullet points, mỗi ý 1 dòng bắt đầu bằng "- "
- "mode": "bullet"
```

Prompt cho **study_notes mode** (25 samples):
```
Tạo 10 cặp (lecture content, study notes) tiếng Việt.
- "document": Nội dung bài học 200-500 từ
- "summary": Ghi chú học tập có cấu trúc: khái niệm chính, ý quan trọng, ví dụ
- "mode": "study_notes"
```

Lưu vào `data/synthetic/raw_lecture.json`.

---

## Ngày 5: Baseline Inference

### Member A: Chạy model chưa fine-tune

**File:** `scripts/predict.py` hoặc notebook mới

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Baseline 1: ViT5 gốc (chưa train summarization)
model_name = "VietAI/vit5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load 3 sample inputs
samples = {
    "meeting": open("data/samples/meeting_note_vi.txt").read(),
    "lecture": open("data/samples/lecture_note_vi.txt").read(),
    "article": open("data/samples/article_vi.txt").read(),
}

for name, text in samples.items():
    input_text = f"tom tat: {text}"
    inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128, num_beams=4)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n=== {name} (zero-shot) ===")
    print(f"Input ({len(text)} chars): {text[:200]}...")
    print(f"Output: {result}")
```

Chạy tương tự cho **Baseline 2**: `VietAI/vit5-base-vietnews-summarization`

**Ghi nhận cho mỗi output:**
- [ ] Có đúng tiếng Việt không?
- [ ] Có tóm tắt được ý chính không?
- [ ] Có hallucination không?
- [ ] Có lặp (repetition) không?
- [ ] Có quá generic không?

### Member B: Review + clean synthetic samples

Đọc lại 100 samples synthetic:
```bash
# Đếm samples
cat data/synthetic/raw_meeting.json | python -c "import json,sys; print(len(json.load(sys.stdin)))"
cat data/synthetic/raw_lecture.json | python -c "import json,sys; print(len(json.load(sys.stdin)))"
```

**Checklist review cho mỗi sample:**
- [ ] Tiếng Việt tự nhiên, không Vietlish?
- [ ] Document đủ dài (>100 từ)?
- [ ] Summary thực sự tóm tắt document, không thêm info?
- [ ] Mode đúng (bullet có "- ", study_notes có cấu trúc)?
- [ ] Không có nội dung nhạy cảm/inappropriate?

Sửa trực tiếp trong file JSON nếu cần.

---

## Ngày 6: So Sánh Baseline + Save Synthetic

### Member A: Tạo bảng so sánh baseline

**File:** `reports/examples/baseline_outputs.md`

```markdown
# Baseline Comparison

## Sample 1: Meeting Note

### Input (trích 200 ký tự đầu)
"Cuộc họp phòng kinh doanh ngày 3/5..."

### Zero-shot ViT5-base
"[output ở đây]"
**Nhận xét:** [tiếng Việt kém / lặp / không tóm tắt được]

### Pretrained ViT5-vietnews-summarization
"[output ở đây]"
**Nhận xét:** [tóm tắt tốt hơn / nhưng không controllable]

### Kết luận
Fine-tuning cần thiết vì: [lý do cụ thể từ observation]

## Sample 2: Lecture Note
[tương tự]

## Sample 3: Article
[tương tự]
```

### Member B: Format synthetic data thành JSONL

```bash
# Merge meeting + lecture thành 1 file
python -c "
import json
meeting = json.load(open('data/synthetic/raw_meeting.json'))
lecture = json.load(open('data/synthetic/raw_lecture.json'))
all_data = meeting + lecture
json.dump(all_data, open('data/synthetic/all_synthetic.json','w'), ensure_ascii=False, indent=2)
print(f'Total: {len(all_data)} samples')
"

# Chạy script validate + split
python scripts/generate_synthetic.py \
  --input data/synthetic/all_synthetic.json \
  --output-dir data/synthetic
```

**Verify:**
```bash
wc -l data/synthetic/train.jsonl data/synthetic/validation.jsonl
# Expected: ~80 train, ~20 validation
```

---

## Ngày 7: Document + FastAPI Skeleton

### Member A: Viết dataset notes

**File:** `reports/examples/dataset_notes.md`

```markdown
# Dataset Decision Notes

## Primary Dataset: VietNews-Abs-Sum
- Source: ithieund/VietNews-Abs-Sum (HuggingFace)
- Domain: Tin tức tiếng Việt
- Samples: [số cụ thể]
- Quality: [đánh giá sau khi inspect]
- Split: [cách chia]

## Synthetic Augmentation Data
- 50 meeting note samples (generated by ChatGPT, reviewed manually)
- 50 lecture/study samples (25 bullet + 25 study_notes)
- Purpose: Domain adaptation cho meeting/lecture use case
- Quality: Reviewed, [X] samples fixed, [Y] samples removed

## Domain Gap Analysis
- VietNews = tin tức tường thuật → tốt cho concise summarization
- Meeting notes = ngôn ngữ informal, có tên người, deadline
- Gap: Model có thể miss format meeting-specific (action items, người phụ trách)
- Mitigation: Synthetic data Phase 2 + post-processing

## Decision Log
- [Ngày]: Chọn VietNews thay WikiLingua vì [lý do cụ thể]
- [Ngày]: Generate 100 synthetic samples cho multi-mode training
```

### Member B: FastAPI skeleton

**File:** `api/main.py`

```python
from fastapi import FastAPI

app = FastAPI(
    title="Vietnamese Summarizer API",
    description="API cho Smart Meeting & Study Notes Summarization",
    version="0.1.0",
)

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": False}

@app.post("/api/summarize")
async def summarize():
    # Placeholder — sẽ implement sau khi có model
    return {"error": "Model not loaded yet. Complete training first."}
```

```bash
# Test
uvicorn api.main:app --reload --port 8000
# Mở localhost:8000/docs → thấy Swagger UI
```

---

## Checklist Cuối Tuần 1

| # | Deliverable | File | Owner | Done? |
|---|---|---|---|---|
| 1 | VietNews train split | `data/processed/train.jsonl` | A | [ ] |
| 2 | VietNews val split | `data/processed/validation.jsonl` | A | [ ] |
| 3 | VietNews test split | `data/processed/test.jsonl` | A | [ ] |
| 4 | Synthetic train split | `data/synthetic/train.jsonl` | B | [ ] |
| 5 | Synthetic val split | `data/synthetic/validation.jsonl` | B | [ ] |
| 6 | Dataset statistics | `notebooks/01_dataset_exploration.ipynb` | A | [ ] |
| 7 | Baseline outputs | `reports/examples/baseline_outputs.md` | A | [ ] |
| 8 | Dataset notes | `reports/examples/dataset_notes.md` | A | [ ] |
| 9 | Synthetic script | `scripts/generate_synthetic.py` | B | [ ] |
| 10 | FastAPI skeleton | `api/main.py` | B | [ ] |
| 11 | Environment works | cả 2 members test OK | Both | [ ] |
