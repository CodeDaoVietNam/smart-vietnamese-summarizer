# Deep Dive Part 2: Data Pipeline, Product Layer & System Design

> Phần này mổ xẻ: dataset, preprocessing, controllable generation, web app, và cách mọi thứ kết nối.

---

## 1. Dataset Deep Dive

### 1.1 WikiLingua Vietnamese

**Là gì:** Dataset gồm các bài hướng dẫn (how-to) từ WikiHow, được dịch sang nhiều ngôn ngữ.

```
Một sample WikiLingua:
{
  "document": "Bước 1: Mở ứng dụng Settings trên điện thoại. 
               Bước 2: Chọn mục Wi-Fi. 
               Bước 3: Nhấn vào tên mạng muốn kết nối...",
  "summary":  "Để kết nối Wi-Fi, mở Settings, chọn Wi-Fi, 
               nhấn tên mạng và nhập mật khẩu."
}
```

**Ưu điểm:**
- Có sẵn trên HuggingFace, dễ load.
- Có cặp document-summary sẵn → supervised learning.
- Văn bản có cấu trúc bước → gần với meeting notes/lecture notes.

**Nhược điểm (quan trọng!):**
- Nhiều bài bị dịch máy → tiếng Việt không tự nhiên.
- Domain là how-to → khác meeting notes thực tế → **domain shift**.
- Một số summary quá ngắn hoặc không capture đủ ý.

**Quyết định:** Thử WikiLingua trước, nếu chất lượng kém → chuyển sang VietNews.

### 1.2 VietNews (Fallback)

```
Một sample VietNews:
{
  "document": "Chiều 5/5, tại Hà Nội, Thủ tướng Phạm Minh Chính 
               đã chủ trì cuộc họp về tình hình kinh tế...",
  "summary":  "Thủ tướng chủ trì họp bàn giải pháp thúc đẩy 
               tăng trưởng kinh tế quý II."
}
```

**Ưu điểm:** Tiếng Việt tự nhiên, domain tin tức gần với meeting notes hơn.
**Nhược điểm:** Không phải how-to/study format.

### 1.3 Data Preprocessing Pipeline

```
INPUT:  Raw dataset từ HuggingFace

PROCESS:
  1. Lọc samples rỗng (document hoặc summary trống)
  2. Lọc quá ngắn (document < 50 ký tự → không đủ nội dung)
  3. Lọc quá dài (document > 10000 ký tự → outlier)
  4. Normalize whitespace (nhiều space/tab → 1 space)
  5. Normalize Unicode (NFD → NFC cho tiếng Việt)
  6. Split: 80% train, 10% validation, 10% test (seed=42)

OUTPUT: 3 file JSONL (train.jsonl, validation.jsonl, test.jsonl)
```

**Tại sao JSONL?** Mỗi dòng là 1 JSON object → dễ stream, dễ đọc từng sample mà không load cả file vào RAM.

**Tại sao seed=42?** Để reproducible — ai chạy cũng ra cùng split → kết quả so sánh được.

### 1.4 Tokenization Cho Training

```
INPUT:  {"document": "Cuộc họp hôm nay...", "summary": "Tóm lại..."}

PROCESS:
  1. Thêm prefix: "tom tat ngan: Cuộc họp hôm nay..."
  2. Tokenize input: 
     tokenizer("tom tat ngan: Cuộc họp hôm nay...", 
               max_length=512, truncation=True, padding="max_length")
     → input_ids: [100, 234, 567, ..., 0, 0, 0]  (padded to 512)
     → attention_mask: [1, 1, 1, ..., 0, 0, 0]    (1=real, 0=padding)
  
  3. Tokenize target:
     tokenizer("Tóm lại...", max_length=128, truncation=True)
     → labels: [789, 321, 654, ..., -100, -100]   (-100 = ignore in loss)

OUTPUT: {"input_ids": [...], "attention_mask": [...], "labels": [...]}
```

**attention_mask:** Nói cho model biết token nào là thật (1) vs padding (0). Model không chú ý đến padding tokens.

**labels = -100:** PyTorch CrossEntropyLoss tự động bỏ qua vị trí có label = -100. Dùng cho padding tokens trong target → không phạt model vì đoán sai padding.

---

## 2. Controllable Generation — Cách Điều Khiển Output

### 2.1 Prefix-Based Control

```
Cùng 1 document, thay đổi prefix → output style khác nhau:

Mode "concise":
  Input: "tom tat ngan gon: {document}"
  → Model sinh đoạn tóm tắt 3-5 câu

Mode "bullet":
  Input: "tom tat thanh cac y chinh: {document}"
  → Model sinh text, post-process thêm "- " đầu mỗi ý

Mode "action_items":
  Input: "trich xuat cac viec can lam: {document}"
  → Model sinh text, post-process format thành checklist

Mode "study_notes":
  Input: "tao ghi chu hoc tap: {document}"
  → Model sinh text, post-process chia thành các mục
```

### 2.2 Tại Sao Prefix Hoạt Động?

```
T5 được pre-train với prefix instructions:
  "summarize: ..."  → model học: prefix này = phải tóm tắt
  "translate: ..."  → model học: prefix này = phải dịch

Fine-tuning với prefix tiếng Việt:
  "tom tat ngan: ..." → model học: prefix này = tóm tắt ngắn gọn

Inference với prefix khác:
  "trich xuat viec can lam: ..." → model CỐ GẮNG hiểu instruction
  Có thể không perfect vì không được train riêng cho prefix này
  → Bổ sung bằng post-processing
```

### 2.3 Post-Processing Layer

```
INPUT:  Raw model output: "Cần hoàn thành báo cáo trước thứ 6. 
                           Gửi email cho khách hàng. 
                           Chuẩn bị slide thuyết trình."

PROCESS theo mode:
  bullet → split by sentence → thêm "- " mỗi dòng
  action_items → split → thêm "☐ " + detect deadline/person
  study_notes → split → nhóm theo topic → thêm heading

OUTPUT (action_items):
  "☐ Hoàn thành báo cáo — Deadline: thứ 6
   ☐ Gửi email cho khách hàng
   ☐ Chuẩn bị slide thuyết trình"
```

**Honest note cho báo cáo:**
Post-processing là **rule-based formatting**, không phải model learning. Model sinh raw text, code format lại. Cần nói rõ trong báo cáo để tránh bị hỏi "model có thật sự học output bullet format không?"

### 2.4 Length Control

```
length="short":  max_new_tokens=64   (~40 từ)
length="medium": max_new_tokens=128  (~80 từ)
length="long":   max_new_tokens=256  (~160 từ)
```

Không phải model "hiểu" ngắn/dài — ta chỉ giới hạn số token sinh ra. Đây là **engineering control**, không phải **learned behavior**.

---

## 3. Product Layer — Từ Model Đến App

### 3.1 Keyword Extraction

```
INPUT:  Document text + Summary text
PROCESS:
  1. Tách từ (word segmentation)
  2. Bỏ stopwords tiếng Việt ("và", "của", "là", "có", "trong"...)
  3. Đếm tần suất (term frequency)
  4. Lấy top 5-10 từ xuất hiện nhiều nhất
OUTPUT: ["cuộc họp", "doanh thu", "quý 3", "tăng trưởng"]
```

**Tại sao dùng heuristic thay vì NER model?**
- NER model (Named Entity Recognition) nặng, tốn thêm RAM + latency.
- Frequency-based heuristic đủ tốt cho highlighting purpose.
- Đây là feature phụ, không phải core deliverable.

### 3.2 Confidence Estimation

```
INPUT:  Model generation output
PROCESS:
  Score 1: Generation probability (nếu model output scores available)
           → average log-prob của tất cả tokens sinh ra
  Score 2: Length sanity
           → output quá ngắn (<10 tokens) hoặc quá dài → penalty
  Score 3: Repetition check
           → nhiều n-gram lặp → confidence thấp
  Score 4: Keyword coverage
           → bao nhiêu keywords input xuất hiện trong output

  Final = weighted average → clamp [0, 100]

OUTPUT: 73 (ví dụ — "chất lượng khá tốt")
```

**QUAN TRỌNG:** Đây là **heuristic proxy**, KHÔNG phải xác suất đúng. Phải note rõ trong báo cáo và UI.

### 3.3 Streamlit App Architecture

```
streamlit_app.py
     │
     ├── Sidebar
     │    ├── Mode selector (selectbox)
     │    ├── Length selector (radio buttons)
     │    └── Sample picker (3 samples có sẵn)
     │
     ├── Main Area
     │    ├── Left column: Text area (input)
     │    ├── Summarize button
     │    └── Right column: Generated output
     │         ├── Summary text (highlighted keywords)
     │         ├── Keywords (tags)
     │         ├── Confidence bar
     │         ├── Latency (ms)
     │         └── Token count
     │
     └── Compare Tab (chạy 4 modes cùng lúc)
```

---

## 4. System Design — Tại Sao Cấu Trúc Code Như Vậy

### 4.1 Tách scripts/ vs src/

```
scripts/  = Entry points (thin wrappers)
  train.py: parse args → gọi src functions → done

src/smart_summarizer/ = Business logic (reusable)
  modeling/trainer.py: chứa training logic thật

Tại sao tách?
→ scripts/ dùng cho CLI: `python scripts/train.py --config ...`
→ src/ dùng cho import: `from smart_summarizer.modeling import ...`
→ App import từ src/, không import từ scripts/
→ Test import từ src/, không chạy scripts/
→ Separation of concerns: entry point ≠ logic
```

### 4.2 Config-Driven Design

```
Hardcoded (tệ):
  model_name = "VietAI/vit5-base"    # muốn đổi → sửa code → dễ bug

Config-driven (tốt):
  config = load_yaml("configs/train.yaml")
  model_name = config["model"]["name"]  # muốn đổi → sửa YAML → an toàn
```

**Lợi ích:**
- Thay đổi hyperparameters mà không sửa code.
- Reproducibility: chia sẻ YAML = chia sẻ exact setup.
- Multiple configs cho multiple experiments.

### 4.3 Tại Sao Cần Testing

```
Không test:
  Sửa postprocess.py → break bullet format → chỉ phát hiện khi demo
  → Đã cuối tuần 4, không kịp fix → mất điểm

Có test:
  Sửa postprocess.py → chạy pytest → test_postprocess.py fail ngay
  → Fix trong 5 phút → safe
```

---

## 5. Workflow Thực Tế — Bạn Sẽ Làm Gì Theo Thứ Tự

```
Tuần 1:
  1. pip install -e ".[dev]" → cài dependencies
  2. python scripts/prepare_data.py → tải + clean data
  3. Mở notebook 01 → xem 20 samples → đánh giá chất lượng
  4. Chạy baseline predict (chưa fine-tune) → lưu ví dụ

Tuần 2:
  5. Upload lên Colab → enable GPU T4
  6. python scripts/train.py → fine-tune 3 epochs (~2-4 giờ)
  7. Download checkpoint về

Tuần 3:
  8. python scripts/evaluate.py → ROUGE scores
  9. Đọc 5 good + 5 bad examples → viết error analysis
  10. streamlit run app/streamlit_app.py → test app

Tuần 4:
  11. Polish app UI
  12. Viết báo cáo (dùng data từ reports/)
  13. Làm slides + rehearse demo
```

---

## 6. Câu Hỏi Viva Thường Gặp & Cách Trả Lời

### "Tại sao chọn Transformer thay vì LSTM?"
→ Self-attention capture long-range dependencies tốt hơn, xử lý song song trên GPU, là state-of-the-art cho text generation.

### "Controllable generation của bạn có thật sự controllable không?"
→ Trung thực: Model được fine-tune cho summarization với prefix "tom tat ngan:". Các mode khác (bullet, action_items) dùng prefix khác + rule-based post-processing. Không phải multi-task supervised training. Đây là engineering approach hợp lý cho constraint 4 tuần.

### "Confidence score có đáng tin không?"
→ Đây là heuristic quality estimate, không phải calibrated probability. Kết hợp generation score + length sanity + repetition check. Mục đích là cho user 1 tín hiệu rough, không claim chính xác tuyệt đối.

### "Dataset có representative cho use case không?"
→ WikiLingua là how-to articles, khác meeting notes. Có domain gap. Đây là limitation. Future work: thu thập meeting notes tiếng Việt thật để fine-tune thêm.

### "Tại sao không dùng GPT-4/Gemini API?"
→ Đề bài yêu cầu fine-tune Transformer, không phải gọi API. Project cần demonstrate hiểu biết về training pipeline, không chỉ inference.
