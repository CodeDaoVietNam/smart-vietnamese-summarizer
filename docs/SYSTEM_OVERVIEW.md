# System Overview: Vietnamese Controllable Summarization

> Tài liệu này giải thích toàn bộ hệ thống từ góc nhìn **Input → Process → Output**.
> Mọi kết luận đều có lý do. Không cảm tính.

---

## 1. Bài Toán Chúng Ta Đang Giải

### Nhìn như một hệ thống

```
INPUT:  Văn bản tiếng Việt dài (meeting notes, lecture, article)
        + Mode (concise | bullet | action_items | study_notes)
        + Length (short | medium | long)

PROCESS: Transformer model (ViT5) đã fine-tune
         → hiểu ngữ nghĩa → chọn ý quan trọng → sinh văn bản mới

OUTPUT: Bản tóm tắt tiếng Việt theo đúng mode và độ dài yêu cầu
        + Keywords, confidence score, latency
```

### Tại sao bài toán này tồn tại?

- Con người mất 10-30 phút đọc 1 tài liệu dài → cần AI rút gọn còn 30 giây.
- Mỗi người cần output khác nhau: sinh viên cần study notes, quản lý cần action items.
- → **Controllable summarization** giải quyết cả hai: tóm tắt + tùy chỉnh output.

---

## 2. Thuật Ngữ Cốt Lõi

### 2.1 NLP (Natural Language Processing)

**Là gì:** Lĩnh vực AI dạy máy tính hiểu và sinh ngôn ngữ tự nhiên.

**Trong project:** Toàn bộ project là một NLP task — nhận text tiếng Việt, xử lý, sinh text mới.

### 2.2 Text Summarization

**Là gì:** Bài toán rút gọn văn bản dài thành ngắn hơn, giữ ý chính.

Có 2 loại:

| Loại | Cách hoạt động | Ví dụ |
|---|---|---|
| **Extractive** | Chọn câu quan trọng từ bài gốc, ghép lại | Lấy câu 1, 5, 12 từ bài |
| **Abstractive** | Hiểu nội dung, viết lại bằng câu mới | Model tự viết câu tóm tắt |

**Project dùng Abstractive** vì:
- Extractive chỉ copy-paste → không linh hoạt cho bullet/action_items/study_notes.
- Abstractive cho phép model **viết lại** theo format mong muốn.
- Transformer architecture phù hợp cho abstractive generation.

### 2.3 Transformer

**Là gì:** Kiến trúc neural network do Google giới thiệu (2017, paper "Attention is All You Need").

**Tại sao quan trọng:** Trước Transformer, NLP dùng RNN/LSTM — xử lý tuần tự, chậm, quên thông tin xa. Transformer dùng **Self-Attention** → xử lý song song, nhớ quan hệ xa.

**Thành phần chính:**

```
┌─────────────────────────────────────────┐
│              TRANSFORMER                │
│                                         │
│  ┌───────────┐      ┌───────────┐       │
│  │  ENCODER  │ ──── │  DECODER  │       │
│  │           │      │           │       │
│  │ Đọc input │      │ Sinh output│      │
│  │ Hiểu nghĩa│      │ từng token │      │
│  └───────────┘      └───────────┘       │
└─────────────────────────────────────────┘
```

- **Encoder:** Đọc toàn bộ input, tạo "hiểu biết" (representation) về văn bản.
- **Decoder:** Dựa trên representation đó, sinh output từng token một.
- **Self-Attention:** Cơ chế cho phép mỗi từ "nhìn" tất cả các từ khác để hiểu ngữ cảnh.

### 2.4 Self-Attention

**Input → Process → Output:**

```
INPUT:   Chuỗi các token (từ/subword)
PROCESS: Mỗi token tính "điểm chú ý" với mọi token khác
         → token nào liên quan mạnh → điểm cao → ảnh hưởng nhiều
OUTPUT:  Mỗi token có representation mới, chứa thông tin ngữ cảnh
```

**Ví dụ:** Câu "Cuộc họp ngày mai **bị hoãn** vì giám đốc **đi công tác**"
- Attention giúp model hiểu "bị hoãn" liên quan đến "đi công tác" dù cách xa nhau.

**Công thức:** `Attention(Q, K, V) = softmax(QK^T / √d_k) × V`
- Q (Query): "Tôi đang tìm gì?"
- K (Key): "Tôi có thông tin gì?"
- V (Value): "Thông tin thực tế là gì?"

### 2.5 Tokenization

**Là gì:** Chuyển text thành dãy số (token IDs) mà model hiểu được.

```
INPUT:   "Cuộc họp quan trọng"
PROCESS: Tokenizer chia thành subwords → map sang ID
OUTPUT:  [1234, 567, 89, 432]
```

**Tại sao không chia theo từ?** Vì từ vựng vô hạn, nhưng subword vocabulary có giới hạn (32K-64K tokens). Từ lạ được chia thành các mảnh nhỏ hơn mà model đã biết.

### 2.6 T5 / ViT5

**T5 (Text-to-Text Transfer Transformer):**
- Mọi NLP task đều được format thành text-to-text.
- Ví dụ: `"summarize: {document}"` → `"{summary}"`

**ViT5:**
- T5 nhưng pre-trained trên **tiếng Việt** (do VietAI phát triển).
- Đã học cấu trúc ngữ pháp, từ vựng, ngữ nghĩa tiếng Việt.

**Tại sao ViT5 thay vì FLAN-T5?**
- T5 gốc trained trên tiếng Anh → hiểu tiếng Việt kém.
- FLAN-T5 multilingual nhưng tiếng Việt chiếm phần rất nhỏ.
- ViT5 focused vào tiếng Việt → tokenizer tốt hơn, representation chính xác hơn.
- **Logic:** Cùng tài nguyên, model chuyên biệt cho ngôn ngữ target luôn tốt hơn model general.

### 2.7 Fine-tuning

```
INPUT:   Pre-trained ViT5 (biết tiếng Việt, chưa biết tóm tắt tốt)
         + Dataset summarization tiếng Việt
PROCESS: Train thêm 3 epochs, model học cách tóm tắt
OUTPUT:  Fine-tuned ViT5 (biết tiếng Việt + biết tóm tắt)
```

**Tại sao không train từ đầu?**
- Train from scratch cần hàng triệu GPU-hours + terabytes data.
- Fine-tuning chỉ cần vài giờ trên 1 GPU + vài nghìn samples.
- Pre-trained model đã có "kiến thức ngôn ngữ" → fine-tuning chỉ dạy thêm "kỹ năng tóm tắt".

### 2.8 Seq2Seq (Sequence-to-Sequence)

Kiến trúc nhận chuỗi input → sinh chuỗi output. T5/ViT5 là Seq2Seq model.

```
INPUT sequence:  "tom tat ngan: Cuộc họp hôm nay thảo luận về..."
OUTPUT sequence: "Cuộc họp quyết định 3 điểm chính..."
```

### 2.9 ROUGE Metrics

Thước đo chất lượng tóm tắt, so sánh output với reference.

| Metric | Đo cái gì | Ý nghĩa |
|---|---|---|
| ROUGE-1 | Overlap từ đơn | Model có dùng đúng từ không? |
| ROUGE-2 | Overlap cặp từ | Model có giữ đúng cụm từ không? |
| ROUGE-L | Longest common subsequence | Model có giữ đúng thứ tự ý không? |

**Giới hạn:** ROUGE đo overlap từ, không đo nghĩa. Hai câu cùng nghĩa nhưng dùng từ khác → ROUGE thấp. Vì vậy cần thêm qualitative evaluation.

### 2.10 Beam Search

Thuật toán sinh text — giữ lại top-k khả năng song song thay vì chỉ chọn 1.

```
Greedy:  Chọn 1 đường → có thể miss đường tốt hơn
Beam=4:  Giữ 4 đường song song → chọn đường tổng điểm cao nhất
```

**Trade-off:** beam càng lớn → quality cao hơn nhưng chậm hơn.

### 2.11 Gradient Accumulation

Kỹ thuật giả lập batch size lớn trên GPU nhỏ.

```
Muốn: batch_size = 16
GPU chỉ chứa: batch_size = 2
Giải pháp: accumulate 8 mini-batches → cập nhật 1 lần
Hiệu quả: 2 × 8 = 16
```

### 2.12 fp16 (Mixed Precision)

Dùng số thực 16-bit thay vì 32-bit → giảm ~40% VRAM, tăng ~30% speed.

---

## 3. Tech Stack

| Tool | Vai trò | Tại sao chọn (lý do logic) |
|---|---|---|
| **Python 3.10+** | Ngôn ngữ chính | Ecosystem ML/NLP lớn nhất |
| **PyTorch** | Deep learning framework | HuggingFace dựa trên PyTorch |
| **HF Transformers** | Load/train/infer model | Có sẵn ViT5, Seq2SeqTrainer |
| **HF Datasets** | Load/xử lý dataset | Tích hợp WikiLingua/VietNews |
| **evaluate + rouge-score** | Tính ROUGE | Thư viện chuẩn, reproducible |
| **Streamlit** | Web UI | Python thuần, deploy nhanh |
| **PyYAML** | Config | Tách config khỏi code |
| **pytest** | Testing | Standard Python testing |
| **Google Colab T4** | GPU training | Miễn phí, 16GB VRAM |

---

## 4. System Thinking: Kiến Trúc Toàn Bộ

### 4.1 Training Pipeline (offline, chạy 1 lần)

```
[HuggingFace Hub] → download dataset
        ↓
[Raw Dataset] → prepare_data.py (lọc, clean, split)
        ↓
[Processed JSONL: train / val / test]
        ↓
[train.py] → tokenize → Seq2SeqTrainer → GPU
        ↓
[Fine-tuned ViT5 Checkpoint]
```

### 4.2 Evaluation Pipeline (offline)

```
[Checkpoint + Test Set] → evaluate.py → generate predictions
        ↓
[So sánh với reference] → ROUGE scores + examples
        ↓
[Error Analysis] → categorize: hallucination, missing, repetition
```

### 4.3 Inference Pipeline (online, mỗi lần user dùng)

```
User Input (text + mode + length)
    → Preprocessing (normalize, truncate)
    → Prefix Injection ("tom tat ngan: ...")
    → Tokenizer (text → token IDs)
    → ViT5 Model (encoder → decoder → beam search)
    → Detokenize (token IDs → text)
    → Post-processing (format theo mode)
    → Keyword Extraction + Confidence Estimation
    → Streamlit UI (hiển thị kết quả)
```

---

## 5. Logical Reasoning: Tại Sao Cho Mọi Quyết Định

### Q1: Tại sao Abstractive thay vì Extractive?

```
Yêu cầu: 4 output modes khác nhau từ cùng 1 input

Extractive → chỉ chọn câu gốc → không thể format thành bullet/action → LOẠI
Abstractive → sinh câu mới → format bất kỳ dạng nào → CHỌN
```

### Q2: Tại sao fine-tune thay vì zero-shot?

```
Zero-shot: Model chưa học summarization → output lặp, lan man → ROUGE thấp
Fine-tune: Model học pattern tóm tắt → output coherent → ROUGE cao hơn

Kết luận: Fine-tune vì có data + có GPU (Colab T4).
```

### Q3: Tại sao 1 model + prefix thay vì 4 model riêng?

```
4 model: 4× training time, 4× storage, 4× RAM → vượt constraint → LOẠI
1 model + prefix: train 1 lần, 1 checkpoint, thay prefix khi inference → CHỌN

Trade-off: Quality mỗi mode thấp hơn 4 model riêng.
Nhưng constraint 4 tuần + 1 GPU → đây là rational choice.
```

### Q4: Tại sao max_source_length = 512?

```
ViT5-base max position: 512 tokens
512 tokens ≈ 300-400 từ tiếng Việt ≈ 1-2 trang A4

Tăng 1024 → VRAM tăng ~4× (attention O(n²)) → T4 OOM → LOẠI
Giảm 256 → mất quá nhiều info → tóm tắt kém → LOẠI
512 = điểm cân bằng giữa info và GPU.
```

### Q5: Tại sao Streamlit thay vì Flask + React?

```
Constraint: 2 người, 4 tuần, focus NLP

Streamlit: Python thuần, 50-100 dòng UI, 1-2 ngày → CHỌN
Flask+React: backend + frontend riêng, 5-7 ngày → lấy thời gian NLP → LOẠI

Web app là deliverable phụ. NLP model là deliverable chính.
```

### Q6: Tại sao learning_rate = 2e-5?

```
LR cao (1e-3): phá pre-trained knowledge (catastrophic forgetting) → LOẠI
LR thấp (1e-7): gần như không học → tốn thời gian vô ích → LOẠI
2e-5: standard fine-tuning LR, validated trên hàng trăm paper → CHỌN
```

---

## 6. Khái Niệm Hay Gặp Khi Làm

### Overfitting vs Underfitting

```
Underfitting: train loss cao, val loss cao → train thêm hoặc tăng model size
Overfitting:  train loss thấp, val loss tăng → early stopping, giảm epochs
```

### Epoch, Batch, Iteration

```
Dataset: 10,000 samples | Batch size: 2 | Grad accumulation: 8
1 iteration = 2 samples
1 effective step = 8 iterations = 16 samples
1 epoch = 10,000 / 2 = 5,000 iterations
3 epochs = 15,000 iterations
```

### Hallucination

Model sinh thông tin **không có trong input**.

```
Input:  "Cuộc họp có A, B, C tham dự"
Output: "Cuộc họp có A, B, C và D tham dự" ← D = hallucination
```

### Truncation

Input dài hơn max_source_length → tokenizer cắt bỏ phần cuối → thông tin cuối bị mất.

---

## 7. Module Map

```
scripts/
  prepare_data.py    ← Chạy 1 lần, tạo data splits
  train.py           ← Chạy 1 lần trên Colab, tạo checkpoint
  evaluate.py        ← Chạy 1 lần, tạo ROUGE scores
  predict.py         ← CLI test nhanh

src/smart_summarizer/
  data/              ← Load dataset, clean text, batch data
  modeling/          ← Load model, training wrapper, generation logic
  evaluation/        ← ROUGE metrics, error categorization
  product/           ← API chính, post-processing, keywords, confidence

app/                 ← Streamlit web UI
```

---

## 8. Checklist Tự Kiểm Tra

Trước khi code, trả lời được các câu này:

- [ ] Transformer khác RNN ở điểm nào?
- [ ] T5 format mọi task thành gì?
- [ ] Fine-tuning khác train from scratch thế nào?
- [ ] ROUGE đo cái gì và giới hạn là gì?
- [ ] Tại sao cần gradient accumulation?
- [ ] Beam search khác greedy search thế nào?
- [ ] Hallucination là gì?
- [ ] Controllable generation hoạt động thế nào trong project?
- [ ] Tại sao chọn ViT5?

---

## 9. Tổng Kết

Project là hệ thống **3 tầng**:

```
Tầng 1: DATA    → Thu thập, clean, split dataset
Tầng 2: MODEL   → Fine-tune ViT5 cho summarization
Tầng 3: PRODUCT → Wrap thành app có UX tốt
```

Mỗi tầng có input-process-output rõ ràng. Mỗi quyết định có lý do logic dựa trên constraints (thời gian, GPU, team size). Tất cả đều là trade-off được cân nhắc.
