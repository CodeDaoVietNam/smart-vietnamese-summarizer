# Deep Dive Part 1: Transformer — Từ Gốc Rễ Đến Ứng Dụng

> File này mổ xẻ chi tiết kiến trúc Transformer, giải thích **tại sao** từng thành phần tồn tại.

---

## 1. Trước Transformer: Vấn Đề Của RNN/LSTM

### RNN (Recurrent Neural Network) hoạt động thế nào?

```
Xử lý tuần tự, từng từ một:

"Cuộc" → [RNN] → h1
"họp"  → [RNN + h1] → h2
"hôm"  → [RNN + h2] → h3
"nay"  → [RNN + h3] → h4
```

- Mỗi bước nhận từ hiện tại + hidden state từ bước trước.
- Hidden state là "bộ nhớ" chứa thông tin các từ đã đọc.

### 3 vấn đề chết người của RNN

**Vấn đề 1: Vanishing Gradient (Gradient biến mất)**

```
Câu dài 100 từ:
Từ thứ 1 → ... → Từ thứ 100

Khi backpropagation, gradient phải đi ngược 100 bước.
Mỗi bước gradient bị nhân với 1 số < 1 → gradient → 0
→ Model KHÔNG THỂ học quan hệ giữa từ 1 và từ 100.
```

Ví dụ thực tế: "Giám đốc Nguyễn Văn A, người đã có 20 năm kinh nghiệm trong ngành công nghệ thông tin và từng làm việc tại nhiều tập đoàn lớn, **đã quyết định** từ chức."
- RNN đọc đến "đã quyết định" → đã quên "Giám đốc Nguyễn Văn A" vì cách quá xa.

**Vấn đề 2: Không song song hóa được**

```
RNN: phải đọc từ 1 xong → mới đọc từ 2 → mới đọc từ 3...
     Thời gian = O(n) tuần tự, không tận dụng được GPU parallel.

Transformer: đọc TẤT CẢ từ cùng lúc
     Thời gian = O(1) cho mỗi layer (parallel trên GPU)
```

**Vấn đề 3: LSTM cải thiện nhưng chưa đủ**

LSTM (Long Short-Term Memory) thêm "cổng" (gate) để kiểm soát bộ nhớ:
- Forget gate: quyết định quên gì
- Input gate: quyết định nhớ gì mới
- Output gate: quyết định output gì

→ Giảm vanishing gradient nhưng vẫn tuần tự, vẫn chậm, vẫn giới hạn ở ~200-500 tokens.

### Transformer giải quyết tất cả

| Vấn đề | RNN/LSTM | Transformer |
|---|---|---|
| Quan hệ xa | Quên dần | Self-attention nhìn trực tiếp mọi vị trí |
| Tốc độ | Tuần tự O(n) | Song song O(1) per layer |
| Scalability | Khó scale | Scale tốt với data và compute |

---

## 2. Kiến Trúc Transformer Chi Tiết

### Tổng quan

```
┌──────────────────────────────────────────────────┐
│                  TRANSFORMER                      │
│                                                   │
│  INPUT TEXT                                       │
│      ↓                                            │
│  [Input Embedding + Positional Encoding]          │
│      ↓                                            │
│  ┌─────────────────────┐                          │
│  │      ENCODER        │ ×N layers (N=6 gốc,     │
│  │                     │         12 cho base)     │
│  │  Multi-Head         │                          │
│  │  Self-Attention     │                          │
│  │       ↓             │                          │
│  │  Add & Norm         │                          │
│  │       ↓             │                          │
│  │  Feed Forward       │                          │
│  │       ↓             │                          │
│  │  Add & Norm         │                          │
│  └─────────────────────┘                          │
│      ↓ (encoder output = "hiểu biết")            │
│  ┌─────────────────────┐                          │
│  │      DECODER        │ ×N layers                │
│  │                     │                          │
│  │  Masked Self-Attn   │ ← chỉ nhìn từ trước     │
│  │       ↓             │                          │
│  │  Add & Norm         │                          │
│  │       ↓             │                          │
│  │  Cross-Attention    │ ← nhìn encoder output    │
│  │       ↓             │                          │
│  │  Add & Norm         │                          │
│  │       ↓             │                          │
│  │  Feed Forward       │                          │
│  │       ↓             │                          │
│  │  Add & Norm         │                          │
│  └─────────────────────┘                          │
│      ↓                                            │
│  [Linear + Softmax → xác suất từng token]         │
│      ↓                                            │
│  OUTPUT TOKEN                                     │
└──────────────────────────────────────────────────┘
```

### 2.1 Input Embedding

**Vấn đề:** Model nhận số, không nhận chữ.

```
INPUT:   "Cuộc họp" → tokenize → [1234, 567]
PROCESS: Mỗi token ID → lookup bảng embedding → vector 768 chiều
OUTPUT:  [[0.12, -0.45, 0.78, ...],   ← vector cho "Cuộc"
          [0.34, 0.67, -0.23, ...]]   ← vector cho "họp"
```

**Tại sao 768 chiều?** Càng nhiều chiều → model biểu diễn càng phong phú. 768 là compromise giữa expressiveness và compute cost. (ViT5-base dùng 768, ViT5-large dùng 1024).

**Embedding có train được:** Ban đầu random, qua training model học được từ nào gần nghĩa → vector gần nhau.

### 2.2 Positional Encoding

**Vấn đề:** Self-Attention xử lý song song → mất thông tin thứ tự từ.
- "Tôi ăn cơm" vs "Cơm ăn tôi" → nếu chỉ có embedding, 2 câu giống nhau!

**Giải pháp:** Cộng thêm vector vị trí vào embedding.

```
Token embedding:     [0.12, -0.45, 0.78, ...]
Position encoding:  +[0.01, 0.02, -0.01, ...]  ← khác nhau cho mỗi vị trí
Final embedding:     [0.13, -0.43, 0.77, ...]
```

**T5/ViT5 dùng Relative Position Bias:** thay vì mã hóa vị trí tuyệt đối, T5 mã hóa khoảng cách tương đối giữa 2 token (token cách nhau 3 vị trí vs 10 vị trí). Linh hoạt hơn cho input có độ dài khác nhau.

### 2.3 Self-Attention — Trái Tim Của Transformer

**Ý tưởng cốt lõi:** Mỗi từ tự hỏi "Trong câu này, từ nào liên quan đến tôi nhất?"

**Bước 1: Tạo Q, K, V**

Mỗi token embedding được nhân với 3 ma trận weight khác nhau:

```
Token "họp" (vector 768d)
  × W_Q (768×64) → Query vector (64d):  "Tôi đang tìm ngữ cảnh gì?"
  × W_K (768×64) → Key vector (64d):    "Tôi cung cấp ngữ cảnh gì?"
  × W_V (768×64) → Value vector (64d):  "Thông tin thực tế của tôi là gì?"
```

**Bước 2: Tính Attention Score**

```
Câu: "Cuộc họp quan trọng"

Q("họp") · K("Cuộc")  = 2.1   ← "họp" chú ý "Cuộc" mức 2.1
Q("họp") · K("họp")   = 5.3   ← "họp" chú ý chính nó mức 5.3
Q("họp") · K("quan")  = 1.8
Q("họp") · K("trọng") = 4.7   ← "họp" chú ý "trọng" mức 4.7
```

**Bước 3: Scale và Softmax**

```
Scores: [2.1, 5.3, 1.8, 4.7]
Scale:  chia cho √64 = 8 → [0.26, 0.66, 0.22, 0.59]
Softmax: chuẩn hóa → [0.12, 0.38, 0.10, 0.40]
                        ↑              ↑
                   "Cuộc" 12%    "trọng" 40%

→ Khi biểu diễn "họp", model chú ý nhiều nhất đến "trọng" (40%)
  và chính nó (38%). Hợp lý! "Cuộc họp quan trọng"
```

**Bước 4: Weighted sum of Values**

```
New representation("họp") = 0.12 × V("Cuộc")
                           + 0.38 × V("họp")
                           + 0.10 × V("quan")
                           + 0.40 × V("trọng")

→ Vector mới của "họp" giờ CHỨA thông tin ngữ cảnh
  từ tất cả các từ trong câu, weighted theo relevance.
```

### 2.4 Multi-Head Attention

**Vấn đề:** 1 attention head chỉ capture 1 loại quan hệ.

```
Head 1: có thể học quan hệ ngữ pháp (chủ-vị)
Head 2: có thể học quan hệ ngữ nghĩa (đồng nghĩa)
Head 3: có thể học quan hệ vị trí (từ kế bên)
...
Head 12: có thể học quan hệ khác

ViT5-base: 12 heads, mỗi head 64 chiều → 12 × 64 = 768 = model dimension
```

Mỗi head có bộ W_Q, W_K, W_V riêng → học pattern riêng → concat lại → Linear → output 768d.

### 2.5 Add & Norm (Residual Connection + Layer Normalization)

**Residual Connection:**
```
output = LayerNorm(x + Attention(x))

Tại sao cộng x gốc?
→ Giúp gradient flow trực tiếp qua shortcut
→ Giải quyết vanishing gradient trong deep network
→ Model 12 layers mà gradient vẫn ổn
```

**Layer Normalization:**
- Chuẩn hóa activation về mean=0, std=1 → training ổn định hơn.

### 2.6 Feed-Forward Network (FFN)

```
FFN(x) = ReLU(x × W1 + b1) × W2 + b2

Dimensions: 768 → 3072 → 768 (expand 4× rồi compress lại)
```

**Tại sao cần FFN sau Attention?**
- Attention capture quan hệ giữa các token.
- FFN transform representation của từng token individually.
- Có thể hiểu: Attention = "nghe ngóng xung quanh", FFN = "suy nghĩ về những gì vừa nghe".

### 2.7 Encoder vs Decoder

| Thành phần | Encoder | Decoder |
|---|---|---|
| Self-Attention | Nhìn tất cả token | Chỉ nhìn token trước (masked) |
| Cross-Attention | Không có | Có — nhìn encoder output |
| Mục đích | Hiểu input | Sinh output |

**Masked Self-Attention trong Decoder:**
```
Sinh từ thứ 3 của output:
  Được nhìn: từ 1, từ 2 (đã sinh)
  Không được nhìn: từ 4, 5, 6... (chưa sinh)
  Lý do: Khi inference, từ tương lai chưa tồn tại!
```

**Cross-Attention:**
```
Decoder token "Tóm" muốn biết input nói gì:
  Q = từ "Tóm" (decoder)
  K, V = từ encoder output (tất cả token input)
  → Decoder "nhìn vào" input để quyết định sinh từ gì tiếp
```

---

## 3. T5: Text-to-Text Transfer Transformer

### Ý tưởng đột phá

Thay vì thiết kế model khác nhau cho mỗi task, T5 format **MỌI** task thành text-to-text:

```
Summarization:  "summarize: {document}"        → "{summary}"
Translation:    "translate to Vietnamese: Hi"  → "Xin chào"
Sentiment:      "sentiment: I love this movie" → "positive"
Q&A:            "question: What is AI? context: AI is..." → "AI is..."
```

### Tại sao thiết kế này thông minh?

1. **Một architecture cho tất cả** — không cần redesign model cho mỗi task.
2. **Transfer learning tự nhiên** — pre-train 1 lần, fine-tune cho bất kỳ task nào.
3. **Prefix là instruction** — model học hiểu "summarize:" nghĩa là phải tóm tắt.

### ViT5 vs T5 vs mT5

| Model | Pre-training data | Tokenizer | Phù hợp cho |
|---|---|---|---|
| T5 | Tiếng Anh (C4 corpus) | English SentencePiece | Tasks tiếng Anh |
| mT5 | 101 ngôn ngữ (mC4) | Multilingual, 250K vocab | Đa ngôn ngữ |
| ViT5 | Tiếng Việt (Vietnamese corpus) | Vietnamese SentencePiece | Tasks tiếng Việt |

**Tại sao ViT5 > mT5 cho tiếng Việt?**
- mT5 chia vocab cho 101 ngôn ngữ → tiếng Việt chỉ được ~2-5% vocab → tokenize kém hiệu quả.
- ViT5 dành 100% vocab cho tiếng Việt → ít token hơn cho cùng câu → fit nhiều text hơn trong 512 tokens.

---

## 4. Training Deep Dive

### 4.1 Pre-training vs Fine-tuning

```
PRE-TRAINING (do VietAI đã làm, ta KHÔNG cần làm):
  Data:    Hàng GB text tiếng Việt (Wikipedia, báo, sách...)
  Task:    Span corruption (che 1 phần text, model đoán lại)
  Thời gian: Hàng tuần trên cluster GPU
  Kết quả: Model hiểu ngữ pháp, từ vựng, ngữ nghĩa tiếng Việt

FINE-TUNING (ta làm trên Colab):
  Data:    Vài nghìn cặp (document, summary) tiếng Việt
  Task:    Summarization cụ thể
  Thời gian: Vài giờ trên 1 GPU T4
  Kết quả: Model hiểu tiếng Việt + BIẾT CÁCH tóm tắt
```

### 4.2 Seq2SeqTrainer (HuggingFace)

Thay vì tự viết training loop, HuggingFace cung cấp Trainer class:

```python
# Tự viết training loop (phức tạp, dễ bug):
for epoch in range(3):
    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        # + logging, evaluation, checkpointing, fp16, gradient accumulation...
        # → 100+ dòng code, nhiều edge case

# Dùng Seq2SeqTrainer (đơn giản, stable):
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,  # chứa lr, epochs, batch_size...
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)
trainer.train()  # 1 dòng, xử lý hết
```

### 4.3 Loss Function: Cross-Entropy

```
Model sinh xác suất cho token tiếp theo:
  P("tóm") = 0.6, P("và") = 0.1, P("cuộc") = 0.2, ...

Target thực tế: "tóm"

Loss = -log(P("tóm")) = -log(0.6) = 0.51

Nếu model đoán đúng hơn: P("tóm") = 0.9 → loss = 0.10 (thấp hơn = tốt hơn)
Nếu model đoán sai:       P("tóm") = 0.1 → loss = 2.30 (cao = tệ)

→ Training giảm loss = dạy model đoán đúng token hơn.
```

### 4.4 Hyperparameters Giải Thích

| Param | Giá trị | Tại sao |
|---|---|---|
| learning_rate | 2e-5 | Standard fine-tuning, không phá pre-trained knowledge |
| epochs | 3 | Đủ học task, không overfit. Fine-tuning thường 2-5 epochs |
| batch_size | 2 | T4 16GB chỉ fit 2 samples ViT5-base + gradients |
| grad_accum | 8 | Effective batch = 2×8 = 16, ổn định hơn batch=2 |
| fp16 | true | Giảm VRAM ~40%, T4 có Tensor Cores hỗ trợ |
| max_source | 512 | Max position của ViT5-base, cân bằng info vs memory |
| max_target | 128 | Summary thường ngắn, 128 tokens ≈ 80-100 từ |

---

## 5. Generation (Sinh Text) Deep Dive

### 5.1 Autoregressive Generation

Decoder sinh text **từng token một**, mỗi token dựa trên tokens trước:

```
Step 1: Input: [START]           → P(token) → chọn "Cuộc"
Step 2: Input: [START, Cuộc]     → P(token) → chọn "họp"
Step 3: Input: [START, Cuộc, họp] → P(token) → chọn "quyết"
...
Step N: → chọn [END] → dừng
```

### 5.2 Greedy vs Beam Search

**Greedy:** Mỗi bước chọn token có xác suất cao nhất.
```
Bước 1: P("Cuộc")=0.4, P("Buổi")=0.35 → chọn "Cuộc"
Bước 2: P("họp")=0.5, P("hội")=0.3    → chọn "họp"

Vấn đề: "Cuộc" có thể dẫn đến dead-end, 
         trong khi "Buổi" dẫn đến câu tổng thể tốt hơn.
```

**Beam Search (beam=4):** Giữ 4 candidates song song.
```
Bước 1: Top 4 = ["Cuộc", "Buổi", "Bản", "Nội"]
Bước 2: Mỗi candidate expand → 4×V options → giữ top 4 overall
...
Cuối: chọn candidate có tổng log-probability cao nhất.
```

**Trade-off trong project:**
- `num_beams=4`: quality tốt, latency chấp nhận (~2-5 giây trên T4).
- `num_beams=1` (greedy): nhanh nhưng quality thấp hơn.
- `num_beams=8+`: quality tăng ít, latency tăng nhiều → không đáng.

### 5.3 Repetition Penalty

**Vấn đề:** Model hay lặp: "Cuộc họp rất quan trọng rất quan trọng rất..."

```
repetition_penalty = 1.2:
  Nếu token "quan" đã xuất hiện → P("quan") /= 1.2 → giảm xác suất chọn lại

no_repeat_ngram_size = 3:
  Cấm lặp bất kỳ 3-gram nào đã xuất hiện.
  "rất quan trọng" đã có → không thể sinh "rất quan trọng" lần 2.
```

---

## 6. Evaluation Deep Dive

### 6.1 ROUGE — Recall-Oriented Understudy for Gisting Evaluation

```
Reference: "Cuộc họp quyết định tăng lương nhân viên"
Prediction: "Cuộc họp đã quyết định việc tăng lương"

ROUGE-1 (unigrams):
  Reference words:  {Cuộc, họp, quyết, định, tăng, lương, nhân, viên}
  Prediction words: {Cuộc, họp, đã, quyết, định, việc, tăng, lương}
  Overlap:          {Cuộc, họp, quyết, định, tăng, lương} = 6 từ
  
  Precision = 6/8 (prediction) = 0.75
  Recall    = 6/8 (reference)  = 0.75
  F1        = 2 × 0.75 × 0.75 / (0.75 + 0.75) = 0.75

ROUGE-2 (bigrams):
  Ref bigrams:  {Cuộc-họp, họp-quyết, quyết-định, định-tăng, tăng-lương, lương-nhân, nhân-viên}
  Pred bigrams: {Cuộc-họp, họp-đã, đã-quyết, quyết-định, định-việc, việc-tăng, tăng-lương}
  Overlap:      {Cuộc-họp, quyết-định, tăng-lương} = 3
  F1 = tính tương tự...

ROUGE-L: Tìm longest common subsequence
  LCS("Cuộc họp quyết định tăng lương nhân viên",
      "Cuộc họp đã quyết định việc tăng lương")
  = "Cuộc họp quyết định tăng lương" (6 từ)
```

### 6.2 Tại Sao Cần Error Analysis Ngoài ROUGE

ROUGE chỉ đo **word overlap**, không đo:
- **Factual correctness:** "Model nói meeting lúc 3pm nhưng thực tế là 2pm" → ROUGE vẫn cao vì các từ khác đúng.
- **Coherence:** Câu có nghĩa không? Mạch lạc không?
- **Coverage:** Có bỏ sót ý quan trọng không?

→ Phải đọc output bằng mắt và phân loại lỗi.
