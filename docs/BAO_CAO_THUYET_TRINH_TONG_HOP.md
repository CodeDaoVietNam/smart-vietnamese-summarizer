# Báo Cáo Thuyết Trình Tổng Hợp
## Smart Vietnamese Controllable Summarizer

Tài liệu này là bản tổng hợp đã chỉnh sửa từ hai file `BAO_CAO_CHITIET_P1.md` và `BAO_CAO_CHITIET_P2.md`. Mục tiêu là dùng làm **kịch bản thuyết trình/bảo vệ**, không phải slide. Nội dung được viết theo hướng rõ ràng, trung thực, có chiều sâu kỹ thuật nhưng tránh overclaim.

---

## 0. Thông Điệp Chính Của Đồ Án

Đồ án xây dựng một hệ thống **tóm tắt văn bản tiếng Việt có điều khiển đầu ra**. Người dùng đưa vào một văn bản tiếng Việt dài, chọn kiểu tóm tắt mong muốn, và hệ thống sinh ra bản tóm tắt tương ứng.

Các mode chính:

```text
concise       -> tóm tắt ngắn gọn thành một đoạn văn tự nhiên
bullet        -> tóm tắt thành các ý chính dạng gạch đầu dòng
action_items  -> trích xuất việc cần làm, người phụ trách, deadline nếu có
study_notes   -> tạo ghi chú học tập có cấu trúc
```

Điểm quan trọng của đồ án không chỉ là fine-tune một mô hình, mà là xây dựng một pipeline AI engineering tương đối hoàn chỉnh:

```text
Data preparation
  -> Phase 1 ViT5 fine-tuning
  -> Phase 2 controllability baseline
  -> LoRA mixed-data adaptation
  -> Evaluation
  -> FastAPI backend
  -> Streamlit frontend
  -> Hybrid neuro-symbolic decoder
```

Kết luận trung thực nên dùng khi thuyết trình:

> Phase 1 học năng lực tóm tắt tiếng Việt tổng quát. Phase 2 full fine-tune giúp thử nghiệm controllability theo mode nhưng bị giới hạn bởi synthetic data ít. LoRA mixed-data giữ lại backbone Phase 1 và học thêm adaptation nhẹ, cân bằng hơn giữa chất lượng tóm tắt tổng quát và khả năng điều khiển đầu ra. Hybrid decoder được thêm ở tầng inference để làm output ổn định hơn, đặc biệt với `action_items` và `study_notes`.

---

## 1. Bài Toán Và Động Lực

### 1.1. Vấn đề thực tế

Trong học tập và công việc, người dùng thường phải xử lý lượng văn bản lớn:

- Một cuộc họp dài có thể tạo ra nhiều trang biên bản.
- Một bài giảng có nhiều khái niệm, ví dụ và lỗi dễ nhầm.
- Một bài báo hoặc báo cáo dài cần được rút gọn để đọc nhanh.

Tuy nhiên, không phải người dùng nào cũng cần cùng một kiểu tóm tắt:

- Quản lý thường cần `action_items`: ai làm gì, deadline khi nào.
- Sinh viên thường cần `study_notes`: khái niệm chính, cần nhớ, ví dụ, lỗi dễ nhầm.
- Người đọc tin tức hoặc báo cáo thường cần `concise` hoặc `bullet`.

Vì vậy, bài toán của đồ án là:

```text
Input:
  Văn bản tiếng Việt dài
  + mode: concise | bullet | action_items | study_notes
  + length: short | medium | long

Output:
  Bản tóm tắt tiếng Việt theo đúng mode
  + keywords
  + heuristic quality estimate
  + latency
  + token count
```

### 1.2. Vì sao gọi là controllable summarization?

Tóm tắt thông thường chỉ học ánh xạ:

```text
document -> summary
```

Trong đồ án này, hệ thống học thêm điều kiện điều khiển:

```text
document + mode prefix -> mode-aware summary
```

Cùng một document, nếu prefix khác nhau thì output mong muốn cũng khác nhau. Đây là lý do hệ thống được gọi là **controllable summarization**.

---

## 2. Vì Sao Chọn Abstractive Summarization?

Có hai hướng tóm tắt chính:

| Hướng | Cách hoạt động | Phù hợp với đồ án? |
|---|---|---|
| Extractive | Chọn câu quan trọng từ văn bản gốc | Không phù hợp cho nhiều mode |
| Abstractive | Hiểu nội dung và sinh câu mới | Phù hợp hơn |

Nếu dùng extractive summarization, hệ thống chủ yếu chỉ copy các câu từ văn bản gốc. Điều này khó đáp ứng các mode có cấu trúc như:

```text
- Người phụ trách: An
  Hành động: rà soát dữ liệu
  Deadline: trước thứ Tư
```

Trong khi đó, abstractive summarization cho phép model viết lại nội dung theo format mới. Vì vậy, đồ án chọn hướng **abstractive sequence-to-sequence summarization**.

---

## 3. Nền Tảng Transformer Và ViT5

### 3.1. Giới hạn của RNN/LSTM

Trước Transformer, nhiều bài toán NLP dùng RNN hoặc LSTM. RNN xử lý văn bản tuần tự:

```text
"Cuộc" -> h1
"họp"  -> h2
"hôm"  -> h3
"nay"  -> h4
```

Vấn đề:

- Khó học quan hệ xa vì gradient suy giảm qua nhiều bước.
- Không song song hóa tốt trên GPU vì phải xử lý từng token theo thứ tự.
- LSTM cải thiện bộ nhớ bằng các gate, nhưng vẫn là mô hình tuần tự.

Trong bài toán tóm tắt, quan hệ xa rất quan trọng. Ví dụ người phụ trách có thể nằm ở đầu câu, deadline nằm ở cuối câu. Vì vậy cần một kiến trúc có thể nhìn toàn bộ chuỗi hiệu quả hơn.

### 3.2. Transformer giải quyết gì?

Transformer dùng self-attention để mỗi token có thể trực tiếp chú ý đến các token khác trong cùng chuỗi. Công thức attention:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Diễn giải:

- `Q` hoặc Query: token hiện tại đang tìm thông tin gì.
- `K` hoặc Key: token khác có thể cung cấp tín hiệu gì.
- `V` hoặc Value: thông tin thực tế được tổng hợp.

Nhờ attention, model có thể học các quan hệ như:

- Chủ thể và hành động.
- Hành động và deadline.
- Khái niệm và ví dụ.
- Nguyên nhân và kết quả.

### 3.3. Encoder và decoder khác nhau thế nào?

Transformer encoder-decoder gồm hai phần:

```text
Encoder:
  Đọc toàn bộ input
  -> tạo contextual representations

Decoder:
  Sinh output từng token
  -> dùng masked self-attention
  -> dùng cross-attention để nhìn lại encoder output
```

Encoder dùng self-attention hai chiều, tức là mỗi token trong input có thể nhìn các token khác. Decoder thì dùng masked self-attention để không nhìn token tương lai khi sinh output.

Khi decoder sinh token hiện tại, nó dùng cross-attention để hỏi encoder:

> Phần nào trong input liên quan nhất đến token tôi đang sinh?

Đây là cơ chế quan trọng giúp mô hình sinh summary bám nội dung văn bản gốc.

### 3.4. Vì sao chọn ViT5?

T5 là kiến trúc text-to-text, tức là mọi task đều được đưa về dạng:

```text
input text -> output text
```

Ví dụ:

```text
"summarize: {document}" -> "{summary}"
```

ViT5 là phiên bản T5 được pre-train cho tiếng Việt. Lý do chọn ViT5:

- Tokenizer phù hợp hơn với tiếng Việt.
- Mô hình đã học ngữ pháp, từ vựng và cấu trúc tiếng Việt.
- Phù hợp với bài toán sinh văn bản tiếng Việt hơn T5 tiếng Anh.
- T5 text-to-text rất tự nhiên cho summarization.

Thông tin kỹ thuật nên trình bày ở mức vừa phải:

```text
Architecture: T5-style encoder-decoder
Hidden dimension: 768
Attention heads: 12
Vocabulary size: khoảng 32K
Model scale: khoảng hơn 200M parameters
```

Không nên nói quá chắc về mọi con số nếu không trích trực tiếp từ model card. Khi bị hỏi, nên nói:

> Em dựa trên cấu hình ViT5-base và artifact HuggingFace, con số parameter thực tế trong log training của hệ thống khoảng 226.8M parameters.

---

## 4. Pipeline Huấn Luyện

### 4.1. Tổng quan pipeline

Luồng huấn luyện có ba hướng chính:

```text
Phase 1:
  VietNews
  -> full fine-tune ViT5
  -> models/vit5-summarizer-v1

Phase 2 full fine-tune baseline:
  synthetic paired data
  -> train từ Phase 1 checkpoint
  -> models/vit5-summarizer-v2

LoRA mixed-data adaptation:
  Phase 1 backbone frozen
  + LoRA adapter
  + mixed data
  -> models/vit5-summarizer-lora
```

Điểm cần nói rõ:

> Phase 2 full fine-tune và LoRA là hai hướng khác nhau. Phase 2 full fine-tune là baseline controllability. LoRA là experiment cải thiện chính để giảm nguy cơ quên kiến thức khi dữ liệu mode còn ít.

### 4.2. Phase 1: Core Vietnamese summarization

Mục tiêu Phase 1 là dạy model kỹ năng tóm tắt tiếng Việt tổng quát.

```text
Base model: VietAI/vit5-base
Dataset: VietNews summarization
Output: models/vit5-summarizer-v1
```

Vì sao dùng VietNews?

- Tiếng Việt tự nhiên hơn nhiều nguồn dịch máy.
- Phù hợp để học tóm tắt văn bản tiếng Việt.
- Có quy mô lớn hơn synthetic data của các mode.

Phase 1 trả lời câu hỏi:

> Làm sao để tóm tắt tiếng Việt cho tốt?

### 4.3. Phase 2 full fine-tune baseline

Mục tiêu Phase 2 là thử dạy model controllability theo mode.

Synthetic data có schema dạng:

```json
{
  "document": "...",
  "summary": "...",
  "mode": "action_items",
  "domain": "meeting_notes",
  "base_id": "..."
}
```

Ý tưởng paired dataset:

```text
Cùng một document
  -> concise target
  -> bullet target
  -> action_items target
  -> study_notes target
```

Vấn đề:

- Synthetic data ít hơn rất nhiều so với VietNews.
- Full fine-tune trên dữ liệu nhỏ có thể làm model học format nhưng giảm năng lực tổng quát.
- Các mode khó như `action_items` và `study_notes` cần dữ liệu chất lượng cao hơn.

Do đó, Phase 2 full fine-tune không được xem là mô hình cuối duy nhất, mà là một baseline để so sánh.

### 4.4. LoRA mixed-data adaptation

LoRA hoặc Low-Rank Adaptation là kỹ thuật fine-tuning hiệu quả tham số. Thay vì cập nhật toàn bộ trọng số gốc `W`, LoRA đóng băng backbone và học một cập nhật hạng thấp:

```text
W' = W + ΔW
ΔW = B A * alpha / r
```

Trong project:

```text
Backbone: models/vit5-summarizer-v1
Adapter: models/vit5-summarizer-lora
r: 8
lora_alpha: 16
target_modules: q, v
learning_rate: 2.0e-4
epochs: 3
```

Số liệu thực tế từ `training_log_lora.json`:

```text
Trainable parameters: 884,736
Total parameters:     226,835,712
Trainable ratio:      0.39%
```

Nên nói:

> LoRA chỉ train khoảng 0.39% tổng số tham số, giúp adaptation nhẹ hơn full fine-tune và giảm nguy cơ catastrophic forgetting.

Không nên nói:

> LoRA chống catastrophic forgetting tuyệt đối.

Nên nói:

> LoRA giảm nguy cơ làm hỏng backbone vì trọng số gốc được giữ frozen, nhưng output vẫn cần đánh giá thực nghiệm.

### 4.5. Mixed data thực tế dùng cho LoRA

Recipe ban đầu có thiết kế thêm XL-Sum, nhưng lần chạy thực tế trong artifact hiện tại không lấy được XL-Sum. Theo `lora_data_report.json`, dữ liệu thực tế là:

```text
Train rows: 5320
Validation rows: 580
Holdout rows: 160

Train source counts:
  VietNews: 4000
  viWikiHow: 1000
  synthetic_curated: 320

Validation source counts:
  VietNews: 400
  viWikiHow: 100
  synthetic_curated: 80
```

Mode distribution train:

```text
concise:       3080
bullet:        1080
action_items:  1080
study_notes:     80
```

Nhận xét trung thực:

> LoRA mixed data giúp tăng dữ liệu cho concise, bullet và action_items, nhưng study_notes vẫn còn ít. Đây là một lý do study_notes vẫn là mode khó nhất.

---

## 5. Inference Pipeline

Khi người dùng gửi request từ web, hệ thống chạy theo luồng:

```text
User text + mode + length
  -> Streamlit frontend
  -> FastAPI backend
  -> SmartSummarizer
  -> clean text
  -> build instruction prefix
  -> tokenize
  -> ViT5/LoRA generate
  -> decode neural draft
  -> hybrid decoder
  -> keywords + heuristic quality estimate
  -> response JSON
```

### 5.1. Prefix-based control

Mỗi mode có prefix riêng:

```python
concise:
  "Tóm tắt ngắn gọn thành một đoạn văn tự nhiên, không dùng bullet: "

bullet:
  "Tóm tắt thành các ý chính dạng bullet, mỗi bullet một ý: "

action_items:
  "Chỉ trích xuất việc cần làm. Mỗi dòng gồm người phụ trách, hành động, deadline nếu có: "

study_notes:
  "Tạo ghi chú học tập. Nêu khái niệm chính, cần nhớ, ví dụ, lỗi dễ nhầm: "
```

Prefix là tín hiệu để model biết cần sinh kiểu output nào.

### 5.2. Generation

Hệ thống dùng các tham số generation như:

```text
num_beams: 4
repetition_penalty: 1.2
no_repeat_ngram_size: 3
max_new_tokens:
  short: 96
  medium: 160
  long: 256
```

Beam search giữ nhiều candidate sequence song song, thường cho output tự nhiên hơn greedy search. Tuy nhiên, beam search không đảm bảo factuality tuyệt đối.

### 5.3. Hybrid neuro-symbolic decoder

Một vấn đề thực tế là model neural có thể sinh nội dung hợp lý nhưng format chưa ổn định. Ví dụ:

- action item thiếu người phụ trách.
- deadline bị gán sai.
- study notes thiếu trường `Lỗi dễ nhầm`.
- bullet bị dính thành một đoạn văn.

Vì vậy, project thêm hybrid decoder:

```text
Neural draft + source text
  -> extractor
  -> critic / validator
  -> renderer
  -> final output
```

Vai trò:

- `extractor`: lấy cấu trúc từ source và draft.
- `critic`: kiểm tra lỗi như thiếu field, prompt leakage, duplicate.
- `renderer`: ép output về format cuối.

Với `action_items`, hệ thống ưu tiên source-first extraction để giảm hallucination. Với `study_notes`, renderer ép đủ bốn trường:

```text
Khái niệm chính: ...
Cần nhớ: ...
Ví dụ: ...
Lỗi dễ nhầm: ...
```

Nên nói rõ:

> Hybrid decoder là inference-time reliability layer. Nó giúp output demo ổn định hơn, nhưng không thay thế human factuality evaluation.

---

## 6. Backend Và Frontend

### 6.1. Backend

Backend dùng FastAPI với các endpoint chính:

```text
GET  /api/health
POST /api/summarize
POST /api/compare-modes
```

Backend chịu trách nhiệm:

- Validate request.
- Load model theo config.
- Gọi `SmartSummarizer`.
- Trả JSON response.

### 6.2. Frontend

Frontend dùng Streamlit:

- Cho phép nhập hoặc chọn sample text.
- Chọn mode và length.
- Generate một summary.
- Compare all modes.
- Hiển thị summary, keywords, latency, input tokens, output tokens và Heuristic Quality Estimate.

Nên tránh gọi hệ thống là production microservices. Cách nói chính xác hơn:

> Hệ thống hiện là modular two-service demo: Streamlit frontend và FastAPI inference backend. Training và evaluation là offline batch jobs.

---

## 7. Evaluation

### 7.1. ROUGE metrics

ROUGE đo overlap giữa prediction và reference summary:

```text
ROUGE-1: overlap từ đơn
ROUGE-2: overlap cặp từ
ROUGE-L: longest common subsequence
```

ROUGE hữu ích để đánh giá summarization, nhưng không đủ cho controllable summarization. Lý do:

- Không đo chắc factuality.
- Không đo mode adherence.
- Không đo format correctness.
- Không đánh giá tốt các output có paraphrase đúng nghĩa nhưng ít overlap từ.

Vì vậy, với các mode như `action_items` và `study_notes`, cần thêm qualitative evaluation và error analysis.

### 7.2. Kết quả định lượng thực tế

Số liệu hiện tại trong `reports/metrics`:

| System | ROUGE-1 | ROUGE-2 | ROUGE-L | Nhận xét |
|---|---:|---:|---:|---|
| Phase 1 ViT5 | 0.5496 | 0.2294 | 0.3436 | Backbone ổn định cho news summarization |
| Phase 2 full fine-tune | 0.5490 | 0.2219 | 0.3336 | Có controllability nhưng ROUGE news giảm nhẹ |
| LoRA mixed-data | 0.5482 | 0.2330 | 0.3411 | Cân bằng hơn Phase 2 full fine-tune |
| LoRA mixed holdout | 0.5695 | 0.2970 | 0.3913 | Tốt trên phân phối gần mixed data |

Diễn giải:

- Phase 1 vẫn là backbone tốt cho tóm tắt tổng quát.
- Phase 2 không bắt buộc phải thắng Phase 1 trên ROUGE vì mục tiêu chính là mode controllability.
- LoRA giữ chất lượng gần Phase 1 hơn Phase 2 full fine-tune, đồng thời học thêm hành vi theo mode.
- Không nên chọn model chỉ dựa vào ROUGE; cần xem mode adherence, factuality và demo quality.

### 7.3. Heuristic Quality Estimate

Trong UI, hệ thống có `Heuristic Quality Estimate`. Đây không phải confidence thật của model.

Nó là điểm heuristic dựa trên các tín hiệu như:

- Độ dài output có hợp lý không.
- Mức độ lặp.
- Keyword coverage.
- Generation score nếu có.

Nên nói:

> Đây là chỉ báo hỗ trợ người dùng, không phải xác suất output đúng. Vì chưa được calibrate bằng human-labeled dataset, factuality vẫn cần đánh giá riêng.

---

## 8. Các Vấn Đề Đã Gặp Phải Và Cách Xử Lý

### 8.1. Synthetic data ít

Vấn đề:

- Dữ liệu synthetic ban đầu rất ít so với VietNews.
- Mode khó như `study_notes` có ít mẫu hơn các mode khác trong mixed LoRA data.

Cách xử lý:

- Tăng synthetic data theo dạng paired dataset.
- Thử LoRA mixed-data để không phụ thuộc hoàn toàn vào synthetic data nhỏ.
- Giữ Phase 1 backbone làm nền.

### 8.2. Output các mode còn giống nhau

Vấn đề:

- `bullet`, `action_items`, `study_notes` đôi khi giống nhau về nội dung.
- Model có thể chỉ thay đổi format mà chưa thật sự hiểu mục đích của mode.

Cách xử lý:

- Dùng prefix rõ hơn.
- Dùng paired data cùng document nhiều target.
- Dùng hybrid renderer để ép format tối thiểu.
- Đánh giá qualitative theo từng mode.

### 8.3. Action items dễ sai factuality

Vấn đề:

- Model có thể bịa người phụ trách.
- Model có thể gán sai deadline.
- Có thể biến một câu context thành task dù không có việc rõ ràng.

Cách xử lý:

- Source-first extraction.
- Nếu thiếu owner/deadline thì dùng `Chưa rõ`, không bịa.
- Với văn bản không có task rõ, trả về thông báo thận trọng.

### 8.4. Study notes khó hơn bullet

Vấn đề:

- Study notes cần cấu trúc học tập chứ không chỉ liệt kê ý chính.
- Dữ liệu học tập thật còn ít.

Cách xử lý:

- Renderer ép 4 trường cố định.
- Extractor tìm câu có dấu hiệu định nghĩa, ví dụ, lỗi dễ nhầm.
- Future work cần thêm lecture/study material thật.

### 8.5. Long input bị giới hạn

Vấn đề:

- `max_source_length = 512`.
- Văn bản dài có thể bị truncate, làm mất thông tin cuối.

Cách xử lý tương lai:

- Chunking: chia văn bản thành đoạn.
- Hierarchical summarization: tóm tắt từng chunk rồi tóm tắt lại.
- Long-context model nếu có tài nguyên.

---

## 9. Hạn Chế Hiện Tại

Các hạn chế cần nói rõ để báo cáo trung thực:

| Hạn chế | Nguyên nhân | Tác động |
|---|---|---|
| Data theo mode còn ít | Synthetic/curated data chưa lớn | Mode controllability chưa hoàn toàn ổn định |
| Study notes yếu hơn | Ít dữ liệu giáo dục thật | Output đôi khi giống bullet |
| Action items cần factuality cao | Đây gần với information extraction | Sai owner/deadline là lỗi nghiêm trọng |
| Context length 512 | Giới hạn model/tài nguyên | Input dài có thể mất ý |
| Heuristic quality chưa calibrate | Không có human-labeled quality dataset | Không phải confidence thật |
| Chưa production-ready | Demo local, chưa deploy scale | Thiếu monitoring, logging, rate limit |

Một câu nên dùng:

> Hệ thống hiện phù hợp để demo end-to-end và phân tích hướng nghiên cứu, nhưng chưa nên coi là production system cho các tài liệu quan trọng nếu không có human review.

---

## 10. Hướng Phát Triển

### 10.1. Ngắn hạn

- Tăng số lượng và chất lượng paired multi-mode data.
- Tạo human evaluation rubric rõ ràng:
  - mode adherence
  - factuality
  - usefulness
  - format correctness
  - conciseness
- Thêm evidence highlighting cho `action_items`, chỉ ra câu nguồn hỗ trợ mỗi item.
- Cải thiện keyword extraction bằng TF-IDF hoặc YAKE.

### 10.2. Trung hạn

- Thử adapter riêng cho từng mode:

```text
ViT5 Phase 1 backbone
  + adapter_concise
  + adapter_bullet
  + adapter_action_items
  + adapter_study_notes
```

- Thêm chunking/hierarchical summarization cho tài liệu dài.
- Batch inference để evaluate nhanh hơn.
- Tạo test set meeting/lecture thật, không chỉ synthetic.

### 10.3. Dài hạn

- Dùng model lớn hơn hoặc instruction-tuned model tiếng Việt.
- RLHF/DPO nếu có human preference data.
- Speech-to-summary:

```text
Audio meeting
  -> ASR
  -> Vietnamese text
  -> controllable summarization
```

- Production deployment:
  - logging
  - monitoring
  - model versioning
  - rate limiting
  - GPU serving optimization

---

## 11. Kịch Bản Thuyết Trình 10-15 Phút

Nếu chỉ có 10-15 phút, không nên đọc toàn bộ tài liệu này. Nên nói theo flow sau:

### 11.1. Mở đầu: 1 phút

> Đề tài của em là hệ thống tóm tắt tiếng Việt có điều khiển đầu ra. Người dùng nhập văn bản, chọn mode như concise, bullet, action_items hoặc study_notes, hệ thống sinh summary tương ứng. Điểm chính của đồ án là xây dựng pipeline end-to-end, từ fine-tune ViT5, thử nghiệm Phase 2/LoRA, đến backend/frontend demo và hybrid decoder.

### 11.2. Kiến trúc tổng quan: 1.5 phút

> Hệ thống gồm hai luồng. Offline pipeline dùng để chuẩn bị dữ liệu, train và evaluate model. Online pipeline gồm Streamlit frontend, FastAPI backend và SmartSummarizer service. Khi user gửi request, backend build instruction, tokenize, generate bằng ViT5/LoRA, rồi qua hybrid decoder để ổn định output.

### 11.3. Transformer/ViT5: 2 phút

> Em chọn ViT5 vì đây là T5 được pre-train cho tiếng Việt. Transformer encoder đọc toàn bộ input bằng self-attention, decoder sinh output từng token. Cross-attention giúp decoder nhìn lại encoder output để bám nội dung gốc. Đây là kiến trúc phù hợp cho abstractive summarization.

### 11.4. Training: 3 phút

> Phase 1 fine-tune ViT5 trên VietNews để học summarization tổng quát. Phase 2 full fine-tune dùng synthetic paired data để học mode controllability nhưng bị giới hạn do dữ liệu ít. Vì vậy em thử LoRA mixed-data: đóng băng Phase 1 backbone và chỉ train adapter nhỏ khoảng 0.39% tham số. LoRA giúp adaptation nhẹ hơn và giảm nguy cơ quên kiến thức cũ.

### 11.5. Inference và hybrid decoder: 2 phút

> Model neural đôi khi sinh output chưa ổn định về format, nhất là action_items và study_notes. Vì vậy em thêm hybrid decoder gồm extractor, critic và renderer. Với action_items, hệ thống ưu tiên trích từ source để giảm hallucination. Với study_notes, renderer ép đủ bốn trường: khái niệm chính, cần nhớ, ví dụ, lỗi dễ nhầm.

### 11.6. Evaluation: 2 phút

> Em dùng ROUGE để đánh giá summarization, nhưng ROUGE không đủ cho controllability. Kết quả Phase 1 ROUGE-L là 0.3436, Phase 2 là 0.3336, LoRA là 0.3411. Điều này cho thấy Phase 1 ổn định cho news summarization, còn LoRA giữ chất lượng gần Phase 1 hơn full fine-tune. Với action_items và study_notes, cần thêm error analysis và human rubric.

### 11.7. Hạn chế và kết luận: 1.5 phút

> Hạn chế chính là data theo mode còn ít, context length 512, action_items cần factuality cao, và Heuristic Quality Estimate không phải confidence thật. Hướng phát triển là mở rộng paired data, human evaluation, adapter riêng theo mode, evidence highlighting và chunking cho tài liệu dài. Kết luận là hệ thống đã chạy end-to-end, có so sánh nhiều hướng model, và LoRA + hybrid decoder là hướng hợp lý nhất trong phạm vi dữ liệu hiện tại.

---

## 12. Câu Hỏi Thầy Có Thể Hỏi Và Cách Trả Lời

### Câu 1. Vì sao Phase 2 không tăng ROUGE?

Vì ROUGE đang đo overlap với reference news summary, trong khi Phase 2 học thêm output style từ synthetic data. Mục tiêu Phase 2 không phải tối ưu riêng VietNews ROUGE, mà là controllability. Vì vậy cần đánh giá thêm mode adherence và factuality.

### Câu 2. Vì sao dùng LoRA?

Vì dữ liệu mode không lớn. Full fine-tune toàn bộ model dễ overfit hoặc làm giảm năng lực tổng quát từ Phase 1. LoRA đóng băng backbone và chỉ train adapter nhỏ, trong project là khoảng 0.39% tổng tham số.

### Câu 3. Hybrid decoder có làm mất tính abstractive không?

Không hoàn toàn. Model vẫn sinh neural draft bằng abstractive generation. Hybrid decoder chỉ là lớp inference-time reliability để ổn định format và giảm lỗi cấu trúc. Với action_items, ưu tiên factuality nên source-first extraction là hợp lý.

### Câu 4. Quality Estimate có phải confidence không?

Không. Đây là heuristic quality estimate, không phải xác suất đúng. Nó chỉ dựa trên các tín hiệu như độ dài, repetition, keyword coverage và generation score nếu có.

### Câu 5. Hệ thống có production-ready chưa?

Chưa. Hiện tại đây là modular demo gồm Streamlit và FastAPI. Production cần thêm monitoring, logging, model versioning, scaling, security, rate limiting và human evaluation nghiêm hơn.

---

## 13. Kết Luận Cuối Cùng

Đồ án đã xây dựng được một hệ thống Vietnamese controllable summarization chạy end-to-end. Phase 1 tạo nền tảng tóm tắt tiếng Việt tổng quát. Phase 2 full fine-tune là baseline để học controllability từ synthetic data. LoRA mixed-data là hướng cân bằng hơn khi dữ liệu mode còn ít, vì giữ lại Phase 1 backbone và chỉ train adapter nhỏ. Hybrid neuro-symbolic decoder giúp output ổn định hơn khi serving, đặc biệt với các mode có cấu trúc như `action_items` và `study_notes`.

Kết luận nên nói khi bảo vệ:

> Hệ thống chưa hoàn hảo và chưa production-ready, nhưng đã thể hiện đầy đủ tư duy AI engineering: có training pipeline, evaluation pipeline, serving architecture, nhiều hướng model để so sánh, error analysis và cách xử lý thực tế cho output structured. Đây là nền tảng tốt để tiếp tục mở rộng bằng dữ liệu tốt hơn, human evaluation và adapter theo mode.

