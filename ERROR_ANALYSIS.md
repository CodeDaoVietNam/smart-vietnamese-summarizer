# Error Analysis: Vietnamese Controllable Summarizer

Tài liệu này phân tích các nhóm lỗi chính của hệ thống tóm tắt tiếng Việt có điều khiển mode. Mục tiêu không chỉ là liệt kê lỗi, mà còn giải thích nguyên nhân kỹ thuật, tác động đến người dùng, mức độ nghiêm trọng và hướng cải thiện phù hợp với kiến trúc hiện tại.

## 1. Tóm tắt kết quả hiện tại

Hệ thống hiện có ba hướng mô hình chính:

| System | Mục tiêu | ROUGE-1 | ROUGE-2 | ROUGE-L | Nhận xét |
|---|---|---:|---:|---:|---|
| Phase 1 ViT5 | Tóm tắt tiếng Việt tổng quát trên VietNews | 0.5496 | 0.2294 | 0.3436 | Backbone ổn định nhất cho news summarization |
| Phase 2 full fine-tune | Học controllability từ synthetic data | 0.5490 | 0.2219 | 0.3336 | Có học mode nhưng ROUGE news giảm nhẹ |
| LoRA mixed-data | Giữ backbone Phase 1 và học adaptation nhẹ | 0.5482 | 0.2330 | 0.3411 | Cân bằng tốt hơn giữa summarization và controllability |
| LoRA mixed holdout | Đánh giá trên holdout mixed data | 0.5695 | 0.2970 | 0.3913 | Tốt trên phân phối gần LoRA mixed data |

Kết luận chính:

- Phase 1 vẫn là nền tảng tốt cho tóm tắt tin tức tổng quát.
- Phase 2 full fine-tune không vượt Phase 1 về ROUGE, điều này hợp lý vì mục tiêu Phase 2 là điều khiển style, không tối ưu riêng news summarization.
- LoRA mixed-data là hướng cân bằng hơn vì giữ được năng lực tổng quát gần Phase 1, đồng thời học thêm hành vi theo mode.
- ROUGE không đủ để đánh giá `action_items` và `study_notes`; cần đánh giá thủ công theo rubric.
- Tầng hybrid neuro-symbolic decoder đã cải thiện format output, nhưng không thay thế được factuality evaluation.

## 2. Phân loại lỗi chính

### 2.1. Lỗi thiếu ý quan trọng

Biểu hiện:

- Summary chỉ giữ một phần nội dung chính.
- Một số thực thể, số liệu, deadline hoặc kết luận quan trọng bị bỏ qua.
- Với văn bản dài, thông tin ở cuối dễ bị mất do input bị truncate.

Ví dụ thường gặp:

```text
Source có nhiều sự kiện hoặc nhiều nhiệm vụ.
Output chỉ nhắc 1-2 ý đầu, bỏ các ý sau.
```

Nguyên nhân:

- Mô hình encoder-decoder bị giới hạn context length.
- Beam search có xu hướng chọn câu an toàn, gần phần đầu văn bản.
- Dữ liệu train Phase 1 là news summarization, thường ưu tiên lead bias.
- Với meeting/project notes, thông tin quan trọng có thể nằm ở giữa hoặc cuối văn bản.

Tác động:

- Với `concise`, summary có thể vẫn tự nhiên nhưng chưa đủ thông tin.
- Với `action_items`, bỏ sót task là lỗi nghiêm trọng vì người dùng cần danh sách việc cần làm đầy đủ.
- Với report/demo, lỗi này dễ bị phát hiện nếu input có nhiều người phụ trách.

Mức độ nghiêm trọng: Cao với `action_items`, Trung bình với `concise` và `bullet`.

Hướng khắc phục:

- Dùng hybrid extractor source-first cho `action_items`.
- Với văn bản dài, thêm chunking hoặc hierarchical summarization.
- Tạo holdout set có các task nằm ở cuối văn bản để test lỗi lead bias.
- Với meeting notes, ưu tiên câu chứa trigger như `phụ trách`, `sẽ`, `cần`, `deadline`, `trước`.

### 2.2. Lỗi hallucination hoặc suy diễn quá mức

Biểu hiện:

- Output thêm thông tin không có trong source.
- Gán sai người phụ trách hoặc deadline.
- Diễn giải một nhận xét thành một task dù source không giao việc rõ ràng.

Ví dụ:

```text
Input chỉ nói: "Nhóm thống nhất không thêm chức năng lưu lịch sử."
Output action_items có thể diễn thành một task với owner không rõ.
```

Nguyên nhân:

- Mô hình sinh ngôn ngữ theo xác suất, không có ràng buộc factuality cứng.
- `action_items` là structured information extraction, không phải summarization thuần.
- Dữ liệu synthetic nhỏ khiến model học format tốt hơn học factual constraint.
- Prompt prefix không đủ để buộc model không bịa owner/deadline.

Tác động:

- Rất nghiêm trọng với `action_items` vì có thể tạo việc không tồn tại.
- Gây mất tin cậy nếu output được dùng trong workflow thật.
- Cần human review trước khi dùng cho quyết định quan trọng.

Mức độ nghiêm trọng: Cao.

Hướng khắc phục:

- Không để model tự quyết định hoàn toàn action item.
- Dùng extractor lấy owner/action/deadline từ source trước.
- Renderer chỉ hiển thị item có evidence trong source.
- Thêm critic cảnh báo `missing_owner`, `missing_deadline`, `context_only_item`.
- Future work: train NER/sequence labeling cho owner/action/deadline.

### 2.3. Lỗi không đúng mode

Biểu hiện:

- `concise` đôi khi giống bullet hoặc quá dài.
- `bullet` đôi khi chỉ là một câu dài có dấu gạch đầu dòng.
- `action_items` đôi khi giống summary thường.
- `study_notes` đôi khi giống bullet summary, chưa giống ghi chú học tập.

Nguyên nhân:

- Mode conditioning bằng prefix còn yếu.
- Phase 2 synthetic data vẫn nhỏ so với Phase 1 VietNews.
- Một LoRA adapter chung phải học nhiều hành vi cùng lúc.
- Một số mode có bản chất gần nhau, đặc biệt `bullet` và `study_notes`.

Tác động:

- Người dùng không thấy rõ giá trị của controllable summarization.
- Compare-all-modes kém thuyết phục nếu các output quá giống nhau.
- Report không nên claim rằng model đã hiểu hoàn toàn từng mode.

Mức độ nghiêm trọng: Trung bình đến cao.

Hướng khắc phục:

- Duy trì mode-specific renderer để đảm bảo format tối thiểu.
- Tăng paired dataset: cùng input có đủ 4 target khác nhau.
- Chấm holdout theo `mode_adherence`.
- Future work: adapter riêng cho `action_items` và `study_notes`.

### 2.4. Lỗi format không ổn định

Biểu hiện:

- Action item dính nhiều field trên cùng một dòng.
- Thiếu nhãn `Người phụ trách`, `Hành động`, `Deadline`.
- Study notes thiếu một trong bốn trường.
- Output copy một phần prompt như `Nêu khái niệm chính...` hoặc `Độ dài...`.

Nguyên nhân:

- Decoder không bị ràng buộc bởi schema.
- Target synthetic chưa đủ nhiều để model học format chắc chắn.
- Beam search tối ưu xác suất chuỗi, không tối ưu tính hợp lệ cấu trúc.
- Prompt leakage có thể xảy ra nếu instruction quá giống target text.

Tác động:

- UI khó đọc.
- Demo dễ bị mất điểm dù nội dung có thể đúng một phần.
- Downstream processing khó sử dụng nếu output không có cấu trúc ổn định.

Mức độ nghiêm trọng: Trung bình.

Hướng khắc phục đã triển khai:

- Thêm hybrid neuro-symbolic decoder.
- Tách extractor, critic và renderer.
- Renderer ép format `action_items`:

```text
- Người phụ trách: ...
  Hành động: ...
  Deadline: ...
```

- Renderer ép format `study_notes`:

```text
Khái niệm chính: ...
Cần nhớ: ...
Ví dụ: ...
Lỗi dễ nhầm: ...
```

Hướng cải thiện thêm:

- Expose critic report ở chế độ debug.
- Highlight evidence sentence trong UI.
- Train structured-output targets dạng JSON-like.

### 2.5. Lỗi trùng lặp và lan man

Biểu hiện:

- Một ý xuất hiện nhiều lần bằng cách diễn đạt hơi khác.
- Summary dài nhưng không thêm thông tin mới.
- Bullet có hai dòng gần nghĩa nhau.

Nguyên nhân:

- Beam search và repetition penalty chưa triệt tiêu hoàn toàn near-duplicate.
- Source có nhiều câu cùng chủ đề.
- Model thường diễn đạt lại ý chính theo nhiều cách.

Tác động:

- Output kém gọn.
- `bullet` mất chất lượng vì các bullet không độc lập.
- `Heuristic Quality Estimate` có thể giảm nếu repetition ratio cao.

Mức độ nghiêm trọng: Trung bình.

Hướng khắc phục đã triển khai:

- Dedupe exact.
- Dedupe near-duplicate bằng word overlap.
- Giới hạn số câu/items theo `length`.

Hướng cải thiện thêm:

- Dùng sentence embedding similarity để dedupe tốt hơn.
- Thêm reranker chọn bullet đa dạng nhất.

### 2.6. Lỗi quá ngắn hoặc quá dài

Biểu hiện:

- Output quá ngắn, chỉ một câu và thiếu ngữ cảnh.
- Output quá dài, gần như copy nhiều phần source.
- `short`, `medium`, `long` khác nhau chưa đều ở mọi mode.

Nguyên nhân:

- `max_new_tokens` chỉ là giới hạn trên, không đảm bảo model dùng hết budget.
- Nếu ép `min_new_tokens`, model dễ sinh thừa hoặc copy prompt.
- Formatter giới hạn output theo số câu/items, nhưng nội dung từng item vẫn có thể dài.

Tác động:

- Người dùng khó kiểm soát độ dài thực tế.
- Với `action_items`, length nên hiểu là số lượng item, không chỉ token.
- Với `study_notes`, length nên hiểu là độ chi tiết từng field.

Mức độ nghiêm trọng: Trung bình.

Hướng khắc phục:

- Không ép `min_new_tokens` quá cao.
- Dùng renderer kiểm soát số câu/items.
- UI giải thích `length behavior` cho người dùng.
- Future work: length-aware training targets.

## 3. Phân tích theo từng mode

### 3.1. Concise

Điểm mạnh:

- Tự nhiên nhất vì gần với dữ liệu VietNews.
- Phase 1 và LoRA đều giữ chất lượng khá ổn.
- Phù hợp với article/general documents.

Lỗi thường gặp:

- Bỏ sót chi tiết phụ quan trọng.
- Đôi khi chọn câu đầu thay vì tổng hợp toàn văn.
- Với input meeting, có thể tóm tắt bối cảnh nhưng bỏ task.

Khuyến nghị:

- Dùng `concise` cho article hoặc project update tổng quát.
- Không dùng `concise` khi mục tiêu là lấy đầy đủ việc cần làm.

### 3.2. Bullet

Điểm mạnh:

- Dễ đọc trong UI.
- Phù hợp với project update, article, lecture.
- Renderer giúp format ổn định.

Lỗi thường gặp:

- Một bullet chứa quá nhiều ý.
- Một số bullet gần nghĩa nhau.
- Đôi khi bullet giống `study_notes` nếu input là lecture.

Khuyến nghị:

- Chấm riêng `format_correctness` và `usefulness`.
- Tiếp tục cải thiện dedupe/reranking.

### 3.3. Action Items

Điểm mạnh:

- Sau khi thêm hybrid decoder, format ổn định hơn rõ rệt.
- Source-first extractor giúp giảm hallucination từ draft.
- Demo tốt với meeting có owner/deadline rõ.

Lỗi thường gặp:

- Owner hoặc deadline không rõ thì phải fallback `Chưa rõ`.
- Câu có nhiều người và nhiều hành động phức tạp vẫn khó parse.
- Nếu source không có task thật, model có thể muốn biến summary thành task; extractor cần chặn.

Khuyến nghị:

- Đây là mode cần human review nhất.
- Nên hiển thị evidence sentence trong tương lai.
- Nên phát triển thành bài toán information extraction có nhãn riêng.

### 3.4. Study Notes

Điểm mạnh:

- Format bốn trường giúp output dễ học và dễ trình bày.
- Phù hợp với lecture/study material.
- Extractor có thể bắt các câu chứa `Ví dụ` và `Lỗi dễ nhầm`.

Lỗi thường gặp:

- Nội dung đôi khi chỉ là câu source được đưa vào field, chưa khái quát sâu.
- `Cần nhớ` và `Khái niệm chính` đôi khi gần nhau.
- Nếu source không có ví dụ rõ, field `Ví dụ` có thể yếu.

Khuyến nghị:

- Cần thêm dữ liệu study-note chất lượng cao.
- Có thể train adapter riêng cho study notes.
- Có thể dùng critic kiểm tra field nào quá giống nhau.

## 4. Phân tích theo hệ thống/model

### 4.1. Phase 1 ViT5

Ưu điểm:

- Tốt nhất cho summarization tổng quát.
- ROUGE-L cao nhất hoặc gần cao nhất trên VietNews.
- Ít bị overfit vào synthetic template.

Nhược điểm:

- Không được train trực tiếp cho nhiều output modes.
- `action_items` và `study_notes` không ổn nếu chỉ dùng Phase 1.

Kết luận:

Phase 1 nên được giữ làm backbone chính.

### 4.2. Phase 2 Full Fine-tune

Ưu điểm:

- Có học thêm prefix/mode behavior.
- Hữu ích như baseline cho controllability.

Nhược điểm:

- Synthetic data nhỏ nên dễ học format bề mặt.
- ROUGE-L trên news giảm nhẹ so với Phase 1.
- Có nguy cơ catastrophic forgetting.

Kết luận:

Phase 2 full fine-tune là baseline quan trọng, nhưng không phải lựa chọn final tốt nhất nếu chỉ có ít synthetic data.

### 4.3. LoRA Mixed-data

Ưu điểm:

- Giữ Phase 1 backbone frozen.
- Train ít tham số hơn.
- ROUGE gần Phase 1 hơn Phase 2 full fine-tune.
- Phù hợp khi dữ liệu controllability còn ít.

Nhược điểm:

- Một adapter chung vẫn phải học nhiều mode khác nhau.
- Study notes chỉ có ít dữ liệu so với concise/bullet/action.
- Không tự giải quyết hoàn toàn structured extraction.

Kết luận:

LoRA mixed-data là hướng final hợp lý nhất hiện tại, kết hợp với hybrid decoder để cải thiện output serving.

## 5. Hạn chế của đánh giá hiện tại

Hiện tại hệ thống có ROUGE và mode comparison report, nhưng manual rubric summary vẫn chưa được chấm:

```text
reports/metrics/holdout_rubric_summary.md
scored_rows = 0 cho cả 4 mode
```

Điều này nghĩa là:

- Chưa có bằng chứng định lượng thủ công cho mode adherence.
- Chưa có số factuality/usefulness/format correctness đáng tin cậy.
- Report nên ghi rõ phần qualitative evaluation dựa trên observed examples, không claim là human evaluation hoàn chỉnh nếu rubric chưa được chấm.

Khuyến nghị trước khi nộp:

- Chấm ít nhất 20 examples, mỗi example đủ 4 mode.
- Mỗi output chấm 5 tiêu chí, mỗi tiêu chí 0-2.
- Tạo bảng trung bình theo mode.

Rubric đề xuất:

| Tiêu chí | 0 điểm | 1 điểm | 2 điểm |
|---|---|---|---|
| Mode adherence | Sai mode | Đúng một phần | Đúng rõ mode |
| Factuality | Bịa/sai thông tin | Có lỗi nhỏ | Đúng với source |
| Usefulness | Ít hữu ích | Hữu ích một phần | Hữu ích rõ ràng |
| Format correctness | Sai format | Format chưa đều | Format đúng |
| Conciseness | Quá dài/ngắn | Chấp nhận được | Gọn và đủ |

## 6. Root cause tổng hợp

Các lỗi của hệ thống đến từ bốn nhóm nguyên nhân chính:

1. Data mismatch.
   Phase 1 học news summarization, trong khi app phục vụ meeting, lecture, project update và study material.

2. Data scale.
   Dữ liệu synthetic/curated cho mode còn ít, đặc biệt với `study_notes`.

3. Task mismatch.
   `action_items` là structured extraction, không phải summarization thuần.

4. Evaluation mismatch.
   ROUGE đo overlap với reference, nhưng không đo tốt controllability, factuality hoặc format correctness.

## 7. Các cải tiến đã thực hiện sau error analysis

Hệ thống đã được cải thiện theo các hướng sau:

- Thêm LoRA mixed-data thay vì chỉ full fine-tune synthetic.
- Thêm structured formatter, sau đó refactor thành hybrid neuro-symbolic decoder.
- Tách extractor, critic và renderer để output theo mode ổn định hơn.
- Source-first action item extraction để giảm hallucination.
- Study-note renderer đảm bảo đủ bốn trường.
- Đổi `Quality Estimate` thành `Heuristic Quality Estimate` để tránh hiểu nhầm.
- Thêm demo samples và UI playbook.
- Thêm test regression cho formatter/extractor.

## 8. Hướng cải thiện tiếp theo theo mức ưu tiên

Ưu tiên 1: hoàn thiện evaluation.

- Chấm holdout rubric.
- So sánh Phase 1, Phase 2, LoRA và LoRA + hybrid decoder.
- Chọn 5 good cases và 5 failure cases cho report.

Ưu tiên 2: tăng chất lượng structured extraction.

- Thêm evidence sentence cho mỗi action item.
- Highlight evidence trong UI.
- Thêm parser tốt hơn cho câu có nhiều owner.
- Tách rõ task thật và câu quyết định chung.

Ưu tiên 3: cải thiện data.

- Tạo thêm action-item targets có owner/action/deadline rõ.
- Tạo study-note targets chất lượng cao hơn.
- Dùng JSON-like structured target để model học schema dễ hơn.

Ưu tiên 4: cải thiện long document handling.

- Chunking.
- Hierarchical summarization.
- Ưu tiên chunk chứa action triggers hoặc lecture cues.

Ưu tiên 5: tối ưu inference.

- Batch evaluation.
- Caching.
- Quantization hoặc giảm beam cho demo CPU.

## 9. Cách trình bày trong báo cáo

Không nên viết:

```text
Mô hình đã tối ưu hoàn toàn cho bốn mode.
```

Nên viết:

```text
Kết quả cho thấy Phase 1 giữ năng lực tóm tắt tiếng Việt tổng quát tốt, trong khi LoRA mixed-data giúp bổ sung khả năng điều khiển output style mà không làm giảm mạnh chất lượng tổng quát. Tuy nhiên, các mode có cấu trúc cao như action_items và study_notes vẫn còn phụ thuộc vào chất lượng dữ liệu và cần thêm tầng kiểm soát ở inference time. Vì vậy, hệ thống bổ sung hybrid neuro-symbolic decoder gồm extractor, critic và renderer để tăng độ ổn định về format và giảm lỗi hallucination trong các trường có cấu trúc.
```

Claim final nên dùng:

```text
Đồ án đã xây dựng được một pipeline Vietnamese controllable summarization hoàn chỉnh, có huấn luyện, đánh giá, triển khai web demo và tầng inference reliability. Hệ thống chưa giải quyết triệt để factuality và controllability, nhưng đã chứng minh hướng kết hợp ViT5/LoRA với hybrid neuro-symbolic decoding là phù hợp khi dữ liệu mode còn hạn chế.
```
