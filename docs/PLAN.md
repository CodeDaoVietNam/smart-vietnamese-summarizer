# Smart Meeting & Study Notes Summarization System

## 1. Tóm Tắt Và Động Lực

Xây dựng một web application tóm tắt văn bản tiếng Việt dựa trên Transformer. Người dùng nhập meeting notes, lecture notes, transcript, bài học hoặc tài liệu dài; hệ thống sinh ra nhiều dạng output như tóm tắt ngắn, bullet points, action items và study notes.

Động lực chính: sinh viên, nhóm học tập và nhóm làm việc thường có rất nhiều nội dung dài nhưng cần nhanh chóng nắm ý chính, việc cần làm và khái niệm quan trọng. Project này biến bài toán `text summarization` thành một sản phẩm AI có điều khiển: cùng một input nhưng output thay đổi theo mục đích người dùng.

Lý do phù hợp với đề:
- Có Transformer thật: dùng mô hình encoder-decoder kiểu T5.
- Có fine-tuning thật: huấn luyện lại trên dataset summarization tiếng Việt.
- Có evaluation thật: dùng ROUGE và phân tích lỗi.
- Có web app thật: Streamlit app có input, mode selector, length selector, keyword highlight, inference time và confidence proxy.
- Có tính sáng tạo: controllable summarization thay vì chỉ tóm tắt một kiểu cố định.

## 2. Bài Toán, Input, Output

Tên bài toán: `Vietnamese Controllable Abstractive Summarization`.

Input chính:
- Một đoạn văn bản tiếng Việt dài, ví dụ meeting notes, lecture notes, transcript, bài báo, tài liệu học tập.
- Tham số điều khiển gồm `mode` và `length`.
- `mode`: `concise`, `bullet`, `action_items`, `study_notes`.
- `length`: `short`, `medium`, `long`.

Output chính:
- `concise`: đoạn tóm tắt 3-5 câu.
- `bullet`: các ý chính dạng gạch đầu dòng.
- `action_items`: việc cần làm, người phụ trách nếu có, deadline nếu có.
- `study_notes`: khái niệm chính, ý quan trọng, ghi chú ôn tập.
- Metadata hiển thị trên web: số token input, thời gian inference, confidence proxy, keyword/entity highlight.

Giải thích `why`:
- Transformer phù hợp vì self-attention giúp mô hình hiểu quan hệ xa trong văn bản dài.
- T5/ViT5 phù hợp vì bài toán là text-to-text: input là văn bản, output cũng là văn bản.
- Controllable generation làm project giống một sản phẩm AI thực tế hơn notebook summarization thông thường.

## 3. Mô Hình, Dataset Và Evaluation

Mô hình chính:
- Dùng `VietAI/vit5-base` làm base model.
- Lý do: ViT5 là mô hình Transformer text-to-text cho tiếng Việt, phù hợp hơn FLAN-T5 khi yêu cầu là Vietnamese input.
- Cấu hình Colab T4: fp16, max input length 512, max target length 128, batch size 2, gradient accumulation 8, 3 epochs.

Dataset chính:
- Dùng `WikiLingua Vietnamese` làm dataset chính vì có cặp article-summary tiếng Việt, hợp với study/how-to notes.
- Split nếu dataset chỉ có train: `80% train`, `10% validation`, `10% test`, seed `42`.
- Dataset dự phòng nếu WikiLingua khó xử lý: `VietNews-Abs-Sum` hoặc `harouzie/vietnews`, phù hợp summarization tiếng Việt và dễ benchmark.

Controlled training format:
- Convert mỗi sample thành instruction-style input.
- Ví dụ input training: `tom tat ngan: {document}`.
- Target: summary gốc từ dataset.
- Với `bullet`, `study_notes`, `action_items`, v1 sẽ dùng cùng model đã fine-tune và điều khiển bằng prefix + post-processing format.
- Không claim action-items là được supervised bằng dataset riêng; báo cáo sẽ nói rõ đây là controllable inference layer trên summarization model.

Metrics:
- Main metrics: `ROUGE-1`, `ROUGE-2`, `ROUGE-L`.
- Optional: `BERTScore` nếu còn thời gian.
- Extra product metrics: latency trung bình, số token input tối đa xử lý được, qualitative examples.

Error analysis bắt buộc:
- Hallucination: model thêm thông tin không có trong input.
- Missing key points: bỏ sót ý quan trọng.
- Entity error: sai tên người, tổ chức, thời gian.
- Repetition: lặp ý.
- Long input failure: input quá dài bị truncation.

## 4. Kiến Trúc Và Vận Hành

Pipeline hệ thống:
```text
User Text + Mode + Length
-> Preprocessing
-> Tokenizer
-> Fine-tuned ViT5
-> Controlled Generation
-> Post-processing
-> Keyword/Entity Highlight
-> Streamlit Web UI
```

Các module chính:
- `data`: load dataset, clean text, split train/validation/test.
- `training`: tokenize, fine-tune ViT5, save checkpoint.
- `evaluation`: generate prediction trên test set, tính ROUGE, lưu bảng kết quả.
- `inference`: hàm `generate_summary(text, mode, length)`.
- `web_app`: Streamlit UI cho demo.

Interface nội bộ:
```python
generate_summary(
    text: str,
    mode: Literal["concise", "bullet", "action_items", "study_notes"],
    length: Literal["short", "medium", "long"]
) -> {
    "summary": str,
    "keywords": list[str],
    "confidence": float,
    "latency_ms": int,
    "input_tokens": int
}
```

Confidence:
- Dùng proxy từ average generation score nếu lấy được `output_scores`.
- Nếu không ổn định, dùng heuristic kết hợp length ratio, repetition penalty và keyword coverage.
- Trong báo cáo ghi rõ đây là `confidence estimate`, không phải xác suất đúng tuyệt đối.

Web app:
- Text area cho input.
- Selector cho mode.
- Slider hoặc segmented control cho length.
- Nút summarize.
- Layout `Original Text | Generated Output`.
- Hiển thị highlighted keywords, inference time, input tokens, confidence estimate.
- Có 2-3 sample tiếng Việt để demo nhanh.

## 5. Kế Hoạch 4 Tuần

Tuần 1: Chuẩn bị và baseline
- Chốt problem statement, input/output, scope.
- Tải dataset WikiLingua Vietnamese.
- Làm preprocessing: remove empty samples, normalize whitespace, lọc văn bản quá ngắn/quá dài.
- Chia train/validation/test.
- Chạy baseline inference bằng `VietAI/vit5-base-vietnews-summarization` hoặc base ViT5 trước fine-tune để có ví dụ so sánh.

Tuần 2: Fine-tuning model
- Fine-tune `VietAI/vit5-base` bằng `Seq2SeqTrainer`.
- Hyperparameters mặc định: learning rate `2e-5`, epochs `3`, batch size `2`, gradient accumulation `8`, fp16 `true`.
- Nếu Colab T4 OOM: giảm batch size về `1`, tăng gradient accumulation lên `16`, giảm input length về `384`.
- Save best checkpoint theo validation loss.
- Lưu tokenizer, config, training logs và model weights.

Tuần 3: Evaluation và controllable output
- Generate summaries trên test set.
- Tính ROUGE-1/2/L.
- Làm bảng so sánh baseline vs fine-tuned model.
- Thêm controlled prompts cho 4 mode.
- Thêm post-processing cho bullet, action items, study notes.
- Viết error analysis với ít nhất 5 ví dụ tốt và 5 ví dụ lỗi.

Tuần 4: Web app, báo cáo và demo
- Xây Streamlit app.
- Thêm sample inputs, before/after layout, keyword highlight, inference time, confidence estimate.
- Hoàn thiện báo cáo theo cấu trúc: Introduction, Dataset, Model, Training, Evaluation, Web App, Error Analysis, Limitations, Future Work.
- Chuẩn bị slide demo 5-7 phút.
- Kiểm tra lại toàn bộ yêu cầu nộp: source code, dataset hoặc link Drive, weights, report.

## 6. Acceptance Criteria

Project được xem là hoàn thành khi:
- Fine-tune được ít nhất 1 mô hình Transformer trên dataset summarization tiếng Việt.
- Có file kết quả evaluation gồm ROUGE-1, ROUGE-2, ROUGE-L trên test set.
- Có web app chạy được với input tiếng Việt.
- Web app hỗ trợ đủ 4 mode: concise, bullet, action_items, study_notes.
- Báo cáo giải thích rõ input, output, why, dataset, model, split, metric, kết quả, lỗi và giới hạn.
- Demo có ít nhất 3 ví dụ: meeting notes, lecture notes, article/study document.

## 7. Assumptions Và Nguồn Tham Khảo

Assumptions:
- Tài nguyên huấn luyện là Google Colab T4.
- Project ưu tiên Vietnamese input.
- Timeline là 4 tuần.
- Streamlit được chọn để giảm thời gian frontend và tập trung vào NLP.
- `action_items` là tính năng controllable inference/post-processing, không phải supervised task riêng.

Nguồn tham khảo chính:
- ViT5 GitHub/VietAI: https://github.com/vietai/ViT5
- ViT5-base Hugging Face: https://huggingface.co/VietAI/vit5-base
- ViT5 VietNews summarization model: https://huggingface.co/VietAI/vit5-base-vietnews-summarization
- WikiLingua dataset: https://huggingface.co/datasets/esdurmus/wiki_lingua
- VietNews abstractive summarization dataset: https://huggingface.co/datasets/ithieund/VietNews-Abs-Sum
- FLAN-T5 reference nếu cần so sánh: https://huggingface.co/google/flan-t5-small


**Nguồn cần thêm**:

- HuggingFace evaluate library: Tính ROUGE — đây là tool chính cho evaluation	https://huggingface.co/docs/evaluate

- rouge-score PyPI: Backend cho evaluate khi tính ROUGE	https://pypi.org/project/rouge-score/

- Attention Is All You Need (paper gốc): Tham khảo bắt buộc trong báo cáo khi nói về Transformer	https://arxiv.org/abs/1706.03762

- ViT5 paper: Cite trong báo cáo khi giải thích model	https://arxiv.org/abs/2205.06457

- Vietnamese stopwords: Cần cho keyword extraction	https://github.com/stopwords/vietnamese-stopwords

- Streamlit docs: Tham khảo khi build app	https://docs.streamlit.io

- FastAPI docs: Tham khảo khi build API (theo PLAN v3)	https://fastapi.tiangolo.com
