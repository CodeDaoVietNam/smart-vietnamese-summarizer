# Operations Architecture And Phase 2 200-Sample Plan

Tài liệu này mô tả cách vận hành đồ án sau khi mở rộng synthetic data Phase 2 lên 200 mẫu.

## 1. Kiến Trúc Vận Hành

Đồ án hiện là **modular 2-service demo**, không phải hệ thống microservices phức tạp.

```text
Streamlit Frontend
  -> FastAPI Backend
  -> SmartSummarizer service layer
  -> ViT5 checkpoint + tokenizer
```

Các script như `prepare_data.py`, `train.py`, `train_phase2.py`, `evaluate.py`, `evaluate_modes.py` là offline batch jobs. Nếu trình bày theo microservice architecture, chỉ nên nói rằng frontend và backend là hai service deploy được tách riêng; training/evaluation là pipeline offline.

Đồ án có 3 luồng vận hành chính.

### Offline training

```text
VietNews raw dataset
  -> scripts/prepare_data.py
  -> data/processed/train.jsonl
  -> data/processed/validation.jsonl
  -> data/processed/test.jsonl
  -> scripts/train.py
  -> models/vit5-summarizer-v1

data/synthetic/reviewed_all.json
  -> scripts/generate_synthetic.py
  -> data/synthetic/train.jsonl
  -> data/synthetic/validation.jsonl
  -> scripts/train_phase2.py
  -> models/vit5-summarizer-v2
```

Phase 1 học kỹ năng tóm tắt tiếng Việt tổng quát từ VietNews. Phase 2 dùng synthetic multi-mode data để model hiểu prefix điều khiển output style.

### Offline evaluation

```text
models/vit5-summarizer-v2
  -> scripts/evaluate.py
  -> reports/metrics/eval_results.json
  -> reports/examples/test_predictions.jsonl

data/samples/qualitative_mode_eval.jsonl
  -> scripts/evaluate_modes.py
  -> reports/examples/mode_comparison_predictions.jsonl
```

ROUGE trên VietNews dùng để so sánh khả năng summarization chung. Mode comparison dùng để đánh giá controllability vì Phase 2 không nhất thiết phải thắng Phase 1 trên news ROUGE.

### Online serving

```text
User text + mode + length
  -> Streamlit app
  -> FastAPI /api/summarize hoặc /api/compare-modes
  -> SmartSummarizer.from_config(configs/app.yaml)
  -> build_instruction(mode prefix)
  -> tokenizer + ViT5 generate()
  -> postprocess_summary()
  -> keywords + quality_estimate + latency
  -> UI result
```

## 2. Synthetic Data Version

Dataset Phase 2 hiện được chuẩn hóa ở:

```text
data/synthetic/reviewed_all.json
data/synthetic/train.jsonl
data/synthetic/validation.jsonl
```

Phase 2 hiện dùng paired controllability dataset:

```text
50 base documents x 4 modes = 200 training rows
```

Cùng một `base_id` có chung `document` nhưng 4 target summary khác nhau cho `concise`, `bullet`, `action_items`, `study_notes`.

Phân bổ sau khi mở rộng:

| Mode | Train | Validation | Total |
|---|---:|---:|---:|
| `concise` | 40 | 10 | 50 |
| `bullet` | 40 | 10 | 50 |
| `action_items` | 40 | 10 | 50 |
| `study_notes` | 40 | 10 | 50 |
| **Total** | **160** | **40** | **200** |

Phân bổ domain:

| Domain | Total |
|---|---:|
| `meeting_notes` | 72 |
| `lecture_notes` | 72 |
| `project_updates` | 28 |
| `study_materials` | 28 |

Để tái tạo dataset 200 mẫu và split 80/20:

```bash
python scripts/build_phase2_synthetic_200.py
python scripts/generate_synthetic.py --input data/synthetic/reviewed_all.json
```

## 3. Training And Evaluation Commands

Train Phase 2 từ checkpoint Phase 1:

```bash
python scripts/train_phase2.py --config configs/train_phase2.yaml
```

Hyperparameters mặc định giữ nguyên cho lần chạy 200 mẫu đầu tiên:

```yaml
epochs: 2
learning_rate: 1.0e-5
gradient_accumulation_steps: 4
warmup_ratio: 0.15
```

Evaluate ROUGE:

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

Evaluate controllability across modes:

```bash
python scripts/evaluate_modes.py \
  --input data/samples/qualitative_mode_eval.jsonl \
  --config configs/app.yaml \
  --length medium \
  --markdown-output reports/examples/mode_comparison_report.md
```

## 4. Backend And Frontend Use Cases

Backend use cases:

- `GET /api/health`: kiểm tra API và model đã load.
- `POST /api/summarize`: sinh một output theo `text`, `mode`, `length`.
- `POST /api/compare-modes`: sinh đủ 4 output mode cho cùng một input để demo controllability.

Frontend use cases:

- Chọn sample meeting, lecture hoặc article.
- Dán văn bản tiếng Việt và chọn mode/length.
- Generate một summary.
- Compare all modes trên cùng input bằng `/api/compare-modes`.
- Hiển thị summary, keywords, quality estimate, token count và latency.
- Báo lỗi rõ khi backend offline hoặc input rỗng.

## 5. Acceptance Criteria

- `concise`: đoạn văn ngắn, tự nhiên, không bullet.
- `bullet`: danh sách ý chính, mỗi dòng một ý.
- `action_items`: ưu tiên việc cần làm, người phụ trách, deadline nếu có.
- `study_notes`: ghi chú học tập, nhấn mạnh khái niệm và lỗi dễ nhầm.
- Cùng một input không được cho 4 output gần như giống nhau rồi chỉ đổi format.

## 6. Experiment Notes

Không gộp Phase 1 và Phase 2 trong lần chạy chính. Nếu cần experiment phụ sau baseline 200 mẫu, có thể thử mixed replay ở Phase 2 với khoảng 70-80% synthetic và 20-30% VietNews concise để giảm nguy cơ quên năng lực summarization gốc.
