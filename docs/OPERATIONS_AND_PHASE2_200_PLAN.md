# Operations Architecture And Phase 2 200-Sample Plan

Tài liệu này mô tả cách vận hành đồ án sau khi mở rộng synthetic data Phase 2 lên 200 mẫu.

## 1. Kiến Trúc Vận Hành

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
  -> FastAPI /summarize
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
| `meeting_notes` | 70 |
| `lecture_notes` | 70 |
| `project_updates` | 30 |
| `study_materials` | 30 |

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
  --length medium
```

## 4. Acceptance Criteria

- `concise`: đoạn văn ngắn, tự nhiên, không bullet.
- `bullet`: danh sách ý chính, mỗi dòng một ý.
- `action_items`: ưu tiên việc cần làm, người phụ trách, deadline nếu có.
- `study_notes`: ghi chú học tập, nhấn mạnh khái niệm và lỗi dễ nhầm.
- Cùng một input không được cho 4 output gần như giống nhau rồi chỉ đổi format.

## 5. Experiment Notes

Không gộp Phase 1 và Phase 2 trong lần chạy chính. Nếu cần experiment phụ sau baseline 200 mẫu, có thể thử mixed replay ở Phase 2 với khoảng 70-80% synthetic và 20-30% VietNews concise để giảm nguy cơ quên năng lực summarization gốc.
