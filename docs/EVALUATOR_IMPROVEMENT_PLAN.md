# Evaluator Improvement Plan

## Summary

Project hiện đã chạy end-to-end: Phase 1, Phase 2, FastAPI, Streamlit và predict script đều có đường chạy rõ. Điểm cần cải thiện không nằm ở kiến trúc app mà nằm ở controllability của model, đặc biệt là `study_notes` và `action_items`.

Mục tiêu đánh giá final là trung thực:

- Phase 1 là nền tảng summarization tiếng Việt tổng quát.
- Phase 2 là adaptation để điều khiển output theo mode.
- Phase 2 không bắt buộc thắng Phase 1 trên ROUGE VietNews.
- Chọn checkpoint final bằng holdout rubric và demo quality, không chỉ bằng `eval_loss`.

## Evaluation Protocol

ROUGE chỉ chạy trên subset cố định để tránh tốn nhiều giờ trên Colab:

```yaml
dataset:
  max_samples: 200
```

Nếu còn thời gian, có thể tăng lên `500`, nhưng không chạy full `67,640` samples cho demo/report.

Holdout controllability dùng:

```bash
python scripts/evaluate_modes.py \
  --input data/samples/holdout_mode_eval.jsonl \
  --length medium \
  --output reports/examples/holdout_mode_comparison_predictions.jsonl \
  --markdown-output reports/examples/holdout_mode_comparison_report.md
```

Sau đó chấm thủ công vào:

```text
reports/examples/holdout_rubric_template.csv
```

Mỗi tiêu chí chấm `0-2`:

- `mode_adherence`
- `factuality`
- `usefulness`
- `format_correctness`
- `conciseness`

Tổng hợp điểm:

```bash
python scripts/summarize_holdout_rubric.py
```

## Experiment Selection

So sánh:

| model | test_subset | rouge1 | rouge2 | rougeL | mode_adherence | factuality | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| Phase 1 | 200 VietNews | điền sau eval | điền sau eval | điền sau eval | N/A | đọc mẫu | no |
| Phase 2 epoch 2 | 200 VietNews + holdout | điền sau eval | điền sau eval | điền sau eval | điền rubric | điền rubric | maybe |
| Phase 2 epoch 3 | 200 VietNews + holdout | điền sau eval | điền sau eval | điền sau eval | điền rubric | điền rubric | final nếu tốt hơn |

Train lại epoch 2 nếu cần baseline sạch:

```bash
python scripts/train_phase2.py --config configs/train_phase2_epoch2.yaml
```

Train experiment epoch 3:

```bash
python scripts/train_phase2.py --config configs/train_phase2.yaml
```

Evaluate epoch 2 baseline:

```bash
python scripts/evaluate.py --config configs/eval_phase2_epoch2.yaml
```

Evaluate epoch 3/final candidate:

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

Ưu tiên chọn checkpoint theo thứ tự:

1. Holdout mode adherence.
2. Factuality.
3. Format correctness.
4. Demo quality.
5. ROUGE không tụt quá mạnh so với Phase 1.

## Demo Guidance

Chọn mode hợp domain:

- Meeting notes: `action_items`, `bullet`.
- Lecture notes: `study_notes`.
- General article: `concise`, `bullet`.

Không dùng output bị copy prompt hoặc bịa người/deadline làm screenshot demo. Nếu `study_notes` sinh chưa tốt trên meeting input, ghi limitation rằng mode này phù hợp hơn với nội dung học tập.

## Final Acceptance

- `concise`: một đoạn văn ngắn, không bullet.
- `bullet`: 3-5 ý chính.
- `action_items`: có người phụ trách, hành động, deadline khi input có task.
- `study_notes`: đúng 4 dòng `Khái niệm chính`, `Cần nhớ`, `Ví dụ`, `Lỗi dễ nhầm`.
- App local chạy được với `/api/health`, `/api/summarize`, `/api/compare-modes`.
