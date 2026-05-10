# Codebase Blueprint: Smart Vietnamese Summarization System

## Summary

Tạo một codebase Python chuyên nghiệp cho project `Smart Meeting & Study Notes Summarization System`, gồm 4 phần chính: data pipeline, fine-tuning, evaluation, và Streamlit web app.

Workspace hiện tại đang trống, nên scaffold sẽ là một project mới hoàn chỉnh, ưu tiên chạy tốt trên Google Colab T4 và local demo.

Stack mặc định:
- Python `3.10+`
- PyTorch + Hugging Face `transformers`
- `datasets`, `evaluate`, `rouge-score`
- `streamlit` cho web app
- `pytest` cho test
- `ruff` cho lint/format
- `VietAI/vit5-base` làm model chính

## Proposed Codebase Structure

```text
smart-vietnamese-summarizer/
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── configs/
│   ├── train_vit5_base.yaml
│   ├── eval.yaml
│   └── app.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│       ├── meeting_note_vi.txt
│       ├── lecture_note_vi.txt
│       └── article_vi.txt
├── models/
│   └── .gitkeep
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── examples/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_training_colab.ipynb
│   └── 03_error_analysis.ipynb
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── export_model.py
├── src/
│   └── smart_summarizer/
│       ├── __init__.py
│       ├── config.py
│       ├── constants.py
│       ├── data/
│       │   ├── dataset_loader.py
│       │   ├── preprocessing.py
│       │   └── collator.py
│       ├── modeling/
│       │   ├── model_loader.py
│       │   ├── trainer.py
│       │   └── generation.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   └── error_analysis.py
│       ├── product/
│       │   ├── summarizer.py
│       │   ├── keyword_extractor.py
│       │   ├── confidence.py
│       │   └── postprocess.py
│       └── utils/
│           ├── logging.py
│           ├── paths.py
│           └── seed.py
├── app/
│   ├── streamlit_app.py
│   ├── components.py
│   └── style.py
└── tests/
    ├── test_preprocessing.py
    ├── test_generation.py
    ├── test_postprocess.py
    └── test_confidence.py
```

## Key Modules And Responsibilities

`configs/train_vit5_base.yaml`
- Chứa toàn bộ training config: model name, dataset name, max lengths, batch size, epochs, learning rate, output directory.
- Default model: `VietAI/vit5-base`.
- Default dataset: `wiki_lingua`, language filter `vietnamese`.
- Colab T4 defaults: batch size `2`, gradient accumulation `8`, fp16 `true`.

`scripts/prepare_data.py`
- Load dataset từ Hugging Face.
- Lọc sample rỗng, quá ngắn, quá dài.
- Chuẩn hóa whitespace.
- Split train/validation/test nếu dataset chưa có split chuẩn.
- Save ra `data/processed/train.jsonl`, `validation.jsonl`, `test.jsonl`.

`scripts/train.py`
- Đọc config YAML.
- Load tokenizer/model.
- Tokenize dataset.
- Fine-tune bằng `Seq2SeqTrainer`.
- Save checkpoint vào `models/vit5-summarizer`.
- Save training logs vào `reports/metrics/training_log.json`.

`scripts/evaluate.py`
- Load model đã fine-tune.
- Generate summary trên test set.
- Tính `ROUGE-1`, `ROUGE-2`, `ROUGE-L`.
- Save metrics vào `reports/metrics/eval_results.json`.
- Save prediction examples vào `reports/examples/test_predictions.jsonl`.

`scripts/predict.py`
- CLI inference nhanh.
- Input: text file hoặc raw text.
- Output: summary theo mode và length.

`src/smart_summarizer/product/summarizer.py`
- API chính cho app.
- Interface bắt buộc:

```python
def generate_summary(
    text: str,
    mode: str = "concise",
    length: str = "medium",
) -> dict:
    ...
```

Return shape:

```python
{
    "summary": str,
    "keywords": list[str],
    "confidence": float,
    "latency_ms": int,
    "input_tokens": int,
    "mode": str,
    "length": str,
}
```

`src/smart_summarizer/modeling/generation.py`
- Map `mode + length` thành instruction prefix.
- Ví dụ:
  - `concise`: `tom tat ngan gon: {text}`
  - `bullet`: `tom tat thanh cac y chinh: {text}`
  - `action_items`: `trich xuat cac viec can lam: {text}`
  - `study_notes`: `tao ghi chu hoc tap: {text}`
- Điều khiển `max_new_tokens`, `num_beams`, `repetition_penalty`.

`src/smart_summarizer/product/postprocess.py`
- Format output theo mode.
- Với `bullet`, đảm bảo mỗi ý là một dòng bắt đầu bằng `-`.
- Với `action_items`, cố gắng chuẩn hóa thành các dòng việc cần làm.
- Với `study_notes`, chia thành các mục ngắn dễ đọc.

`src/smart_summarizer/product/keyword_extractor.py`
- V1 dùng heuristic nhẹ: lấy cụm danh từ/từ khóa dựa trên tần suất sau khi bỏ stopwords tiếng Việt.
- Không phụ thuộc model NER nặng để tránh phức tạp.
- Output dùng để highlight trên web app.

`src/smart_summarizer/product/confidence.py`
- Confidence proxy, không claim là xác suất đúng tuyệt đối.
- Ưu tiên lấy generation score nếu available.
- Fallback heuristic: keyword coverage, repetition ratio, output length sanity.
- Clamp về thang `0-100`.

`app/streamlit_app.py`
- Web UI chính.
- Có text area input, mode selector, length selector, button summarize.
- Có sample picker: meeting, lecture, article.
- Có layout 2 cột: original text và generated output.
- Hiển thị token count, latency, confidence estimate, keywords.
- Có tab `Compare Modes` để chạy cùng input qua 4 mode.

## CLI Commands

Cài môi trường:

```bash
pip install -e ".[dev]"
```

Chuẩn bị dữ liệu:

```bash
python scripts/prepare_data.py --config configs/train_vit5_base.yaml
```

Fine-tune:

```bash
python scripts/train.py --config configs/train_vit5_base.yaml
```

Evaluate:

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

Predict thử:

```bash
python scripts/predict.py --text-file data/samples/meeting_note_vi.txt --mode bullet --length medium
```

Chạy web app:

```bash
streamlit run app/streamlit_app.py
```

## Config Defaults

`train_vit5_base.yaml` nên có các giá trị mặc định:

```yaml
project_name: smart-vietnamese-summarizer
seed: 42

model:
  name: VietAI/vit5-base
  output_dir: models/vit5-summarizer

dataset:
  name: wiki_lingua
  language: vietnamese
  train_file: data/processed/train.jsonl
  validation_file: data/processed/validation.jsonl
  test_file: data/processed/test.jsonl

tokenization:
  max_source_length: 512
  max_target_length: 128

training:
  epochs: 3
  learning_rate: 2.0e-5
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 8
  fp16: true
  eval_strategy: epoch
  save_strategy: epoch
  predict_with_generate: true

generation:
  num_beams: 4
  repetition_penalty: 1.2
  no_repeat_ngram_size: 3
```

## Data Flow

Training flow:

```text
Hugging Face Dataset
-> prepare_data.py
-> processed JSONL
-> train.py
-> fine-tuned checkpoint
-> evaluate.py
-> ROUGE metrics + predictions
```

Application flow:

```text
Streamlit input
-> generate_summary()
-> preprocess
-> tokenizer
-> ViT5 checkpoint
-> controlled generation
-> postprocess
-> confidence + keyword extraction
-> UI result
```

## Testing Plan

Unit tests:
- `test_preprocessing.py`: empty text, whitespace normalization, long text truncation.
- `test_generation.py`: mode/length mapping creates correct generation parameters.
- `test_postprocess.py`: bullet/action/study output formatting is stable.
- `test_confidence.py`: confidence always returns number from `0` to `100`.

Integration tests:
- Load tokenizer and model config without running full training.
- Run inference with a tiny model mock or very short sample.
- Validate `generate_summary()` always returns required keys.

Manual acceptance tests:
- Meeting note input returns concise summary and action items.
- Lecture note input returns study notes.
- Long article input does not crash.
- Empty input shows friendly validation message.
- Web app displays latency, token count, confidence and keywords.

## Report Artifacts Produced By Codebase

The codebase must generate these files for báo cáo:
- `reports/metrics/eval_results.json`: ROUGE scores.
- `reports/metrics/training_log.json`: training loss, validation loss.
- `reports/examples/test_predictions.jsonl`: input, reference, prediction.
- `reports/examples/error_analysis.md`: hallucination, missing key points, repetition, entity errors.
- `reports/figures/`: optional charts for loss curve and ROUGE comparison.

## Professional Standards

Code style:
- Typed function signatures for public functions.
- No hard-coded paths inside core modules; paths come from config.
- Scripts are thin entrypoints; business logic lives in `src/`.
- Logging uses a shared logger.
- Random seed centralized in `utils/seed.py`.
- App imports from package, not from scripts.
- Dataset/model artifacts are not committed to git.

Git ignore:
- Ignore `data/raw`, `data/processed`, `models`, `.env`, cache folders, notebook checkpoints.
- Keep placeholder `.gitkeep` where needed.

README must include:
- Project overview.
- Problem definition: input/output.
- Installation.
- Dataset preparation.
- Training.
- Evaluation.
- Web app usage.
- Example screenshots.
- Limitations and future work.

## Assumptions

- Codebase sẽ được scaffold mới trong `/home/ductien/Documents/Transformer`.
- Vietnamese input là scope chính.
- Colab T4 là training target chính.
- Streamlit là web framework mặc định.
- V1 tập trung vào summarization; action items và study notes là controllable generation + post-processing, không phải task supervised riêng.
- Dataset chính là WikiLingua Vietnamese, fallback là VietNews nếu loading hoặc chất lượng không phù hợp.
