# Smart Meeting & Study Notes Summarization System

A controllable Transformer-based Vietnamese summarization project for the final assignment **Transformer-based NLP Application**.

The system accepts Vietnamese meeting notes, lecture notes, articles, or study documents and generates one of four output styles:

- `concise`: short paragraph summary
- `bullet`: bullet-point summary
- `action_items`: tasks, follow-ups, and deadlines when present
- `study_notes`: key concepts and exam-friendly notes

## Why This Project

This project is more than a basic text summarizer. It demonstrates a practical AI product workflow:

- A real NLP task: abstractive summarization
- A real Transformer model: ViT5
- Real fine-tuning on Vietnamese summarization data
- Quantitative evaluation with ROUGE
- A FastAPI backend with a Streamlit frontend and controllable output modes

## Input And Output

Input:

```text
Vietnamese long-form text + output mode + summary length
```

Output:

```python
{
    "summary": "...",
    "keywords": ["..."],
    "quality_estimate": 87.5,
    "latency_ms": 1200,
    "input_tokens": 312,
    "mode": "bullet",
    "length": "medium",
}
```

## Project Structure

```text
configs/                 YAML configs for training, evaluation, and app
api/                     FastAPI serving layer
data/samples/            Demo Vietnamese inputs
data/synthetic/          Synthetic meeting and lecture data for phase 2
scripts/                 Thin CLI entrypoints
src/smart_summarizer/    Core package logic
app/                     Streamlit application
tests/                   Unit tests
reports/                 Metrics, predictions, figures, and examples
models/                  Fine-tuned model artifacts
```

For the full system map, training/inference diagrams, artifact strategy, and production checklist, read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For Colab, install directly in the notebook:

```bash
pip install -e ".[dev]"
```

## Prepare Data

```bash
python scripts/prepare_data.py --config configs/train_phase1.yaml
```

This loads VietNews, normalizes examples, creates train/validation/test splits, and writes:

```text
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test.jsonl
```

## Train

```bash
python scripts/train.py --config configs/train_phase1.yaml
```

Default training target is Google Colab T4:

- Model: `VietAI/vit5-base`
- Phase 1 dataset: `ithieund/VietNews-Abs-Sum`
- Batch size: `2`
- Gradient accumulation: `8`
- Epochs: `3`
- FP16: enabled

The Phase 1 checkpoint is saved to:

```text
models/vit5-summarizer-v1
```

## Phase 2 Multi-Mode Adaptation

```bash
python scripts/build_phase2_synthetic_200.py
python scripts/generate_synthetic.py --input data/synthetic/reviewed_all.json
python scripts/train_phase2.py --config configs/train_phase2.yaml
```

Phase 2 uses a paired controllability dataset to adapt the model to the four output modes:

- `concise`
- `bullet`
- `action_items`
- `study_notes`

The current Phase 2 dataset has 200 reviewed rows from 50 base documents:

```text
50 base documents x 4 modes
160 train / 40 validation
50 samples per output mode
```

To rebuild the reviewed synthetic dataset and stratified split:

```bash
python scripts/build_phase2_synthetic_200.py
python scripts/generate_synthetic.py --input data/synthetic/reviewed_all.json
```

The final checkpoint is saved to:

```text
models/vit5-summarizer-v2
```

## Evaluate

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

Evaluation outputs:

```text
reports/metrics/eval_results.json
reports/examples/test_predictions.jsonl
```

Main metrics:

- ROUGE-1
- ROUGE-2
- ROUGE-L

For qualitative controllability evaluation across all four output modes:

```bash
python scripts/evaluate_modes.py --input data/samples/qualitative_mode_eval.jsonl --length medium
```

This writes both JSONL predictions and a grouped Markdown report for reading mode differences side by side.

## Predict

```bash
python scripts/predict.py --text-file data/samples/meeting_note_vi.txt --mode bullet --length medium
```

## Run The Backend And Web App

Start the FastAPI backend first:

```bash
uvicorn api.main:app --reload
```

Then open the Streamlit frontend:

```bash
streamlit run app/streamlit_app.py
```

The app supports:

- Sample Vietnamese inputs
- Output mode selection
- Length control
- Compare-all-modes view through `POST /api/compare-modes`
- Keyword display
- Quality Estimate
- Input token count
- Inference latency

## Testing

```bash
pytest
```

Optional style checks:

```bash
ruff check .
```

## Report Guide

Recommended report chapters:

1. Introduction and motivation
2. Problem definition: input, output, task type
3. Dataset: VietNews, synthetic adaptation data, preprocessing, split
4. Model: ViT5 and Transformer encoder-decoder architecture
5. Training setup: Phase 1 on VietNews, Phase 2 on synthetic data, Colab T4 environment
6. Evaluation: ROUGE scores and qualitative examples
7. Error analysis: hallucination, missing key points, repetition, long input failure
8. Web application: FastAPI backend, Streamlit frontend, inference pipeline
9. Limitations and future work

## Limitations

- `action_items` and `study_notes` are implemented through controllable generation plus post-processing, not a fully supervised extraction dataset.
- Very long inputs are truncated to fit the model context window.
- Quality Estimate is a heuristic signal, not a calibrated probability.
- VietNews is a better Vietnamese base dataset than WikiLingua, but it still has domain gap with real meeting notes.
- Generated summaries still require human verification for high-stakes use.

## Future Work

- Add speech-to-text for real meeting audio.
- Add retrieval-augmented summarization for long documents.
- Add supervised data for action-item extraction.
- Add multilingual support.
- Add BERTScore and human evaluation.
