# Synthetic Data Guide

This folder stores reviewed synthetic data for Phase 2 multi-mode adaptation.

Recommended files:

- `train.jsonl`
- `validation.jsonl`
- `meeting_samples.jsonl`
- `lecture_samples.jsonl`
- `PROMPTS.md`

Each row should look like:

```json
{"base_id": "meeting_01", "document": "...", "summary": "...", "mode": "bullet", "domain": "meeting_notes"}
```

Supported `mode` values:

- `concise`
- `bullet`
- `action_items`
- `study_notes`

Current reviewed dataset:

- `reviewed_all.json`: 400 reviewed samples.
- `train.jsonl`: 320 samples after base-aware stratified split.
- `validation.jsonl`: 80 samples after base-aware stratified split.
- `base_id`: groups the same input document across all 4 output modes.
- Each mode has 100 total samples: `concise`, `bullet`, `action_items`, `study_notes`.
- Each base document has exactly 4 rows, one per mode.
- `data/samples/holdout_mode_eval.jsonl`: 20 documents not used for training.
- `reports/examples/holdout_rubric_template.csv`: manual rubric template for scoring mode quality.

Workflow:

1. Generate or rebuild 100 base documents x 4 modes.
2. Review and clean the text manually.
3. Save the reviewed samples into a JSON array or JSONL file.
4. Convert them into base-aware train/validation JSONL files:

```bash
python scripts/build_phase2_synthetic_400.py
python scripts/generate_synthetic.py --input data/synthetic/reviewed_all.json
```

5. Run Phase 2 training:

```bash
python scripts/train_phase2.py --config configs/train_phase2.yaml
```

If you do not have reviewed samples yet, the script still supports the tiny starter dataset:

```bash
python scripts/generate_synthetic.py
```
