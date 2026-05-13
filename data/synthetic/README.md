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
{"document": "...", "summary": "...", "mode": "bullet"}
```

Supported `mode` values:

- `concise`
- `bullet`
- `action_items`
- `study_notes`

Current reviewed dataset:

- `reviewed_all.json`: 200 reviewed samples.
- `train.jsonl`: 160 samples after stratified split.
- `validation.jsonl`: 40 samples after stratified split.
- Each mode has 50 total samples: `concise`, `bullet`, `action_items`, `study_notes`.

Workflow:

1. Generate or rebuild 200 synthetic meeting, lecture, project, and study samples.
2. Review and clean the text manually.
3. Save the reviewed samples into a JSON array or JSONL file.
4. Convert them into balanced train/validation JSONL files:

```bash
python scripts/build_phase2_synthetic_200.py
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
