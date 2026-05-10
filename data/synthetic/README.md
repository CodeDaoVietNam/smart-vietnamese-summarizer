# Synthetic Data Guide

This folder stores reviewed synthetic data for Phase 2 multi-mode adaptation.

Recommended files:

- `train.jsonl`
- `validation.jsonl`
- `meeting_samples.jsonl`
- `lecture_samples.jsonl`

Each row should look like:

```json
{"document": "...", "summary": "...", "mode": "bullet"}
```

Supported `mode` values:

- `concise`
- `bullet`
- `action_items`
- `study_notes`

Workflow:

1. Generate 50-100 synthetic meeting and lecture samples.
2. Review and clean the text manually.
3. Merge them into `train.jsonl` and `validation.jsonl`.
4. Run `python scripts/train_phase2.py --config configs/train_phase2.yaml`.
