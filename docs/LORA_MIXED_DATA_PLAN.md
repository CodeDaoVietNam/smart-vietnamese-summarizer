# LoRA Mixed-Data Phase 2 Plan

## Summary

LoRA is an additional Phase 2 experiment, not a replacement for the existing full fine-tune baseline. The system keeps `models/vit5-summarizer-v1` as a frozen backbone and trains one shared adapter on a larger mixed dataset.

```text
ViT5 Phase 1 backbone
  + shared LoRA adapter
  + mode prefix
  -> controllable summary
```

## Data Recipe

Default builder command:

```bash
python scripts/build_lora_mixed_data.py
```

It writes:

```text
data/lora/train.jsonl
data/lora/validation.jsonl
data/lora/holdout.jsonl
reports/metrics/lora_data_report.json
```

The intended train mix is:

```text
3,000 VietNews concise/replay
1,000 XL-Sum Vietnamese concise
1,000 viWikiHow action-step style
1,000 pseudo bullet rows from VietNews
320 synthetic curated rows
```

Use `--skip-remote` only for local smoke tests. The final experiment should include the remote XL-Sum and viWikiHow sources.

## Training And Evaluation

Train:

```bash
python scripts/train_lora.py --config configs/train_lora.yaml
```

Evaluate:

```bash
python scripts/evaluate.py --config configs/eval_lora.yaml
python scripts/evaluate.py --config configs/eval_lora_holdout.yaml
python scripts/evaluate_modes.py \
  --config configs/app_lora.yaml \
  --input data/samples/holdout_mode_eval.jsonl \
  --length medium \
  --output reports/examples/holdout_mode_comparison_lora.jsonl \
  --markdown-output reports/examples/holdout_mode_comparison_lora.md
```

Compare:

```text
Phase 1 backbone
Phase 2 full fine-tune on curated synthetic data
Phase 2 LoRA adapter on mixed data
```

Selection priority:

```text
factuality -> mode adherence -> format correctness -> ROUGE stability -> demo quality
```

## Report Framing

Explain that the original 400-row Phase 2 dataset is useful but small. LoRA reduces catastrophic forgetting by freezing the Phase 1 backbone and training a small adapter on broader mixed data.
