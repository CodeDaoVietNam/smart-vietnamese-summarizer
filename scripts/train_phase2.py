from __future__ import annotations

import argparse
import json

from _bootstrap import add_project_paths

add_project_paths()

from datasets import DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

from smart_summarizer.config import deep_get, load_config
from smart_summarizer.data.collator import build_seq2seq_collator
from smart_summarizer.data.dataset_loader import load_jsonl_dataset
from smart_summarizer.evaluation.metrics import compute_rouge
from smart_summarizer.modeling.trainer import tokenize_seq2seq_dataset
from smart_summarizer.utils.logging import get_logger
from smart_summarizer.utils.paths import ensure_parent, resolve_path
from smart_summarizer.utils.seed import set_seed


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 fine-tuning on synthetic multi-mode data.")
    parser.add_argument("--config", default="configs/train_phase2.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    model_name = deep_get(config, "model.name", "models/vit5-summarizer-v1")
    output_dir = resolve_path(deep_get(config, "model.output_dir", "models/vit5-summarizer-v2"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer and model from %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    prefixes = deep_get(config, "prefixes", {}) or {}
    raw_datasets = DatasetDict(
        {
            "train": load_jsonl_dataset(deep_get(config, "dataset.train_file")),
            "validation": load_jsonl_dataset(deep_get(config, "dataset.validation_file")),
        }
    )

    max_source_length = int(deep_get(config, "tokenization.max_source_length", 512))
    max_target_length = int(deep_get(config, "tokenization.max_target_length", 128))
    tokenized = DatasetDict(
        {
            split: tokenize_seq2seq_dataset(
                dataset,
                tokenizer,
                max_source_length,
                max_target_length,
                prefixes=prefixes,
            )
            for split, dataset in raw_datasets.items()
        }
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(deep_get(config, "training.epochs", 2)),
        learning_rate=float(deep_get(config, "training.learning_rate", 1e-5)),
        per_device_train_batch_size=int(deep_get(config, "training.per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(deep_get(config, "training.per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(deep_get(config, "training.gradient_accumulation_steps", 4)),
        fp16=bool(deep_get(config, "training.fp16", True)),
        evaluation_strategy=deep_get(config, "training.eval_strategy", "epoch"),
        save_strategy=deep_get(config, "training.save_strategy", "epoch"),
        save_total_limit=int(deep_get(config, "training.save_total_limit", 2)),
        load_best_model_at_end=bool(deep_get(config, "training.load_best_model_at_end", True)),
        metric_for_best_model=deep_get(config, "training.metric_for_best_model", "eval_loss"),
        logging_steps=int(deep_get(config, "training.logging_steps", 10)),
        predict_with_generate=bool(deep_get(config, "training.predict_with_generate", True)),
        warmup_ratio=float(deep_get(config, "training.warmup_ratio", 0.15)),
        report_to="none",
    )

    data_collator = build_seq2seq_collator(tokenizer, model)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = [
            [token if token != -100 else tokenizer.pad_token_id for token in label]
            for label in labels
        ]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return compute_rouge(decoded_preds, decoded_labels)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting phase 2 training.")
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    log_path = ensure_parent("reports/metrics/training_log_phase2.json")
    payload = {
        "train_metrics": train_result.metrics,
        "log_history": trainer.state.log_history,
    }
    with log_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    logger.info("Phase 2 training completed. Model saved to %s", output_dir)


if __name__ == "__main__":
    main()
