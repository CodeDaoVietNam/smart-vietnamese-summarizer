from __future__ import annotations

import argparse
import inspect
import json

try:
    from _bootstrap import add_project_paths
except ModuleNotFoundError:
    from scripts._bootstrap import add_project_paths

add_project_paths()

from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

from smart_summarizer.config import deep_get, load_config
from smart_summarizer.data.collator import build_seq2seq_collator
from smart_summarizer.data.dataset_loader import load_jsonl_dataset
from smart_summarizer.evaluation.metrics import compute_rouge
from smart_summarizer.modeling.model_loader import load_tokenizer
from smart_summarizer.modeling.trainer import tokenize_seq2seq_dataset
from smart_summarizer.utils.logging import get_logger
from smart_summarizer.utils.paths import ensure_parent, resolve_path
from smart_summarizer.utils.seed import set_seed


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a shared LoRA adapter for phase 2.")
    parser.add_argument("--config", default="configs/train_lora.yaml")
    return parser.parse_args()


def build_training_args_kwargs(config: dict, output_dir: str) -> dict:
    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": float(deep_get(config, "training.epochs", 3)),
        "learning_rate": float(deep_get(config, "training.learning_rate", 2e-4)),
        "per_device_train_batch_size": int(deep_get(config, "training.per_device_train_batch_size", 2)),
        "per_device_eval_batch_size": int(deep_get(config, "training.per_device_eval_batch_size", 2)),
        "gradient_accumulation_steps": int(deep_get(config, "training.gradient_accumulation_steps", 4)),
        "fp16": bool(deep_get(config, "training.fp16", True)),
        "save_strategy": deep_get(config, "training.save_strategy", "epoch"),
        "save_total_limit": int(deep_get(config, "training.save_total_limit", 2)),
        "load_best_model_at_end": bool(deep_get(config, "training.load_best_model_at_end", True)),
        "metric_for_best_model": deep_get(config, "training.metric_for_best_model", "eval_loss"),
        "logging_steps": int(deep_get(config, "training.logging_steps", 25)),
        "predict_with_generate": bool(deep_get(config, "training.predict_with_generate", True)),
        "warmup_ratio": float(deep_get(config, "training.warmup_ratio", 0.05)),
        "report_to": "none",
    }

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = deep_get(config, "training.eval_strategy", "epoch")
    elif "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = deep_get(config, "training.eval_strategy", "epoch")

    return kwargs


def build_lora_config(config: dict) -> LoraConfig:
    return LoraConfig(
        r=int(deep_get(config, "peft.r", 8)),
        lora_alpha=int(deep_get(config, "peft.lora_alpha", 16)),
        lora_dropout=float(deep_get(config, "peft.lora_dropout", 0.05)),
        target_modules=list(deep_get(config, "peft.target_modules", ["q", "v"])),
        bias=deep_get(config, "peft.bias", "none"),
        task_type=TaskType.SEQ_2_SEQ_LM,
    )


def trainable_parameter_report(model) -> dict[str, int | float]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    ratio = trainable / total if total else 0.0
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": ratio,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    model_name = deep_get(config, "model.name", "models/vit5-summarizer-v1")
    output_dir = resolve_path(deep_get(config, "model.output_dir", "models/vit5-summarizer-lora"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer and frozen backbone from %s", model_name)
    tokenizer = load_tokenizer(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    base_model.config.use_cache = False
    model = get_peft_model(base_model, build_lora_config(config))
    parameter_report = trainable_parameter_report(model)
    logger.info("LoRA trainable parameter report: %s", parameter_report)

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
        **build_training_args_kwargs(config, str(output_dir))
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

    logger.info("Starting LoRA training.")
    train_result = trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    log_path = ensure_parent(deep_get(config, "outputs.training_log", "reports/metrics/training_log_lora.json"))
    payload = {
        "train_metrics": train_result.metrics,
        "parameter_report": parameter_report,
        "log_history": trainer.state.log_history,
    }
    with log_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    logger.info("LoRA training completed. Adapter saved to %s", output_dir)


if __name__ == "__main__":
    main()
