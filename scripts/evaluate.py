from __future__ import annotations

import argparse
import json

from _bootstrap import add_project_paths

add_project_paths()

from smart_summarizer.config import deep_get, load_config
from smart_summarizer.data.dataset_loader import load_jsonl
from smart_summarizer.evaluation.error_analysis import quick_error_tags
from smart_summarizer.evaluation.metrics import compute_rouge
from smart_summarizer.modeling.generation import build_instruction, generation_kwargs
from smart_summarizer.modeling.model_loader import load_seq2seq_model, load_tokenizer, resolve_existing_model
from smart_summarizer.utils.logging import get_logger
from smart_summarizer.utils.paths import ensure_parent
from smart_summarizer.utils.seed import set_seed


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned summarization model.")
    parser.add_argument("--config", default="configs/eval.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    model_path = resolve_existing_model(
        deep_get(config, "model.name", "models/vit5-summarizer-v2"),
        deep_get(config, "model.fallback_name", "VietAI/vit5-base"),
    )
    tokenizer = load_tokenizer(model_path)
    model, _ = load_seq2seq_model(model_path)
    rows = load_jsonl(deep_get(config, "dataset.test_file", "data/processed/test.jsonl"))
    max_samples = deep_get(config, "dataset.max_samples")
    if max_samples:
        rows = rows[: int(max_samples)]

    mode = deep_get(config, "generation.mode", "concise")
    length = deep_get(config, "generation.length", "medium")
    max_source_length = int(deep_get(config, "tokenization.max_source_length", 512))

    predictions: list[str] = []
    references: list[str] = []
    prediction_rows: list[dict[str, str | list[str]]] = []

    kwargs = generation_kwargs(
        length=length,
        num_beams=int(deep_get(config, "generation.num_beams", 4)),
        repetition_penalty=float(deep_get(config, "generation.repetition_penalty", 1.2)),
        no_repeat_ngram_size=int(deep_get(config, "generation.no_repeat_ngram_size", 3)),
    )

    logger.info("Generating predictions for %s examples.", len(rows))
    for row in rows:
        instruction = build_instruction(row["document"], mode=mode)
        encoded = tokenizer(
            instruction,
            return_tensors="pt",
            max_length=max_source_length,
            truncation=True,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        output = model.generate(**encoded, **kwargs)
        prediction = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        reference = row["summary"]
        predictions.append(prediction)
        references.append(reference)
        prediction_rows.append(
            {
                "document": row["document"],
                "reference": reference,
                "prediction": prediction,
                "error_tags": quick_error_tags(row["document"], prediction),
            }
        )

    metrics = compute_rouge(predictions, references)
    metrics_path = ensure_parent(deep_get(config, "outputs.metrics_file"))
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    predictions_path = ensure_parent(deep_get(config, "outputs.predictions_file"))
    with predictions_path.open("w", encoding="utf-8") as file:
        for row in prediction_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Evaluation completed: %s", metrics)


if __name__ == "__main__":
    main()
