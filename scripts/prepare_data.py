from __future__ import annotations

import argparse

from _bootstrap import add_project_paths

add_project_paths()

from smart_summarizer.config import deep_get, load_config
from smart_summarizer.data.dataset_loader import (
    ensure_train_validation_test,
    load_remote_dataset,
    save_jsonl,
)
from smart_summarizer.utils.logging import get_logger
from smart_summarizer.utils.seed import set_seed


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Vietnamese summarization data.")
    parser.add_argument("--config", default="configs/train_phase1.yaml")
    return parser.parse_args()


def maybe_limit(dataset, limit: int | None):
    if limit:
        return dataset.select(range(min(limit, len(dataset))))
    return dataset


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    dataset_name = deep_get(config, "dataset.name", "ithieund/VietNews-Abs-Sum")
    language = deep_get(config, "dataset.language", "vietnamese")
    logger.info("Loading dataset %s/%s", dataset_name, language)
    dataset = ensure_train_validation_test(load_remote_dataset(dataset_name, language), seed=seed)

    limits = deep_get(config, "dataset.max_samples", {}) or {}
    for split_name in ("train", "validation", "test"):
        dataset[split_name] = maybe_limit(dataset[split_name], limits.get(split_name))

    outputs = {
        "train": deep_get(config, "dataset.train_file"),
        "validation": deep_get(config, "dataset.validation_file"),
        "test": deep_get(config, "dataset.test_file"),
    }
    for split_name, output_path in outputs.items():
        logger.info("Saving %s examples to %s", len(dataset[split_name]), output_path)
        save_jsonl(dataset[split_name], output_path)

    logger.info("Data preparation completed.")


if __name__ == "__main__":
    main()
