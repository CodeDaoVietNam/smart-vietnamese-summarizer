from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_project_paths

add_project_paths()

from smart_summarizer.constants import SUMMARY_MODES
from smart_summarizer.product.summarizer import SmartSummarizer
from smart_summarizer.utils.paths import ensure_parent


def load_jsonl(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run qualitative controllability evaluation across all output modes."
    )
    parser.add_argument("--input", default="data/samples/qualitative_mode_eval.jsonl")
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--length", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--output", default="reports/examples/mode_comparison_predictions.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    summarizer = SmartSummarizer.from_config(args.config)
    output_path = ensure_parent(args.output)

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            for mode in SUMMARY_MODES:
                result = summarizer.generate_summary(row["document"], mode=mode, length=args.length)
                payload = {
                    "id": row.get("id", ""),
                    "domain": row.get("domain", ""),
                    "mode": mode,
                    "length": args.length,
                    "document": row["document"],
                    "prediction": result["summary"],
                    "quality_estimate": result["quality_estimate"],
                    "latency_ms": result["latency_ms"],
                }
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Saved mode comparison predictions to {output_path}")


if __name__ == "__main__":
    main()
