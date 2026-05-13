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
    parser.add_argument("--markdown-output", default="reports/examples/mode_comparison_report.md")
    return parser.parse_args()


def write_markdown_report(path: str, rows: list[dict[str, str | int | float]]) -> None:
    output_path = ensure_parent(path)
    grouped: dict[str, list[dict[str, str | int | float]]] = {}
    for row in rows:
        grouped.setdefault(str(row["id"]), []).append(row)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("# Mode Comparison Report\n\n")
        file.write(
            "Mỗi input được chạy qua đủ 4 mode để kiểm tra controllability. "
            "Kết quả tốt là các mode khác nhau về mục đích, không chỉ khác format.\n\n"
        )
        for sample_id, sample_rows in grouped.items():
            domain = sample_rows[0]["domain"]
            document = str(sample_rows[0]["document"])
            file.write(f"## {sample_id} ({domain})\n\n")
            file.write("**Input preview:** ")
            file.write(document[:500].replace("\n", " "))
            file.write("\n\n")
            by_mode = {str(row["mode"]): row for row in sample_rows}
            for mode in SUMMARY_MODES:
                row = by_mode[mode]
                file.write(f"### {mode}\n\n")
                file.write(str(row["prediction"]).strip())
                file.write("\n\n")
                file.write(
                    f"`quality_estimate={row['quality_estimate']}` "
                    f"`latency_ms={row['latency_ms']}`\n\n"
                )


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    summarizer = SmartSummarizer.from_config(args.config)
    output_path = ensure_parent(args.output)
    prediction_rows: list[dict[str, str | int | float]] = []

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
                prediction_rows.append(payload)
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    write_markdown_report(args.markdown_output, prediction_rows)
    print(f"Saved mode comparison predictions to {output_path}")
    print(f"Saved grouped markdown report to {args.markdown_output}")


if __name__ == "__main__":
    main()
