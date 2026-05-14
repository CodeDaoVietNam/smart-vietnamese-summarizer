from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from _bootstrap import add_project_paths

add_project_paths()

from smart_summarizer.utils.paths import ensure_parent


CRITERIA = (
    "mode_adherence",
    "factuality",
    "usefulness",
    "format_correctness",
    "conciseness",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize manual holdout rubric scores by mode.")
    parser.add_argument("--input", default="reports/examples/holdout_rubric_template.csv")
    parser.add_argument("--output", default="reports/metrics/holdout_rubric_summary.md")
    return parser.parse_args()


def parse_score(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    score = float(value)
    if score < 0 or score > 2:
        raise ValueError(f"Rubric scores must be between 0 and 2, got {score}.")
    return score


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.input).open("r", encoding="utf-8")))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)

    output_path = ensure_parent(args.output)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("# Holdout Rubric Summary\n\n")
        file.write("| mode | scored_rows | avg_mode_adherence | avg_factuality | avg_usefulness | avg_format_correctness | avg_conciseness | avg_total |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")

        for mode in ("concise", "bullet", "action_items", "study_notes"):
            mode_rows = grouped.get(mode, [])
            scored_rows = [
                row
                for row in mode_rows
                if all(parse_score(row.get(criterion, "")) is not None for criterion in CRITERIA)
            ]
            if not scored_rows:
                file.write(f"| {mode} | 0 | - | - | - | - | - | - |\n")
                continue

            averages = {
                criterion: sum(parse_score(row[criterion]) or 0.0 for row in scored_rows) / len(scored_rows)
                for criterion in CRITERIA
            }
            avg_total = sum(averages.values())
            file.write(
                f"| {mode} | {len(scored_rows)} | "
                f"{averages['mode_adherence']:.2f} | "
                f"{averages['factuality']:.2f} | "
                f"{averages['usefulness']:.2f} | "
                f"{averages['format_correctness']:.2f} | "
                f"{averages['conciseness']:.2f} | "
                f"{avg_total:.2f} |\n"
            )

    print(f"Saved holdout rubric summary to {output_path}")


if __name__ == "__main__":
    main()
