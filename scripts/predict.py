from __future__ import annotations

import argparse

from _bootstrap import add_project_paths

add_project_paths()

from smart_summarizer.product.summarizer import SmartSummarizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run summarization inference.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file")
    parser.add_argument("--mode", default="concise", choices=["concise", "bullet", "action_items", "study_notes"])
    parser.add_argument("--length", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--config", default="configs/app.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.text
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as file:
            text = file.read()

    summarizer = SmartSummarizer.from_config(args.config)
    result = summarizer.generate_summary(text or "", mode=args.mode, length=args.length)
    print(result["summary"])
    print()
    print(f"Keywords: {', '.join(result['keywords'])}")
    print(f"Heuristic Quality Estimate: {result['quality_estimate']}%")
    print(f"Latency: {result['latency_ms']} ms")


if __name__ == "__main__":
    main()
