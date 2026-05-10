from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from _bootstrap import add_project_paths

add_project_paths()

from smart_summarizer.utils.paths import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model artifacts for submission.")
    parser.add_argument("--model-dir", default="models/vit5-summarizer-v2")
    parser.add_argument("--output-dir", default="submission/model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = resolve_path(args.model_dir)
    target = resolve_path(args.output_dir)
    if not source.exists():
        raise FileNotFoundError(f"Model directory not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    print(f"Exported model from {source} to {target}")


if __name__ == "__main__":
    main()
