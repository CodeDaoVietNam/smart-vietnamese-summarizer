from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

try:
    from _bootstrap import add_project_paths
except ModuleNotFoundError:
    from scripts._bootstrap import add_project_paths

add_project_paths()

from smart_summarizer.data.dataset_loader import load_jsonl, normalize_pair
from smart_summarizer.data.preprocessing import clean_text
from smart_summarizer.utils.paths import ensure_parent
from smart_summarizer.utils.seed import set_seed


MODES = {"concise", "bullet", "action_items", "study_notes"}
SENTENCE_RE = re.compile(r"(?<=[.!?。])\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed data for shared LoRA adaptation.")
    parser.add_argument("--output-dir", default="data/lora")
    parser.add_argument("--report-file", default="reports/metrics/lora_data_report.json")
    parser.add_argument("--synthetic-train", default="data/synthetic/train.jsonl")
    parser.add_argument("--synthetic-validation", default="data/synthetic/validation.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-remote", action="store_true")
    parser.add_argument("--strict-counts", action="store_true")
    parser.add_argument("--vietnews-train", type=int, default=3000)
    parser.add_argument("--xlsum-train", type=int, default=1000)
    parser.add_argument("--wikihow-train", type=int, default=1000)
    parser.add_argument("--pseudo-bullet-train", type=int, default=1000)
    parser.add_argument("--vietnews-validation", type=int, default=300)
    parser.add_argument("--xlsum-validation", type=int, default=100)
    parser.add_argument("--wikihow-validation", type=int, default=100)
    parser.add_argument("--pseudo-bullet-validation", type=int, default=100)
    parser.add_argument("--holdout-size", type=int, default=200)
    return parser.parse_args()


def stable_hash(text: str) -> str:
    normalized = clean_text(text).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_RE.split(clean_text(text))
    return [part.strip() for part in parts if len(part.split()) >= 4]


def pseudo_bullet_summary(row: dict[str, str]) -> str:
    candidates = split_sentences(row.get("summary", "")) or split_sentences(row.get("document", ""))
    if not candidates:
        candidates = [clean_text(row.get("summary") or row.get("document", ""))]
    return "\n".join(f"- {item}" for item in candidates[:5])


def action_summary(row: dict[str, str]) -> str:
    candidates = split_sentences(row.get("summary", "")) or split_sentences(row.get("document", ""))
    actions = candidates[:3] or [clean_text(row.get("summary") or "Hoàn thành các bước chính.")]
    return "\n".join(
        (
            f"- Người phụ trách: Người học\n"
            f"  Hành động: {action.rstrip('.')}\n"
            f"  Deadline: Khi thực hiện hướng dẫn"
        )
        for action in actions
    )


def lora_row(
    row: dict[str, str],
    *,
    mode: str,
    source: str,
    task_type: str,
    domain: str,
    summary_transform: Callable[[dict[str, str]], str] | None = None,
) -> dict[str, str]:
    summary = summary_transform(row) if summary_transform else row["summary"]
    return {
        "document": clean_text(row["document"]),
        "summary": summary.strip(),
        "mode": mode,
        "domain": domain,
        "source": source,
        "task_type": task_type,
    }


def load_jsonl_if_exists(path: str) -> list[dict[str, Any]]:
    resolved = Path(path)
    if not resolved.exists():
        return []
    return load_jsonl(resolved)


def normalize_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in raw_rows:
        pair = normalize_pair(row)
        if pair:
            rows.append(pair)
    return rows


def load_local_vietnews() -> list[dict[str, str]]:
    raw_rows: list[dict[str, Any]] = []
    for path in (
        "data/processed/train.jsonl",
        "data/processed/validation.jsonl",
        "data/processed/test.jsonl",
        "data/processed/train_colab.jsonl",
        "data/processed/validation_colab.jsonl",
        "data/processed/test_colab.jsonl",
    ):
        raw_rows.extend(load_jsonl_if_exists(path))
    return normalize_rows(raw_rows)


def load_remote_rows(
    dataset_name: str,
    config: str | None = None,
    max_rows: int | None = None,
) -> list[dict[str, str]]:
    from datasets import load_dataset

    raw = load_dataset(dataset_name, config) if config else load_dataset(dataset_name)
    rows: list[dict[str, str]] = []
    for split in raw.values():
        for example in split:
            pair = normalize_pair(example)
            if pair:
                rows.append(pair)
            if max_rows and len(rows) >= max_rows:
                return rows
    return rows


def load_xlsum_vietnamese_rows(max_rows: int | None = None) -> list[dict[str, str]]:
    """Load XL-Sum Vietnamese without relying on deprecated dataset scripts."""
    from datasets import load_dataset

    data_files = {
        "train": "hf://datasets/csebuetnlp/xlsum/vietnamese/xlsum-train.parquet",
        "validation": "hf://datasets/csebuetnlp/xlsum/vietnamese/xlsum-validation.parquet",
        "test": "hf://datasets/csebuetnlp/xlsum/vietnamese/xlsum-test.parquet",
    }
    try:
        raw = load_dataset("parquet", data_files=data_files)
    except Exception:
        raw = load_dataset("GEM/xlsum", "vietnamese")

    rows: list[dict[str, str]] = []
    for split in raw.values():
        for example in split:
            pair = normalize_pair(example)
            if pair:
                rows.append(pair)
            if max_rows and len(rows) >= max_rows:
                return rows
    return rows


def take_unique(
    source_rows: list[dict[str, str]],
    count: int,
    used_hashes: set[str],
    randomizer: random.Random,
) -> list[dict[str, str]]:
    pool = list(source_rows)
    randomizer.shuffle(pool)
    selected: list[dict[str, str]] = []
    for row in pool:
        row_hash = stable_hash(row["document"])
        if row_hash in used_hashes:
            continue
        used_hashes.add(row_hash)
        selected.append(row)
        if len(selected) >= count:
            break
    return selected


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_rows(
    output: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    *,
    count: int,
    used_hashes: set[str],
    randomizer: random.Random,
    mode: str,
    source: str,
    task_type: str,
    domain: str,
    summary_transform: Callable[[dict[str, str]], str] | None = None,
) -> int:
    selected = take_unique(source_rows, count, used_hashes, randomizer)
    output.extend(
        lora_row(
            row,
            mode=mode,
            source=source,
            task_type=task_type,
            domain=domain,
            summary_transform=summary_transform,
        )
        for row in selected
    )
    return len(selected)


def load_synthetic(path: str) -> list[dict[str, str]]:
    rows = []
    for row in load_jsonl_if_exists(path):
        if row.get("mode") in MODES and row.get("document") and row.get("summary"):
            rows.append(
                {
                    "document": clean_text(row["document"]),
                    "summary": str(row["summary"]).strip(),
                    "mode": str(row["mode"]),
                    "domain": str(row.get("domain") or "synthetic"),
                    "source": "synthetic_curated",
                    "task_type": f"synthetic_{row['mode']}",
                }
            )
    return rows


def summarize_split(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "mode_counts": dict(Counter(row["mode"] for row in rows)),
        "task_type_counts": dict(Counter(row["task_type"] for row in rows)),
        "avg_document_words": round(sum(len(row["document"].split()) for row in rows) / max(1, len(rows)), 2),
        "avg_summary_words": round(sum(len(row["summary"].split()) for row in rows) / max(1, len(rows)), 2),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    randomizer = random.Random(args.seed)

    vietnews = load_local_vietnews()
    if not vietnews and not args.skip_remote:
        vietnews_needed = (
            args.vietnews_train
            + args.vietnews_validation
            + args.pseudo_bullet_train
            + args.pseudo_bullet_validation
            + args.holdout_size
            + 500
        )
        vietnews = load_remote_rows("ithieund/VietNews-Abs-Sum", max_rows=vietnews_needed)

    xlsum: list[dict[str, str]] = []
    wikihow: list[dict[str, str]] = []
    if not args.skip_remote:
        xlsum_needed = args.xlsum_train + args.xlsum_validation + args.holdout_size + 500
        wikihow_needed = args.wikihow_train + args.wikihow_validation + args.holdout_size + 500
        xlsum = load_xlsum_vietnamese_rows(max_rows=xlsum_needed)
        wikihow = load_remote_rows("ithieund/viWikiHow-Abs-Sum", max_rows=wikihow_needed)

    synthetic_train = load_synthetic(args.synthetic_train)
    synthetic_validation = load_synthetic(args.synthetic_validation)

    train_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    holdout_rows: list[dict[str, str]] = []
    used_hashes: set[str] = set()
    actual_counts: dict[str, int] = {}

    actual_counts["train_vietnews"] = add_rows(
        train_rows,
        vietnews,
        count=args.vietnews_train,
        used_hashes=used_hashes,
        randomizer=randomizer,
        mode="concise",
        source="vietnews",
        task_type="concise_replay",
        domain="news",
    )
    actual_counts["train_xlsum"] = add_rows(
        train_rows,
        xlsum,
        count=args.xlsum_train,
        used_hashes=used_hashes,
        randomizer=randomizer,
        mode="concise",
        source="xlsum_vi",
        task_type="concise_replay",
        domain="news",
    )
    actual_counts["train_wikihow"] = add_rows(
        train_rows,
        wikihow,
        count=args.wikihow_train,
        used_hashes=used_hashes,
        randomizer=randomizer,
        mode="action_items",
        source="viwikihow",
        task_type="action_steps",
        domain="how_to",
        summary_transform=action_summary,
    )
    actual_counts["train_pseudo_bullet"] = add_rows(
        train_rows,
        vietnews,
        count=args.pseudo_bullet_train,
        used_hashes=used_hashes,
        randomizer=randomizer,
        mode="bullet",
        source="vietnews",
        task_type="pseudo_bullet",
        domain="news",
        summary_transform=pseudo_bullet_summary,
    )
    train_rows.extend(synthetic_train)

    actual_counts["validation_vietnews"] = add_rows(
        validation_rows,
        vietnews,
        count=args.vietnews_validation,
        used_hashes=used_hashes,
        randomizer=randomizer,
        mode="concise",
        source="vietnews",
        task_type="concise_replay",
        domain="news",
    )
    actual_counts["validation_xlsum"] = add_rows(
        validation_rows,
        xlsum,
        count=args.xlsum_validation,
        used_hashes=used_hashes,
        randomizer=randomizer,
        mode="concise",
        source="xlsum_vi",
        task_type="concise_replay",
        domain="news",
    )
    actual_counts["validation_wikihow"] = add_rows(
        validation_rows,
        wikihow,
        count=args.wikihow_validation,
        used_hashes=used_hashes,
        randomizer=randomizer,
        mode="action_items",
        source="viwikihow",
        task_type="action_steps",
        domain="how_to",
        summary_transform=action_summary,
    )
    actual_counts["validation_pseudo_bullet"] = add_rows(
        validation_rows,
        vietnews,
        count=args.pseudo_bullet_validation,
        used_hashes=used_hashes,
        randomizer=randomizer,
        mode="bullet",
        source="vietnews",
        task_type="pseudo_bullet",
        domain="news",
        summary_transform=pseudo_bullet_summary,
    )
    validation_rows.extend(synthetic_validation)

    holdout_vietnews = round(args.holdout_size * 0.4)
    holdout_xlsum = round(args.holdout_size * 0.2)
    holdout_wikihow = round(args.holdout_size * 0.2)
    holdout_pseudo = args.holdout_size - holdout_vietnews - holdout_xlsum - holdout_wikihow
    holdout_plan = (
        ("vietnews", vietnews, holdout_vietnews, "concise", "concise_holdout", "news", None),
        ("xlsum_vi", xlsum, holdout_xlsum, "concise", "concise_holdout", "news", None),
        ("viwikihow", wikihow, holdout_wikihow, "action_items", "action_holdout", "how_to", action_summary),
        ("vietnews", vietnews, holdout_pseudo, "bullet", "pseudo_bullet_holdout", "news", pseudo_bullet_summary),
    )
    for source, rows, count, mode, task_type, domain, transform in holdout_plan:
        actual = add_rows(
            holdout_rows,
            rows,
            count=count,
            used_hashes=used_hashes,
            randomizer=randomizer,
            mode=mode,
            source=source,
            task_type=task_type,
            domain=domain,
            summary_transform=transform,
        )
        actual_counts[f"holdout_{task_type}"] = actual

    randomizer.shuffle(train_rows)
    randomizer.shuffle(validation_rows)
    randomizer.shuffle(holdout_rows)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "validation.jsonl", validation_rows)
    write_jsonl(output_dir / "holdout.jsonl", holdout_rows)

    report = {
        "requested": vars(args),
        "actual_counts": actual_counts,
        "splits": {
            "train": summarize_split(train_rows),
            "validation": summarize_split(validation_rows),
            "holdout": summarize_split(holdout_rows),
        },
        "source_pool_sizes": {
            "vietnews": len(vietnews),
            "xlsum_vi": len(xlsum),
            "viwikihow": len(wikihow),
            "synthetic_train": len(synthetic_train),
            "synthetic_validation": len(synthetic_validation),
        },
    }
    report_path = ensure_parent(args.report_file)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    if args.strict_counts:
        expected_train = (
            args.vietnews_train
            + args.xlsum_train
            + args.wikihow_train
            + args.pseudo_bullet_train
            + len(synthetic_train)
        )
        expected_validation = (
            args.vietnews_validation
            + args.xlsum_validation
            + args.wikihow_validation
            + args.pseudo_bullet_validation
            + len(synthetic_validation)
        )
        if len(train_rows) != expected_train or len(validation_rows) != expected_validation:
            raise ValueError("Mixed LoRA data did not reach requested train/validation counts.")
        if len(holdout_rows) != args.holdout_size:
            raise ValueError("Mixed LoRA holdout did not reach requested size.")

    print(f"Saved LoRA train rows: {len(train_rows)}")
    print(f"Saved LoRA validation rows: {len(validation_rows)}")
    print(f"Saved LoRA holdout rows: {len(holdout_rows)}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
