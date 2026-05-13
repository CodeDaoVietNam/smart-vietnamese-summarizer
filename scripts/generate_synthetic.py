from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import add_project_paths

add_project_paths()

from smart_summarizer.constants import SUMMARY_MODES
from smart_summarizer.utils.paths import ensure_parent


STARTER_TEMPLATES = [
    {
        "document": (
            "Nhóm dự án họp để thống nhất kế hoạch tuần này. An phụ trách chuẩn bị báo cáo, "
            "Bình hoàn thiện backend API, Chi kiểm tra giao diện frontend trước thứ Sáu."
        ),
        "summary": (
            "Nhóm thống nhất kế hoạch tuần. An làm báo cáo, Bình hoàn thiện API, "
            "Chi kiểm tra frontend trước thứ Sáu."
        ),
        "mode": "concise",
        "domain": "meeting_notes",
    },
    {
        "document": (
            "Giảng viên nhấn mạnh rằng Transformer dùng self-attention để mô hình hóa quan hệ xa. "
            "Sinh viên cần hiểu encoder, decoder và cơ chế beam search trước buổi kiểm tra."
        ),
        "summary": (
            "- Transformer dùng self-attention.\n"
            "- Cần hiểu encoder và decoder.\n"
            "- Cần nắm beam search trước buổi kiểm tra."
        ),
        "mode": "study_notes",
        "domain": "lecture_notes",
    },
    {
        "document": (
            "Trong cuộc họp, nhóm thống nhất cần hoàn thành báo cáo cuối kỳ, gửi bản nháp cho giảng viên "
            "và chuẩn bị demo trước tối thứ Năm."
        ),
        "summary": (
            "- Hoàn thành báo cáo cuối kỳ.\n"
            "- Gửi bản nháp cho giảng viên.\n"
            "- Chuẩn bị demo trước tối thứ Năm."
        ),
        "mode": "action_items",
        "domain": "meeting_notes",
    },
    {
        "document": (
            "Buổi thảo luận học tập tập trung vào attention, positional encoding và lý do ViT5 phù hợp cho tiếng Việt."
        ),
        "summary": (
            "- Attention là thành phần cốt lõi.\n"
            "- Positional encoding giúp giữ thông tin thứ tự.\n"
            "- ViT5 phù hợp cho tác vụ tiếng Việt."
        ),
        "mode": "bullet",
        "domain": "study_materials",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create starter synthetic data or convert reviewed synthetic JSON/JSONL "
            "into train/validation JSONL files for phase 2."
        )
    )
    parser.add_argument(
        "--input",
        help=(
            "Optional path to reviewed synthetic data in JSON array or JSONL format. "
            "If omitted, the script writes a tiny starter dataset."
        ),
    )
    parser.add_argument("--train-file", default="data/synthetic/train.jsonl")
    parser.add_argument("--validation-file", default="data/synthetic/validation.jsonl")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-document-chars", type=int, default=120)
    parser.add_argument("--min-summary-chars", type=int, default=20)
    return parser.parse_args()


def load_rows(path: str) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be an array of objects.")
    return payload


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def validate_row(
    row: dict[str, Any],
    min_document_chars: int,
    min_summary_chars: int,
) -> dict[str, Any] | None:
    document = normalize_text(row.get("document"))
    summary = str(row.get("summary") or "").strip()
    mode = normalize_text(row.get("mode")).lower()

    if not document or not summary or not mode:
        return None
    if len(document) < min_document_chars or len(summary) < min_summary_chars:
        return None
    if mode not in SUMMARY_MODES:
        return None

    normalized = dict(row)
    normalized["document"] = document
    normalized["summary"] = summary
    normalized["mode"] = mode
    if "domain" in normalized:
        normalized["domain"] = normalize_text(normalized["domain"]).lower()
    return normalized


def write_rows(path: str, rows: list[dict[str, Any]]) -> None:
    output = ensure_parent(path)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split(
    rows: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if all(row.get("base_id") for row in rows):
        return base_id_split(rows, validation_ratio, seed)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)

    randomizer = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for mode, samples in grouped.items():
        randomizer.shuffle(samples)
        if len(samples) == 1:
            train_rows.extend(samples)
            continue

        validation_size = max(1, round(len(samples) * validation_ratio))
        validation_size = min(validation_size, len(samples) - 1)
        validation_rows.extend(samples[:validation_size])
        train_rows.extend(samples[validation_size:])
        print(
            f"Mode {mode}: train={len(samples[validation_size:])}, "
            f"validation={len(samples[:validation_size])}"
        )

    randomizer.shuffle(train_rows)
    randomizer.shuffle(validation_rows)
    return train_rows, validation_rows


def base_id_split(
    rows: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_id"])].append(row)

    for base_id, samples in grouped.items():
        modes = {sample["mode"] for sample in samples}
        if modes != set(SUMMARY_MODES):
            raise ValueError(f"{base_id} must contain all modes, got {sorted(modes)}.")

    by_domain: dict[str, list[str]] = defaultdict(list)
    for base_id, samples in grouped.items():
        by_domain[str(samples[0].get("domain", "unknown"))].append(base_id)

    randomizer = random.Random(seed)
    validation_base_ids: set[str] = set()
    for domain, base_ids in sorted(by_domain.items()):
        randomizer.shuffle(base_ids)
        validation_size = round(len(base_ids) * validation_ratio)
        if len(base_ids) > 1:
            validation_size = max(1, min(validation_size, len(base_ids) - 1))
        validation_base_ids.update(base_ids[:validation_size])
        print(
            f"Domain {domain}: train_bases={len(base_ids) - validation_size}, "
            f"validation_bases={validation_size}"
        )

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for base_id, samples in grouped.items():
        target = validation_rows if base_id in validation_base_ids else train_rows
        target.extend(samples)

    randomizer.shuffle(train_rows)
    randomizer.shuffle(validation_rows)
    return train_rows, validation_rows


def print_distribution(label: str, rows: list[dict[str, Any]]) -> None:
    counter = Counter(row["mode"] for row in rows)
    base_count = len({row.get("base_id") for row in rows if row.get("base_id")})
    if base_count:
        print(f"{label}: {len(rows)} rows / {base_count} base docs -> {dict(counter)}")
    else:
        print(f"{label}: {len(rows)} rows -> {dict(counter)}")


def build_dataset_from_input(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = load_rows(args.input)
    cleaned_rows = [
        normalized
        for row in raw_rows
        if (normalized := validate_row(row, args.min_document_chars, args.min_summary_chars))
    ]

    if not cleaned_rows:
        raise ValueError("No valid reviewed synthetic samples found after validation.")

    print(f"Loaded {len(raw_rows)} reviewed samples, kept {len(cleaned_rows)} valid rows.")
    train_rows, validation_rows = stratified_split(cleaned_rows, args.validation_ratio, args.seed)
    print_distribution("Train split", train_rows)
    print_distribution("Validation split", validation_rows)
    return train_rows, validation_rows


def build_starter_dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = STARTER_TEMPLATES[:3]
    validation_rows = STARTER_TEMPLATES[3:]
    print("No reviewed input provided. Writing tiny starter synthetic dataset.")
    print_distribution("Train split", train_rows)
    print_distribution("Validation split", validation_rows)
    return train_rows, validation_rows


def main() -> None:
    args = parse_args()
    if args.input:
        train_rows, validation_rows = build_dataset_from_input(args)
    else:
        train_rows, validation_rows = build_starter_dataset()

    write_rows(args.train_file, train_rows)
    write_rows(args.validation_file, validation_rows)

    print(f"Saved train synthetic data to {args.train_file}")
    print(f"Saved validation synthetic data to {args.validation_file}")


if __name__ == "__main__":
    main()
