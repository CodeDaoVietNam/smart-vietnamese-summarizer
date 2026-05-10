from __future__ import annotations

import argparse
import json

from smart_summarizer.utils.paths import ensure_parent


TEMPLATES = [
    {
        "document": (
            "Nhóm dự án họp để thống nhất kế hoạch tuần này. An phụ trách chuẩn bị báo cáo, "
            "Bình hoàn thiện backend API, Chi kiểm tra giao diện frontend trước thứ Sáu."
        ),
        "summary": "Nhóm thống nhất kế hoạch tuần. An làm báo cáo, Bình hoàn thiện API, Chi kiểm tra frontend trước thứ Sáu.",
        "mode": "concise",
    },
    {
        "document": (
            "Giảng viên nhấn mạnh rằng Transformer dùng self-attention để mô hình hóa quan hệ xa. "
            "Sinh viên cần hiểu encoder, decoder và cơ chế beam search trước buổi kiểm tra."
        ),
        "summary": "- Transformer dùng self-attention.\n- Cần hiểu encoder và decoder.\n- Cần nắm beam search trước buổi kiểm tra.",
        "mode": "study_notes",
    },
    {
        "document": (
            "Trong cuộc họp, nhóm thống nhất cần hoàn thành báo cáo cuối kỳ, gửi bản nháp cho giảng viên "
            "và chuẩn bị demo trước tối thứ Năm."
        ),
        "summary": "- Hoàn thành báo cáo cuối kỳ.\n- Gửi bản nháp cho giảng viên.\n- Chuẩn bị demo trước tối thứ Năm.",
        "mode": "action_items",
    },
    {
        "document": (
            "Buổi thảo luận học tập tập trung vào attention, positional encoding và lý do ViT5 phù hợp cho tiếng Việt."
        ),
        "summary": "- Attention là thành phần cốt lõi.\n- Positional encoding giúp giữ thông tin thứ tự.\n- ViT5 phù hợp cho tác vụ tiếng Việt.",
        "mode": "bullet",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create starter synthetic samples for phase 2.")
    parser.add_argument("--train-file", default="data/synthetic/train.jsonl")
    parser.add_argument("--validation-file", default="data/synthetic/validation.jsonl")
    return parser.parse_args()


def write_rows(path: str, rows: list[dict[str, str]]) -> None:
    output = ensure_parent(path)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    write_rows(args.train_file, TEMPLATES[:3])
    write_rows(args.validation_file, TEMPLATES[3:])
    print(f"Starter synthetic data written to {args.train_file} and {args.validation_file}")


if __name__ == "__main__":
    main()
