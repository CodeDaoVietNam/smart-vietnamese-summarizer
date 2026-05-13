from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


MODE_DOMAIN_ADDITIONS = {
    ("concise", "meeting_notes"): 20,
    ("concise", "lecture_notes"): 10,
    ("concise", "project_updates"): 4,
    ("concise", "study_materials"): 6,
    ("bullet", "meeting_notes"): 12,
    ("bullet", "lecture_notes"): 16,
    ("bullet", "project_updates"): 6,
    ("bullet", "study_materials"): 6,
    ("action_items", "meeting_notes"): 24,
    ("action_items", "project_updates"): 12,
    ("action_items", "study_materials"): 4,
    ("study_notes", "lecture_notes"): 34,
    ("study_notes", "study_materials"): 6,
}

TARGET_MODE_COUNTS = {
    "concise": 50,
    "bullet": 50,
    "action_items": 50,
    "study_notes": 50,
}

TARGET_DOMAIN_COUNTS = {
    "meeting_notes": 70,
    "lecture_notes": 70,
    "project_updates": 30,
    "study_materials": 30,
}

BASE_REVIEWED_COUNT = 40

MEETING_TOPICS = [
    ("nhóm đồ án NLP", "Minh", "Lan", "Huy", "bản demo tóm tắt", "thứ Năm"),
    ("nhóm vận hành sản phẩm", "Thảo", "Quân", "Linh", "lỗi đăng nhập và OTP", "cuối ngày mai"),
    ("nhóm marketing tuyển sinh", "An", "Bình", "Chi", "landing page chiến dịch", "thứ Sáu"),
    ("ban tổ chức hội thảo", "Phúc", "Mai", "Duy", "kịch bản check-in", "sáng thứ Hai"),
    ("nhóm chăm sóc khách hàng", "Vy", "Nam", "Trúc", "bộ câu trả lời mẫu", "chiều thứ Tư"),
    ("nhóm kiểm thử phần mềm", "Khoa", "Trang", "Long", "danh sách lỗi ưu tiên", "trước buổi demo"),
    ("nhóm nội dung học tập", "Ngọc", "Tú", "Sơn", "bộ ghi chú ôn thi", "cuối tuần"),
    ("nhóm quản lý dữ liệu", "Hà", "Đức", "Yến", "quy trình kiểm tra chất lượng", "ngày 15"),
]

PROJECT_TOPICS = [
    ("ứng dụng tóm tắt văn bản", "API xử lý ổn định hơn", "giao diện so sánh mode còn chậm", "thêm log latency"),
    ("dashboard chăm sóc khách hàng", "bộ lọc ticket đã rõ hơn", "biểu đồ theo ngày còn lệch múi giờ", "rà soát truy vấn"),
    ("hệ thống học trực tuyến", "module bài tập đã mở cho lớp thử nghiệm", "đồng bộ điểm đôi lúc trễ", "kiểm tra hàng đợi xử lý"),
    ("chatbot nội bộ", "câu trả lời cho FAQ chính xác hơn", "câu hỏi dài vẫn bị bỏ sót ý", "bổ sung bước tách yêu cầu"),
    ("cổng đăng ký sự kiện", "luồng đăng ký đã ngắn hơn", "email xác nhận chưa ổn định", "kiểm tra dịch vụ gửi mail"),
    ("bộ công cụ phân tích phản hồi", "pipeline làm sạch dữ liệu đã chạy tự động", "nhãn lỗi còn chưa nhất quán", "review lại guideline gán nhãn"),
]

LECTURE_TOPICS = [
    ("self-attention", "mỗi token xét quan hệ với các token khác", "hiểu được phụ thuộc xa trong câu", "nhầm với attention thông thường"),
    ("encoder-decoder", "encoder đọc input còn decoder sinh output", "phù hợp bài toán seq2seq", "quên luồng dữ liệu giữa hai phần"),
    ("positional encoding", "bổ sung thông tin thứ tự vào embedding", "Transformer không xử lý tuần tự như RNN", "nghĩ rằng mô hình tự biết vị trí"),
    ("beam search", "giữ nhiều giả thuyết sinh văn bản cùng lúc", "giảm rủi ro chọn câu kém từ bước đầu", "chọn beam quá lớn gây chậm"),
    ("ROUGE", "đo độ trùng lặp từ hoặc chuỗi con với reference", "hữu ích cho so sánh tự động", "không đo đầy đủ ý nghĩa"),
    ("fine-tuning", "điều chỉnh mô hình pre-trained trên dataset mục tiêu", "tiết kiệm tài nguyên hơn train từ đầu", "dễ overfit nếu dữ liệu nhỏ"),
    ("tokenization", "chia văn bản thành token hoặc subword", "giúp mô hình xử lý từ hiếm", "nhầm token với từ nguyên vẹn"),
    ("gradient accumulation", "cộng dồn gradient qua nhiều mini-batch", "giả lập batch lớn trên GPU nhỏ", "quên chia effective batch size"),
]

STUDY_TOPICS = [
    ("ôn tập Transformer", "vẽ sơ đồ encoder-decoder", "self-attention và positional encoding", "nhầm vai trò của encoder"),
    ("ôn tập cơ sở dữ liệu", "chia bài theo chuẩn hóa và giao dịch", "khóa chính, khóa ngoại, ACID", "học thuộc định nghĩa quá dài"),
    ("ôn tập học máy", "lập bảng so sánh mô hình", "overfitting, regularization, validation", "chỉ nhớ tên thuật toán"),
    ("ôn tập NLP", "ghi ví dụ input-output cho từng task", "tokenization, embedding, seq2seq", "không phân biệt training và inference"),
    ("ôn tập kiểm thử phần mềm", "gom lỗi theo mức độ nghiêm trọng", "unit test, integration test, regression test", "viết test thiếu expected behavior"),
    ("ôn tập mạng máy tính", "tóm tắt theo từng tầng giao thức", "TCP, UDP, DNS, HTTP", "học rời rạc từng khái niệm"),
]


def load_json_array(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return payload


def write_json_array(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(list(rows), file, ensure_ascii=False, indent=2)
        file.write("\n")


def pick(items: list[tuple[str, ...]], index: int) -> tuple[str, ...]:
    return items[index % len(items)]


def meeting_document(index: int, mode_hint: str) -> str:
    team, owner_a, owner_b, owner_c, focus, deadline = pick(MEETING_TOPICS, index)
    return (
        f"Cuộc họp của {team} tập trung vào việc rà soát tiến độ {focus} và thống nhất ưu tiên trong tuần. "
        f"{owner_a} báo cáo rằng phần chính đã hoàn thành nhưng vẫn còn một số tình huống biên cần kiểm tra kỹ hơn. "
        f"{owner_b} đề xuất tách các lỗi theo mức độ ảnh hưởng để nhóm không mất thời gian vào những việc ít quan trọng. "
        f"{owner_c} nhắc rằng tài liệu demo cũng cần cập nhật song song, vì người xem sẽ hỏi vì sao nhóm chọn cách xử lý hiện tại. "
        f"Sau khi thảo luận, cả nhóm thống nhất giữ phạm vi vừa đủ, ưu tiên ổn định trải nghiệm chính, ghi lại rủi ro còn tồn tại "
        f"và hoàn thành phần việc được giao trước {deadline}. Nội dung này phù hợp để tạo output dạng {mode_hint}."
    )


def project_document(index: int, mode_hint: str) -> str:
    name, progress, issue, next_step = pick(PROJECT_TOPICS, index)
    return (
        f"Báo cáo cập nhật dự án {name} cho biết nhóm đã đạt một số tiến triển rõ ràng trong sprint hiện tại. "
        f"Điểm tích cực nhất là {progress}, nhờ đó nhóm thử nghiệm có thể kiểm tra luồng chính mà không cần thao tác thủ công quá nhiều. "
        f"Tuy vậy, vấn đề còn tồn tại là {issue}, làm cho trải nghiệm chưa ổn định khi dữ liệu đầu vào thay đổi. "
        f"Nhóm kỹ thuật đề xuất {next_step}, còn nhóm sản phẩm muốn ghi nhận phản hồi người dùng trước khi mở rộng thêm tính năng. "
        f"Trong tuần tới, mục tiêu là giữ phạm vi nhỏ, sửa lỗi có ảnh hưởng trực tiếp đến demo và chuẩn bị số liệu so sánh trước buổi báo cáo. "
        f"Nội dung này được dùng để huấn luyện mode {mode_hint}."
    )


def lecture_document(index: int, mode_hint: str) -> str:
    concept, meaning, benefit, common_error = pick(LECTURE_TOPICS, index)
    return (
        f"Bài giảng hôm nay giải thích khái niệm {concept} trong bối cảnh mô hình Transformer và các hệ thống xử lý ngôn ngữ tự nhiên. "
        f"Giảng viên nhấn mạnh rằng {meaning}, vì vậy sinh viên cần nhìn khái niệm này như một phần của toàn bộ pipeline chứ không học rời rạc. "
        f"Lợi ích chính là {benefit}, đặc biệt khi mô hình phải xử lý văn bản dài hoặc sinh ra bản tóm tắt có cấu trúc. "
        f"Giảng viên cũng cảnh báo lỗi hay gặp là {common_error}, khiến phần giải thích trong báo cáo thiếu chính xác. "
        f"Cuối buổi, sinh viên được yêu cầu tự viết lại định nghĩa ngắn, thêm ví dụ đầu vào đầu ra và ghi chú câu hỏi còn chưa chắc để ôn tập sau. "
        f"Đoạn này phù hợp với mode {mode_hint}."
    )


def study_document(index: int, mode_hint: str) -> str:
    topic, method, focus, mistake = pick(STUDY_TOPICS, index)
    return (
        f"Tài liệu hướng dẫn {topic} khuyên sinh viên không nên chép lại toàn bộ nội dung bài giảng theo thứ tự ban đầu. "
        f"Cách hiệu quả hơn là {method}, sau đó rút ra những ý có khả năng xuất hiện trong câu hỏi tự luận hoặc phần trình bày miệng. "
        f"Người học cần chú ý nhóm kiến thức gồm {focus}, vì đây là các điểm thường liên kết với nhau trong bài kiểm tra. "
        f"Tài liệu cũng nhắc rằng lỗi phổ biến là {mistake}, làm cho ghi chú dài nhưng khó dùng khi cần ôn nhanh. "
        f"Mỗi mục nên có một định nghĩa ngắn, một ví dụ nhỏ và một cảnh báo dễ nhầm để người đọc có thể nhớ lại kiến thức trong vài phút. "
        f"Nội dung này hỗ trợ huấn luyện mode {mode_hint}."
    )


def build_document(domain: str, index: int, mode: str) -> str:
    if domain == "meeting_notes":
        return meeting_document(index, mode)
    if domain == "project_updates":
        return project_document(index, mode)
    if domain == "lecture_notes":
        return lecture_document(index, mode)
    if domain == "study_materials":
        return study_document(index, mode)
    raise ValueError(f"Unsupported domain: {domain}")


def concise_summary(domain: str, index: int) -> str:
    if domain == "meeting_notes":
        team, owner_a, owner_b, owner_c, focus, deadline = pick(MEETING_TOPICS, index)
        return (
            f"{team.capitalize()} thống nhất ưu tiên ổn định {focus}, phân chia việc cho {owner_a}, "
            f"{owner_b} và {owner_c}, đồng thời cập nhật tài liệu trước {deadline}."
        )
    if domain == "project_updates":
        name, progress, issue, next_step = pick(PROJECT_TOPICS, index)
        return (
            f"Dự án {name} đã có tiến triển ở luồng chính nhưng vẫn còn vấn đề {issue}; "
            f"nhóm sẽ {next_step} và chuẩn bị số liệu cho buổi báo cáo tới."
        )
    if domain == "lecture_notes":
        concept, meaning, benefit, _ = pick(LECTURE_TOPICS, index)
        return (
            f"Bài giảng làm rõ {concept}: {meaning}. Khái niệm này quan trọng vì {benefit} "
            f"và cần được ôn cùng ví dụ input-output."
        )
    topic, method, focus, _ = pick(STUDY_TOPICS, index)
    return (
        f"Tài liệu {topic} khuyên sinh viên {method}, tập trung vào {focus} "
        f"và ghi chú ngắn kèm ví dụ để ôn tập hiệu quả."
    )


def bullet_summary(domain: str, index: int) -> str:
    if domain == "meeting_notes":
        team, owner_a, owner_b, owner_c, focus, deadline = pick(MEETING_TOPICS, index)
        return (
            f"- {team.capitalize()} rà soát tiến độ {focus}.\n"
            f"- {owner_a} kiểm tra các tình huống biên còn lại.\n"
            f"- {owner_b} phân loại lỗi theo mức độ ảnh hưởng.\n"
            f"- {owner_c} cập nhật tài liệu demo trước {deadline}."
        )
    if domain == "project_updates":
        name, progress, issue, next_step = pick(PROJECT_TOPICS, index)
        return (
            f"- Dự án {name} đã ghi nhận tiến triển: {progress}.\n"
            f"- Vấn đề còn lại là {issue}.\n"
            f"- Nhóm kỹ thuật sẽ {next_step}.\n"
            f"- Tuần tới ưu tiên sửa lỗi ảnh hưởng trực tiếp đến demo."
        )
    if domain == "lecture_notes":
        concept, meaning, benefit, common_error = pick(LECTURE_TOPICS, index)
        return (
            f"- {concept} có ý nghĩa: {meaning}.\n"
            f"- Lợi ích chính: {benefit}.\n"
            f"- Lỗi hay gặp: {common_error}.\n"
            f"- Khi ôn tập nên thêm ví dụ input-output."
        )
    topic, method, focus, mistake = pick(STUDY_TOPICS, index)
    return (
        f"- Chủ đề ôn tập: {topic}.\n"
        f"- Cách học nên dùng: {method}.\n"
        f"- Cần tập trung vào {focus}.\n"
        f"- Tránh lỗi: {mistake}."
    )


def action_summary(domain: str, index: int) -> str:
    if domain == "meeting_notes":
        _, owner_a, owner_b, owner_c, focus, deadline = pick(MEETING_TOPICS, index)
        return (
            f"- {owner_a} kiểm tra các tình huống biên của {focus} trước {deadline}.\n"
            f"- {owner_b} phân loại lỗi theo mức độ ảnh hưởng và gửi danh sách ưu tiên.\n"
            f"- {owner_c} cập nhật tài liệu demo để phản ánh quyết định mới.\n"
            f"- Cả nhóm chạy thử luồng chính trước buổi báo cáo."
        )
    if domain == "project_updates":
        name, _, issue, next_step = pick(PROJECT_TOPICS, index)
        return (
            f"- Nhóm kỹ thuật {next_step} cho dự án {name}.\n"
            f"- Nhóm sản phẩm tổng hợp phản hồi liên quan đến {issue}.\n"
            f"- Trưởng nhóm chốt danh sách lỗi ảnh hưởng trực tiếp đến demo.\n"
            f"- Các bên cập nhật tiến độ trong buổi họp sprint tiếp theo."
        )
    topic, method, focus, _ = pick(STUDY_TOPICS, index)
    return (
        f"- Nhóm học tập tạo checklist {topic} theo phương pháp {method}.\n"
        f"- Mỗi thành viên chuẩn bị ví dụ ngắn cho {focus}.\n"
        f"- Người phụ trách tổng hợp ghi chú trước buổi ôn nhóm.\n"
        f"- Cả nhóm rà soát lại các lỗi dễ nhầm trước ngày kiểm tra."
    )


def study_summary(domain: str, index: int) -> str:
    if domain == "lecture_notes":
        concept, meaning, benefit, common_error = pick(LECTURE_TOPICS, index)
        return (
            f"- Khái niệm chính: {concept}.\n"
            f"- Cần nhớ: {meaning}.\n"
            f"- Ý nghĩa: {benefit}.\n"
            f"- Lỗi dễ nhầm: {common_error}.\n"
            f"- Nên tự viết một ví dụ input-output để ôn tập."
        )
    topic, method, focus, mistake = pick(STUDY_TOPICS, index)
    return (
        f"- Chủ đề: {topic}.\n"
        f"- Cách ghi chú: {method}.\n"
        f"- Trọng tâm: {focus}.\n"
        f"- Tránh: {mistake}.\n"
        f"- Mỗi mục nên có định nghĩa ngắn và ví dụ nhỏ."
    )


def build_summary(mode: str, domain: str, index: int) -> str:
    if mode == "concise":
        return concise_summary(domain, index)
    if mode == "bullet":
        return bullet_summary(domain, index)
    if mode == "action_items":
        return action_summary(domain, index)
    if mode == "study_notes":
        return study_summary(domain, index)
    raise ValueError(f"Unsupported mode: {mode}")


def validate_targets(rows: list[dict[str, str]]) -> None:
    mode_counts = Counter(row["mode"] for row in rows)
    domain_counts = Counter(row["domain"] for row in rows)
    if mode_counts != TARGET_MODE_COUNTS:
        raise ValueError(f"Unexpected mode counts: {dict(mode_counts)}")
    if domain_counts != TARGET_DOMAIN_COUNTS:
        raise ValueError(f"Unexpected domain counts: {dict(domain_counts)}")
    for index, row in enumerate(rows, start=1):
        for field in ("document", "summary", "mode", "domain"):
            if not row.get(field):
                raise ValueError(f"Row {index} is missing {field}.")
        if len(row["document"].split()) < 80:
            raise ValueError(f"Row {index} document is too short.")
        if len(row["summary"]) >= len(row["document"]):
            raise ValueError(f"Row {index} summary is not shorter than document.")


def build_additions(start_id: int = 1) -> list[dict[str, str]]:
    additions: list[dict[str, str]] = []
    running_index = start_id
    for (mode, domain), count in MODE_DOMAIN_ADDITIONS.items():
        for local_index in range(count):
            source_index = running_index + local_index
            additions.append(
                {
                    "document": build_document(domain, source_index, mode),
                    "summary": build_summary(mode, domain, source_index),
                    "mode": mode,
                    "domain": domain,
                }
            )
        running_index += count
    return additions


def build_qualitative_rows() -> list[dict[str, str]]:
    specs = [
        ("meeting_01", "meeting_notes", 101),
        ("meeting_02", "meeting_notes", 102),
        ("meeting_03", "meeting_notes", 103),
        ("lecture_01", "lecture_notes", 104),
        ("lecture_02", "lecture_notes", 105),
        ("lecture_03", "lecture_notes", 106),
        ("project_01", "project_updates", 107),
        ("project_02", "project_updates", 108),
        ("study_01", "study_materials", 109),
        ("study_02", "study_materials", 110),
    ]
    rows = []
    for sample_id, domain, index in specs:
        rows.append(
            {
                "id": sample_id,
                "domain": domain,
                "document": build_document(domain, index, "qualitative comparison"),
            }
        )
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand phase 2 synthetic data to 200 reviewed samples.")
    parser.add_argument("--input", default="data/synthetic/reviewed_all.json")
    parser.add_argument("--output", default="data/synthetic/reviewed_all.json")
    parser.add_argument("--qualitative-output", default="data/samples/qualitative_mode_eval.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = load_json_array(input_path)
    if len(rows) < BASE_REVIEWED_COUNT:
        raise ValueError(
            f"{input_path} must contain at least {BASE_REVIEWED_COUNT} seed reviewed samples."
        )
    base_rows = rows[:BASE_REVIEWED_COUNT]
    additions = build_additions()
    expanded = base_rows + additions
    validate_targets(expanded)

    write_json_array(output_path, expanded)
    write_jsonl(Path(args.qualitative_output), build_qualitative_rows())

    print(f"Saved {len(expanded)} reviewed samples to {output_path}")
    print(f"Mode counts: {dict(Counter(row['mode'] for row in expanded))}")
    print(f"Domain counts: {dict(Counter(row['domain'] for row in expanded))}")
    print(f"Saved qualitative evaluation set to {args.qualitative_output}")


if __name__ == "__main__":
    main()
