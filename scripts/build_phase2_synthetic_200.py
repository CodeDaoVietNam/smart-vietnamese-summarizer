from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODES = ("concise", "bullet", "action_items", "study_notes")

TARGET_MODE_COUNTS = {
    "concise": 50,
    "bullet": 50,
    "action_items": 50,
    "study_notes": 50,
}

TARGET_DOMAIN_COUNTS = {
    "meeting_notes": 72,
    "lecture_notes": 72,
    "project_updates": 28,
    "study_materials": 28,
}

FORBIDDEN_META_PHRASES = (
    "phù hợp với mode",
    "phù hợp để tạo output",
    "dùng để huấn luyện",
    "hỗ trợ huấn luyện",
    "đoạn này",
    "nội dung này",
)


@dataclass(frozen=True)
class BaseExample:
    base_id: str
    domain: str
    title: str
    document: str
    concise: str
    bullet: str
    action_items: str
    study_notes: str


MEETING_CONTEXTS = [
    {
        "team": "nhóm đồ án NLP",
        "topic": "chuẩn bị demo controllable summarization",
        "owner_a": "Minh",
        "task_a": "sửa lỗi API khi văn bản đầu vào quá dài",
        "owner_b": "Lan",
        "task_b": "cập nhật giao diện so sánh bốn mode",
        "owner_c": "Huy",
        "task_c": "viết lại phần kiến trúc trong báo cáo",
        "deadline": "tối thứ Năm",
        "risk": "demo có thể bị chậm nếu model load lại nhiều lần",
        "decision": "giữ phạm vi sản phẩm ở tóm tắt có điều khiển",
    },
    {
        "team": "nhóm vận hành sản phẩm",
        "topic": "xử lý phản hồi sau bản cập nhật ứng dụng",
        "owner_a": "Thảo",
        "task_a": "tổng hợp các lỗi đăng nhập và OTP",
        "owner_b": "Quân",
        "task_b": "kiểm tra log backend trong giờ cao điểm",
        "owner_c": "Linh",
        "task_c": "soạn email thông báo tình trạng cho khách hàng",
        "deadline": "cuối ngày mai",
        "risk": "người dùng có thể gửi nhiều ticket trùng nhau",
        "decision": "ưu tiên sửa lỗi đăng nhập trước khi thêm tính năng mới",
    },
    {
        "team": "nhóm marketing tuyển sinh",
        "topic": "lên kế hoạch truyền thông cho chiến dịch tháng tới",
        "owner_a": "An",
        "task_a": "hoàn thiện ba video ngắn về trải nghiệm học viên",
        "owner_b": "Bình",
        "task_b": "rà soát landing page và form đăng ký",
        "owner_c": "Chi",
        "task_c": "theo dõi tỷ lệ chuyển đổi từng kênh quảng cáo",
        "deadline": "thứ Sáu",
        "risk": "ngân sách quảng cáo bị phân tán vào nhóm người dùng chưa phù hợp",
        "decision": "tập trung thông điệp vào kết quả học tập thực tế",
    },
    {
        "team": "ban tổ chức hội thảo",
        "topic": "chuẩn bị sự kiện giới thiệu dự án cuối khóa",
        "owner_a": "Phúc",
        "task_a": "chốt kịch bản check-in và phân luồng khách mời",
        "owner_b": "Mai",
        "task_b": "cập nhật slide giới thiệu chương trình",
        "owner_c": "Duy",
        "task_c": "kiểm tra âm thanh, máy chiếu và đường truyền mạng",
        "deadline": "sáng thứ Hai",
        "risk": "khách đến cùng lúc có thể gây ùn tắc ở quầy check-in",
        "decision": "mở bàn hỗ trợ sớm hơn ba mươi phút",
    },
    {
        "team": "nhóm chăm sóc khách hàng",
        "topic": "chuẩn hóa câu trả lời cho các câu hỏi thường gặp",
        "owner_a": "Vy",
        "task_a": "gom nhóm ticket theo chủ đề",
        "owner_b": "Nam",
        "task_b": "viết lại câu trả lời cho lỗi thanh toán",
        "owner_c": "Trúc",
        "task_c": "kiểm tra giọng văn trước khi đưa vào hệ thống",
        "deadline": "chiều thứ Tư",
        "risk": "câu trả lời quá kỹ thuật có thể làm khách hàng khó hiểu",
        "decision": "ưu tiên ngôn ngữ ngắn gọn và có bước xử lý rõ",
    },
    {
        "team": "nhóm kiểm thử phần mềm",
        "topic": "chốt danh sách lỗi trước khi đóng băng bản demo",
        "owner_a": "Khoa",
        "task_a": "kiểm thử lại chức năng upload file",
        "owner_b": "Trang",
        "task_b": "ghi nhận lỗi hiển thị tiếng Việt trên giao diện",
        "owner_c": "Long",
        "task_c": "xác nhận nút so sánh mode hoạt động ổn định",
        "deadline": "trước buổi demo",
        "risk": "lỗi nhỏ ở giao diện có thể làm người xem hiểu sai kết quả",
        "decision": "không thêm tính năng mới trong tuần này",
    },
    {
        "team": "nhóm nội dung học tập",
        "topic": "biên tập bộ ghi chú ôn thi cho sinh viên năm nhất",
        "owner_a": "Ngọc",
        "task_a": "rút gọn phần định nghĩa dài",
        "owner_b": "Tú",
        "task_b": "thêm ví dụ minh họa cho từng khái niệm",
        "owner_c": "Sơn",
        "task_c": "rà soát lỗi thuật ngữ trong tài liệu",
        "deadline": "cuối tuần",
        "risk": "ghi chú quá dài khiến sinh viên khó ôn nhanh",
        "decision": "mỗi chương chỉ giữ một trang tóm tắt chính",
    },
    {
        "team": "nhóm quản lý dữ liệu",
        "topic": "thiết kế quy trình kiểm tra chất lượng dữ liệu",
        "owner_a": "Hà",
        "task_a": "định nghĩa tiêu chí loại bỏ bản ghi lỗi",
        "owner_b": "Đức",
        "task_b": "viết script thống kê dữ liệu thiếu",
        "owner_c": "Yến",
        "task_c": "tạo báo cáo chất lượng dữ liệu theo tuần",
        "deadline": "ngày 15",
        "risk": "dữ liệu thiếu nhãn sẽ làm kết quả đánh giá kém tin cậy",
        "decision": "kiểm tra dữ liệu trước khi đưa vào huấn luyện",
    },
    {
        "team": "nhóm nghiên cứu người dùng",
        "topic": "tổng hợp phản hồi về bản thử nghiệm giao diện",
        "owner_a": "My",
        "task_a": "phỏng vấn thêm năm người dùng mới",
        "owner_b": "Khánh",
        "task_b": "mã hóa phản hồi theo nhóm vấn đề",
        "owner_c": "Tâm",
        "task_c": "đề xuất thay đổi cho màn hình kết quả",
        "deadline": "thứ Ba tuần tới",
        "risk": "mẫu khảo sát hiện còn thiên về người dùng kỹ thuật",
        "decision": "bổ sung phản hồi từ sinh viên và nhân viên văn phòng",
    },
]

LECTURE_CONTEXTS = [
    {
        "concept": "self-attention",
        "definition": "mỗi token xét quan hệ với các token khác trong cùng chuỗi",
        "benefit": "giúp mô hình nắm được phụ thuộc xa trong văn bản",
        "example": "từ 'hoãn' liên hệ với nguyên nhân nằm ở cuối câu",
        "mistake": "nhầm self-attention với attention giữa encoder và decoder",
        "exercise": "vẽ ma trận chú ý đơn giản cho một câu ngắn",
    },
    {
        "concept": "encoder-decoder",
        "definition": "encoder đọc input còn decoder sinh output từng token",
        "benefit": "phù hợp với bài toán biến đổi một chuỗi thành chuỗi khác",
        "example": "văn bản dài được biến thành bản tóm tắt ngắn",
        "mistake": "mô tả hai khối riêng lẻ nhưng quên luồng dữ liệu giữa chúng",
        "exercise": "chú thích đường đi từ document đến summary",
    },
    {
        "concept": "positional encoding",
        "definition": "bổ sung thông tin thứ tự vào embedding của token",
        "benefit": "giúp Transformer phân biệt vị trí dù xử lý song song",
        "example": "hai câu có cùng từ nhưng đổi thứ tự có thể đổi nghĩa",
        "mistake": "nghĩ rằng attention tự biết vị trí của từ",
        "exercise": "so sánh câu trước và sau khi đảo vị trí cụm từ",
    },
    {
        "concept": "beam search",
        "definition": "giữ nhiều giả thuyết sinh văn bản tại mỗi bước",
        "benefit": "giảm rủi ro chọn nhánh kém do quyết định quá sớm",
        "example": "beam size bốn giữ bốn câu ứng viên song song",
        "mistake": "tăng beam quá lớn làm chậm mà chưa chắc hay hơn",
        "exercise": "mô phỏng ba bước sinh token với beam size hai",
    },
    {
        "concept": "ROUGE",
        "definition": "đo độ trùng lặp từ hoặc chuỗi con với bản tham chiếu",
        "benefit": "cho phép so sánh tự động nhiều mô hình tóm tắt",
        "example": "ROUGE-L xét chuỗi con chung dài nhất",
        "mistake": "xem ROUGE như đánh giá đầy đủ về ý nghĩa",
        "exercise": "so sánh hai bản tóm tắt cùng nghĩa nhưng khác từ",
    },
    {
        "concept": "fine-tuning",
        "definition": "tiếp tục huấn luyện mô hình pre-trained trên dữ liệu mục tiêu",
        "benefit": "tiết kiệm tài nguyên hơn nhiều so với train từ đầu",
        "example": "ViT5 học thêm cách tóm tắt từ dataset VietNews",
        "mistake": "dùng learning rate quá lớn làm mất kiến thức đã học",
        "exercise": "giải thích vì sao Phase 2 dùng learning rate nhỏ hơn Phase 1",
    },
    {
        "concept": "tokenization",
        "definition": "chuyển văn bản thành token hoặc subword để model xử lý",
        "benefit": "giúp mô hình xử lý từ hiếm và biến thể tiếng Việt",
        "example": "một từ dài có thể được chia thành nhiều mảnh nhỏ",
        "mistake": "đồng nhất token với từ hoàn chỉnh trong câu",
        "exercise": "quan sát tokenizer xử lý một câu tiếng Việt có dấu",
    },
    {
        "concept": "gradient accumulation",
        "definition": "cộng dồn gradient qua nhiều mini-batch trước khi cập nhật",
        "benefit": "giả lập batch size lớn trên GPU nhỏ",
        "example": "batch hai và accumulation tám tạo effective batch mười sáu",
        "mistake": "quên tính effective batch khi so sánh thí nghiệm",
        "exercise": "tính số update step khi thay đổi accumulation",
    },
    {
        "concept": "post-processing",
        "definition": "làm sạch hoặc định dạng output sau khi model sinh văn bản",
        "benefit": "giúp kết quả nhất quán hơn với mode người dùng chọn",
        "example": "bullet mode được chuẩn hóa thành các dòng bắt đầu bằng dấu gạch",
        "mistake": "nhầm post-processing với năng lực học sâu của model",
        "exercise": "chỉ ra phần nào do model sinh và phần nào do rule xử lý",
    },
]

PROJECT_CONTEXTS = [
    {
        "project": "ứng dụng tóm tắt văn bản",
        "progress": "API xử lý ổn định hơn với input cỡ vừa",
        "issue": "giao diện so sánh mode còn chậm khi gọi bốn lần liên tiếp",
        "owner_a": "Minh",
        "task_a": "thử endpoint compare-modes",
        "owner_b": "Lan",
        "task_b": "thêm trạng thái loading riêng cho từng mode",
        "deadline": "thứ Sáu",
        "metric": "latency trung bình và số lỗi request",
    },
    {
        "project": "dashboard chăm sóc khách hàng",
        "progress": "bộ lọc ticket theo chủ đề đã dễ dùng hơn",
        "issue": "biểu đồ theo ngày còn lệch múi giờ khi export báo cáo",
        "owner_a": "Quân",
        "task_a": "rà soát truy vấn thống kê",
        "owner_b": "Vy",
        "task_b": "kiểm tra lại nhãn tiếng Việt trên dashboard",
        "deadline": "cuối tuần",
        "metric": "số ticket được phân loại đúng",
    },
    {
        "project": "hệ thống học trực tuyến",
        "progress": "module bài tập đã mở cho lớp thử nghiệm",
        "issue": "đồng bộ điểm đôi lúc trễ khi nhiều sinh viên nộp cùng lúc",
        "owner_a": "Hà",
        "task_a": "kiểm tra hàng đợi xử lý điểm",
        "owner_b": "Tú",
        "task_b": "viết hướng dẫn cho giảng viên",
        "deadline": "thứ Hai",
        "metric": "thời gian đồng bộ điểm sau khi nộp bài",
    },
    {
        "project": "chatbot nội bộ",
        "progress": "câu trả lời cho FAQ phổ biến chính xác hơn",
        "issue": "câu hỏi dài có nhiều yêu cầu vẫn bị bỏ sót ý",
        "owner_a": "Khánh",
        "task_a": "bổ sung bước tách ý trước khi gọi model",
        "owner_b": "My",
        "task_b": "gán nhãn nhóm câu hỏi thất bại",
        "deadline": "chiều thứ Năm",
        "metric": "tỷ lệ câu trả lời được người dùng đánh giá hữu ích",
    },
    {
        "project": "cổng đăng ký sự kiện",
        "progress": "luồng đăng ký đã ngắn hơn hai bước",
        "issue": "email xác nhận chưa ổn định với một số địa chỉ trường học",
        "owner_a": "Duy",
        "task_a": "kiểm tra dịch vụ gửi mail",
        "owner_b": "Mai",
        "task_b": "viết lại nội dung email xác nhận",
        "deadline": "sáng thứ Ba",
        "metric": "tỷ lệ email gửi thành công",
    },
    {
        "project": "bộ công cụ phân tích phản hồi",
        "progress": "pipeline làm sạch dữ liệu đã chạy tự động mỗi ngày",
        "issue": "nhãn lỗi còn chưa nhất quán giữa các reviewer",
        "owner_a": "Trúc",
        "task_a": "review lại guideline gán nhãn",
        "owner_b": "Nam",
        "task_b": "tính độ đồng thuận giữa reviewer",
        "deadline": "ngày 18",
        "metric": "độ đồng thuận nhãn và số mẫu bị loại",
    },
    {
        "project": "hệ thống lưu trữ tài liệu",
        "progress": "chức năng tìm kiếm theo tiêu đề đã nhanh hơn",
        "issue": "file scan có lỗi OCR làm trích xuất keyword chưa chính xác",
        "owner_a": "Sơn",
        "task_a": "lọc các file OCR chất lượng thấp",
        "owner_b": "Yến",
        "task_b": "kiểm tra lại danh sách keyword quan trọng",
        "deadline": "thứ Tư",
        "metric": "tỷ lệ tìm đúng tài liệu trong top năm kết quả",
    },
]

STUDY_CONTEXTS = [
    {
        "topic": "ôn tập Transformer",
        "method": "vẽ sơ đồ encoder-decoder trước khi học công thức",
        "focus": "self-attention, positional encoding và beam search",
        "mistake": "nhầm vai trò của encoder với decoder",
        "owner": "nhóm trưởng học tập",
        "deadline": "tối Chủ nhật",
    },
    {
        "topic": "ôn tập cơ sở dữ liệu",
        "method": "chia bài theo chuẩn hóa, giao dịch và chỉ mục",
        "focus": "khóa chính, khóa ngoại, ACID và index",
        "mistake": "học thuộc định nghĩa nhưng không nêu được ví dụ",
        "owner": "bạn phụ trách môn cơ sở dữ liệu",
        "deadline": "buổi ôn nhóm",
    },
    {
        "topic": "ôn tập học máy",
        "method": "lập bảng so sánh mô hình và lỗi thường gặp",
        "focus": "overfitting, regularization và validation",
        "mistake": "chỉ nhớ tên thuật toán mà không hiểu khi nào dùng",
        "owner": "nhóm thống kê",
        "deadline": "thứ Bảy",
    },
    {
        "topic": "ôn tập NLP",
        "method": "ghi ví dụ input-output cho từng task",
        "focus": "tokenization, embedding, seq2seq và evaluation",
        "mistake": "không phân biệt training pipeline và inference pipeline",
        "owner": "nhóm NLP",
        "deadline": "tối thứ Sáu",
    },
    {
        "topic": "ôn tập kiểm thử phần mềm",
        "method": "gom lỗi theo mức độ nghiêm trọng và loại test",
        "focus": "unit test, integration test và regression test",
        "mistake": "viết test thiếu expected behavior rõ ràng",
        "owner": "nhóm QA",
        "deadline": "trước buổi lab",
    },
    {
        "topic": "ôn tập mạng máy tính",
        "method": "tóm tắt theo từng tầng giao thức",
        "focus": "TCP, UDP, DNS và HTTP",
        "mistake": "học rời rạc từng khái niệm mà không nối được luồng request",
        "owner": "nhóm mạng máy tính",
        "deadline": "ngày mai",
    },
    {
        "topic": "ôn tập kỹ thuật phần mềm",
        "method": "liên hệ từng khái niệm với ví dụ trong đồ án",
        "focus": "use case, architecture, testing và deployment",
        "mistake": "mô tả kiến trúc quá chung mà không chỉ ra trách nhiệm từng lớp",
        "owner": "nhóm báo cáo",
        "deadline": "hai ngày trước khi thuyết trình",
    },
]


def write_json_array(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(list(rows), file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def meeting_base(index: int, context: dict[str, str]) -> BaseExample:
    document = (
        f"Cuộc họp của {context['team']} tập trung vào {context['topic']}. "
        f"{context['owner_a']} báo cáo phần đã hoàn thành và nhận nhiệm vụ {context['task_a']}. "
        f"{context['owner_b']} đề xuất ưu tiên các việc có ảnh hưởng trực tiếp đến trải nghiệm người dùng, "
        f"đồng thời sẽ {context['task_b']}. {context['owner_c']} nhắc rằng tài liệu và demo cần được cập nhật cùng lúc, "
        f"nên sẽ {context['task_c']}. Cả nhóm thống nhất quyết định chính là {context['decision']}. "
        f"Rủi ro được ghi nhận là {context['risk']}. Các đầu việc phải hoàn thành trước {context['deadline']}, "
        f"sau đó nhóm sẽ chạy thử lại luồng chính và ghi nhận các lỗi còn tồn tại."
    )
    concise = (
        f"{sentence_case(context['team'])} thống nhất {context['decision']}, phân công {context['owner_a']}, "
        f"{context['owner_b']} và {context['owner_c']} xử lý các việc quan trọng trước {context['deadline']}."
    )
    bullet = (
        f"- {sentence_case(context['team'])} tập trung vào {context['topic']}.\n"
        f"- Quyết định chính: {context['decision']}.\n"
        f"- Rủi ro cần theo dõi: {context['risk']}.\n"
        f"- Các đầu việc phải hoàn thành trước {context['deadline']}."
    )
    action_items = (
        f"- {context['owner_a']} {context['task_a']} trước {context['deadline']}.\n"
        f"- {context['owner_b']} {context['task_b']} trước {context['deadline']}.\n"
        f"- {context['owner_c']} {context['task_c']} trước {context['deadline']}.\n"
        f"- Cả nhóm chạy thử lại luồng chính và ghi nhận lỗi còn tồn tại."
    )
    study_notes = (
        f"- Bối cảnh: {context['topic']}.\n"
        f"- Quyết định cần nhớ: {context['decision']}.\n"
        f"- Rủi ro: {context['risk']}.\n"
        f"- Cách ôn/demo: nêu rõ người phụ trách, deadline và lý do ưu tiên."
    )
    return BaseExample(f"meeting_{index:02d}", "meeting_notes", context["topic"], document, concise, bullet, action_items, study_notes)


def lecture_base(index: int, context: dict[str, str]) -> BaseExample:
    document = (
        f"Bài giảng hôm nay giải thích khái niệm {context['concept']} trong hệ thống Transformer. "
        f"Giảng viên định nghĩa ngắn gọn rằng {context['definition']}. Ý nghĩa chính của khái niệm này là "
        f"{context['benefit']}, nhất là khi xử lý văn bản dài hoặc sinh bản tóm tắt có cấu trúc. "
        f"Ví dụ được đưa ra là {context['example']}. Giảng viên cũng nhấn mạnh lỗi dễ gặp: {context['mistake']}. "
        f"Cuối buổi, sinh viên được giao bài tập {context['exercise']} và tự viết lại khái niệm bằng lời của mình."
    )
    concise = (
        f"Bài giảng làm rõ {context['concept']}: {context['definition']}. Khái niệm này quan trọng vì "
        f"{context['benefit']}, nhưng sinh viên cần tránh lỗi {context['mistake']}."
    )
    bullet = (
        f"- Chủ đề chính: {context['concept']}.\n"
        f"- Định nghĩa: {context['definition']}.\n"
        f"- Lợi ích: {context['benefit']}.\n"
        f"- Ví dụ: {context['example']}."
    )
    action_items = (
        f"- Sinh viên hoàn thành bài tập: {context['exercise']}.\n"
        f"- Sinh viên tự viết lại định nghĩa {context['concept']} bằng lời của mình.\n"
        f"- Nhóm học tập chuẩn bị một ví dụ về {context['concept']} trước buổi ôn.\n"
        f"- Người phụ trách ghi chú thêm lỗi dễ nhầm: {context['mistake']}."
    )
    study_notes = (
        f"- Khái niệm chính: {context['concept']}.\n"
        f"- Cần nhớ: {context['definition']}.\n"
        f"- Ý nghĩa: {context['benefit']}.\n"
        f"- Ví dụ: {context['example']}.\n"
        f"- Lỗi dễ nhầm: {context['mistake']}."
    )
    return BaseExample(f"lecture_{index:02d}", "lecture_notes", context["concept"], document, concise, bullet, action_items, study_notes)


def project_base(index: int, context: dict[str, str]) -> BaseExample:
    document = (
        f"Báo cáo sprint của dự án {context['project']} cho biết nhóm đã có tiến triển: {context['progress']}. "
        f"Tuy nhiên, vấn đề còn tồn tại là {context['issue']}, làm cho trải nghiệm chưa ổn định trong một số trường hợp. "
        f"{context['owner_a']} nhận nhiệm vụ {context['task_a']}, còn {context['owner_b']} sẽ {context['task_b']}. "
        f"Nhóm thống nhất theo dõi chỉ số {context['metric']} để quyết định có cần mở rộng giải pháp hay không. "
        f"Các kết quả thử nghiệm và nhận xét của người dùng sẽ được tổng hợp trước {context['deadline']}."
    )
    concise = (
        f"Dự án {context['project']} đã tiến triển ở luồng chính nhưng vẫn còn vấn đề {context['issue']}. "
        f"Nhóm sẽ theo dõi {context['metric']} và hoàn tất các việc ưu tiên trước {context['deadline']}."
    )
    bullet = (
        f"- Dự án: {context['project']}.\n"
        f"- Tiến triển: {context['progress']}.\n"
        f"- Vấn đề còn lại: {context['issue']}.\n"
        f"- Chỉ số theo dõi: {context['metric']}."
    )
    action_items = (
        f"- {context['owner_a']} {context['task_a']} trước {context['deadline']}.\n"
        f"- {context['owner_b']} {context['task_b']} trước {context['deadline']}.\n"
        f"- Nhóm sản phẩm tổng hợp phản hồi người dùng trước {context['deadline']}.\n"
        f"- Trưởng nhóm cập nhật chỉ số {context['metric']} trong báo cáo sprint."
    )
    study_notes = (
        f"- Bối cảnh dự án: {context['project']}.\n"
        f"- Điểm đã tốt: {context['progress']}.\n"
        f"- Vấn đề cần phân tích: {context['issue']}.\n"
        f"- Chỉ số đánh giá: {context['metric']}.\n"
        f"- Bài học: quyết định cải tiến nên dựa trên số đo và phản hồi thật."
    )
    return BaseExample(f"project_{index:02d}", "project_updates", context["project"], document, concise, bullet, action_items, study_notes)


def study_base(index: int, context: dict[str, str]) -> BaseExample:
    document = (
        f"Tài liệu hướng dẫn {context['topic']} khuyên sinh viên không nên chép lại toàn bộ bài giảng theo thứ tự ban đầu. "
        f"Phương pháp được đề xuất là {context['method']}, sau đó chọn ra các ý có khả năng xuất hiện trong câu hỏi tự luận. "
        f"Phần trọng tâm gồm {context['focus']}. Tài liệu cảnh báo lỗi thường gặp là {context['mistake']}. "
        f"{context['owner']} sẽ tổng hợp phiên bản ghi chú cuối cùng trước {context['deadline']}. "
        f"Mỗi mục trong ghi chú cần có định nghĩa ngắn, ví dụ nhỏ và một cảnh báo dễ nhầm để người học ôn nhanh."
    )
    concise = (
        f"Tài liệu {context['topic']} khuyến nghị sinh viên {context['method']}, tập trung vào {context['focus']} "
        f"và tránh lỗi {context['mistake']}."
    )
    bullet = (
        f"- Chủ đề: {context['topic']}.\n"
        f"- Phương pháp học: {context['method']}.\n"
        f"- Trọng tâm: {context['focus']}.\n"
        f"- Lỗi cần tránh: {context['mistake']}."
    )
    action_items = (
        f"- {sentence_case(context['owner'])} tổng hợp ghi chú cuối cùng trước {context['deadline']}.\n"
        f"- Mỗi thành viên chuẩn bị một ví dụ cho phần {context['focus']}.\n"
        f"- Cả nhóm rà soát lỗi dễ nhầm: {context['mistake']}.\n"
        f"- Người phụ trách gửi bản ôn tập ngắn cho nhóm trước buổi học."
    )
    study_notes = (
        f"- Chủ đề ôn tập: {context['topic']}.\n"
        f"- Cách học: {context['method']}.\n"
        f"- Cần nhớ: {context['focus']}.\n"
        f"- Lỗi dễ nhầm: {context['mistake']}.\n"
        f"- Mẹo ôn nhanh: mỗi mục chỉ giữ định nghĩa, ví dụ và cảnh báo."
    )
    return BaseExample(f"study_{index:02d}", "study_materials", context["topic"], document, concise, bullet, action_items, study_notes)


def cycle_contexts(items: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    return [items[index % len(items)] for index in range(count)]


def build_base_examples() -> list[BaseExample]:
    examples: list[BaseExample] = []
    examples.extend(
        meeting_base(index + 1, context)
        for index, context in enumerate(cycle_contexts(MEETING_CONTEXTS, 18))
    )
    examples.extend(
        lecture_base(index + 1, context)
        for index, context in enumerate(cycle_contexts(LECTURE_CONTEXTS, 18))
    )
    examples.extend(
        project_base(index + 1, context)
        for index, context in enumerate(cycle_contexts(PROJECT_CONTEXTS, 7))
    )
    examples.extend(
        study_base(index + 1, context)
        for index, context in enumerate(cycle_contexts(STUDY_CONTEXTS, 7))
    )
    return examples


def row_for_mode(example: BaseExample, mode: str) -> dict[str, str]:
    return {
        "base_id": example.base_id,
        "document": example.document,
        "summary": getattr(example, mode),
        "mode": mode,
        "domain": example.domain,
    }


def build_rows() -> list[dict[str, str]]:
    return [
        row_for_mode(example, mode)
        for example in build_base_examples()
        for mode in MODES
    ]


def validate_targets(rows: list[dict[str, str]]) -> None:
    mode_counts = Counter(row["mode"] for row in rows)
    domain_counts = Counter(row["domain"] for row in rows)
    base_modes: dict[str, set[str]] = defaultdict(set)
    base_documents: dict[str, str] = {}

    if len(rows) != 200:
        raise ValueError(f"Expected 200 rows, got {len(rows)}.")
    if mode_counts != TARGET_MODE_COUNTS:
        raise ValueError(f"Unexpected mode counts: {dict(mode_counts)}")
    if domain_counts != TARGET_DOMAIN_COUNTS:
        raise ValueError(f"Unexpected domain counts: {dict(domain_counts)}")

    for index, row in enumerate(rows, start=1):
        for field in ("base_id", "document", "summary", "mode", "domain"):
            if not row.get(field):
                raise ValueError(f"Row {index} is missing {field}.")
        if len(row["document"].split()) < 95:
            raise ValueError(f"Row {index} document is too short.")
        if len(row["summary"]) >= len(row["document"]):
            raise ValueError(f"Row {index} summary is not shorter than document.")
        lowered = row["document"].lower()
        if any(phrase in lowered for phrase in FORBIDDEN_META_PHRASES):
            raise ValueError(f"Row {index} contains synthetic meta text.")

        base_id = row["base_id"]
        base_modes[base_id].add(row["mode"])
        if base_id in base_documents and base_documents[base_id] != row["document"]:
            raise ValueError(f"Base document mismatch for {base_id}.")
        base_documents[base_id] = row["document"]

    if len(base_modes) != 50:
        raise ValueError(f"Expected 50 base documents, got {len(base_modes)}.")
    for base_id, modes in base_modes.items():
        if modes != set(MODES):
            raise ValueError(f"{base_id} does not have all modes: {sorted(modes)}")


def build_qualitative_rows() -> list[dict[str, str]]:
    selected_ids = (
        "meeting_01",
        "meeting_02",
        "meeting_03",
        "lecture_01",
        "lecture_02",
        "lecture_03",
        "project_01",
        "project_02",
        "study_01",
        "study_02",
    )
    examples = {example.base_id: example for example in build_base_examples()}
    return [
        {
            "id": base_id,
            "domain": examples[base_id].domain,
            "document": examples[base_id].document,
        }
        for base_id in selected_ids
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paired 200-row synthetic dataset for phase 2 controllability."
    )
    parser.add_argument("--output", default="data/synthetic/reviewed_all.json")
    parser.add_argument("--qualitative-output", default="data/samples/qualitative_mode_eval.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows()
    validate_targets(rows)

    write_json_array(Path(args.output), rows)
    write_jsonl(Path(args.qualitative_output), build_qualitative_rows())

    print(f"Saved {len(rows)} paired reviewed samples to {args.output}")
    print(f"Base documents: {len({row['base_id'] for row in rows})}")
    print(f"Mode counts: {dict(Counter(row['mode'] for row in rows))}")
    print(f"Domain counts: {dict(Counter(row['domain'] for row in rows))}")
    print(f"Saved qualitative evaluation set to {args.qualitative_output}")


if __name__ == "__main__":
    main()
