from smart_summarizer.product.formatters import format_summary, validate_mode_output


def test_action_items_repair_inline_fields() -> None:
    draft = "Người phụ trách: Phương Hành động: viết tooltip giải thích Deadline: chiều thứ Sáu"
    result = format_summary(draft, source="", mode="action_items")
    assert "- Người phụ trách: Phương" in result
    assert "  Hành động: viết tooltip giải thích" in result
    assert "  Deadline: chiều thứ Sáu" in result


def test_action_items_fill_missing_deadline() -> None:
    result = format_summary("An chuẩn bị báo cáo.", source="", mode="action_items")
    assert "Deadline: Chưa rõ" in result


def test_action_items_no_task_message() -> None:
    result = format_summary("Không có việc cần làm rõ ràng.", source="", mode="action_items")
    assert result == "Không có việc cần làm rõ ràng."


def test_action_items_prefers_named_owners_from_source() -> None:
    source = (
        "Trong cuộc họp, nhóm cần hoàn thành demo trước thứ Sáu. "
        "An phụ trách rà soát dữ liệu trước thứ Tư. "
        "Bình chịu trách nhiệm kiểm tra backend trước thứ Năm."
    )
    result = format_summary("Người phụ trách: Trong Hành động: hoàn thành demo", source=source, mode="action_items")
    assert "Người phụ trách: An" in result
    assert "Người phụ trách: Bình" in result
    assert "Người phụ trách: Trong" not in result


def test_action_items_splits_multiple_owner_tasks() -> None:
    source = (
        "Trong cuộc họp sáng thứ Hai, nhóm thống nhất demo cần hoàn thành trước 17h thứ Sáu. "
        "An phụ trách rà soát dữ liệu trước tối thứ Tư. "
        "Bình chịu trách nhiệm kiểm tra backend trước chiều thứ Năm. "
        "Chi sẽ hoàn thiện frontend. "
        "Dũng phụ trách viết nhận xét kết quả."
    )
    result = format_summary("", source=source, mode="action_items", length="long")

    assert "Người phụ trách: An" in result
    assert "Người phụ trách: Bình" in result
    assert "Người phụ trách: Chi" in result
    assert "Người phụ trách: Dũng" in result
    assert "Người phụ trách: Chưa rõ" not in result


def test_action_items_splits_tasks_joined_by_conjunctions() -> None:
    source = (
        "An phụ trách rà soát dữ liệu trước tối thứ Tư, "
        "Bình chịu trách nhiệm kiểm tra backend trước chiều thứ Năm, "
        "còn Chi sẽ hoàn thiện frontend."
    )
    result = format_summary("", source=source, mode="action_items", length="long")

    assert "Người phụ trách: An" in result
    assert "Người phụ trách: Bình" in result
    assert "Người phụ trách: Chi" in result


def test_study_notes_repair_missing_label() -> None:
    result = format_summary(
        "Khái niệm chính: Self-attention. Cần nhớ: token xét token khác. Ví dụ: câu dài.",
        source="Lỗi dễ nhầm là nghĩ attention tự biết vị trí.",
        mode="study_notes",
    )
    assert "Khái niệm chính:" in result
    assert "Cần nhớ:" in result
    assert "Ví dụ:" in result
    assert "Lỗi dễ nhầm:" in result


def test_bullet_dedupes_and_marks_items() -> None:
    result = format_summary("Ý thứ nhất. Ý thứ nhất. Ý thứ hai.", source="", mode="bullet")
    lines = result.splitlines()
    assert lines == ["- Ý thứ nhất.", "- Ý thứ hai."]


def test_bullet_length_changes_item_limit() -> None:
    text = "Ý một. Ý hai. Ý ba. Ý bốn. Ý năm. Ý sáu."
    short = format_summary(text, source="", mode="bullet", length="short")
    long = format_summary(text, source="", mode="bullet", length="long")
    assert len(short.splitlines()) == 2
    assert len(long.splitlines()) == 6


def test_concise_length_changes_sentence_limit() -> None:
    text = "Câu một. Câu hai. Câu ba. Câu bốn."
    short = format_summary(text, source="", mode="concise", length="short")
    long = format_summary(text, source="", mode="concise", length="long")
    assert short == "Câu một."
    assert "Câu bốn." in long


def test_concise_strips_mode_labels_and_bullets() -> None:
    result = format_summary("- Khái niệm chính: Transformer xử lý chuỗi.", source="", mode="concise")
    assert not result.startswith("-")
    assert "Khái niệm chính:" not in result


def test_formatter_removes_length_instruction_leakage() -> None:
    draft = (
        "Độ dài: chi tiết hơn, giữ thêm bối cảnh và các ý phụ quan trọng. "
        "Transformer dùng self-attention để liên kết các token."
    )
    result = format_summary(draft, source="", mode="concise", length="long")
    assert "Độ dài:" not in result
    assert "Transformer dùng self-attention" in result


def test_formatter_removes_generic_finetune_hallucination() -> None:
    draft = (
        "Sau khi fine-tune, mô hình học cách ánh xạ văn bản dài thành bản tóm tắt ngắn. "
        "Self-attention giúp token tham chiếu đến token khác trong cùng chuỗi."
    )
    result = format_summary(draft, source="", mode="bullet", length="medium")
    assert "fine-tune" not in result.lower()
    assert "Self-attention" in result


def test_formatter_does_not_remove_content_word_tom_tat() -> None:
    source = (
        "Trong cuộc họp, demo của hệ thống tóm tắt tiếng Việt cần hoàn thành trước thứ Sáu. "
        "An phụ trách rà soát dữ liệu trước tối thứ Tư."
    )
    result = format_summary("", source=source, mode="action_items", length="medium")
    assert "Người phụ trách: An" in result
    assert "rà soát dữ liệu" in result


def test_study_notes_removes_prompt_fragments() -> None:
    result = format_summary(
        "Khái niệm chính: Nêu khái niệm chính, attention là cơ chế tập trung vào token.",
        source="Giảng viên nhấn mạnh lỗi dễ nhầm là xem attention như lời giải thích tuyệt đối.",
        mode="study_notes",
    )
    assert "Nêu khái niệm chính" not in result
    assert "attention là cơ chế" in result


def test_bullet_removes_khai_niem_bullet_label() -> None:
    result = format_summary(
        "Khái niệm bullet: attention là cơ chế tập trung. Ý thứ hai.",
        source="",
        mode="bullet",
    )
    assert "Khái niệm bullet" not in result
    assert result.startswith("- attention")


def test_bullet_removes_near_duplicate_items() -> None:
    result = format_summary(
        (
            "attention là cơ chế giúp mô hình tập trung vào chuỗi đầu vào. "
            "Trong bài học, attention là cơ chế giúp mô hình tập trung vào chuỗi đầu vào khi sinh đầu ra."
        ),
        source="",
        mode="bullet",
        length="medium",
    )
    assert len(result.splitlines()) == 1


def test_validate_mode_output_reports_missing_fields() -> None:
    validation = validate_mode_output("Khái niệm chính: Transformer.", mode="study_notes")
    assert not validation["is_valid"]
    assert "Cần nhớ" in validation["missing_fields"]
