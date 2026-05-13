from smart_summarizer.product.postprocess import postprocess_summary


def test_bullet_postprocess_adds_markers() -> None:
    result = postprocess_summary("Ý thứ nhất. Ý thứ hai.", mode="bullet")
    assert result.splitlines()[0].startswith("- ")


def test_concise_postprocess_keeps_plain_text() -> None:
    assert postprocess_summary("  Một tóm tắt ngắn. ", mode="concise") == "Một tóm tắt ngắn."


def test_action_items_are_line_based() -> None:
    result = postprocess_summary("An chuẩn bị báo cáo. Bình kiểm tra demo.", mode="action_items")
    assert "- An chuẩn bị báo cáo." in result


def test_action_items_preserve_no_task_message() -> None:
    result = postprocess_summary("Không có việc cần làm rõ ràng.", mode="action_items")
    assert result == "Không có việc cần làm rõ ràng."


def test_study_notes_use_study_labels() -> None:
    result = postprocess_summary("Self-attention. Mỗi token xét token khác.", mode="study_notes")
    assert result.splitlines()[0].startswith("Khái niệm chính:")
