from scripts.build_lora_mixed_data import action_summary, lora_row, pseudo_bullet_summary, stable_hash


def test_stable_hash_normalizes_whitespace_and_case() -> None:
    assert stable_hash("  Xin Chào  ") == stable_hash("xin chào")


def test_pseudo_bullet_summary_builds_bullets() -> None:
    row = {
        "document": "Một. Hai. Ba.",
        "summary": "Ý thứ nhất đủ dài. Ý thứ hai cũng đủ dài.",
    }
    result = pseudo_bullet_summary(row)
    assert result.startswith("- ")
    assert "\n- " in result


def test_action_summary_uses_owner_action_deadline_format() -> None:
    row = {
        "document": "Bước một là chuẩn bị nguyên liệu. Bước hai là nấu trong mười phút.",
        "summary": "Chuẩn bị nguyên liệu. Nấu trong mười phút.",
    }
    result = action_summary(row)
    assert "Người phụ trách:" in result
    assert "Hành động:" in result
    assert "Deadline:" in result


def test_lora_row_has_required_metadata() -> None:
    row = {"document": "Nội dung đầu vào đủ dài.", "summary": "Tóm tắt ngắn."}
    result = lora_row(
        row,
        mode="concise",
        source="unit",
        task_type="test",
        domain="news",
    )
    assert result["mode"] == "concise"
    assert result["source"] == "unit"
    assert result["task_type"] == "test"
