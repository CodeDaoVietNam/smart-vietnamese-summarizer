from pathlib import Path

from smart_summarizer.product.critics import critic_action_items, critic_study_notes
from smart_summarizer.product.extractors import extract_action_items, extract_study_notes
from smart_summarizer.product.renderers import render_action_items, render_study_notes


def test_action_extractor_prefers_source_over_noisy_draft() -> None:
    source = (
        "An phụ trách rà soát dữ liệu trước tối thứ Tư, "
        "Bình chịu trách nhiệm kiểm tra backend trước chiều thứ Năm, "
        "còn Chi sẽ hoàn thiện frontend."
    )
    draft = "Người phụ trách: Trong Hành động: hoàn thành demo Deadline: sáng thứ Hai"

    items = extract_action_items(source=source, draft=draft, length="long")
    rendered = render_action_items(items)

    assert [item.owner for item in items] == ["An", "Bình", "Chi"]
    assert "Người phụ trách: Trong" not in rendered
    assert critic_action_items(items).is_valid


def test_action_extractor_filters_context_only_sentences() -> None:
    source = (
        "Trong cuộc họp, nhóm thống nhất demo cần hoàn thành trước thứ Sáu. "
        "An phụ trách rà soát dữ liệu trước tối thứ Tư."
    )
    items = extract_action_items(source=source, draft="", length="medium")

    assert [item.owner for item in items] == ["An"]
    assert "Chưa rõ" not in render_action_items(items)


def test_study_extractor_maps_example_and_misconception() -> None:
    source = Path("data/samples/web_demo/03_lecture_transformer_attention.txt").read_text(
        encoding="utf-8"
    )
    notes = extract_study_notes(source=source, draft="", length="long")
    rendered = render_study_notes(notes)

    assert "Khái niệm chính:" in rendered
    assert "Cần nhớ:" in rendered
    assert "Ví dụ:" in rendered
    assert "Lỗi dễ nhầm:" in rendered
    assert "sinh viên nộp báo cáo" in notes.example
    assert "lỗi dễ nhầm" in notes.misconception.lower()
    assert critic_study_notes(notes).is_valid
