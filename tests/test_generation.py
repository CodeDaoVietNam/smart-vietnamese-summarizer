import pytest

from smart_summarizer.modeling.generation import build_instruction, generation_kwargs


def test_build_instruction_uses_mode_prefix() -> None:
    instruction = build_instruction("Nội dung cần tóm tắt", mode="bullet")
    assert instruction.startswith("Tóm tắt thành các ý chính dạng bullet, mỗi bullet một ý:")


def test_generation_kwargs_uses_length_budget() -> None:
    kwargs = generation_kwargs(length="short")
    assert kwargs["max_new_tokens"] == 64
    assert kwargs["num_beams"] == 4


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        build_instruction("text", mode="unknown")
