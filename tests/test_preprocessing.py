from smart_summarizer.data.preprocessing import clean_text, is_valid_pair, truncate_by_words


def test_clean_text_normalizes_whitespace() -> None:
    assert clean_text("  Xin   chào   Việt Nam .  ") == "Xin chào Việt Nam."


def test_invalid_empty_pair() -> None:
    assert not is_valid_pair("", "")


def test_truncate_by_words() -> None:
    assert truncate_by_words("một hai ba bốn", 2) == "một hai"
