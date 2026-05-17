from smart_summarizer.product.keyword_extractor import extract_keywords


def test_extract_keywords_includes_meaningful_bigrams() -> None:
    keywords = extract_keywords(
        "Mô hình học máy dùng xử lý ngôn ngữ tự nhiên. Học máy hỗ trợ tóm tắt văn bản.",
        max_keywords=6,
    )
    assert any("học máy" == keyword for keyword in keywords)
