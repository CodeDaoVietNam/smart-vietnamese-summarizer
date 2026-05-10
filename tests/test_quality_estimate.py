from smart_summarizer.product.quality_estimate import compute_quality_estimate


def test_quality_estimate_range() -> None:
    score = compute_quality_estimate(
        "An sẽ chuẩn bị báo cáo trước thứ Sáu và nhóm chạy thử demo.",
        "An chuẩn bị báo cáo trước thứ Sáu.",
        ["an", "báo", "cáo"],
    )
    assert 0 <= score <= 100


def test_empty_quality_estimate_is_zero() -> None:
    assert compute_quality_estimate("", "") == 0.0
