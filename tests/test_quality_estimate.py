from smart_summarizer.product.quality_estimate import compute_quality_estimate, quality_estimate_from_scores


def test_quality_estimate_range() -> None:
    score = compute_quality_estimate(
        "An sẽ chuẩn bị báo cáo trước thứ Sáu và nhóm chạy thử demo.",
        "An chuẩn bị báo cáo trước thứ Sáu.",
        ["an", "báo", "cáo"],
    )
    assert 0 <= score <= 100


def test_empty_quality_estimate_is_zero() -> None:
    assert compute_quality_estimate("", "") == 0.0


def test_generation_score_proxy_is_bounded() -> None:
    assert quality_estimate_from_scores([100.0, -100.0]) is not None
    assert 0 <= quality_estimate_from_scores([100.0, -100.0]) <= 100
