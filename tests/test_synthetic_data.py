import json
from collections import Counter, defaultdict
from pathlib import Path


MODES = {"concise", "bullet", "action_items", "study_notes"}


def load_jsonl(path: str) -> list[dict[str, str]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_reviewed_synthetic_400_distribution() -> None:
    rows = json.loads(Path("data/synthetic/reviewed_all.json").read_text(encoding="utf-8"))

    assert len(rows) == 400
    assert Counter(row["mode"] for row in rows) == {
        "concise": 100,
        "bullet": 100,
        "action_items": 100,
        "study_notes": 100,
    }
    assert Counter(row["domain"] for row in rows) == {
        "meeting_notes": 120,
        "lecture_notes": 120,
        "project_updates": 80,
        "general_articles": 80,
    }


def test_each_base_id_has_all_modes() -> None:
    rows = json.loads(Path("data/synthetic/reviewed_all.json").read_text(encoding="utf-8"))
    grouped: dict[str, set[str]] = defaultdict(set)
    documents: dict[str, str] = {}

    for row in rows:
        grouped[row["base_id"]].add(row["mode"])
        documents.setdefault(row["base_id"], row["document"])
        assert documents[row["base_id"]] == row["document"]

    assert len(grouped) == 100
    assert all(modes == MODES for modes in grouped.values())


def test_train_validation_split_has_no_base_leakage() -> None:
    train_rows = load_jsonl("data/synthetic/train.jsonl")
    validation_rows = load_jsonl("data/synthetic/validation.jsonl")

    assert len(train_rows) == 320
    assert len(validation_rows) == 80
    assert {row["base_id"] for row in train_rows}.isdisjoint(
        {row["base_id"] for row in validation_rows}
    )
    assert Counter(row["mode"] for row in train_rows) == {mode: 80 for mode in MODES}
    assert Counter(row["mode"] for row in validation_rows) == {mode: 20 for mode in MODES}


def test_holdout_eval_set_distribution() -> None:
    rows = load_jsonl("data/samples/holdout_mode_eval.jsonl")

    assert len(rows) == 20
    assert Counter(row["domain"] for row in rows) == {
        "meeting_notes": 5,
        "lecture_notes": 5,
        "project_updates": 5,
        "general_articles": 5,
    }
