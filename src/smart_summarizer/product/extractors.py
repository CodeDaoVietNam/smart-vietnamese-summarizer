from __future__ import annotations

from dataclasses import dataclass
import re

from smart_summarizer.data.preprocessing import clean_text


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+")
LABEL_RE = re.compile(
    r"(?i)\b(khái niệm chính|khái niệm bullet|cần nhớ|ví dụ|ý nghĩa|lỗi dễ nhầm|người phụ trách|hành động|deadline)\s*:"
)
INSTRUCTION_LEAKAGE_RE = re.compile(
    r"(?i)\b("
    r"độ dài\s*:\s*(?:rất ngắn|vừa phải|chi tiết hơn)[^.?!。]*(?:[.?!。]|$)|"
    r"(?:^|(?<=[.?!。])\s*)tóm tắt(?:\s+ngắn gọn|\s+thành)[^.?!。]{0,120}(?:[.?!。]|$)|"
    r"chỉ trích xuất việc cần làm[^.?!。]*(?:[.?!。]|$)|"
    r"tạo ghi chú học tập[^.?!。]*(?:[.?!。]|$)|"
    r"nêu khái niệm chính,?\s*|"
    r"mỗi bullet một ý,?\s*|"
    r"sau khi fine-?tune,\s*mô hình học cách ánh xạ văn bản dài thành bản tóm tắt ngắn\.?"
    r")"
)
ACTION_LABEL_RE = re.compile(r"(?i)(người phụ trách|hành động|deadline)\s*:")
DATE_RE = re.compile(
    r"(?i)\b("
    r"trước\s+[^,.。;]+|"
    r"sáng\s+[^,.。;]+|chiều\s+[^,.。;]+|tối\s+[^,.。;]+|"
    r"cuối\s+[^,.。;]+|ngày\s+\d{1,2}|thứ\s+[A-Za-zÀ-ỹ0-9 ]+"
    r")"
)
TASK_RE = re.compile(
    r"(?i)\b("
    r"phụ trách|nhận nhiệm vụ|sẽ|cần|chuẩn bị|kiểm tra|hoàn thiện|"
    r"cập nhật|tổng hợp|rà soát|thiết kế|viết|gửi|xác nhận|thống kê|"
    r"soạn|chạy thử|đo|lọc|review|chịu trách nhiệm"
    r")\b"
)
OWNER_STOP_WORDS = {
    "Buổi",
    "Cả",
    "Dự",
    "Ngoài",
    "Nhóm",
    "Thứ",
    "Trong",
    "Việc",
}
OWNER_TRIGGER_RE = re.compile(
    r"^([A-ZÀ-Ỹ][A-Za-zÀ-ỹ]{1,20})\s+"
    r"(phụ trách|chịu trách nhiệm|sẽ|cần|nhận nhiệm vụ|đảm nhiệm)\b",
    re.IGNORECASE,
)
OWNER_TRIGGER_ANYWHERE_RE = re.compile(
    r"(?=(?:^|[,.;]\s+|(?:\s+(?:và|còn)\s+))"
    r"[A-ZÀ-Ỹ][A-Za-zÀ-ỹ]{1,20}\s+"
    r"(?:phụ trách|chịu trách nhiệm|sẽ|cần|nhận nhiệm vụ|đảm nhiệm)\b)",
    re.IGNORECASE,
)

STUDY_LABELS = ("Khái niệm chính", "Cần nhớ", "Ví dụ", "Lỗi dễ nhầm")
LENGTH_SENTENCE_LIMITS = {
    "short": 1,
    "medium": 2,
    "long": 4,
}
LENGTH_ITEM_LIMITS = {
    "short": 2,
    "medium": 4,
    "long": 6,
}
STUDY_WORD_LIMITS = {
    "short": 18,
    "medium": 34,
    "long": 64,
}


@dataclass(frozen=True)
class ActionItem:
    owner: str
    action: str
    deadline: str
    evidence: str


@dataclass(frozen=True)
class StudyNotes:
    concept: str
    remember: str
    example: str
    misconception: str


def remove_instruction_leakage(text: str) -> str:
    previous = text or ""
    while True:
        cleaned = INSTRUCTION_LEAKAGE_RE.sub("", previous)
        cleaned = clean_text(cleaned).strip(" -*•:")
        if cleaned == previous:
            return cleaned
        previous = cleaned


def split_sentences(text: str) -> list[str]:
    text = remove_instruction_leakage(text)
    lines = [clean_text(line).strip(" -*•\t") for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) > 1:
        return lines

    text = clean_text(text)
    if not text:
        return []

    parts = SENTENCE_SPLIT_RE.split(text)
    if len(parts) == 1:
        parts = re.split(r"(?:^|\s+)[-*•]\s+", text)
    return [part.strip(" -*•\t") for part in parts if part.strip(" -*•\t")]


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = clean_text(item).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(clean_text(item))
    return result


def word_set(text: str) -> set[str]:
    return {word for word in re.findall(r"\w+", clean_text(text).lower()) if len(word) > 2}


def is_near_duplicate(left: str, right: str) -> bool:
    left_words = word_set(left)
    right_words = word_set(right)
    if not left_words or not right_words:
        return False
    overlap = len(left_words & right_words) / min(len(left_words), len(right_words))
    return overlap >= 0.72


def dedupe_near_preserve_order(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        duplicate_index = next(
            (index for index, existing in enumerate(result) if is_near_duplicate(item, existing)),
            None,
        )
        if duplicate_index is None:
            result.append(item)
        elif len(item) > len(result[duplicate_index]):
            result[duplicate_index] = item
    return result


def strip_mode_labels(text: str) -> str:
    text = remove_instruction_leakage(text)
    return clean_text(LABEL_RE.sub("", text).strip(" -*•:"))


def first_sentences(text: str, limit: int = 3) -> list[str]:
    return split_sentences(text)[:limit]


def limit_words(text: str, limit: int) -> str:
    words = clean_text(text).split()
    if len(words) <= limit:
        return clean_text(text)
    return " ".join(words[:limit]).rstrip(" ,;:") + "..."


def length_value(length: str, mapping: dict[str, int], default: str = "medium") -> int:
    return mapping.get(length, mapping[default])


def find_first_matching(sentences: list[str], patterns: tuple[str, ...]) -> str | None:
    for sentence in sentences:
        if any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in patterns):
            return sentence
    return None


def has_action_signal(text: str) -> bool:
    return bool(TASK_RE.search(text or ""))


def extract_deadline(text: str) -> str:
    match = DATE_RE.search(text or "")
    if not match:
        return "Chưa rõ"
    return clean_text(match.group(1).rstrip("."))


def extract_owner(text: str) -> str:
    owner_match = re.search(r"(?i)người phụ trách\s*:\s*([^:\n]+?)(?=\s+hành động\s*:|$)", text)
    if owner_match:
        owner = clean_text(owner_match.group(1).strip(" -.;,"))
        trigger_match = OWNER_TRIGGER_RE.search(owner)
        if trigger_match:
            owner = trigger_match.group(1)
        owner_words = owner.split()
        if owner_words and owner_words[0] in OWNER_STOP_WORDS:
            return "Chưa rõ"
        if len(owner_words) > 2:
            return " ".join(owner_words[:2])
        return owner

    sentence = clean_text(text)
    trigger_match = OWNER_TRIGGER_RE.search(sentence)
    if trigger_match and trigger_match.group(1) not in OWNER_STOP_WORDS:
        return trigger_match.group(1)

    sentence = re.sub(r"(?i)\b(sẽ|cần|nhận nhiệm vụ|phụ trách)\b.*", "", sentence).strip(" -.;,")
    words = sentence.split()
    if 1 <= len(words) <= 4:
        if words[0] in OWNER_STOP_WORDS:
            return "Chưa rõ"
        return sentence
    if words and words[0][:1].isupper() and words[0] not in OWNER_STOP_WORDS:
        return words[0]
    return "Chưa rõ"


def extract_action(text: str) -> str:
    action_match = re.search(r"(?i)hành động\s*:\s*(.+?)(?=\s+deadline\s*:|$)", text)
    if action_match:
        return clean_text(action_match.group(1).strip(" -.;,"))

    cleaned = ACTION_LABEL_RE.sub("", clean_text(text)).strip(" -.;,")
    cleaned = DATE_RE.sub("", cleaned).strip(" -.;,")
    cleaned = re.sub(r"(?i)^người phụ trách\s*", "", cleaned).strip(" -.;,")
    return cleaned or "Chưa rõ"


def split_action_clauses(text: str) -> list[str]:
    sentence = clean_text(text)
    starts = [match.start() for match in OWNER_TRIGGER_ANYWHERE_RE.finditer(sentence)]
    if not starts:
        return [sentence]

    clauses: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(sentence)
        clause = sentence[start:end].strip(" ,.;")
        clause = re.sub(r"^(?i:và|còn)\s+", "", clause).strip(" ,.;")
        if clause:
            clauses.append(clause)
    return clauses or [sentence]


def candidate_action_sentences(text: str) -> list[str]:
    candidates: list[str] = []
    for item in split_sentences(text):
        if not has_action_signal(item):
            continue
        candidates.extend(split_action_clauses(item))
    return [item for item in candidates if has_action_signal(item)]


def is_context_only_action(text: str) -> bool:
    lowered = clean_text(text).lower()
    return (
        extract_owner(text) == "Chưa rõ"
        and any(marker in lowered for marker in ("trong cuộc họp", "nhóm thống nhất", "cả nhóm đồng ý"))
    )


def normalize_action_item(text: str) -> ActionItem:
    return ActionItem(
        owner=extract_owner(text),
        action=extract_action(text),
        deadline=extract_deadline(text),
        evidence=clean_text(text),
    )


def extract_action_items(source: str, draft: str, length: str = "medium") -> list[ActionItem]:
    combined = f"{draft}\n{source or ''}"
    if "không có việc cần làm" in combined.lower():
        return []

    limit = length_value(length, LENGTH_ITEM_LIMITS)
    # Source sentences are usually more factual than a noisy generated draft.
    candidates = candidate_action_sentences(source or "")
    if len(candidates) < 2:
        candidates.extend(candidate_action_sentences(draft))
    candidates = dedupe_preserve_order(candidates)
    candidates = [item for item in candidates if not is_context_only_action(item)]

    items = [normalize_action_item(item) for item in candidates]
    owned_items = [item for item in items if item.owner != "Chưa rõ"]
    if owned_items:
        items = owned_items
    return items[:limit]


def parse_labeled_lines(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    pattern = re.compile(
        r"(?i)(khái niệm chính|cần nhớ|ví dụ|lỗi dễ nhầm)\s*:\s*"
        r"(.+?)(?=(?:khái niệm chính|cần nhớ|ví dụ|lỗi dễ nhầm)\s*:|$)"
    )
    for label, value in pattern.findall(text or ""):
        canonical = {
            "khái niệm chính": "Khái niệm chính",
            "cần nhớ": "Cần nhớ",
            "ví dụ": "Ví dụ",
            "lỗi dễ nhầm": "Lỗi dễ nhầm",
        }[label.lower()]
        result[canonical] = clean_text(value.strip(" -.;"))
    return result


def extract_study_notes(source: str, draft: str, length: str = "medium") -> StudyNotes:
    labeled = parse_labeled_lines(draft)
    generated_sentences = split_sentences(draft)
    source_sentences = split_sentences(source or "")
    fallback_sentences = dedupe_preserve_order(generated_sentences + source_sentences)
    word_limit = length_value(length, STUDY_WORD_LIMITS)
    source_hints = {
        "Khái niệm chính": find_first_matching(source_sentences, (r"\blà\b", r"cơ chế", r"khái niệm")),
        "Cần nhớ": find_first_matching(
            source_sentences,
            (r"thay vì", r"khác với", r"điểm quan trọng", r"nhấn mạnh", r"multi-head", r"positional"),
        ),
        "Ví dụ": find_first_matching(source_sentences, (r"ví dụ", r"chẳng hạn")),
        "Lỗi dễ nhầm": find_first_matching(source_sentences, (r"lỗi dễ nhầm", r"dễ nhầm", r"không phải", r"thực tế")),
    }

    values: dict[str, str] = {}
    for index, label in enumerate(STUDY_LABELS):
        if labeled.get(label):
            values[label] = limit_words(strip_mode_labels(labeled[label]), word_limit)
        elif source_hints.get(label):
            values[label] = limit_words(strip_mode_labels(source_hints[label]), word_limit)
        elif index < len(fallback_sentences):
            values[label] = limit_words(strip_mode_labels(fallback_sentences[index]), word_limit)
        else:
            values[label] = "Chưa nêu rõ trong văn bản"

    return StudyNotes(
        concept=values["Khái niệm chính"],
        remember=values["Cần nhớ"],
        example=values["Ví dụ"],
        misconception=values["Lỗi dễ nhầm"],
    )
