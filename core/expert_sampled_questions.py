from __future__ import annotations

from typing import Dict, FrozenSet, List, Set, Tuple


EXPERT_SAMPLED_QUESTION_IDS_BY_SPLIT: Dict[str, List[str]] = {
    "chapter_intro": [
        "1/iii#2",
    ],
    "chapter_1": [
        "1.2/main",
        "1.5/main",
    ],
    "chapter_2": [
        "2.7/i#2",
    ],
    "chapter_3": [
        "3.1/ii#2",
        "3.7/i",
    ],
    "chapter_4": [
        "4.5/iii",
        "4.5/ii",
        "4.7/main",
    ],
    "chapter_5": [
        "5.3/iii",
    ],
    "chapter_6": [
        "6.3/main",
        "6.2/main",
    ],
    "chapter_7": [
        "7.2/2",
    ],
    "chapter_8": [
        "8.10/2",
        "8.6/2#2",
        "8.10/4#1",
    ],
    "chapter_9": [
        "9.1/2#1",
    ],
    "chapter_10": [
        "10.2/2#2",
        "10.6/1",
        "10.3/main#1",
    ],
}


def big_question_id(question_id: str) -> str:
    return str(question_id or "").split("/", 1)[0].strip()


def build_sampled_big_questions_by_split() -> Dict[str, Set[str]]:
    selected: Dict[str, Set[str]] = {}
    for split, question_ids in EXPERT_SAMPLED_QUESTION_IDS_BY_SPLIT.items():
        selected[split] = {
            big_question_id(question_id)
            for question_id in question_ids
            if big_question_id(question_id)
        }
    return selected


EXPERT_SAMPLED_BIG_QUESTIONS_BY_SPLIT = build_sampled_big_questions_by_split()
EXPERT_SAMPLED_QUESTION_PAIRS: FrozenSet[Tuple[str, str]] = frozenset(
    (split, question_id)
    for split, question_ids in EXPERT_SAMPLED_QUESTION_IDS_BY_SPLIT.items()
    for question_id in question_ids
)
