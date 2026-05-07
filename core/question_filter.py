from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _split_question_id_tokens(raw_text: str) -> List[str]:
    if not raw_text:
        return []
    return [token.strip() for token in re.split(r"[\s,]+", raw_text) if token.strip()]


def load_question_id_filter(
    question_ids: Optional[str] = None,
    question_ids_file: Optional[str] = None,
) -> Optional[Set[str]]:
    tokens: Set[str] = set()
    explicit_filter_requested = False

    if question_ids:
        explicit_filter_requested = True
        tokens.update(_split_question_id_tokens(question_ids))

    if question_ids_file is not None:
        explicit_filter_requested = True
        path = Path(question_ids_file)
        if not path.exists():
            raise RuntimeError(f"Question id file not found: {path}")
        tokens.update(_split_question_id_tokens(path.read_text(encoding="utf-8")))

    if explicit_filter_requested:
        return tokens
    return None


def filter_rows_by_question_ids(
    rows: List[Dict[str, Any]],
    allowed_question_ids: Optional[Set[str]],
) -> List[Dict[str, Any]]:
    if allowed_question_ids is None:
        return rows
    return [row for row in rows if str(row.get("id", "")).strip() in allowed_question_ids]


def merge_rows_by_question_id(
    existing_rows: List[Dict[str, Any]],
    replacement_rows: List[Dict[str, Any]],
    ordered_question_ids: List[str],
    *,
    keep_unordered_existing: bool = True,
) -> List[Dict[str, Any]]:
    existing_by_id = {
        str(row.get("id", "")).strip(): row
        for row in existing_rows
        if str(row.get("id", "")).strip()
    }
    replacement_by_id = {
        str(row.get("id", "")).strip(): row
        for row in replacement_rows
        if str(row.get("id", "")).strip()
    }
    merged_by_id = dict(existing_by_id)
    merged_by_id.update(replacement_by_id)

    final_rows: List[Dict[str, Any]] = []
    used_ids: Set[str] = set()
    for qid in ordered_question_ids:
        if qid in merged_by_id and qid not in used_ids:
            final_rows.append(merged_by_id[qid])
            used_ids.add(qid)

    for row in existing_rows:
        qid = str(row.get("id", "")).strip()
        if not qid:
            if keep_unordered_existing:
                final_rows.append(row)
            continue
        if qid in merged_by_id and qid not in used_ids:
            final_rows.append(merged_by_id[qid])
            used_ids.add(qid)

    for row in replacement_rows:
        qid = str(row.get("id", "")).strip()
        if not qid:
            if keep_unordered_existing:
                final_rows.append(row)
            continue
        if qid not in used_ids:
            final_rows.append(row)
            used_ids.add(qid)

    return final_rows
