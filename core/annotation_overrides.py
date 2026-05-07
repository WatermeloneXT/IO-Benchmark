from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def review_annotations_root(paths: Dict[str, Any]) -> Path:
    hf_json_dir = Path(paths["hf_json_dir"]).resolve()
    base = Path(paths.get("review_annotations_dir") or (hf_json_dir.parent / "review_annotations")).resolve()
    return base / "post_step5"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _is_meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def _json_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001
        return str(value).strip()


def _extract_answer_overrides(raw_value: Any) -> Tuple[str, str, str]:
    if isinstance(raw_value, dict):
        answer_text = ""
        sympy_text = ""
        compare_text = ""
        for key in ("reference_answer", "answer", "final_answer", "text", "value"):
            value = _normalize_text(raw_value.get(key))
            if value:
                answer_text = value
                break
        for key in ("reference_answer_sympy", "answer_sympy", "sympy", "final_answer_for_compare", "value_sympy"):
            value = _normalize_text(raw_value.get(key))
            if value:
                sympy_text = value
                break
        for key in ("final_answer_for_compare", "reference_answer_sympy", "reference_answer", "answer", "final_answer", "value"):
            value = _normalize_text(raw_value.get(key))
            if value:
                compare_text = value
                break
        if not answer_text and raw_value:
            answer_text = _json_string(raw_value)
        if not compare_text:
            compare_text = sympy_text or answer_text
        return answer_text, sympy_text, compare_text

    raw_text = _json_string(raw_value)
    if not raw_text:
        return "", "", ""
    return raw_text, raw_text, raw_text


def load_split_annotation_overrides(paths: Dict[str, Any], split_name: str) -> Dict[str, Dict[str, Any]]:
    root = review_annotations_root(paths)
    overrides: Dict[str, Dict[str, Any]] = {}
    if not root.exists():
        return overrides

    for path in sorted(root.glob(f"*/{split_name}.json")):
        doc = _read_json(path)
        entries = doc.get("entries", {})
        if not isinstance(entries, dict):
            continue
        annotator = _normalize_text(doc.get("annotator")) or path.parent.name
        for question_id, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            qid = _normalize_text(question_id) or _normalize_text(entry.get("question_id"))
            if not qid:
                continue
            rewritten_question = _normalize_text(entry.get("annotator_rewritten_question"))
            rewritten_solution = _normalize_text(entry.get("annotator_rewritten_solution"))
            rewritten_answer = entry.get("annotator_rewritten_answer", entry.get("annotation"))
            annotator_comment = _normalize_text(entry.get("annotator_comment", entry.get("comment")))
            overrides[qid] = {
                "annotator": annotator,
                "annotator_rewritten_question": rewritten_question,
                "annotator_rewritten_solution": rewritten_solution,
                "annotator_rewritten_answer": rewritten_answer,
                "annotator_comment": annotator_comment,
                "saved_at": _normalize_text(entry.get("saved_at")),
                "has_question_override": _is_meaningful(rewritten_question),
                "has_solution_override": _is_meaningful(rewritten_solution),
                "has_answer_override": _is_meaningful(rewritten_answer),
            }
    return overrides


def apply_annotation_override(row: Dict[str, Any], override: Dict[str, Any] | None) -> Dict[str, Any]:
    out = dict(row)
    if not override:
        out.setdefault("annotator_rewritten_question", "")
        out.setdefault("annotator_rewritten_solution", "")
        out.setdefault("annotator_rewritten_answer", {})
        out.setdefault("annotator_comment", "")
        out.setdefault("annotator_override_active", False)
        return out

    rewritten_question = _normalize_text(override.get("annotator_rewritten_question"))
    rewritten_solution = _normalize_text(override.get("annotator_rewritten_solution"))
    rewritten_answer = override.get("annotator_rewritten_answer", {})
    annotator_comment = _normalize_text(override.get("annotator_comment"))
    answer_text, sympy_text, compare_text = _extract_answer_overrides(rewritten_answer)

    out["annotator"] = _normalize_text(override.get("annotator"))
    out["annotator_rewritten_question"] = rewritten_question
    out["annotator_rewritten_solution"] = rewritten_solution
    out["annotator_rewritten_answer"] = rewritten_answer
    out["annotator_comment"] = annotator_comment
    out["annotator_override_saved_at"] = _normalize_text(override.get("saved_at"))
    out["annotator_override_active"] = any(
        [
            _is_meaningful(rewritten_question),
            _is_meaningful(rewritten_solution),
            _is_meaningful(rewritten_answer),
        ]
    )
    out["annotator_rewritten_answer_reference"] = answer_text
    out["annotator_rewritten_answer_sympy"] = sympy_text
    out["annotator_rewritten_answer_compare"] = compare_text
    return out


def apply_split_annotation_overrides(rows: List[Dict[str, Any]], paths: Dict[str, Any], split_name: str) -> List[Dict[str, Any]]:
    overrides = load_split_annotation_overrides(paths, split_name)
    if not overrides:
        return [apply_annotation_override(row, None) for row in rows]
    return [apply_annotation_override(row, overrides.get(str(row.get("id", "")).strip())) for row in rows]


def resolve_effective_question(row: Dict[str, Any]) -> str:
    rewritten = _normalize_text(row.get("annotator_rewritten_question"))
    if rewritten:
        return rewritten
    return _normalize_text(row.get("question_final") or row.get("question_standalone"))


def resolve_effective_reference_reasoning(row: Dict[str, Any]) -> str:
    rewritten = _normalize_text(row.get("annotator_rewritten_solution"))
    if rewritten:
        return rewritten
    return _normalize_text(row.get("reference_reasoning"))


def resolve_effective_reference_answer(row: Dict[str, Any]) -> str:
    if _is_meaningful(row.get("annotator_rewritten_answer")):
        return _normalize_text(row.get("annotator_rewritten_answer_reference"))
    return _normalize_text(row.get("reference_answer"))


def resolve_effective_reference_answer_sympy(row: Dict[str, Any]) -> str:
    if _is_meaningful(row.get("annotator_rewritten_answer")):
        return _normalize_text(row.get("annotator_rewritten_answer_sympy"))
    return _normalize_text(row.get("reference_answer_sympy"))


def resolve_effective_final_answer_for_compare(row: Dict[str, Any]) -> str:
    if _is_meaningful(row.get("annotator_rewritten_answer")):
        return _normalize_text(row.get("annotator_rewritten_answer_compare"))
    return _normalize_text(row.get("final_answer_for_compare") or row.get("reference_answer_sympy") or row.get("reference_answer"))
