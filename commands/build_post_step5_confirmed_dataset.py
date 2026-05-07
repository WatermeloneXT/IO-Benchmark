import json
from collections import defaultdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.expert_sampled_questions import EXPERT_SAMPLED_QUESTION_PAIRS

EXPORT_FILE = ROOT / "econ_ai/data/review_annotations/post_step5_exports/annotated_curated_dataset.jsonl"
EXPERT_DIR = ROOT / "econ_ai/data/review_annotations/post_step5_expert/expert"
HF_JSON_DIR = ROOT / "econ_ai/data/hf_json"
OUT_DIR = ROOT / "econ_ai/data/review_annotations/post_step5_confirmed_dataset"

EXPERT_EXPLICITLY_CORRECT_IDS = {
    "1.5/main",
    "10.2/2#2",
    "4.5/iii",
    "5.3/iii",
    "6.2/main",
    "6.3/main",
    "7.2/2",
    "8.10/2",
}

EXPERT_AMBIGUOUS_EXCLUDED_IDS = {
    "3.7/i",
}

MULTI_OUTPUT_JSON_OVERRIDES = {
    "5.3/i": {
        "keys": ["q1", "q2", "q3"],
        "reference_answer_json": {
            "q1": "1/4",
            "q2": "1/4",
            "q3": "1/4",
        },
        "reference_answer_json_modes": {
            "q1": "sympy",
            "q2": "sympy",
            "q3": "sympy",
        },
    },
    "5.5/ii#1": {
        "keys": ["q1", "q2"],
        "reference_answer_json": {
            "q1": "(2-a)/7",
            "q2": "(5+a)/21",
        },
        "reference_answer_json_modes": {
            "q1": "sympy",
            "q2": "sympy",
        },
    },
    "5.9/i": {
        "keys": ["P1", "P2"],
        "reference_answer_json": {
            "P1": "0",
            "P2": "0",
        },
        "reference_answer_json_modes": {
            "P1": "sympy",
            "P2": "sympy",
        },
    },
    "10.7/main#1": {
        "keys": ["Pi_0", "Pi_1", "Pi_2"],
        "reference_answer_json": {
            "Pi_0": "t",
            "Pi_1": "t + Delta_s**2/(9*t)",
            "Pi_2": "t",
        },
        "reference_answer_json_modes": {
            "Pi_0": "sympy",
            "Pi_1": "sympy",
            "Pi_2": "sympy",
        },
        "replace_text": {
            "Return the triple $(\\Pi_0,\\Pi_1,\\Pi_2)$.": "",
        },
    },
}

QUESTION_OVERRIDES = {
    "1.7/i": {
        "reference_answer_sympy": "r*Integral(c_w_s*exp(-r*(s-t)), (s, t, oo))",
        "final_answer_for_compare": "r*Integral(c_w_s*exp(-r*(s-t)), (s, t, oo))",
    },
    "2.6/main#1": {
        "question_final": (
            "A dynamic perspective, however, leads to a greater goodwill.\n\n"
            "True or False: As the discount factor $\\delta$ increases, the first-period price increases."
        ),
        "question_type": "judge",
        "answer_kind": "bool",
        "comparison_mode": "exact",
        "reference_answer": False,
        "reference_answer_sympy": "N/A",
        "final_answer_for_compare": "False",
        "reference_reasoning": (
            "False. The optimal first-period price decreases as the discount factor $\\delta$ increases."
        ),
        "reference_answer_json": {},
        "reference_answer_json_modes": {},
    },
}


def normalize_text(value) -> str:
    return " ".join(str(value).replace("\n", " ").split()).strip()


def is_blank(value) -> bool:
    return not normalize_text(value)


def is_comment_5(row: dict) -> bool:
    value = row.get("annotator_comment")
    if str(value).strip() == "5":
        return True
    codes = row.get("annotator_comment_codes")
    if isinstance(codes, list):
        return any(str(code).strip() == "5" for code in codes)
    return False


def load_export_rows() -> dict[str, dict]:
    rows = {}
    with EXPORT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("original_id", "")).strip()
            if qid:
                rows[qid] = row
    return rows


def load_hf_rows() -> dict[str, dict]:
    rows = {}
    for path in sorted(HF_JSON_DIR.glob("chapter_*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = str(row.get("id", "")).strip()
                if qid:
                    rows[qid] = row
    return rows


def is_silent_expert_confirmation(item: dict) -> bool:
    return (
        is_blank(item.get("annotator_comment", ""))
        and is_blank(item.get("annotator_rewritten_question", ""))
        and is_blank(item.get("annotator_rewritten_solution", ""))
        and is_blank(item.get("annotator_rewritten_answer", ""))
    )


def load_expert_confirmations() -> tuple[dict[str, dict], set[str], list[dict]]:
    expert = {}
    silent_confirmed_ids: set[str] = set()
    ignored_non_sampled_rows: list[dict] = []
    for path in sorted(EXPERT_DIR.glob("chapter_*.json")):
        split_name = path.stem
        data = json.loads(path.read_text(encoding="utf-8"))
        for qid, item in data.get("entries", {}).items():
            if (split_name, qid) not in EXPERT_SAMPLED_QUESTION_PAIRS:
                ignored_non_sampled_rows.append(
                    {
                        "split": split_name,
                        "id": qid,
                        "reason": "expert_context_not_sampled_subquestion",
                    }
                )
                continue
            if is_silent_expert_confirmation(item):
                silent_confirmed_ids.add(qid)
            expert[qid] = {
                "expert_source_file": str(path.relative_to(ROOT)),
                "expert_annotator": data.get("annotator", "expert"),
                "expert_comment": item.get("annotator_comment", ""),
                "expert_rewritten_answer": item.get("annotator_rewritten_answer", ""),
                "expert_rewritten_question": item.get("annotator_rewritten_question", ""),
                "expert_rewritten_solution": item.get("annotator_rewritten_solution", ""),
                "expert_saved_at": item.get("saved_at", ""),
            }
    return expert, silent_confirmed_ids, ignored_non_sampled_rows


def chapter_sort_key(split_name: str):
    if split_name == "chapter_intro":
        return (-1, split_name)
    if split_name.startswith("chapter_"):
        suffix = split_name[len("chapter_") :]
        if suffix.isdigit():
            return (int(suffix), split_name)
    return (9999, split_name)


def append_json_output_instruction(question_final: str, keys: list[str]) -> str:
    instruction = (
        "Return the final answer as a JSON object with exactly these keys: "
        f"{', '.join(keys)}. Use double-quoted JSON keys and string values."
    )
    normalized_question = question_final.rstrip()
    if instruction in normalized_question:
        return normalized_question
    return f"{normalized_question}\n\n{instruction}"


def apply_post_build_overrides(record: dict) -> dict:
    qid = str(record.get("id", "")).strip()
    json_override = MULTI_OUTPUT_JSON_OVERRIDES.get(qid)
    if json_override:
        question_final = str(record.get("question_final", ""))
        for old_text, new_text in json_override.get("replace_text", {}).items():
            question_final = question_final.replace(old_text, new_text)
        question_final = append_json_output_instruction(question_final, json_override["keys"])
        record["question_final"] = question_final.strip()
        record["answer_kind"] = "json"
        record["comparison_mode"] = "json"
        record["reference_answer"] = dict(json_override["reference_answer_json"])
        record["reference_answer_sympy"] = "N/A"
        record["reference_answer_json"] = dict(json_override["reference_answer_json"])
        record["reference_answer_json_modes"] = dict(json_override["reference_answer_json_modes"])
        record["final_answer_for_compare"] = dict(json_override["reference_answer_json"])

    question_override = QUESTION_OVERRIDES.get(qid)
    if question_override:
        record.update(question_override)

    return record


def _coerce_reference_answer(existing_value, raw_value: str):
    text = normalize_text(raw_value)
    if not text:
        return existing_value
    if isinstance(existing_value, bool):
        lowered = text.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    if isinstance(existing_value, (dict, list)):
        try:
            return json.loads(text)
        except Exception:
            return text
    return text


def materialize_effective_annotation_fields(record: dict, effective: dict, annotator_overrides: dict) -> dict:
    question_text = normalize_text(effective.get("question")) or normalize_text(annotator_overrides.get("question"))
    if question_text:
        record["question_standalone"] = question_text
        record["question_final"] = question_text

    reasoning_text = normalize_text(effective.get("reference_reasoning")) or normalize_text(annotator_overrides.get("solution"))
    if reasoning_text:
        record["reference_reasoning"] = reasoning_text

    answer_reference = normalize_text(effective.get("reference_answer")) or normalize_text(annotator_overrides.get("answer_reference"))
    if answer_reference:
        record["reference_answer"] = _coerce_reference_answer(record.get("reference_answer"), answer_reference)

    answer_sympy = normalize_text(effective.get("reference_answer_sympy")) or normalize_text(annotator_overrides.get("answer_sympy"))
    if answer_sympy:
        record["reference_answer_sympy"] = answer_sympy

    answer_compare = normalize_text(effective.get("final_answer_for_compare")) or normalize_text(annotator_overrides.get("answer_compare"))
    if answer_compare:
        record["final_answer_for_compare"] = answer_compare

    return record


def build_hf_like_record(base_row: dict, export_row: dict) -> dict:
    record = dict(base_row)
    effective = export_row.get("effective_after_annotation", {}) or {}
    annotator_overrides = export_row.get("annotator_overrides", {}) or {}
    record["annotator"] = export_row.get("annotator", "")
    record["annotator_comment"] = export_row.get("annotator_comment", "")
    record["annotator_comment_codes"] = export_row.get("annotator_comment_codes", [])
    record["annotator_comment_descriptions"] = export_row.get("annotator_comment_descriptions", [])
    record["annotator_overrides"] = annotator_overrides
    record["effective_after_annotation"] = effective
    record["before_annotation"] = export_row.get("before_annotation", {})
    record["step5"] = export_row.get("step5", {})
    record["renumbered_index"] = export_row.get("renumbered_index")
    record["renumbered_split_index"] = export_row.get("renumbered_split_index")
    record = materialize_effective_annotation_fields(record, effective, annotator_overrides)
    return apply_post_build_overrides(record)


def main() -> None:
    export_rows = load_export_rows()
    hf_rows = load_hf_rows()
    expert_rows, silent_confirmed_ids, ignored_non_sampled_expert_rows = load_expert_confirmations()

    selected = {}
    source_counts = defaultdict(int)
    missing_hf_base = []

    for qid, row in export_rows.items():
        if not is_comment_5(row):
            continue
        base_row = hf_rows.get(qid)
        if base_row is None:
            missing_hf_base.append(qid)
            continue
        record = build_hf_like_record(base_row, row)
        record["confirmation_bucket"] = "non_expert_comment_5"
        record["confirmation_status"] = "confirmed_without_expert"
        record["confirmation_note"] = "Selected because annotator_comment indicates category 5 (no expert confirmation needed)."
        record["confirmation_source_file"] = (
            f"econ_ai/data/review_annotations/post_step5/{row.get('annotator')}/chapter_{row.get('chapter')}.json"
        )
        selected[qid] = record
        source_counts["non_expert_comment_5"] += 1

    missing_expert_base = []
    missing_expert_confirmation = []
    for qid in sorted(EXPERT_EXPLICITLY_CORRECT_IDS):
        base_hf_row = hf_rows.get(qid)
        if base_hf_row is None:
            missing_hf_base.append(qid)
            continue
        base_row = export_rows.get(qid)
        if base_row is None:
            missing_expert_base.append(qid)
            continue
        expert_meta = expert_rows.get(qid)
        if expert_meta is None:
            missing_expert_confirmation.append(qid)
            continue
        record = build_hf_like_record(base_hf_row, base_row)
        record["confirmation_bucket"] = "expert_explicitly_correct"
        record["confirmation_status"] = "confirmed_by_expert"
        record["confirmation_note"] = "Selected because expert explicitly marked it totally correct with no rewrite fields populated."
        record.update(expert_meta)
        selected[qid] = record
        source_counts["expert_explicitly_correct"] += 1

    for qid in sorted(silent_confirmed_ids):
        if qid in EXPERT_AMBIGUOUS_EXCLUDED_IDS:
            continue
        if qid in EXPERT_EXPLICITLY_CORRECT_IDS:
            continue
        base_hf_row = hf_rows.get(qid)
        if base_hf_row is None:
            missing_hf_base.append(qid)
            continue
        base_row = export_rows.get(qid)
        if base_row is None:
            missing_expert_base.append(qid)
            continue
        expert_meta = expert_rows.get(qid)
        if expert_meta is None:
            missing_expert_confirmation.append(qid)
            continue
        record = build_hf_like_record(base_hf_row, base_row)
        record["confirmation_bucket"] = "expert_silent_no_change"
        record["confirmation_status"] = "confirmed_by_expert"
        record["confirmation_note"] = (
            "Selected because expert left comment and rewrite fields empty, which we treat as an implicit confirmation."
        )
        record.update(expert_meta)
        selected[qid] = record
        source_counts["expert_silent_no_change"] += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_split = defaultdict(list)
    for qid, row in selected.items():
        split = str(row.get("split", "")).strip()
        if not split:
            chapter = str(row.get("chapter", "")).strip()
            split = f"chapter_{chapter}" if chapter else "chapter_unknown"
        by_split[split].append((qid, row))

    manifest = {
        "source_export_file": str(EXPORT_FILE.relative_to(ROOT)),
        "source_expert_dir": str(EXPERT_DIR.relative_to(ROOT)),
        "source_hf_json_dir": str(HF_JSON_DIR.relative_to(ROOT)),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "selection_criteria": {
            "non_expert_comment_5": "annotator_comment == 5 or annotator_comment_codes contains 5",
            "expert_explicitly_correct_ids": sorted(EXPERT_EXPLICITLY_CORRECT_IDS),
            "expert_sampled_subquestion_pairs": [
                {"split": split_name, "id": qid}
                for split_name, qid in sorted(EXPERT_SAMPLED_QUESTION_PAIRS)
            ],
            "expert_silent_no_change": "expert review entry has empty comment and empty rewritten question/solution/answer",
            "expert_ambiguous_excluded_ids": sorted(EXPERT_AMBIGUOUS_EXCLUDED_IDS),
            "multi_output_json_overrides": sorted(MULTI_OUTPUT_JSON_OVERRIDES),
            "question_overrides": sorted(QUESTION_OVERRIDES),
        },
        "output_format": "hf_json-compatible rows copied from data/hf_json with confirmation metadata and post-build overrides applied",
        "source_counts": dict(source_counts),
        "total_rows": len(selected),
        "rows_per_split": {},
        "ignored_non_sampled_expert_rows": ignored_non_sampled_expert_rows,
        "missing_hf_base_rows": missing_hf_base,
        "missing_expert_base_rows": missing_expert_base,
        "missing_expert_confirmation_rows": missing_expert_confirmation,
    }

    for split in sorted(by_split, key=chapter_sort_key):
        rows = [row for _, row in sorted(by_split[split], key=lambda item: item[0])]
        out_path = OUT_DIR / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["rows_per_split"][split] = len(rows)

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
