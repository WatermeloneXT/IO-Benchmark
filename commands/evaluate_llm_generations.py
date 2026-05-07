#!/usr/bin/env python3
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cost_logging import get_run_context, register_run_outputs
from core.annotation_overrides import (
    apply_split_annotation_overrides,
    resolve_effective_question,
    resolve_effective_reference_answer,
    resolve_effective_reference_answer_sympy,
    resolve_effective_reference_reasoning,
)
from core.llm_utils import (
    azure_chat_call,
    create_client,
    load_config,
    resolve_generate_reasoning_request,
    resolve_judge_model,
    resolve_solver_model,
)
from core.model_layout import resolve_generation_input, solver_llm_evaluation_file, solver_llm_summary_file
from core.path_overrides import apply_dataset_path_overrides
from core.prompts import SYSTEM_LLM_EVAL
from core.question_filter import filter_rows_by_question_ids, load_question_id_filter, merge_rows_by_question_id
from core.result_metadata import build_result_metadata
from core.solver_variants import build_solver_artifact_label


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(value: Any, default: str = "") -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text if text else default


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value or "").strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None


def normalize_reason(text: Any) -> str:
    raw = normalize_text(text, default="No reason provided.")
    first_sentence = re.split(r"(?<=[.!?])\s+", raw, maxsplit=1)[0]
    words = first_sentence.split()
    if len(words) > 40:
        first_sentence = " ".join(words[:40]).rstrip(".,;:!?")
    return first_sentence


def parse_points(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        try:
            return float(Fraction(str(value).strip()))
        except Exception:
            return 1.0


def resolve_assignment_group(row: Dict[str, Any], fallback_split: str = "") -> str:
    split_name = normalize_text(row.get("split", ""), default=fallback_split)
    chapter_name = normalize_text(row.get("chapter", ""))
    if split_name != "chapter_assignment" and chapter_name != "assignment":
        return ""
    for key in ("original_id", "problem_number", "id"):
        raw = normalize_text(row.get(key, ""))
        if not raw:
            continue
        prefix = raw.split("#", 1)[0].split("/", 1)[0].strip()
        if prefix.isdigit():
            return prefix
    return ""


def summarize_assignment_groups(split_rows: List[Dict[str, Any]], split_name: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for row in split_rows:
        group = resolve_assignment_group(row, split_name)
        if not group:
            continue
        stats = groups.setdefault(
            group,
            {
                "total": 0,
                "correct": 0,
                "reasoning_correct": 0,
                "accuracy": 0.0,
                "reasoning_accuracy": 0.0,
                "total_points": 0.0,
                "earned_points": 0.0,
                "weighted_accuracy": 0.0,
            },
        )
        stats["total"] += 1
        if bool(row.get("answer_correct", False)):
            stats["correct"] += 1
        if bool(row.get("reasoning_correct", False)):
            stats["reasoning_correct"] += 1
        stats["total_points"] += parse_points(row.get("points", 1.0))
        stats["earned_points"] += parse_points(row.get("points_earned", 0.0))

    ordered = sorted(groups.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0]))
    out: Dict[str, Dict[str, Any]] = {}
    for group, stats in ordered:
        stats["accuracy"] = (stats["correct"] / stats["total"]) if stats["total"] else 0.0
        stats["reasoning_accuracy"] = (stats["reasoning_correct"] / stats["total"]) if stats["total"] else 0.0
        stats["weighted_accuracy"] = (
            stats["earned_points"] / stats["total_points"]
        ) if stats["total_points"] else 0.0
        out[group] = stats
    return out


def merge_assignment_group_summaries(
    summary: Dict[str, Dict[str, Any]],
    group_summary: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    for group, stats in group_summary.items():
        merged = summary.setdefault(
            group,
            {
                "total": 0,
                "correct": 0,
                "reasoning_correct": 0,
                "accuracy": 0.0,
                "reasoning_accuracy": 0.0,
                "total_points": 0.0,
                "earned_points": 0.0,
                "weighted_accuracy": 0.0,
            },
        )
        merged["total"] += int(stats.get("total", 0))
        merged["correct"] += int(stats.get("correct", 0))
        merged["reasoning_correct"] += int(stats.get("reasoning_correct", 0))
        merged["total_points"] += parse_points(stats.get("total_points", 0.0))
        merged["earned_points"] += parse_points(stats.get("earned_points", 0.0))

    ordered = sorted(summary.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0]))
    out: Dict[str, Dict[str, Any]] = {}
    for group, merged in ordered:
        merged["accuracy"] = (merged["correct"] / merged["total"]) if merged["total"] else 0.0
        merged["reasoning_accuracy"] = (
            merged["reasoning_correct"] / merged["total"]
        ) if merged["total"] else 0.0
        merged["weighted_accuracy"] = (
            merged["earned_points"] / merged["total_points"]
        ) if merged["total_points"] else 0.0
        out[group] = merged
    return out


def resolve_reference_answer(row: Dict[str, Any]) -> str:
    answer = normalize_text(resolve_effective_reference_answer(row), default="")
    if answer:
        return answer
    return normalize_text(resolve_effective_reference_answer_sympy(row), default="N/A")


def resolve_predict_answer(row: Dict[str, Any]) -> str:
    for key in ("final_answer_for_compare", "answer_boxed", "answer_sympy"):
        answer = normalize_text(row.get(key, ""))
        if answer:
            return answer
    return "N/A"


def resolve_detail(is_correct: bool, reasoning_correct: bool) -> str:
    if is_correct and reasoning_correct:
        return "ok"
    if (not is_correct) and reasoning_correct:
        return "wrong_answer"
    if is_correct and (not reasoning_correct):
        return "bad_reasoning"
    return "both_wrong"


def align_key(row: Dict[str, Any], fallback_split: str) -> Tuple[str, str]:
    rid = normalize_text(row.get("id", ""))
    split = normalize_text(row.get("split", ""), default=fallback_split)
    return rid, split


def judge_one(
    client: Any,
    judge_model: str,
    question: str,
    reference_reasoning: str,
    reference_answer: str,
    model_response: str,
    predict_answer: str,
    split: str = "",
    question_id: str = "",
    reasoning_request: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, bool, str]:
    user_payload = {
        "question": question,
        "reference_reasoning": reference_reasoning,
        "reference_answer": reference_answer,
        "predict_reasoning": model_response,
        "predict_answer": predict_answer,
    }
    user_prompt = "Evaluate this QA pair and return strict JSON only:\n" + json.dumps(user_payload, ensure_ascii=False)

    for _ in range(3):
        content = azure_chat_call(
            client=client,
            model=judge_model,
            system=SYSTEM_LLM_EVAL,
            user=user_prompt,
            temperature=0.0,
            max_tokens=512,
            max_retries=1,
            telemetry={
                "operation": "evaluate_llm_judge",
                "split": split,
                "question_id": question_id,
            },
            reasoning_request=reasoning_request,
        )
        candidate = content.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-zA-Z]*\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)
            candidate = candidate.strip()
        try:
            obj = json.loads(candidate)
        except Exception:  # noqa: BLE001
            continue

        is_correct = parse_bool(obj.get("is_correct"))
        reasoning_correct = parse_bool(obj.get("reasoning_correct"))
        judge_reason = normalize_reason(obj.get("judge_reason", "No reason provided."))
        if is_correct is None or reasoning_correct is None:
            continue
        return is_correct, reasoning_correct, judge_reason

    return False, False, "Judge output parse failed."


def run(
    config_path: str = "config.yaml",
    split: Optional[str] = None,
    force: bool = False,
    solver_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    model: Optional[str] = None,
    hf_json_dir: Optional[str] = None,
    by_model_dir: Optional[str] = None,
    question_ids: Optional[str] = None,
    question_ids_file: Optional[str] = None,
    reasoning_mode: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    solver_reasoning_effort: Optional[str] = None,
    solver_max_solve_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = apply_dataset_path_overrides(
        load_config(config_path),
        hf_json_dir=hf_json_dir,
        by_model_dir=by_model_dir,
    )
    paths = cfg["paths"]
    hf_json_dir = Path(paths["hf_json_dir"])

    legacy_model = (model or "").strip()
    solver_model_base = resolve_solver_model(cfg, requested_model=solver_model or legacy_model)
    effective_solver_max_solve_tokens = int(
        solver_max_solve_tokens or int((cfg.get("generate") or {}).get("max_solve_tokens", 4096) or 4096)
    )
    solver_model_name = build_solver_artifact_label(
        solver_model_base,
        reasoning_effort=solver_reasoning_effort,
        max_solve_tokens=effective_solver_max_solve_tokens,
    )
    judge_model_name = resolve_judge_model(cfg, requested_model=judge_model or legacy_model)
    evaluate_llm_cfg = cfg.get("evaluate_llm", {}) or {}
    judge_reasoning_request = resolve_generate_reasoning_request(
        cfg,
        judge_model_name,
        phase="solve",
        requested_mode=reasoning_mode if reasoning_mode is not None else evaluate_llm_cfg.get("reasoning_mode"),
        requested_effort=(
            reasoning_effort if reasoning_effort is not None else evaluate_llm_cfg.get("reasoning_effort")
        ),
    )
    judge_request_kwargs = dict(judge_reasoning_request.get("request_kwargs", {}) or {})
    if judge_reasoning_request.get("omit_temperature"):
        judge_request_kwargs["omit_temperature"] = True
    question_id_filter = load_question_id_filter(question_ids=question_ids, question_ids_file=question_ids_file)
    partial_overwrite = bool(question_id_filter)
    run_ctx = get_run_context()
    written_outputs: List[Path] = []

    dataset_files = sorted([p for p in hf_json_dir.glob("*.jsonl") if p.is_file()], key=lambda p: p.name)
    if split:
        dataset_files = [p for p in dataset_files if p.stem == split]
    else:
        dataset_files = [p for p in dataset_files if not p.stem.endswith("_balanced")]
    if not dataset_files:
        raise RuntimeError(f"No dataset jsonl files found under: {hf_json_dir}")

    client = None
    summary: Dict[str, Any] = {
        "model": solver_model_name,
        "solver_model": solver_model_base,
        "solver_model_artifact": solver_model_name,
        "solver_reasoning_effort": str(solver_reasoning_effort or "").strip().lower(),
        "solver_max_solve_tokens": effective_solver_max_solve_tokens,
        "judge_model": judge_model_name,
        "force": bool(force),
        "total": 0,
        "correct": 0,
        "reasoning_correct": 0,
        "accuracy": 0.0,
        "reasoning_accuracy": 0.0,
        "total_points": 0.0,
        "earned_points": 0.0,
        "weighted_accuracy": 0.0,
        "splits": {},
    }
    overall_assignment_groups: Dict[str, Dict[str, Any]] = {}

    def record_split_summary(
        *,
        split_name: str,
        split_rows: List[Dict[str, Any]],
        skipped_existing: bool,
    ) -> None:
        split_total = len(split_rows)
        split_correct = sum(1 for row in split_rows if bool(row.get("answer_correct", False)))
        split_reasoning_correct = sum(1 for row in split_rows if bool(row.get("reasoning_correct", False)))
        split_accuracy = (split_correct / split_total) if split_total else 0.0
        split_reasoning_accuracy = (split_reasoning_correct / split_total) if split_total else 0.0
        total_points = sum(parse_points(row.get("points", 1.0)) for row in split_rows)
        earned_points = sum(parse_points(row.get("points_earned", 0.0)) for row in split_rows)
        weighted_accuracy = (earned_points / total_points) if total_points else 0.0
        assignment_groups = summarize_assignment_groups(split_rows, split_name)
        summary["splits"][split_name] = {
            "total": split_total,
            "correct": split_correct,
            "reasoning_correct": split_reasoning_correct,
            "accuracy": split_accuracy,
            "reasoning_accuracy": split_reasoning_accuracy,
            "total_points": total_points,
            "earned_points": earned_points,
            "weighted_accuracy": weighted_accuracy,
            "skipped_existing": skipped_existing,
        }
        if assignment_groups:
            summary["splits"][split_name]["assignment_groups"] = assignment_groups
            merged_groups = merge_assignment_group_summaries(overall_assignment_groups, assignment_groups)
            overall_assignment_groups.clear()
            overall_assignment_groups.update(merged_groups)
        summary["total"] += split_total
        summary["correct"] += split_correct
        summary["reasoning_correct"] += split_reasoning_correct
        summary["total_points"] += total_points
        summary["earned_points"] += earned_points
        if skipped_existing:
            print(f"[evaluate-llm] {split_name}: skip existing -> {out_file}")
        else:
            print(
                f"[evaluate-llm] {split_name}: "
                f"answer={split_correct}/{split_total} ({split_accuracy:.4f}), "
                f"reasoning={split_reasoning_correct}/{split_total} ({split_reasoning_accuracy:.4f})"
            )
        for group, stats in assignment_groups.items():
            print(
                f"[evaluate-llm] {split_name} assignment_{group}: "
                f"earned_points={stats['earned_points']:.4f}/{stats['total_points']:.4f} "
                f"weighted_accuracy={stats['weighted_accuracy']:.4f}"
            )

    for ds_file in dataset_files:
        split_name = ds_file.stem
        out_file = solver_llm_evaluation_file(paths, solver_model_name, judge_model_name, split_name)
        all_ds_rows = apply_split_annotation_overrides(read_jsonl(ds_file), paths, split_name)
        ds_rows = filter_rows_by_question_ids(all_ds_rows, question_id_filter)
        ordered_question_ids = [str(row.get("id", "")).strip() for row in all_ds_rows if str(row.get("id", "")).strip()]
        current_split_question_ids = set(ordered_question_ids)
        existing_split_rows = read_jsonl(out_file) if out_file.exists() else []
        existing_by_id = {
            str(row.get("id", "")).strip(): row
            for row in existing_split_rows
            if str(row.get("id", "")).strip()
        }
        if not ds_rows:
            if partial_overwrite and existing_split_rows:
                record_split_summary(
                    split_name=split_name,
                    split_rows=existing_split_rows,
                    skipped_existing=True,
                )
            continue
        rows_to_evaluate = ds_rows
        if out_file.exists() and (not force) and (not partial_overwrite):
            missing_dataset_rows = [
                row
                for row in ds_rows
                if str(row.get("id", "")).strip() not in existing_by_id
            ]
            stale_existing_ids = sorted(
                qid
                for qid in existing_by_id
                if qid and qid not in current_split_question_ids
            )
            if not missing_dataset_rows and not stale_existing_ids:
                record_split_summary(
                    split_name=split_name,
                    split_rows=existing_split_rows,
                    skipped_existing=True,
                )
                continue
            rows_to_evaluate = missing_dataset_rows
            if stale_existing_ids:
                print(
                    f"[evaluate-llm] {split_name}: dropping {len(stale_existing_ids)} stale existing rows "
                    "not present in current dataset"
                )
            if rows_to_evaluate:
                print(
                    f"[evaluate-llm] {split_name}: evaluating {len(rows_to_evaluate)} missing rows "
                    f"(existing={len(existing_split_rows)})"
                )
            else:
                final_split_rows = merge_rows_by_question_id(
                    existing_split_rows,
                    [],
                    ordered_question_ids,
                    keep_unordered_existing=False,
                )
                write_jsonl(out_file, final_split_rows)
                written_outputs.append(out_file)
                record_split_summary(
                    split_name=split_name,
                    split_rows=final_split_rows,
                    skipped_existing=False,
                )
                continue

        generation_file = resolve_generation_input(paths, solver_model_name, split_name)
        if not generation_file.exists():
            raise RuntimeError(
                f"Missing generation file for split={split_name}, solver_model={solver_model_name}: {generation_file}. "
                "Run generate first with the same --solver-model."
            )
        if client is None:
            client = create_client(cfg, model_name=judge_model_name)

        all_pred_rows = read_jsonl(generation_file)
        pred_rows = filter_rows_by_question_ids(
            all_pred_rows,
            question_id_filter if partial_overwrite else {str(row.get("id", "")).strip() for row in rows_to_evaluate},
        )

        ds_by_key = {
            align_key(row, split_name): row
            for row in rows_to_evaluate
            if align_key(row, split_name)[0]
        }
        pred_by_key = {
            align_key(row, split_name): row
            for row in pred_rows
            if align_key(row, split_name)[0]
        }

        split_rows: List[Dict[str, Any]] = []
        for (qid, qsplit), ds_row in sorted(ds_by_key.items(), key=lambda x: (x[0][1], x[0][0])):
            pred_row = pred_by_key.get((qid, qsplit))
            question = normalize_text(resolve_effective_question(ds_row), default="N/A")
            question_type = normalize_text(ds_row.get("question_type", ""), default="N/A")
            answer_kind = normalize_text(ds_row.get("answer_kind", ""), default="N/A")
            reference_reasoning = normalize_text(resolve_effective_reference_reasoning(ds_row), default="N/A")
            reference_answer = resolve_reference_answer(ds_row)

            if pred_row is None:
                predict_answer = "N/A"
                is_correct = False
                reasoning_correct = False
                judge_reason = "Prediction missing."
            else:
                predict_answer = resolve_predict_answer(pred_row)
                model_response = normalize_text(pred_row.get("model_response", ""), default="N/A")
                is_correct, reasoning_correct, judge_reason = judge_one(
                    client=client,
                    judge_model=judge_model_name,
                    question=question,
                    reference_reasoning=reference_reasoning,
                    reference_answer=reference_answer,
                    model_response=model_response,
                    predict_answer=predict_answer,
                    split=qsplit,
                    question_id=qid,
                    reasoning_request=judge_request_kwargs,
                )

            split_rows.append(
                {
                    "id": qid,
                    "split": qsplit,
                    "assignment_group": resolve_assignment_group(ds_row, split_name),
                    "question_type": question_type,
                    "answer_kind": answer_kind,
                    "reference_reasoning": reference_reasoning,
                    "predict_reasoning": (
                        normalize_text(pred_row.get("model_response", ""), default="N/A")
                        if pred_row is not None
                        else "N/A"
                    ),
                    "reference_answer": reference_answer,
                    "predict_answer": predict_answer,
                    "answer_correct": is_correct,
                    "reasoning_correct": reasoning_correct,
                    "detail": resolve_detail(is_correct=is_correct, reasoning_correct=reasoning_correct),
                    "judge_reason": judge_reason,
                    "points": parse_points(ds_row.get("points", 1.0)),
                    "points_earned": parse_points(ds_row.get("points", 1.0)) if is_correct else 0.0,
                }
            )

        for row in split_rows:
            qid = str(row.get("id", "")).strip()
            qsplit = str(row.get("split", "")).strip() or split_name
            row.update(
                build_result_metadata(
                    stage="evaluate-llm",
                    solver_model=solver_model_name,
                    judge_model=judge_model_name,
                    split=qsplit,
                    question_id=qid,
                )
            )

        final_split_rows = split_rows
        if partial_overwrite or (out_file.exists() and (not force)):
            merge_base_rows = existing_split_rows
            if partial_overwrite:
                target_question_ids = {
                    str(row.get("id", "")).strip()
                    for row in rows_to_evaluate
                    if str(row.get("id", "")).strip()
                }
                merge_base_rows = [
                    row
                    for row in existing_split_rows
                    if str(row.get("id", "")).strip() not in target_question_ids
                ]
            final_split_rows = merge_rows_by_question_id(
                merge_base_rows,
                split_rows,
                ordered_question_ids,
                keep_unordered_existing=False,
            )
        write_jsonl(out_file, final_split_rows)
        written_outputs.append(out_file)
        record_split_summary(
            split_name=split_name,
            split_rows=final_split_rows,
            skipped_existing=False,
        )

    summary["accuracy"] = (summary["correct"] / summary["total"]) if summary["total"] else 0.0
    summary["reasoning_accuracy"] = (
        (summary["reasoning_correct"] / summary["total"]) if summary["total"] else 0.0
    )
    summary["weighted_accuracy"] = (
        summary["earned_points"] / summary["total_points"]
    ) if summary["total_points"] else 0.0
    if overall_assignment_groups:
        summary["assignment_groups"] = overall_assignment_groups

    summary["workflow_id"] = str(run_ctx.get("workflow_id", "") or "")
    summary["run_id"] = str(run_ctx.get("run_id", "") or "")
    summary["stage"] = "evaluate-llm"
    summary_file = solver_llm_summary_file(paths, solver_model_name, judge_model_name)
    write_json(summary_file, summary)
    written_outputs.append(summary_file)
    register_run_outputs(written_outputs)
    print(
        f"[evaluate-llm] overall: answer={summary['correct']}/{summary['total']} ({summary['accuracy']:.4f}), "
        f"reasoning={summary['reasoning_correct']}/{summary['total']} ({summary['reasoning_accuracy']:.4f})"
    )
    print(f"[evaluate-llm] summary -> {summary_file}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-based evaluation for generated econ answers and reasoning.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    parser.add_argument(
        "--solver-model",
        default=None,
        help="Solver model whose generated answers will be judged.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model used for LLM evaluation. Defaults to config evaluate_llm.judge_model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="[Legacy alias] When set, applies to both solver and judge model unless explicitly overridden.",
    )
    parser.add_argument(
        "--hf-json-dir",
        default=None,
        help="Optional input dataset directory override. Example: data/review_annotations/post_step5_confirmed_dataset",
    )
    parser.add_argument(
        "--by-model-dir",
        default=None,
        help="Optional output root override to keep results for different datasets separate.",
    )
    parser.add_argument(
        "--question-ids",
        default=None,
        help="Optional question id filter, comma-separated. Example: 1.7/i,3.1/i",
    )
    parser.add_argument(
        "--question-ids-file",
        default=None,
        help="Optional text file containing question ids to run, separated by commas or whitespace.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if split evaluation files already exist.",
    )
    parser.add_argument(
        "--reasoning-mode",
        default=None,
        choices=["auto", "off", "on"],
        help="Judge-model reasoning mode. Uses the model's native API interface and errors if unsupported.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Judge-model native reasoning effort value. No provider mapping is applied.",
    )
    args = parser.parse_args()
    run(
        config_path=args.config,
        split=args.split,
        force=args.force,
        solver_model=args.solver_model,
        judge_model=args.judge_model,
        model=args.model,
        hf_json_dir=args.hf_json_dir,
        by_model_dir=args.by_model_dir,
        question_ids=args.question_ids,
        question_ids_file=args.question_ids_file,
        reasoning_mode=args.reasoning_mode,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    main()
