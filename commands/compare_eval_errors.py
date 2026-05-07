#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cost_logging import get_run_context, register_run_outputs
from core.annotation_overrides import (
    apply_split_annotation_overrides,
    resolve_effective_question,
    resolve_effective_reference_answer,
    resolve_effective_reference_answer_sympy,
)
from core.llm_utils import load_config, resolve_judge_model, resolve_solver_model
from core.model_layout import (
    resolve_llm_evaluation_input,
    resolve_rule_evaluation_input,
    solver_compare_report_file,
    solver_compare_summary_file,
)
from core.solver_variants import build_solver_artifact_label
from core.path_overrides import apply_dataset_path_overrides
from core.result_metadata import build_result_metadata


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


def normalize_text(value: Any, default: str = "N/A") -> str:
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


def resolve_reference_answer(row: Dict[str, Any]) -> str:
    answer = normalize_text(resolve_effective_reference_answer(row), default="")
    if answer:
        return answer
    return normalize_text(resolve_effective_reference_answer_sympy(row))


def resolve_rule_file(paths: Dict[str, Any], split_name: str, model_name: str) -> Path:
    return resolve_rule_evaluation_input(paths, model_name, split_name)


def resolve_note(
    rule_exists: bool,
    llm_exists: bool,
    rule_ok: Optional[bool],
    llm_answer_ok: Optional[bool],
    llm_reasoning_ok: Optional[bool],
) -> str:
    if not rule_exists and not llm_exists:
        return "missing_both"
    if not rule_exists:
        return "missing_rule"
    if not llm_exists:
        return "missing_llm"
    if rule_ok is None:
        return "invalid_rule_result"
    if llm_answer_ok is None or llm_reasoning_ok is None:
        return "invalid_llm_result"

    if rule_ok and llm_answer_ok and llm_reasoning_ok:
        return "same_pass"
    if (not rule_ok) and (not llm_answer_ok):
        return "same_fail_answer"
    if rule_ok and llm_answer_ok and (not llm_reasoning_ok):
        return "rule_pass_llm_fail_reasoning"
    if (not rule_ok) and llm_answer_ok:
        return "rule_fail_llm_pass_answer"
    if rule_ok and (not llm_answer_ok):
        return "rule_pass_llm_fail_answer"
    return "different"


def run(
    config_path: str = "config.yaml",
    solver_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    model: Optional[str] = None,
    split: Optional[str] = None,
    out: Optional[str] = None,
    hf_json_dir: Optional[str] = None,
    by_model_dir: Optional[str] = None,
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

    base_model_name = resolve_solver_model(cfg, requested_model=solver_model or model)
    model_name = build_solver_artifact_label(
        base_model_name,
        reasoning_effort=solver_reasoning_effort,
        max_solve_tokens=solver_max_solve_tokens or int((cfg.get("generate") or {}).get("max_solve_tokens", 4096) or 4096),
    )
    judge_model_name = resolve_judge_model(cfg, requested_model=judge_model)
    run_ctx = get_run_context()

    dataset_files = sorted([p for p in hf_json_dir.glob("*.jsonl") if p.is_file()], key=lambda p: p.name)
    if split:
        dataset_files = [p for p in dataset_files if p.stem == split]
    else:
        dataset_files = [p for p in dataset_files if not p.stem.endswith("_balanced")]

    if not dataset_files:
        raise RuntimeError(f"No dataset jsonl files found under: {hf_json_dir}")

    compare_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "model": model_name,
        "solver_model": model_name,
        "judge_model": judge_model_name,
        "total_compared": 0,
        "rows_output": 0,
        "rule_wrong_count": 0,
        "llm_wrong_answer_count": 0,
        "llm_wrong_reasoning_count": 0,
        "llm_any_wrong_count": 0,
        "answer_judgment_same_count": 0,
        "overall_judgment_same_count": 0,
        "splits": {},
    }

    for ds_file in dataset_files:
        split_name = ds_file.stem
        rule_file = resolve_rule_file(paths, split_name, model_name)
        llm_file = resolve_llm_evaluation_input(paths, model_name, judge_model_name, split_name)

        ds_rows = apply_split_annotation_overrides(read_jsonl(ds_file), paths, split_name)
        rule_rows = read_jsonl(rule_file)
        llm_rows = read_jsonl(llm_file)

        ds_by_id = {normalize_text(r.get("id", ""), default=""): r for r in ds_rows if normalize_text(r.get("id", ""), default="")}
        rule_by_id = {normalize_text(r.get("id", ""), default=""): r for r in rule_rows if normalize_text(r.get("id", ""), default="")}
        llm_by_id = {normalize_text(r.get("id", ""), default=""): r for r in llm_rows if normalize_text(r.get("id", ""), default="")}

        all_ids: Set[str] = set(ds_by_id.keys()) | set(rule_by_id.keys()) | set(llm_by_id.keys())
        split_total = len(all_ids)
        split_rule_wrong = 0
        split_llm_wrong_answer = 0
        split_llm_wrong_reasoning = 0
        split_llm_any_wrong = 0
        split_answer_same = 0
        split_overall_same = 0
        split_rows_output = 0

        for qid in sorted(all_ids):
            ds_row = ds_by_id.get(qid, {})
            rule_row = rule_by_id.get(qid)
            llm_row = llm_by_id.get(qid)

            question = normalize_text(resolve_effective_question(ds_row))
            question_type = normalize_text(ds_row.get("question_type", ""))
            answer_kind = normalize_text(ds_row.get("answer_kind", ""))
            reference_answer = resolve_reference_answer(ds_row)

            rule_exists = rule_row is not None
            llm_exists = llm_row is not None

            rule_is_correct = parse_bool((rule_row or {}).get("is_correct"))
            rule_detail = normalize_text((rule_row or {}).get("detail", "N/A"))
            rule_prediction = normalize_text((rule_row or {}).get("prediction", "N/A"))
            rule_reference = normalize_text((rule_row or {}).get("reference", "N/A"))

            llm_answer_correct = parse_bool((llm_row or {}).get("answer_correct"))
            llm_reasoning_correct = parse_bool((llm_row or {}).get("reasoning_correct"))
            llm_detail = normalize_text((llm_row or {}).get("detail", "N/A"))
            llm_judge_reason = normalize_text((llm_row or {}).get("judge_reason", "N/A"))
            llm_predict_answer = normalize_text((llm_row or {}).get("predict_answer", "N/A"))
            llm_reference_reasoning = normalize_text((llm_row or {}).get("reference_reasoning", "N/A"))
            llm_predict_reasoning = normalize_text((llm_row or {}).get("predict_reasoning", "N/A"))

            predict_answer = llm_predict_answer if llm_exists else rule_prediction

            rule_wrong = rule_is_correct is False
            llm_wrong_answer = llm_answer_correct is False
            llm_wrong_reasoning = llm_reasoning_correct is False
            llm_any_wrong = llm_wrong_answer or llm_wrong_reasoning

            answer_judgment_same = (
                (rule_is_correct is not None)
                and (llm_answer_correct is not None)
                and (rule_is_correct == llm_answer_correct)
            )
            overall_judgment_same = (
                (rule_is_correct is not None)
                and (llm_answer_correct is not None)
                and (llm_reasoning_correct is not None)
                and (
                    (rule_is_correct and llm_answer_correct and llm_reasoning_correct)
                    or ((not rule_is_correct) and (not llm_answer_correct))
                )
            )

            note = resolve_note(
                rule_exists=rule_exists,
                llm_exists=llm_exists,
                rule_ok=rule_is_correct,
                llm_answer_ok=llm_answer_correct,
                llm_reasoning_ok=llm_reasoning_correct,
            )

            if not (rule_wrong or llm_any_wrong):
                continue

            row = {
                "split": split_name,
                "id": qid,
                "question_type": question_type,
                "answer_kind": answer_kind,
                "question": question,
                "reference_answer": reference_answer,
                "predict_answer": predict_answer,
                "rule_is_correct": rule_is_correct,
                "rule_detail": rule_detail,
                "rule_prediction": rule_prediction,
                "rule_reference": rule_reference,
                "llm_answer_correct": llm_answer_correct,
                "llm_reasoning_correct": llm_reasoning_correct,
                "llm_detail": llm_detail,
                "llm_judge_reason": llm_judge_reason,
                "llm_reference_reasoning": llm_reference_reasoning,
                "llm_predict_reasoning": llm_predict_reasoning,
                "rule_wrong": rule_wrong,
                "llm_wrong_answer": llm_wrong_answer,
                "llm_wrong_reasoning": llm_wrong_reasoning,
                "llm_any_wrong": llm_any_wrong,
                "answer_judgment_same": answer_judgment_same,
                "overall_judgment_same": overall_judgment_same,
                "comparison_note": note,
            }
            row.update(
                build_result_metadata(
                    stage="compare-eval-errors",
                    solver_model=model_name,
                    judge_model=judge_model_name,
                    split=split_name,
                    question_id=qid,
                )
            )
            compare_rows.append(row)

            split_rows_output += 1
            if rule_wrong:
                split_rule_wrong += 1
            if llm_wrong_answer:
                split_llm_wrong_answer += 1
            if llm_wrong_reasoning:
                split_llm_wrong_reasoning += 1
            if llm_any_wrong:
                split_llm_any_wrong += 1
            if answer_judgment_same:
                split_answer_same += 1
            if overall_judgment_same:
                split_overall_same += 1

        summary["total_compared"] += split_total
        summary["rows_output"] += split_rows_output
        summary["rule_wrong_count"] += split_rule_wrong
        summary["llm_wrong_answer_count"] += split_llm_wrong_answer
        summary["llm_wrong_reasoning_count"] += split_llm_wrong_reasoning
        summary["llm_any_wrong_count"] += split_llm_any_wrong
        summary["answer_judgment_same_count"] += split_answer_same
        summary["overall_judgment_same_count"] += split_overall_same
        summary["splits"][split_name] = {
            "total_compared": split_total,
            "rows_output": split_rows_output,
            "rule_wrong_count": split_rule_wrong,
            "llm_wrong_answer_count": split_llm_wrong_answer,
            "llm_wrong_reasoning_count": split_llm_wrong_reasoning,
            "llm_any_wrong_count": split_llm_any_wrong,
            "answer_judgment_same_count": split_answer_same,
            "overall_judgment_same_count": split_overall_same,
            "rule_file": rule_file.as_posix(),
            "llm_file": llm_file.as_posix(),
        }

        print(
            f"[compare-eval-errors] {split_name}: "
            f"output={split_rows_output}, rule_wrong={split_rule_wrong}, "
            f"llm_wrong_answer={split_llm_wrong_answer}, llm_wrong_reasoning={split_llm_wrong_reasoning}"
        )

    if out:
        out_file = Path(out).expanduser().resolve()
    else:
        out_file = solver_compare_report_file(paths, model_name, judge_model_name).resolve()
    summary_file = (
        out_file.with_name(out_file.stem + "__summary.json")
        if out
        else solver_compare_summary_file(paths, model_name, judge_model_name).resolve()
    )

    summary["workflow_id"] = str(run_ctx.get("workflow_id", "") or "")
    summary["run_id"] = str(run_ctx.get("run_id", "") or "")
    summary["stage"] = "compare-eval-errors"
    summary["compare_file"] = out_file.as_posix()
    summary["summary_file"] = summary_file.as_posix()

    write_jsonl(out_file, compare_rows)
    write_json(summary_file, summary)
    register_run_outputs([out_file, summary_file])
    print(f"[compare-eval-errors] wrote -> {out_file}")
    print(f"[compare-eval-errors] summary -> {summary_file}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rule-based and LLM-evaluation wrong cases.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model used by generation/evaluation artifacts.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model used by evaluate-llm artifacts. Defaults to config evaluate_llm.judge_model.",
    )
    parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
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
    parser.add_argument("--out", default=None, help="Optional output file path (.jsonl)")
    args = parser.parse_args()
    run(
        config_path=args.config,
        solver_model=args.solver_model,
        judge_model=args.judge_model,
        split=args.split,
        out=args.out,
        hf_json_dir=args.hf_json_dir,
        by_model_dir=args.by_model_dir,
    )


if __name__ == "__main__":
    main()
