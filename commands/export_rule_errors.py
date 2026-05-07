#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_utils import load_config, resolve_solver_model
from core.model_layout import resolve_generation_input, resolve_rule_evaluation_input
from core.path_overrides import apply_dataset_path_overrides
from core.question_filter import filter_rows_by_question_ids, load_question_id_filter
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


def natural_key(text: str) -> List[Any]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def dataset_tag_from_hf_dir(hf_json_dir: Path) -> str:
    dataset_tag = hf_json_dir.name.strip() or "dataset"
    if dataset_tag.endswith("_dataset"):
        dataset_tag = dataset_tag[: -len("_dataset")]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_tag).strip("._-") or "dataset"


def default_output_path(hf_json_dir: Path, solver_model: str) -> Path:
    dataset_tag = dataset_tag_from_hf_dir(hf_json_dir)
    return PROJECT_ROOT / "data" / "error_reason_annotation" / f"{dataset_tag}_{solver_model}_rule_errors.jsonl"


def build_export_row(
    split_name: str,
    dataset_row: Dict[str, Any],
    generation_row: Dict[str, Any],
    evaluation_row: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "split": split_name,
        "id": str(evaluation_row.get("id", "")).strip(),
        "question_type": str(
            dataset_row.get("question_type")
            or generation_row.get("question_type")
            or evaluation_row.get("question_type")
            or ""
        ).strip(),
        "answer_kind": str(
            dataset_row.get("answer_kind")
            or generation_row.get("answer_kind")
            or evaluation_row.get("answer_kind")
            or ""
        ).strip(),
        "question": str(generation_row.get("question_final") or dataset_row.get("question_final") or "").strip(),
        "reference_answer": str(
            dataset_row.get("reference_answer")
            or generation_row.get("reference_answer")
            or evaluation_row.get("reference")
            or ""
        ).strip(),
        "predict_answer": str(
            evaluation_row.get("prediction")
            or generation_row.get("final_answer_for_compare")
            or generation_row.get("answer_boxed")
            or ""
        ).strip(),
        "reference_reasoning": str(dataset_row.get("reference_reasoning") or "").strip(),
        "model_reasoning": str(generation_row.get("model_response") or "").strip(),
        "rule_detail": str(evaluation_row.get("detail") or "").strip(),
        "rule_prediction": str(evaluation_row.get("prediction") or "").strip(),
        "rule_reference": str(evaluation_row.get("reference") or "").strip(),
        "reason": "",
    }


def run(
    config_path: str = "config.yaml",
    solver_model: Optional[str] = None,
    model: Optional[str] = None,
    split: Optional[str] = None,
    hf_json_dir: Optional[str] = None,
    by_model_dir: Optional[str] = None,
    question_ids: Optional[str] = None,
    question_ids_file: Optional[str] = None,
    out: Optional[str] = None,
    solver_reasoning_effort: Optional[str] = None,
    solver_max_solve_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = apply_dataset_path_overrides(
        load_config(config_path),
        hf_json_dir=hf_json_dir,
        by_model_dir=by_model_dir,
    )
    paths = cfg["paths"]
    hf_json_dir_path = Path(paths["hf_json_dir"])
    base_model_name = resolve_solver_model(cfg, requested_model=solver_model or model)
    model_name = build_solver_artifact_label(
        base_model_name,
        reasoning_effort=solver_reasoning_effort,
        max_solve_tokens=solver_max_solve_tokens or int((cfg.get("generate") or {}).get("max_solve_tokens", 4096) or 4096),
    )
    question_id_filter = load_question_id_filter(
        question_ids=question_ids,
        question_ids_file=question_ids_file,
    )

    dataset_files = sorted(
        [p for p in hf_json_dir_path.glob("*.jsonl") if p.is_file()],
        key=lambda p: natural_key(p.stem),
    )
    if split:
        dataset_files = [p for p in dataset_files if p.stem == split]
    if not dataset_files:
        raise RuntimeError(f"No dataset jsonl files found under {hf_json_dir_path} for split={split or 'all'}.")

    export_rows: List[Dict[str, Any]] = []
    split_counts: Dict[str, int] = {}

    for dataset_file in dataset_files:
        split_name = dataset_file.stem
        generation_file = resolve_generation_input(paths, model_name, split_name)
        evaluation_file = resolve_rule_evaluation_input(paths, model_name, split_name)
        if not generation_file.exists():
            raise FileNotFoundError(
                f"generation file not found: {generation_file}. Run generate first with the same --solver-model."
            )
        if not evaluation_file.exists():
            raise FileNotFoundError(
                f"rule evaluation file not found: {evaluation_file}. Run evaluate first with the same --solver-model."
            )

        dataset_rows = filter_rows_by_question_ids(read_jsonl(dataset_file), question_id_filter)
        generation_rows = filter_rows_by_question_ids(read_jsonl(generation_file), question_id_filter)
        evaluation_rows = filter_rows_by_question_ids(read_jsonl(evaluation_file), question_id_filter)

        dataset_by_id = {
            str(row.get("id", "")).strip(): row
            for row in dataset_rows
            if str(row.get("id", "")).strip()
        }
        generation_by_id = {
            str(row.get("id", "")).strip(): row
            for row in generation_rows
            if str(row.get("id", "")).strip()
        }

        for evaluation_row in evaluation_rows:
            if bool(evaluation_row.get("is_correct", False)):
                continue
            qid = str(evaluation_row.get("id", "")).strip()
            if not qid:
                continue
            export_rows.append(
                build_export_row(
                    split_name=split_name,
                    dataset_row=dataset_by_id.get(qid, {}),
                    generation_row=generation_by_id.get(qid, {}),
                    evaluation_row=evaluation_row,
                )
            )
            split_counts[split_name] = split_counts.get(split_name, 0) + 1

    export_rows.sort(key=lambda row: (natural_key(str(row.get("split", ""))), natural_key(str(row.get("id", "")))))

    out_path = Path(out).resolve() if out else default_output_path(hf_json_dir_path, model_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in export_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "solver_model": model_name,
        "hf_json_dir": str(hf_json_dir_path),
        "split": split or "",
        "count": len(export_rows),
        "out_file": str(out_path),
        "split_counts": split_counts,
    }
    print(f"[export-rule-errors] count={summary['count']} -> {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export current wrong rule-eval questions into a JSONL for manual error labeling.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model used for generation/evaluation file lookup.",
    )
    parser.add_argument(
        "--hf-json-dir",
        default=None,
        help="Optional input dataset directory override. Example: data/review_annotations/post_step5_confirmed_dataset",
    )
    parser.add_argument(
        "--by-model-dir",
        default=None,
        help="Optional output root override to read generations/evaluations from a custom dataset-specific location.",
    )
    parser.add_argument(
        "--question-ids",
        default=None,
        help="Optional question id filter, comma-separated. Example: 1.7/i,3.1/i",
    )
    parser.add_argument(
        "--question-ids-file",
        default=None,
        help="Optional text file containing question ids to export, separated by commas or whitespace.",
    )
    parser.add_argument("--out", default=None, help="Optional output JSONL path")
    args = parser.parse_args()
    run(
        config_path=args.config,
        split=args.split,
        solver_model=args.solver_model,
        hf_json_dir=args.hf_json_dir,
        by_model_dir=args.by_model_dir,
        question_ids=args.question_ids,
        question_ids_file=args.question_ids_file,
        out=args.out,
    )


if __name__ == "__main__":
    main()
