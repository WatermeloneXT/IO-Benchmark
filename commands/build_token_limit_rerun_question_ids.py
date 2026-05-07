#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_utils import load_config, resolve_solver_model
from core.model_layout import solver_cost_dir, solver_generations_dir, solver_reports_dir
from core.path_overrides import apply_dataset_path_overrides
from core.solver_variants import build_solver_artifact_label


DEFAULT_MIN_TOKEN_RATIO = 0.98


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


def write_text_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def natural_key(text: str) -> List[Any]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def default_output_paths(base_dir: Path, split_name: Optional[str]) -> Tuple[Path, Path]:
    label = (split_name or "all").strip() or "all"
    return base_dir / f"{label}.jsonl", base_dir / f"{label}__question_ids.txt"


def generation_files_for_model(generations_dir: Path, split: Optional[str]) -> List[Path]:
    files = sorted([p for p in generations_dir.glob("*.jsonl") if p.is_file()], key=lambda p: p.name)
    if split:
        files = [p for p in files if p.stem == split]
    return files


def build_cost_index(cost_logs_dir: Path) -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_run_and_qid: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    by_qid: Dict[str, Dict[str, Any]] = {}
    if not cost_logs_dir.exists():
        return by_run_and_qid, by_qid

    for day_dir in sorted([p for p in cost_logs_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        for log_file in sorted([p for p in day_dir.glob("*.jsonl") if p.is_file()], key=lambda p: p.name):
            for row in read_jsonl(log_file):
                if row.get("stage") != "generate":
                    continue
                if row.get("operation") != "generate_solve":
                    continue
                if row.get("status") != "success":
                    continue
                qid = str(row.get("question_id", "")).strip()
                if not qid:
                    continue
                workflow_id = str(row.get("workflow_id", "")).strip()
                run_id = str(row.get("run_id", "")).strip()
                key = (workflow_id, run_id, qid)
                by_run_and_qid[key] = row
                by_qid[qid] = row
    return by_run_and_qid, by_qid


def is_missing_final_answer(row: Dict[str, Any]) -> bool:
    if not str(row.get("model_response", "") or "").strip():
        return True
    boxed = str(row.get("answer_boxed", "") or "").strip()
    final_answer = str(row.get("final_answer_for_compare", "") or "").strip()
    return boxed == "N/A" or final_answer == "N/A"


def effective_max_solve_tokens(
    row: Dict[str, Any],
    *,
    default_max_solve_tokens: int,
) -> int:
    meta = row.get("meta") or {}
    value = meta.get("max_solve_tokens", default_max_solve_tokens)
    try:
        return int(value)
    except Exception:
        return int(default_max_solve_tokens)


def threshold_tokens_for_row(
    *,
    max_solve_tokens: int,
    min_token_ratio: float,
    min_output_tokens: Optional[int],
) -> int:
    if min_output_tokens is not None:
        return int(min_output_tokens)
    return max(1, int(math.ceil(float(max_solve_tokens) * float(min_token_ratio))))


def run(
    config_path: str = "config.yaml",
    solver_model: Optional[str] = None,
    model: Optional[str] = None,
    split: Optional[str] = None,
    hf_json_dir: Optional[str] = None,
    by_model_dir: Optional[str] = None,
    out: Optional[str] = None,
    out_json: Optional[str] = None,
    min_token_ratio: float = DEFAULT_MIN_TOKEN_RATIO,
    min_output_tokens: Optional[int] = None,
    solver_reasoning_effort: Optional[str] = None,
    solver_max_solve_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = apply_dataset_path_overrides(
        load_config(config_path),
        hf_json_dir=hf_json_dir,
        by_model_dir=by_model_dir,
    )
    paths = cfg["paths"]
    base_model_name = resolve_solver_model(cfg, requested_model=solver_model or model)
    effective_solver_max_solve_tokens = int(
        solver_max_solve_tokens or int((cfg.get("generate") or {}).get("max_solve_tokens", 4096) or 4096)
    )
    model_name = build_solver_artifact_label(
        base_model_name,
        reasoning_effort=solver_reasoning_effort,
        max_solve_tokens=effective_solver_max_solve_tokens,
    )
    generations_dir = solver_generations_dir(paths, model_name)
    if not generations_dir.exists():
        raise RuntimeError(
            f"No generations directory found for solver_model={model_name}: {generations_dir}. "
            "Run generate first with the same --solver-model."
        )

    files = generation_files_for_model(generations_dir, split)
    if not files:
        raise RuntimeError(
            f"No generation jsonl files found for solver_model={model_name} under {generations_dir}. "
            f"split={split or 'all'}"
        )

    default_max_solve_tokens = effective_solver_max_solve_tokens
    cost_logs_dir = solver_cost_dir(paths, model_name) / "call_logs"
    by_run_and_qid, by_qid = build_cost_index(cost_logs_dir)

    report_dir = solver_reports_dir(paths, model_name) / "token_limit_reruns"
    default_json_path, default_txt_path = default_output_paths(report_dir, split)
    out_json_path = Path(out_json).resolve() if out_json else default_json_path
    out_txt_path = Path(out).resolve() if out else default_txt_path

    selected_rows: List[Dict[str, Any]] = []
    split_counts: Dict[str, Dict[str, int]] = {}
    missing_answer_total = 0
    missing_without_cost = 0
    missing_below_threshold = 0

    for gen_file in files:
        split_name = gen_file.stem
        split_stats = split_counts.setdefault(
            split_name,
            {
                "rows_scanned": 0,
                "missing_answer_rows": 0,
                "selected_rows": 0,
            },
        )
        for row in read_jsonl(gen_file):
            split_stats["rows_scanned"] += 1
            if not is_missing_final_answer(row):
                continue

            missing_answer_total += 1
            split_stats["missing_answer_rows"] += 1
            qid = str(row.get("id", "")).strip()
            workflow_id = str(row.get("workflow_id", "") or row.get("meta", {}).get("workflow_id", "") or "").strip()
            run_id = str(row.get("run_id", "") or row.get("meta", {}).get("run_id", "") or "").strip()
            cost_row = by_run_and_qid.get((workflow_id, run_id, qid)) or by_qid.get(qid)
            if not cost_row:
                missing_without_cost += 1
                continue

            usage_output_tokens = cost_row.get("usage_output_tokens")
            try:
                usage_output_tokens = int(usage_output_tokens)
            except Exception:
                missing_without_cost += 1
                continue

            max_solve_tokens = effective_max_solve_tokens(
                row,
                default_max_solve_tokens=default_max_solve_tokens,
            )
            threshold_tokens = threshold_tokens_for_row(
                max_solve_tokens=max_solve_tokens,
                min_token_ratio=min_token_ratio,
                min_output_tokens=min_output_tokens,
            )
            if usage_output_tokens < threshold_tokens:
                missing_below_threshold += 1
                continue

            split_stats["selected_rows"] += 1
            selected_rows.append(
                {
                    "id": qid,
                    "split": str(row.get("split", "") or split_name),
                    "solver_model": model_name,
                    "usage_output_tokens": usage_output_tokens,
                    "max_solve_tokens": max_solve_tokens,
                    "threshold_tokens": threshold_tokens,
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "request_id": str(cost_row.get("request_id", "") or ""),
                    "timestamp_utc": str(cost_row.get("timestamp_utc", "") or ""),
                    "model_response_empty": not bool(str(row.get("model_response", "") or "").strip()),
                    "answer_boxed": str(row.get("answer_boxed", "") or "").strip() or "N/A",
                    "final_answer_for_compare": str(row.get("final_answer_for_compare", "") or "").strip() or "N/A",
                    "selection_reason": "missing_final_answer_and_near_token_limit",
                }
            )

    selected_rows.sort(key=lambda item: (natural_key(str(item.get("split", ""))), natural_key(str(item.get("id", "")))))
    selected_ids = [str(item.get("id", "")).strip() for item in selected_rows if str(item.get("id", "")).strip()]
    write_jsonl(out_json_path, selected_rows)
    write_text_lines(out_txt_path, selected_ids)

    summary = {
        "solver_model": model_name,
        "generations_dir": str(generations_dir),
        "cost_logs_dir": str(cost_logs_dir),
        "split": split or "",
        "min_token_ratio": float(min_token_ratio),
        "min_output_tokens": int(min_output_tokens) if min_output_tokens is not None else None,
        "default_max_solve_tokens": int(default_max_solve_tokens),
        "rows_scanned": sum(item["rows_scanned"] for item in split_counts.values()),
        "missing_answer_rows": int(missing_answer_total),
        "missing_without_cost": int(missing_without_cost),
        "missing_below_threshold": int(missing_below_threshold),
        "selected_count": len(selected_rows),
        "out_json": str(out_json_path),
        "out_txt": str(out_txt_path),
        "splits": split_counts,
    }
    print(
        f"[build-token-limit-rerun-question-ids] selected={summary['selected_count']} "
        f"missing_answer_rows={summary['missing_answer_rows']} "
        f"missing_without_cost={summary['missing_without_cost']} "
        f"missing_below_threshold={summary['missing_below_threshold']} "
        f"-> {out_txt_path}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a question-id rerun file for rows that failed to produce a final answer "
            "and appear to have exhausted most of the solve token budget."
        )
    )
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model used for generation file lookup.",
    )
    parser.add_argument(
        "--hf-json-dir",
        default=None,
        help="Optional input dataset directory override.",
    )
    parser.add_argument(
        "--by-model-dir",
        default=None,
        help="Optional output root override to read generations from a custom dataset-specific location.",
    )
    parser.add_argument("--out", default=None, help="Optional output txt path for question ids.")
    parser.add_argument("--out-json", default=None, help="Optional output JSONL path with selection details.")
    parser.add_argument(
        "--min-token-ratio",
        type=float,
        default=DEFAULT_MIN_TOKEN_RATIO,
        help="Treat a row as token-limited when usage_output_tokens >= max_solve_tokens * ratio. Default: 0.98",
    )
    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=None,
        help="Optional absolute threshold override. If set, ignores --min-token-ratio.",
    )
    parser.add_argument(
        "--solver-reasoning-effort",
        default=None,
        help="Artifact label override for solver outputs generated with an explicit native reasoning effort.",
    )
    parser.add_argument(
        "--solver-max-solve-tokens",
        type=int,
        default=None,
        help="Artifact label override for solver outputs generated with a non-default max solve token budget.",
    )
    args = parser.parse_args()
    run(
        config_path=args.config,
        split=args.split,
        solver_model=args.solver_model,
        hf_json_dir=args.hf_json_dir,
        by_model_dir=args.by_model_dir,
        out=args.out,
        out_json=args.out_json,
        min_token_ratio=args.min_token_ratio,
        min_output_tokens=args.min_output_tokens,
        solver_reasoning_effort=args.solver_reasoning_effort,
        solver_max_solve_tokens=args.solver_max_solve_tokens,
    )


if __name__ == "__main__":
    main()
