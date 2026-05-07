#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_utils import load_config, resolve_solver_model
from core.model_layout import solver_rule_evaluations_dir
from core.path_overrides import apply_dataset_path_overrides
from core.solver_variants import build_solver_artifact_label


DEFAULT_DETAIL_PATTERNS = [
    "parse_failed",
    "json_prediction_invalid",
    "json_reference_invalid",
    "sympy_not_installed",
]


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
    return PROJECT_ROOT / "data" / "rerun_question_ids" / f"{dataset_tag}_{solver_model}_rerun.txt"


def default_manual_updates_path(hf_json_dir: Path) -> Path:
    dataset_tag = dataset_tag_from_hf_dir(hf_json_dir)
    return PROJECT_ROOT / "data" / "rerun_question_ids" / f"{dataset_tag}_dataset_updates_rerun.txt"


def read_question_ids_txt(path: Path) -> List[str]:
    if not path.exists():
        return []
    tokens = [tok.strip() for tok in re.split(r"[\s,]+", path.read_text(encoding="utf-8")) if tok.strip()]
    return tokens


def should_include_row(detail: str, patterns: List[str]) -> bool:
    text = str(detail or "").strip().lower()
    if not text:
        return False
    return any(pattern.lower() in text for pattern in patterns)


def run(
    config_path: str = "config.yaml",
    solver_model: Optional[str] = None,
    model: Optional[str] = None,
    split: Optional[str] = None,
    hf_json_dir: Optional[str] = None,
    by_model_dir: Optional[str] = None,
    out: Optional[str] = None,
    detail_patterns: Optional[List[str]] = None,
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
    eval_dir = solver_rule_evaluations_dir(paths, model_name)

    eval_files = sorted([p for p in eval_dir.glob("*.jsonl") if p.is_file()], key=lambda p: p.name)
    if split:
        eval_files = [p for p in eval_files if p.stem == split]
    if not eval_files:
        raise RuntimeError(
            f"No rule evaluation files found for solver_model={model_name} under {eval_dir}. "
            "Run evaluate first with the same --solver-model and --by-model-dir."
        )

    patterns = detail_patterns or list(DEFAULT_DETAIL_PATTERNS)
    selected: List[Tuple[str, str]] = []
    for eval_file in eval_files:
        for row in read_jsonl(eval_file):
            qid = str(row.get("id", "")).strip()
            detail = str(row.get("detail", "")).strip()
            if qid and should_include_row(detail, patterns):
                split_name = str(row.get("split", "")).strip() or eval_file.stem
                selected.append((split_name, qid))

    manual_updates_path = default_manual_updates_path(hf_json_dir_path)
    manual_ids = read_question_ids_txt(manual_updates_path)
    for qid in manual_ids:
        selected.append(("", qid))

    selected_by_qid: Dict[str, str] = {}
    for split_name, qid in selected:
        current_split = selected_by_qid.get(qid, "")
        if current_split:
            continue
        selected_by_qid[qid] = split_name
    selected = sorted(
        ((split_name, qid) for qid, split_name in selected_by_qid.items()),
        key=lambda item: (natural_key(item[0]), natural_key(item[1])),
    )
    out_path = Path(out).resolve() if out else default_output_path(hf_json_dir_path, model_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(f"{qid}\n" for _, qid in selected), encoding="utf-8")

    summary = {
        "solver_model": model_name,
        "eval_dir": str(eval_dir),
        "split": split or "",
        "detail_patterns": patterns,
        "manual_updates_file": str(manual_updates_path) if manual_updates_path.exists() else "",
        "manual_updates_count": len(manual_ids),
        "count": len(selected),
        "out_file": str(out_path),
    }
    print(
        f"[build-rerun-question-ids] selected={summary['count']} "
        f"manual_updates={summary['manual_updates_count']} "
        f"patterns={patterns} -> {out_path}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a question-id file for targeted reruns from rule evaluation results.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model used for evaluation file lookup.",
    )
    parser.add_argument(
        "--hf-json-dir",
        default=None,
        help="Optional input dataset directory override. Example: data/review_annotations/post_step5_confirmed_dataset",
    )
    parser.add_argument(
        "--by-model-dir",
        default=None,
        help="Optional output root override to read evaluations from a custom dataset-specific location.",
    )
    parser.add_argument("--out", default=None, help="Optional output txt path for question ids.")
    parser.add_argument(
        "--detail-pattern",
        dest="detail_patterns",
        action="append",
        default=None,
        help="Substring filter for rule eval detail. Can be passed multiple times. "
        "Defaults to parse/conversion-style failures.",
    )
    args = parser.parse_args()
    run(
        config_path=args.config,
        split=args.split,
        solver_model=args.solver_model,
        hf_json_dir=args.hf_json_dir,
        by_model_dir=args.by_model_dir,
        out=args.out,
        detail_patterns=args.detail_patterns,
    )


if __name__ == "__main__":
    main()
