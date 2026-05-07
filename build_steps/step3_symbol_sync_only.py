#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.llm_utils import create_client, load_config
from core.symbol_contract import parse_symbol_contract

from build_steps.common import (
    natural_key,
    read_jsonl,
    resolve_target_splits,
    stage_file,
    write_jsonl,
)
from build_steps.step3_symbol_sync import align_stage3_question_symbols


def parse_problem_filters(problem: Optional[str]) -> List[str]:
    if not isinstance(problem, str) or not problem.strip():
        return []
    return [x.strip() for x in problem.split(",") if x.strip()]


def row_matches_problem(row: Dict[str, Any], filters: List[str]) -> bool:
    if not filters:
        return True
    problem_number = str(row.get("problem_number", "")).strip()
    base_id = str(row.get("id", "")).strip()
    original_id = str(row.get("original_id", "")).strip()
    for f in filters:
        if problem_number == f:
            return True
        if base_id == f or original_id == f:
            return True
        if base_id.startswith(f + "/") or original_id.startswith(f + "/"):
            return True
    return False


def _resolve_contract(stage3_row: Dict[str, Any], final_row: Optional[Dict[str, Any]]) -> tuple[List[str], Dict[str, str]]:
    c = stage3_row.get("symbol_contract", {})
    allowed, defs = parse_symbol_contract(c)
    if allowed:
        return allowed, defs
    if isinstance(final_row, dict):
        raw = {
            "allowed_symbols": final_row.get("symbol_contract_allowed_symbols", []),
            "symbol_definitions": final_row.get("symbol_contract_definitions", {}),
        }
        allowed2, defs2 = parse_symbol_contract(raw)
        if allowed2:
            return allowed2, defs2
    return [], {}


def run(
    config_path: str = "config.yaml",
    chapter: Optional[str] = None,
    target_splits: Optional[List[str]] = None,
    problem: Optional[str] = None,
) -> List[str]:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    chapters_root = Path(paths["chapters_root"])
    hf_json_dir = Path(paths["hf_json_dir"])
    pipeline_dir = Path(paths.get("pipeline_dir") or (hf_json_dir.parent / "pipeline"))
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    hf_json_dir.mkdir(parents=True, exist_ok=True)

    splits = resolve_target_splits(chapters_root, chapter=chapter, target_splits=target_splits)
    if not splits:
        splits = sorted(
            [p.name.replace("_stage3_transform.jsonl", "") for p in pipeline_dir.glob("*_stage3_transform.jsonl")],
            key=natural_key,
        )
    if not splits:
        print("[step3-symbol-sync] no target splits found.")
        return []

    max_retries = int(cfg["azure"].get("max_retries", 4))
    build_cfg = cfg.get("build", {})
    model = cfg["azure"]["deployment_name"]
    problem_filters = parse_problem_filters(problem)

    client: Any = None
    processed: List[str] = []

    for split in splits:
        stage3_file = stage_file(pipeline_dir, split, "stage3_transform")
        final_file = hf_json_dir / f"{split}.jsonl"
        if not stage3_file.exists():
            print(f"[step3-symbol-sync] skip {split}: missing {stage3_file.name}")
            continue
        if not final_file.exists():
            print(f"[step3-symbol-sync] skip {split}: missing {final_file.name}")
            continue

        stage3_rows = read_jsonl(stage3_file)
        final_rows = read_jsonl(final_file)
        if not stage3_rows or not final_rows:
            print(f"[step3-symbol-sync] skip {split}: empty stage3/final rows")
            continue

        final_by_id = {str(r.get("id", "")).strip(): r for r in final_rows if str(r.get("id", "")).strip()}
        changed = 0
        targeted = 0
        missing_final = 0

        for row in stage3_rows:
            if str(row.get("question_type", "")).strip().lower() != "value":
                continue
            if not row_matches_problem(row, problem_filters):
                continue
            qid = str(row.get("id", "")).strip()
            if not qid:
                continue

            final_row = final_by_id.get(qid)
            if final_row is None:
                missing_final += 1
                continue

            allowed_symbols, symbol_definitions = _resolve_contract(row, final_row)
            if not allowed_symbols:
                continue

            if client is None:
                client = create_client(cfg)
            targeted += 1

            before_question = str(row.get("converted_question", "")).strip()
            preserved_before = str(row.get("converted_question_before_symbol_sync", "")).strip()
            if not preserved_before:
                preserved_before = before_question

            sync = align_stage3_question_symbols(
                client=client,
                model=model,
                max_retries=max_retries,
                temperature=float(build_cfg.get("symbol_sync_temperature", 0.0)),
                converted_question=before_question,
                reference_answer_sympy=str(row.get("reference_answer_sympy", "")).strip(),
                allowed_symbols=allowed_symbols,
                symbol_definitions=symbol_definitions,
                split=split,
                question_id=qid,
            )
            final_question = str(sync.get("converted_question_final", before_question)).strip() or before_question
            if final_question != before_question:
                changed += 1

            row["converted_question_before_symbol_sync"] = preserved_before
            row["converted_question"] = final_question
            row["symbol_text_sync_applied"] = bool(sync.get("applied", False))
            row["symbol_text_sync_changed_by_llm"] = bool(sync.get("changed_by_llm", False))
            row["symbol_text_sync_status"] = str(sync.get("status", "skipped"))
            row["symbol_text_sync_reason"] = str(sync.get("reason", ""))
            row["symbol_text_sync_used_symbols"] = list(sync.get("used_symbols", []))
            row["symbol_text_sync_json"] = sync.get("raw_json", {})

            final_row["converted_question_before_symbol_sync"] = preserved_before
            final_row["question_standalone"] = final_question
            final_row["question_final"] = final_question
            final_row["symbol_text_sync_applied"] = bool(sync.get("applied", False))
            final_row["symbol_text_sync_status"] = str(sync.get("status", "skipped"))
            final_row["symbol_text_sync_reason"] = str(sync.get("reason", ""))
            final_row["symbol_text_sync_used_symbols"] = list(sync.get("used_symbols", []))
            final_row["symbol_text_sync_json"] = sync.get("raw_json", {})

            print(f"[step3-symbol-sync] {split} -> {qid} status={row['symbol_text_sync_status']} changed={final_question != before_question}")

        write_jsonl(stage3_file, stage3_rows)
        write_jsonl(final_file, final_rows)
        print(
            f"[step3-symbol-sync] wrote {split}: targeted={targeted} changed={changed} missing_final={missing_final} "
            f"-> {stage3_file.name}, {final_file.name}"
        )
        processed.append(split)

    return sorted(list(dict.fromkeys(processed)), key=natural_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run only stage3 symbol text sync on existing stage3/hf_json outputs."
    )
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--chapter", default=None, help="Optional chapter selector, e.g. 6 or chapter_6")
    parser.add_argument(
        "--problem",
        default=None,
        help="Optional problem filter(s), comma-separated. Examples: 6.8 or 6.8,6.9",
    )
    args = parser.parse_args()
    run(config_path=args.config, chapter=args.chapter, problem=args.problem)


if __name__ == "__main__":
    main()
