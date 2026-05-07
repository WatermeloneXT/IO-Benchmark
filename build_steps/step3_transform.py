#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.llm_utils import create_client, load_config

from build_steps.common import (
    compact_text,
    natural_key,
    read_jsonl,
    resolve_target_splits,
    stage_file,
    write_jsonl,
)
from build_steps.llm_ops import llm_convert_to_eval_item
from build_steps.step3_symbol_sync import align_stage3_question_symbols


def _build_from_stage2_row(
    row: Dict[str, Any],
    cfg: Dict[str, Any],
    client: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model = cfg["azure"]["deployment_name"]
    max_retries = int(cfg["azure"].get("max_retries", 4))
    build_cfg = cfg.get("build", {})

    base_id = str(row.get("id", "")).strip()
    problem_latex = str(row.get("question_original", "")).strip()
    corrected_answer = str(row.get("reference_answer_corrected", "N/A")).strip() or "N/A"
    pairs = row.get("pairs", [])
    if not isinstance(pairs, list) or not pairs:
        pairs = [{"sub_index": 1, "question": problem_latex, "answer": corrected_answer}]

    total_pairs = len(pairs)
    split = str(row.get("split", "")).strip()

    stage3_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []

    for p_idx, pair in enumerate(pairs, start=1):
        pair_question = str(pair.get("question", "")).strip() or problem_latex
        pair_answer = compact_text(str(pair.get("answer", "")), default=corrected_answer)
        final_id = base_id if total_pairs == 1 else f"{base_id}#{p_idx}"

        stage3 = llm_convert_to_eval_item(
            client=client,
            model=model,
            max_retries=max_retries,
            temperature=float(build_cfg.get("transform_temperature", 0.0)),
            sympy_temperature=float(build_cfg.get("convert_temperature", 0.0)),
            question_latex=pair_question,
            reference_answer=pair_answer,
            split=split,
            question_id=final_id,
        )
        stage3_retry_count = 0
        if stage3["question_type"] == "value" and not bool(stage3.get("symbol_contract_validation_passed", True)):
            stage3_retry_count = 1
            mismatch = stage3.get("symbol_contract_mismatch_symbols", [])
            repair_hint = f"reference_answer_sympy has symbols outside allowed_symbols: {mismatch}"
            stage3 = llm_convert_to_eval_item(
                client=client,
                model=model,
                max_retries=max_retries,
                temperature=float(build_cfg.get("transform_temperature", 0.0)),
                sympy_temperature=float(build_cfg.get("convert_temperature", 0.0)),
                question_latex=pair_question,
                reference_answer=pair_answer,
                repair_hint=repair_hint,
                split=split,
                question_id=final_id,
            )

        converted_question_before_symbol_sync = stage3["converted_question"]
        symbol_sync = {
            "converted_question_final": converted_question_before_symbol_sync,
            "applied": False,
            "changed_by_llm": False,
            "status": "skipped",
            "reason": "",
            "used_symbols": [],
            "raw_json": {},
        }
        if stage3["question_type"] == "value":
            contract = stage3.get("symbol_contract", {}) or {}
            symbol_sync = align_stage3_question_symbols(
                client=client,
                model=model,
                max_retries=max_retries,
                temperature=float(build_cfg.get("symbol_sync_temperature", 0.0)),
                converted_question=converted_question_before_symbol_sync,
                reference_answer_sympy=stage3.get("reference_answer_sympy", ""),
                allowed_symbols=contract.get("allowed_symbols", []),
                symbol_definitions=contract.get("symbol_definitions", {}),
                split=split,
                question_id=final_id,
            )
            stage3["converted_question"] = symbol_sync["converted_question_final"]

        stage3_row = {
            "id": final_id,
            "original_id": base_id,
            "pair_index": p_idx,
            "pair_count": total_pairs,
            "chapter": row.get("chapter"),
            "split": split,
            "problem_number": row.get("problem_number"),
            "sub_id": row.get("sub_id"),
            "source_path": row.get("source_path"),
            "pair_question": pair_question,
            "pair_answer": pair_answer,
            "question_type": stage3["question_type"],
            "comparison_mode": stage3["comparison_mode"],
            "converted_question": stage3["converted_question"],
            "converted_question_before_symbol_sync": converted_question_before_symbol_sync,
            "symbol_text_sync_applied": symbol_sync.get("applied", False),
            "symbol_text_sync_changed_by_llm": symbol_sync.get("changed_by_llm", False),
            "symbol_text_sync_status": symbol_sync.get("status", "skipped"),
            "symbol_text_sync_reason": symbol_sync.get("reason", ""),
            "symbol_text_sync_used_symbols": symbol_sync.get("used_symbols", []),
            "symbol_text_sync_json": symbol_sync.get("raw_json", {}),
            "reference_reasoning": stage3["reference_reasoning"],
            "converted_answer": stage3["converted_answer"],
            "comparable_final_answer": stage3["comparable_final_answer"],
            "reference_answer_sympy": stage3["reference_answer_sympy"],
            "symbol_contract": stage3.get("symbol_contract", {}),
            "symbol_contract_validation_passed": stage3.get("symbol_contract_validation_passed", True),
            "symbol_contract_mismatch_symbols": stage3.get("symbol_contract_mismatch_symbols", []),
            "stage3_retry_count": stage3_retry_count,
            "llm_json": stage3["raw_json"],
        }
        stage3_rows.append(stage3_row)

        out_row = {
            "id": final_id,
            "original_id": base_id,
            "pair_index": p_idx,
            "pair_count": total_pairs,
            "chapter": row.get("chapter"),
            "split": split,
            "problem_number": row.get("problem_number"),
            "sub_id": row.get("sub_id"),
            "source_path": row.get("source_path"),
            "source_problem_path": row.get("source_problem_path", ""),
            "source_type": "out_split",
            "question_original": problem_latex,
            "reference_answer_original": row.get("reference_answer_original", "N/A"),
            "reference_answer_corrected": corrected_answer,
            "reference_answer_generated_by_llm": row.get("reference_answer_generated_by_llm", False),
            "split_question": pair_question,
            "split_answer": pair_answer,
            "split_generated_by_llm": row.get("split_generated_by_llm", False),
            "converted_question_before_symbol_sync": converted_question_before_symbol_sync,
            "symbol_text_sync_applied": symbol_sync.get("applied", False),
            "symbol_text_sync_status": symbol_sync.get("status", "skipped"),
            "symbol_text_sync_reason": symbol_sync.get("reason", ""),
            "symbol_text_sync_used_symbols": symbol_sync.get("used_symbols", []),
            "symbol_text_sync_json": json.dumps(symbol_sync.get("raw_json", {}), ensure_ascii=False),
            "question_standalone": stage3["converted_question"],
            "question_final": stage3["converted_question"],
            "question_type": stage3["question_type"],
            "answer_kind": stage3["answer_kind"],
            "comparison_mode": stage3["comparison_mode"],
            "reference_reasoning": stage3["reference_reasoning"],
            "reference_answer": stage3["comparable_final_answer"],
            "final_answer_for_compare": stage3["comparable_final_answer"],
            "reference_answer_sympy": stage3["reference_answer_sympy"],
            "symbol_contract_allowed_symbols": (stage3.get("symbol_contract", {}) or {}).get("allowed_symbols", []),
            "symbol_contract_definitions": (stage3.get("symbol_contract", {}) or {}).get("symbol_definitions", {}),
            "symbol_contract_validation_passed": stage3.get("symbol_contract_validation_passed", True),
            "symbol_contract_mismatch_symbols": stage3.get("symbol_contract_mismatch_symbols", []),
            "stage3_retry_count": stage3_retry_count,
            "stage3_failed_symbol_contract": (
                stage3["question_type"] == "value" and not bool(stage3.get("symbol_contract_validation_passed", True))
            ),
            "stage1_confidence": row.get("stage1_confidence", "low"),
            "extract_confidence": row.get("stage1_confidence", "low"),
            "stage1_json": json.dumps(row.get("stage1_json", {}), ensure_ascii=False),
            "stage2_json": json.dumps(row.get("stage2_json", {}), ensure_ascii=False),
            "stage3_json": json.dumps(stage3["raw_json"], ensure_ascii=False),
        }
        src_problem = str(row.get("source_problem_latex", "")).strip()
        if src_problem:
            out_row["source_problem_latex"] = src_problem
        if bool(build_cfg.get("include_source_answer", True)):
            out_row["source_answer_latex"] = row.get("source_answer_latex", "")
            out_row["reference_answer_latex_from_by_problem"] = row.get("reference_answer_latex_from_by_problem", "")
        final_rows.append(out_row)

    return stage3_rows, final_rows


def parse_problem_filters(problem: Optional[str]) -> List[str]:
    if not isinstance(problem, str) or not problem.strip():
        return []
    return [x.strip() for x in problem.split(",") if x.strip()]


def row_matches_problem(row: Dict[str, Any], filters: List[str]) -> bool:
    if not filters:
        return True
    problem_number = str(row.get("problem_number", "")).strip()
    base_id = str(row.get("id", "")).strip()
    for f in filters:
        if problem_number == f:
            return True
        if base_id == f:
            return True
        if base_id.startswith(f + "/"):
            return True
    return False


def base_ids_from_stage2_rows(rows: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for row in rows:
        base_id = str(row.get("id", "")).strip()
        if base_id:
            out.add(base_id)
    return out


def row_belongs_to_base_ids(row: Dict[str, Any], base_ids: Set[str]) -> bool:
    rid = str(row.get("id", "")).strip()
    if not rid:
        return False
    if rid in base_ids:
        return True
    for base in base_ids:
        if rid.startswith(base + "#"):
            return True
    return False


def merge_replace_rows(
    existing_rows: List[Dict[str, Any]],
    updated_rows: List[Dict[str, Any]],
    replace_base_ids: Set[str],
) -> List[Dict[str, Any]]:
    kept = [r for r in existing_rows if not row_belongs_to_base_ids(r, replace_base_ids)]
    merged = kept + updated_rows
    merged.sort(key=lambda r: natural_key(str(r.get("id", ""))))
    return merged


def build_stage3_for_split(
    split: str,
    stage2_rows: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    client: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not stage2_rows:
        return [], []

    build_cfg = cfg.get("build", {})
    max_workers = int(build_cfg.get("max_workers", 1) or 1)

    stage3_indexed: List[Tuple[int, Dict[str, Any]]] = []
    final_indexed: List[Tuple[int, Dict[str, Any]]] = []

    def _process_one(idx: int, row: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]], str]:
        qid = str(row.get("id", "")).strip()
        s3_rows, final_rows = _build_from_stage2_row(row=row, cfg=cfg, client=client)
        return idx, s3_rows, final_rows, f"[step3] {split} {idx}/{len(stage2_rows)} -> {qid} ({len(final_rows)} item(s))"

    if max_workers <= 1:
        for idx, row in enumerate(stage2_rows, start=1):
            i, s3_rows, final_rows, msg = _process_one(idx, row)
            print(msg)
            for x in s3_rows:
                stage3_indexed.append((i, x))
            for x in final_rows:
                final_indexed.append((i, x))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(_process_one, idx, row): idx for idx, row in enumerate(stage2_rows, start=1)}
            for fut in as_completed(future_map):
                try:
                    i, s3_rows, final_rows, msg = fut.result()
                    print(msg)
                    for x in s3_rows:
                        stage3_indexed.append((i, x))
                    for x in final_rows:
                        final_indexed.append((i, x))
                except Exception as err:  # noqa: BLE001
                    i = future_map[fut]
                    print(f"[step3] {split} {i}/{len(stage2_rows)} -> failed: {err}")

    stage3_indexed.sort(key=lambda x: (x[0], natural_key(str(x[1].get("id", "")))))
    final_indexed.sort(key=lambda x: (x[0], natural_key(str(x[1].get("id", "")))))
    return [r for _, r in stage3_indexed], [r for _, r in final_indexed]


def run(
    config_path: str = "config.yaml",
    chapter: Optional[str] = None,
    target_splits: Optional[List[str]] = None,
    problem: Optional[str] = None,
    skip_existing: bool = False,
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
            [p.name.replace("_stage2_split.jsonl", "") for p in pipeline_dir.glob("*_stage2_split.jsonl")],
            key=natural_key,
        )

    if not splits:
        print("[step3] no target splits found.")
        return []

    client: Any = None
    processed: List[str] = []
    problem_filters = parse_problem_filters(problem)

    for split in splits:
        in_file = stage_file(pipeline_dir, split, "stage2_split")
        if not in_file.exists():
            print(f"[step3] skip {split}: missing {in_file.name}")
            continue
        stage3_file = stage_file(pipeline_dir, split, "stage3_transform")
        final_file = hf_json_dir / f"{split}.jsonl"
        if (
            skip_existing
            and (not problem_filters)
            and stage3_file.exists()
            and final_file.exists()
            and stage3_file.stat().st_size > 0
            and final_file.stat().st_size > 0
        ):
            print(f"[step3] skip {split}: existing outputs -> {stage3_file}, {final_file}")
            processed.append(split)
            continue

        stage2_rows_all = read_jsonl(in_file)
        stage2_rows = [r for r in stage2_rows_all if row_matches_problem(r, problem_filters)]
        if problem_filters:
            print(f"[step3] {split}: filtered {len(stage2_rows)}/{len(stage2_rows_all)} rows by problem={problem_filters}")
        if not stage2_rows:
            continue

        if client is None:
            client = create_client(cfg)
        stage3_rows, final_rows = build_stage3_for_split(split, stage2_rows, cfg, client)
        if not final_rows:
            continue

        if problem_filters and stage3_file.exists():
            existing_stage3 = read_jsonl(stage3_file)
            replace_base_ids = base_ids_from_stage2_rows(stage2_rows)
            stage3_rows = merge_replace_rows(existing_stage3, stage3_rows, replace_base_ids)
        if problem_filters and final_file.exists():
            existing_final = read_jsonl(final_file)
            replace_base_ids = base_ids_from_stage2_rows(stage2_rows)
            final_rows = merge_replace_rows(existing_final, final_rows, replace_base_ids)
        write_jsonl(stage3_file, stage3_rows)
        write_jsonl(final_file, final_rows)
        print(f"[step3] wrote {len(stage3_rows)} rows -> {stage3_file}")
        print(f"[step3] wrote {len(final_rows)} rows -> {final_file}")
        processed.append(split)

    return sorted(list(dict.fromkeys(processed)), key=natural_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage3: convert to evaluation-ready items.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--chapter", default=None, help="Optional chapter selector, e.g. 6 or chapter_6")
    parser.add_argument(
        "--problem",
        default=None,
        help="Optional problem filter(s), comma-separated. Examples: 6.8 or 6.8,6.9",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip split if stage3 and final split outputs already exist and are non-empty.",
    )
    args = parser.parse_args()
    run(config_path=args.config, chapter=args.chapter, problem=args.problem, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
