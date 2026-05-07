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

from core.file_naming import parse_split_and_model_tag, sanitize_model_tag
from core.llm_utils import azure_chat_call, create_client, load_config, resolve_solver_model
from core.model_layout import (
    legacy_generations_dir,
    resolve_generation_input,
    resolve_rule_evaluation_input,
    solver_generation_file,
    solver_generations_dir,
)
from core.path_overrides import apply_dataset_path_overrides
from core.prompts import SYSTEM_LATEX_TO_SYMPY
from core.question_filter import load_question_id_filter
from core.solver_variants import build_solver_artifact_label
from core.symbol_contract import detect_symbol_mismatch, extract_symbol_tokens, normalize_allowed_symbols, parse_symbol_contract
from core.sympy_format import normalize_sympy_expression


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


def extract_last_boxed(text: str) -> str:
    marker = r"\boxed{"
    last = (text or "").rfind(marker)
    if last < 0:
        return "N/A"

    i = last + len(marker)
    depth = 1
    buf: List[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            buf.append(ch)
        else:
            buf.append(ch)
        i += 1
    if depth != 0:
        return "N/A"
    boxed = "".join(buf).strip()
    return boxed or "N/A"


def convert_boxed_to_sympy(
    *,
    client: Any,
    model: str,
    boxed_answer: str,
    temperature: float,
    max_retries: int,
    no_llm: bool,
) -> str:
    base = normalize_sympy_expression((boxed_answer or "").strip())
    if not base or base == "N/A":
        return "N/A"
    if no_llm:
        return base
    try:
        out = azure_chat_call(
            client=client,
            model=model,
            system=SYSTEM_LATEX_TO_SYMPY,
            user=f"Convert this LaTeX to SymPy:\n{base}",
            temperature=temperature,
            max_tokens=512,
            max_retries=max_retries,
        )
        out_text = (out or "").strip()
        norm = normalize_sympy_expression(out_text)
        if (not out_text) or (not norm) or (norm == "N/A"):
            return base
        return norm
    except Exception:  # noqa: BLE001
        return base


def should_reconvert_row(
    row: Dict[str, Any],
    eval_row: Optional[Dict[str, Any]],
    only_bad: bool,
) -> bool:
    answer_kind = str(row.get("answer_kind", "sympy")).strip().lower()
    comparison_mode = str(row.get("comparison_mode", "sympy")).strip().lower()
    if answer_kind == "bool" or comparison_mode == "exact":
        return False
    if not only_bad:
        return True

    answer_sympy = normalize_sympy_expression(str(row.get("answer_sympy", "")))
    if not answer_sympy or answer_sympy == "N/A":
        return True

    detail = str((eval_row or {}).get("detail", "")).strip()
    if detail.startswith("sympy_parse_failed"):
        return True
    return False


def resolve_allowed_symbols(row: Dict[str, Any], answer_sympy: str) -> List[str]:
    allowed = normalize_allowed_symbols(row.get("symbol_contract_allowed_symbols", []))
    if allowed:
        return allowed
    raw_contract = {
        "allowed_symbols": row.get("symbol_contract_allowed_symbols", []),
        "symbol_definitions": row.get("symbol_contract_definitions", {}),
    }
    if not raw_contract["allowed_symbols"]:
        raw_contract = row.get("symbol_contract", {})
    allowed2, _ = parse_symbol_contract(raw_contract)
    if allowed2:
        return allowed2
    return extract_symbol_tokens(answer_sympy)


def run(
    config_path: str = "config.yaml",
    split: Optional[str] = None,
    solver_model: Optional[str] = None,
    model: Optional[str] = None,
    hf_json_dir: Optional[str] = None,
    by_model_dir: Optional[str] = None,
    question_ids: Optional[str] = None,
    question_ids_file: Optional[str] = None,
    only_bad: bool = False,
    dry_run: bool = False,
    no_llm: bool = False,
    solver_reasoning_effort: Optional[str] = None,
    solver_max_solve_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = apply_dataset_path_overrides(
        load_config(config_path),
        hf_json_dir=hf_json_dir,
        by_model_dir=by_model_dir,
    )
    paths = cfg["paths"]
    gen_cfg = cfg.get("generate", {})

    base_model_name = resolve_solver_model(cfg, requested_model=solver_model or model)
    model_name = build_solver_artifact_label(
        base_model_name,
        reasoning_effort=solver_reasoning_effort,
        max_solve_tokens=solver_max_solve_tokens or int((cfg.get("generate") or {}).get("max_solve_tokens", 4096) or 4096),
    )
    question_id_filter = load_question_id_filter(question_ids=question_ids, question_ids_file=question_ids_file)

    client = None if no_llm else create_client(cfg, model_name=model_name)
    files: List[Path] = []
    if split:
        preferred = solver_generation_file(paths, model_name, split)
        if preferred.exists():
            files = [preferred]
        else:
            fallback = resolve_generation_input(paths, model_name, split)
            if fallback.exists():
                files = [fallback]
        if not files:
            raise RuntimeError(f"No generation file found for split={split}, model={model_name}.")
    else:
        primary_dir = solver_generations_dir(paths, model_name)
        files = sorted([p for p in primary_dir.glob("*.jsonl") if p.is_file()], key=lambda p: p.name)
        if not files:
            legacy_dir = legacy_generations_dir(paths)
            legacy_all = sorted([p for p in legacy_dir.glob("*.jsonl") if p.is_file()], key=lambda p: p.name)
            model_tag_target = sanitize_model_tag(model_name)
            tagged: List[Path] = []
            plain: List[Path] = []
            for p in legacy_all:
                _, model_tag = parse_split_and_model_tag(p.name)
                if model_tag is None:
                    plain.append(p)
                elif model_tag == model_tag_target:
                    tagged.append(p)
            files = tagged if tagged else plain

    if not files:
        raise RuntimeError(
            f"No generation files found for model={model_name} under "
            f"{solver_generations_dir(paths, model_name)} or {legacy_generations_dir(paths)}"
        )

    summary: Dict[str, Any] = {
        "model": model_name,
        "splits": {},
        "rows_total": 0,
        "rows_targeted": 0,
        "rows_changed": 0,
        "dry_run": bool(dry_run),
        "only_bad": bool(only_bad),
        "no_llm": bool(no_llm),
    }

    for gen_file in files:
        split_name, _ = parse_split_and_model_tag(gen_file.name)
        eval_file = resolve_rule_evaluation_input(paths, model_name, split_name)
        eval_by_id = {str(r.get("id", "")).strip(): r for r in read_jsonl(eval_file)}

        rows = read_jsonl(gen_file)
        changed = 0
        targeted = 0
        updated_rows: List[Dict[str, Any]] = []
        for row in rows:
            out_row = dict(row)
            qid = str(out_row.get("id", "")).strip()
            if question_id_filter and qid not in question_id_filter:
                updated_rows.append(out_row)
                continue
            eval_row = eval_by_id.get(qid)
            if not should_reconvert_row(out_row, eval_row, only_bad=only_bad):
                updated_rows.append(out_row)
                continue

            targeted += 1
            boxed = str(out_row.get("answer_boxed", "")).strip()
            if not boxed or boxed == "N/A":
                boxed = extract_last_boxed(str(out_row.get("model_response", "")))
                out_row["answer_boxed"] = boxed

            new_sympy = convert_boxed_to_sympy(
                client=client,
                model=model_name,
                boxed_answer=boxed,
                temperature=float(gen_cfg.get("convert_temperature", 0.0)),
                max_retries=int(cfg["azure"].get("max_retries", 4)),
                no_llm=no_llm,
            )
            new_sympy = normalize_sympy_expression(new_sympy)
            old_sympy = normalize_sympy_expression(str(out_row.get("answer_sympy", "")))
            if old_sympy != new_sympy:
                changed += 1

            out_row["answer_sympy"] = new_sympy
            out_row["final_answer_for_compare"] = new_sympy if new_sympy != "N/A" else boxed

            allowed = resolve_allowed_symbols(out_row, new_sympy)
            out_row["symbol_contract_allowed_symbols"] = allowed
            if allowed:
                mismatch_symbols = detect_symbol_mismatch(str(out_row["final_answer_for_compare"]), allowed)
                out_row["symbol_mismatch"] = bool(mismatch_symbols)
                out_row["mismatch_symbols"] = mismatch_symbols
            else:
                out_row["symbol_mismatch"] = False
                out_row["mismatch_symbols"] = []

            updated_rows.append(out_row)

        if not dry_run:
            write_jsonl(gen_file, updated_rows)

        summary["splits"][split_name] = {
            "generation_file": gen_file.as_posix(),
            "evaluation_file": eval_file.as_posix(),
            "rows_total": len(rows),
            "rows_targeted": targeted,
            "rows_changed": changed,
        }
        summary["rows_total"] += len(rows)
        summary["rows_targeted"] += targeted
        summary["rows_changed"] += changed
        print(
            f"[reconvert] {split_name}: total={len(rows)} targeted={targeted} changed={changed}"
            + (" (dry-run)" if dry_run else "")
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run sympy conversion on existing generation files without re-solving.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--split", default=None, help="Optional split name, e.g. chapter_6")
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
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model for generation/evaluation file naming. Overrides config models.default_solver_model.",
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
    parser.add_argument("--only-bad", action="store_true", help="Only reconvert rows that are likely problematic.")
    parser.add_argument("--dry-run", action="store_true", help="Compute changes but do not write files.")
    parser.add_argument("--no-llm", action="store_true", help="Do not call conversion API; only normalize current boxed answers.")
    args = parser.parse_args()
    run(
        config_path=args.config,
        split=args.split,
        solver_model=args.solver_model,
        hf_json_dir=args.hf_json_dir,
        by_model_dir=args.by_model_dir,
        question_ids=args.question_ids,
        question_ids_file=args.question_ids_file,
        only_bad=args.only_bad,
        dry_run=args.dry_run,
        no_llm=args.no_llm,
    )


if __name__ == "__main__":
    main()
