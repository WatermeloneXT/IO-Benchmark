#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cost_logging import register_run_outputs
from core.annotation_overrides import (
    apply_split_annotation_overrides,
    resolve_effective_question,
    resolve_effective_reference_answer,
    resolve_effective_reference_answer_sympy,
)
from core.file_naming import resolve_existing_model_jsonl
from core.llm_utils import (
    azure_chat_call,
    create_client,
    load_config,
    resolve_generate_reasoning_request,
    resolve_solver_model,
)
from core.model_layout import (
    legacy_llm_evaluations_dir,
    legacy_reports_dir,
    legacy_rule_evaluations_dir,
    legacy_generations_dir,
    resolve_rule_evaluation_input,
    solver_compare_reports_dir,
    solver_generation_file,
    solver_reports_dir,
    solver_root,
)
from core.path_overrides import apply_dataset_path_overrides
from core.prompts import SYSTEM_LATEX_TO_SYMPY, SYSTEM_SOLVER_WITH_BOX, solve_user_prompt
from core.question_filter import filter_rows_by_question_ids, load_question_id_filter, merge_rows_by_question_id
from core.result_metadata import build_result_metadata
from core.solver_variants import build_solver_artifact_label, validate_native_reasoning_effort
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


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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


def failure_artifact_paths(base_dir: Path, split_name: str) -> Tuple[Path, Path]:
    return base_dir / f"{split_name}.jsonl", base_dir / f"{split_name}__question_ids.txt"


def clear_failure_artifacts(base_dir: Path, split_name: str) -> None:
    detail_path, qid_path = failure_artifact_paths(base_dir, split_name)
    if detail_path.exists():
        detail_path.unlink()
    if qid_path.exists():
        qid_path.unlink()


def write_failure_artifacts(
    base_dir: Path,
    split_name: str,
    failures: List[Dict[str, Any]],
) -> Tuple[Path, Path]:
    detail_path, qid_path = failure_artifact_paths(base_dir, split_name)
    write_jsonl(detail_path, failures)
    qids = [str(item.get("id", "")).strip() for item in failures if str(item.get("id", "")).strip()]
    write_text_lines(qid_path, qids)
    return detail_path, qid_path


def _drop_question_ids_from_jsonl(path: Path, question_ids: Set[str]) -> bool:
    if not path.exists() or not question_ids:
        return False
    rows = read_jsonl(path)
    kept_rows = [row for row in rows if str(row.get("id", "")).strip() not in question_ids]
    if len(kept_rows) == len(rows):
        return False
    if kept_rows:
        write_jsonl(path, kept_rows)
    else:
        path.unlink()
    return True


def invalidate_evaluation_artifacts(
    paths: Dict[str, Any],
    solver_model: str,
    split_name: str,
    question_ids: Sequence[str],
) -> Dict[str, List[Path]]:
    affected_question_ids = {str(qid).strip() for qid in question_ids if str(qid).strip()}
    row_files: List[Path] = []
    aggregate_files: List[Path] = []

    eval_roots = [
        solver_root(paths, solver_model) / "evaluations",
        legacy_rule_evaluations_dir(paths),
        legacy_llm_evaluations_dir(paths),
    ]
    seen_paths: Set[Path] = set()
    for root in eval_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob(f"{split_name}.jsonl")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            if _drop_question_ids_from_jsonl(path, affected_question_ids):
                row_files.append(path)
        for path in sorted(root.rglob("summary.json")):
            if path.is_file():
                resolved = path.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                path.unlink()
                aggregate_files.append(path)

    compare_dirs = [
        solver_compare_reports_dir(paths, solver_model),
        legacy_reports_dir(paths) / "eval_error_compare",
    ]
    seen_compare: Set[Path] = set()
    for compare_dir in compare_dirs:
        if not compare_dir.exists():
            continue
        for path in sorted(compare_dir.iterdir()):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen_compare:
                continue
            seen_compare.add(resolved)
            path.unlink()
            aggregate_files.append(path)

    return {
        "row_files": row_files,
        "aggregate_files": aggregate_files,
    }


def split_input_files(hf_json_dir: Path, split: Optional[str] = None) -> List[Path]:
    files = sorted([p for p in hf_json_dir.glob("*.jsonl") if p.is_file()], key=lambda p: p.name)
    if split:
        return [p for p in files if p.stem == split]
    return [p for p in files if not p.stem.endswith("_balanced")]


def load_done_ids(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    for row in read_jsonl(path):
        qid = str(row.get("id", "")).strip()
        if qid:
            done.add(qid)
    return done


def extract_last_boxed(text: str) -> str:
    marker = r"\boxed{"
    last = text.rfind(marker)
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


def normalize_bool_answer(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    s = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:text|mathrm|operatorname|mathbf|mathit)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("$", "")
    s = re.sub(r"[`\s]", "", s).lower()
    if s in {"true", "t", "yes", "y", "1"}:
        return "True"
    if s in {"false", "f", "no", "n", "0"}:
        return "False"
    return None


def normalize_text_answer(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()) or "N/A"


def _maybe_wrap_json_object_fragment(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    if s.startswith("{") or s.endswith("}"):
        return s
    # Conservative repair: only wrap strings that look like the inside of a JSON object,
    # e.g. `"q1": "1/4", "q2": "1/4"`.
    if not re.match(r'^\s*"[^"]+"\s*:', s):
        return s
    wrapped = "{" + s + "}"
    try:
        parsed = json.loads(wrapped)
    except Exception:
        return s
    return wrapped if isinstance(parsed, dict) else s


def canonicalize_json_answer(value: Any) -> Tuple[Optional[Dict[str, str]], str]:
    raw_obj: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "N/A":
            return None, "N/A"
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        text = _maybe_wrap_json_object_fragment(text)
        try:
            raw_obj = json.loads(text)
        except Exception:
            try:
                raw_obj = ast.literal_eval(text)
            except Exception:
                return None, normalize_text_answer(text)
    if not isinstance(raw_obj, dict):
        return None, normalize_text_answer(raw_obj)

    normalized: Dict[str, str] = {}
    for key, item in raw_obj.items():
        out_key = str(key).strip()
        if not out_key:
            continue
        if isinstance(item, (dict, list)):
            normalized[out_key] = json.dumps(item, ensure_ascii=False, sort_keys=True)
        else:
            normalized[out_key] = normalize_text_answer(item)
    return normalized, json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def maybe_to_sympy(
    client: Any,
    model: str,
    max_retries: int,
    temperature: float,
    max_tokens: int,
    boxed_answer: str,
    split: str = "",
    question_id: str = "",
    reasoning_request: Optional[Dict[str, Any]] = None,
) -> str:
    ans = normalize_sympy_expression((boxed_answer or "").strip())
    if not ans or ans == "N/A":
        return "N/A"
    try:
        out = azure_chat_call(
            client=client,
            model=model,
            system=SYSTEM_LATEX_TO_SYMPY,
            user=f"Convert this LaTeX to SymPy:\n{ans}",
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            reasoning_request=reasoning_request,
            telemetry={
                "operation": "generate_convert_sympy",
                "split": split,
                "question_id": question_id,
            },
        )
        out_text = (out or "").strip()
        norm = normalize_sympy_expression(out_text)
        if (not out_text) or (not norm) or (norm == "N/A"):
            return ans
        return norm
    except Exception:  # noqa: BLE001
        return ans or "N/A"


def run_for_split(
    input_file: Path,
    output_file: Path,
    evaluation_file: Path,
    cfg: Dict[str, Any],
    client: Any,
    model_name: str,
    artifact_model_name: Optional[str] = None,
    solve_reasoning_request: Optional[Dict[str, Any]] = None,
    convert_reasoning_request: Optional[Dict[str, Any]] = None,
    solve_reasoning_meta: Optional[Dict[str, Any]] = None,
    convert_reasoning_meta: Optional[Dict[str, Any]] = None,
    resume_override: Optional[bool] = None,
    question_id_filter: Optional[Set[str]] = None,
    failure_dir: Optional[Path] = None,
) -> Tuple[int, List[Dict[str, Any]]]:
    artifact_model_name = str(artifact_model_name or model_name).strip() or model_name
    all_rows = read_jsonl(input_file)
    all_rows = apply_split_annotation_overrides(all_rows, cfg["paths"], input_file.stem)
    rows = filter_rows_by_question_ids(all_rows, question_id_filter)
    if not rows:
        if failure_dir is not None:
            clear_failure_artifacts(failure_dir, input_file.stem)
        return 0, []
    gen_cfg = cfg["generate"]
    azure_retries = int((cfg.get("azure") or {}).get("max_retries", 4) or 4)
    anthropic_retries = int((cfg.get("anthropic") or {}).get("max_retries", azure_retries) or azure_retries)
    provider = str(getattr(client, "_econ_provider", "") or "").strip().lower()
    max_retries = anthropic_retries if provider == "anthropic_foundry" else azure_retries
    resume = bool(gen_cfg.get("resume", True)) if resume_override is None else bool(resume_override)
    max_workers = int(gen_cfg.get("max_workers", 1) or 1)
    partial_overwrite = bool(question_id_filter)
    existing_rows = read_jsonl(output_file) if output_file.exists() else []

    if output_file.exists() and not resume and not partial_overwrite:
        output_file.unlink()
    effective_resume = resume
    done = load_done_ids(output_file) if effective_resume else set()
    if partial_overwrite:
        done = set()

    pending: List[Tuple[int, Dict[str, Any]]] = []
    for idx, row in enumerate(rows, start=1):
        qid = str(row.get("id", "")).strip()
        if qid and qid not in done:
            pending.append((idx, row))

    if pending:
        pending_qids = [str(row.get("id", "")).strip() for _, row in pending if str(row.get("id", "")).strip()]
        invalidated = invalidate_evaluation_artifacts(cfg["paths"], artifact_model_name, input_file.stem, pending_qids)
        row_files = invalidated.get("row_files", [])
        aggregate_files = invalidated.get("aggregate_files", [])
        if row_files or aggregate_files:
            print(
                f"[generate] invalidated evaluation artifacts for split={input_file.stem}, model={model_name}: "
                f"question_scoped_files={len(row_files)}, aggregate_files={len(aggregate_files)}, "
                f"question_ids={len(pending_qids)}"
            )

    def _resolve_allowed_symbols(row: Dict[str, Any]) -> List[str]:
        direct = row.get("symbol_contract_allowed_symbols", [])
        allowed = normalize_allowed_symbols(direct)
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
        ref_sympy = str(row.get("reference_answer_sympy", "")).strip()
        return extract_symbol_tokens(ref_sympy)

    def _solve_one(idx: int, row: Dict[str, Any]) -> Tuple[int, str, Optional[Dict[str, Any]], str]:
        qid = str(row.get("id", "")).strip()
        if not qid:
            return idx, "", None, f"[generate] {input_file.name} {idx}/{len(rows)} -> skip(empty id)"
        question_final = str(row.get("question_final", "")).strip()
        effective_question_final = resolve_effective_question(row) or question_final
        answer_kind = str(row.get("answer_kind", "sympy")).strip().lower()
        if answer_kind not in {"sympy", "bool", "text", "json"}:
            answer_kind = "sympy"
        comparison_mode = str(row.get("comparison_mode", "sympy")).strip().lower()
        if comparison_mode not in {"sympy", "exact", "json"}:
            if answer_kind == "sympy":
                comparison_mode = "sympy"
            elif answer_kind == "json":
                comparison_mode = "json"
            else:
                comparison_mode = "exact"
        allowed_symbols = _resolve_allowed_symbols(row)
        split_name = str(row.get("split", "")).strip() or input_file.stem
        reference_answer_json = row.get("reference_answer_json", {})
        json_keys = sorted(reference_answer_json.keys()) if isinstance(reference_answer_json, dict) else []

        prompt = solve_user_prompt(
            effective_question_final,
            answer_kind,
            allowed_symbols=allowed_symbols,
            json_keys=json_keys,
        )
        response = azure_chat_call(
            client=client,
            model=model_name,
            system=SYSTEM_SOLVER_WITH_BOX,
            user=prompt,
            temperature=float(gen_cfg.get("solve_temperature", 0.0)),
            max_tokens=int(gen_cfg.get("max_solve_tokens", 4096)),
            max_retries=max_retries,
            reasoning_request=solve_reasoning_request,
            telemetry={
                "operation": "generate_solve",
                "split": split_name,
                "question_id": qid,
            },
        )

        boxed_answer = extract_last_boxed(response)
        boxed_answer = re.sub(r"\s+", " ", boxed_answer).strip() if boxed_answer != "N/A" else "N/A"
        answer_sympy = "N/A"
        answer_json: Dict[str, str] | None = None
        final_answer_for_compare = boxed_answer
        if answer_kind == "bool" or comparison_mode == "exact":
            final_answer_for_compare = normalize_bool_answer(boxed_answer) or boxed_answer
        elif comparison_mode == "json":
            answer_json, final_answer_for_compare = canonicalize_json_answer(boxed_answer)
        else:
            answer_sympy = maybe_to_sympy(
                client=client,
                model=model_name,
                max_retries=max_retries,
                temperature=float(gen_cfg.get("convert_temperature", 0.0)),
                max_tokens=int(gen_cfg.get("max_convert_tokens", 512)),
                boxed_answer=boxed_answer,
                split=split_name,
                question_id=qid,
                reasoning_request=convert_reasoning_request,
            )
            answer_sympy = normalize_sympy_expression(answer_sympy)
            final_answer_for_compare = answer_sympy if answer_sympy != "N/A" else boxed_answer

        mismatch_symbols: List[str] = []
        symbol_mismatch = False
        if comparison_mode == "sympy" and allowed_symbols:
            mismatch_symbols = detect_symbol_mismatch(str(final_answer_for_compare), allowed_symbols)
            symbol_mismatch = len(mismatch_symbols) > 0

        effective_reference_answer = resolve_effective_reference_answer(row) or row.get("reference_answer", "N/A")
        effective_reference_answer_sympy = resolve_effective_reference_answer_sympy(row) or row.get("reference_answer_sympy", "N/A")

        out_obj = {
            "id": qid,
            "split": split_name,
            "chapter": row.get("chapter"),
            "question_final": effective_question_final or question_final,
            "question_final_used": effective_question_final,
            "annotator_rewritten_question": row.get("annotator_rewritten_question", ""),
            "annotator_rewritten_solution": row.get("annotator_rewritten_solution", ""),
            "annotator_rewritten_answer": row.get("annotator_rewritten_answer", {}),
            "annotator_override_active": bool(row.get("annotator_override_active", False)),
            "answer_kind": answer_kind,
            "question_type": row.get("question_type", "value"),
            "comparison_mode": comparison_mode,
            "model_response": response,
            "answer_boxed": boxed_answer,
            "answer_sympy": answer_sympy,
            "answer_json": answer_json,
            "final_answer_for_compare": final_answer_for_compare,
            "reference_answer": effective_reference_answer,
            "reference_answer_sympy": effective_reference_answer_sympy,
            "reference_answer_json": row.get("reference_answer_json", {}),
            "reference_answer_json_modes": row.get("reference_answer_json_modes", {}),
            "symbol_contract_allowed_symbols": allowed_symbols,
            "symbol_mismatch": symbol_mismatch,
            "mismatch_symbols": mismatch_symbols,
            "points": row.get("points", 1.0),
            "meta": {
                "model": model_name,
                "solver_model_artifact": artifact_model_name,
                "solve_temperature": gen_cfg.get("solve_temperature", 0.0),
                "max_solve_tokens": gen_cfg.get("max_solve_tokens", 4096),
                "convert_temperature": gen_cfg.get("convert_temperature", 0.0),
                "max_convert_tokens": gen_cfg.get("max_convert_tokens", 512),
                "solve_reasoning": solve_reasoning_meta or {},
                "convert_reasoning": convert_reasoning_meta or {},
            },
        }
        out_obj.update(
            build_result_metadata(
                stage="generate",
                solver_model=artifact_model_name,
                split=split_name,
                question_id=qid,
            )
        )
        return idx, qid, out_obj, f"[generate] {input_file.name} {idx}/{len(rows)} -> {qid}"

    results: List[Tuple[int, str, Dict[str, Any], str]] = []
    failures: List[Dict[str, Any]] = []
    if max_workers <= 1:
        for idx, row in pending:
            try:
                i, qid, out_obj, msg = _solve_one(idx, row)
                print(msg)
                if out_obj is not None and qid:
                    results.append((i, qid, out_obj, msg))
            except Exception as e:  # noqa: BLE001
                qid = str(row.get("id", "")).strip()
                err_msg = f"[generate] {input_file.name} {idx}/{len(rows)} -> failed: {e}"
                print(err_msg)
                failures.append(
                    {
                        "split": input_file.stem,
                        "id": qid,
                        "row_index": idx,
                        "error": str(e),
                        "output_file": str(output_file),
                    }
                )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(_solve_one, idx, row): (idx, row) for idx, row in pending}
            for fut in as_completed(future_map):
                try:
                    i, qid, out_obj, msg = fut.result()
                    print(msg)
                    if out_obj is not None and qid:
                        results.append((i, qid, out_obj, msg))
                except Exception as e:  # noqa: BLE001
                    i, row = future_map[fut]
                    qid = str(row.get("id", "")).strip()
                    err_msg = f"[generate] {input_file.name} {i}/{len(rows)} -> failed: {e}"
                    print(err_msg)
                    failures.append(
                        {
                            "split": input_file.stem,
                            "id": qid,
                            "row_index": i,
                            "error": str(e),
                            "output_file": str(output_file),
                        }
                    )

    if failure_dir is not None:
        if failures:
            detail_path, qid_path = write_failure_artifacts(failure_dir, input_file.stem, failures)
            print(
                f"[generate] warning: {len(failures)} row(s) failed in split={input_file.stem}. "
                f"Failure detail -> {detail_path}; rerun ids -> {qid_path}"
            )
        else:
            clear_failure_artifacts(failure_dir, input_file.stem)

    results.sort(key=lambda x: x[0])
    if partial_overwrite:
        result_rows = [out_obj for _, _, out_obj, _ in results]
        ordered_question_ids = [str(row.get("id", "")).strip() for row in all_rows if str(row.get("id", "")).strip()]
        merged_rows = merge_rows_by_question_id(
            existing_rows,
            result_rows,
            ordered_question_ids,
            keep_unordered_existing=False,
        )
        write_jsonl(output_file, merged_rows)
    else:
        for _, qid, out_obj, _ in results:
            append_jsonl(output_file, out_obj)
            done.add(qid)
    return len(rows), failures


def run(
    config_path: str = "config.yaml",
    solver_model: Optional[str] = None,
    split: Optional[str] = None,
    model: Optional[str] = None,
    hf_json_dir: Optional[str] = None,
    by_model_dir: Optional[str] = None,
    question_ids: Optional[str] = None,
    question_ids_file: Optional[str] = None,
    skip_existing: bool = False,
    force: bool = False,
    reasoning_mode: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_solve_tokens: Optional[int] = None,
) -> None:
    if skip_existing and force:
        raise ValueError("--skip-existing and --force cannot be used together.")
    cfg = apply_dataset_path_overrides(
        load_config(config_path),
        hf_json_dir=hf_json_dir,
        by_model_dir=by_model_dir,
    )
    if max_solve_tokens is not None:
        cfg = dict(cfg)
        gen_cfg = dict(cfg.get("generate", {}))
        gen_cfg["max_solve_tokens"] = int(max_solve_tokens)
        cfg["generate"] = gen_cfg
    paths = cfg["paths"]
    hf_json_dir = Path(paths["hf_json_dir"])
    model_name = resolve_solver_model(cfg, requested_model=solver_model or model)
    explicit_effort = str(reasoning_effort or "").strip()
    if explicit_effort:
        validate_native_reasoning_effort(cfg, model_name, explicit_effort)
    effective_max_solve_tokens = int((cfg.get("generate") or {}).get("max_solve_tokens", 4096) or 4096)
    artifact_model_name = build_solver_artifact_label(
        model_name,
        reasoning_effort=explicit_effort,
        max_solve_tokens=effective_max_solve_tokens,
    )
    question_id_filter = load_question_id_filter(question_ids=question_ids, question_ids_file=question_ids_file)
    legacy_gen_dir = legacy_generations_dir(paths)
    failure_dir = solver_reports_dir(paths, artifact_model_name) / "generate_failures"
    generated_outputs: List[Path] = []
    failure_summaries: List[Tuple[str, int, Path]] = []
    solve_reasoning = resolve_generate_reasoning_request(
        cfg,
        model_name,
        phase="solve",
        requested_mode=reasoning_mode,
        requested_effort=reasoning_effort,
    )
    convert_reasoning = resolve_generate_reasoning_request(cfg, model_name, phase="convert")

    client: Any = None
    input_files = split_input_files(hf_json_dir, split=split)
    if not input_files:
        if split:
            raise RuntimeError(f"Split not found under {hf_json_dir}: {split}")
        raise RuntimeError(f"No split jsonl files found under: {hf_json_dir}")

    solve_reasoning_request = dict(solve_reasoning.get("request_kwargs") or {})
    if solve_reasoning.get("omit_temperature"):
        solve_reasoning_request["omit_temperature"] = True
    convert_reasoning_request = dict(convert_reasoning.get("request_kwargs") or {})
    if convert_reasoning.get("omit_temperature"):
        convert_reasoning_request["omit_temperature"] = True
    gen_cfg = cfg["generate"]
    azure_retries = int((cfg.get("azure") or {}).get("max_retries", 4) or 4)
    anthropic_retries = int((cfg.get("anthropic") or {}).get("max_retries", azure_retries) or azure_retries)
    max_workers = int(gen_cfg.get("max_workers", 1) or 1)
    split_contexts: List[Dict[str, Any]] = []

    def _resolve_allowed_symbols(row: Dict[str, Any]) -> List[str]:
        direct = row.get("symbol_contract_allowed_symbols", [])
        allowed = normalize_allowed_symbols(direct)
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
        ref_sympy = str(row.get("reference_answer_sympy", "")).strip()
        return extract_symbol_tokens(ref_sympy)

    def _prepare_split_context(input_file: Path) -> Dict[str, Any]:
        split_name = input_file.stem
        output_file = solver_generation_file(paths, artifact_model_name, split_name)
        legacy_output_file = resolve_existing_model_jsonl(
            legacy_gen_dir,
            split_name,
            artifact_model_name,
            allow_legacy_model_suffix=True,
            allow_legacy_plain_split=True,
        )
        if not output_file.exists() and legacy_output_file.exists() and legacy_output_file != output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_output_file, output_file)
            print(f"[generate] migrated legacy generation -> {output_file}")
        evaluation_file = resolve_rule_evaluation_input(paths, artifact_model_name, split_name)
        if skip_existing and output_file.exists() and output_file.stat().st_size > 0:
            return {
                "action": "skip_existing",
                "split_name": split_name,
                "output_file": output_file,
            }

        all_rows = read_jsonl(input_file)
        all_rows = apply_split_annotation_overrides(all_rows, cfg["paths"], split_name)
        rows = filter_rows_by_question_ids(all_rows, question_id_filter)
        if not rows:
            if failure_dir is not None:
                clear_failure_artifacts(failure_dir, split_name)
            return {
                "action": "skip_no_rows",
                "split_name": split_name,
                "output_file": output_file,
            }

        provider = str(getattr(client, "_econ_provider", "") or "").strip().lower() if client is not None else ""
        max_retries = anthropic_retries if provider == "anthropic_foundry" else azure_retries
        resume = bool(gen_cfg.get("resume", True)) if force is False else False
        partial_overwrite = bool(question_id_filter)
        existing_rows = read_jsonl(output_file) if output_file.exists() else []

        if output_file.exists() and not resume and not partial_overwrite:
            output_file.unlink()
            existing_rows = []

        done = load_done_ids(output_file) if resume else set()
        if partial_overwrite:
            done = set()

        pending: List[Tuple[int, Dict[str, Any]]] = []
        for idx, row in enumerate(rows, start=1):
            qid = str(row.get("id", "")).strip()
            if qid and qid not in done:
                pending.append((idx, row))

        if pending:
            pending_qids = [str(row.get("id", "")).strip() for _, row in pending if str(row.get("id", "")).strip()]
            invalidated = invalidate_evaluation_artifacts(cfg["paths"], artifact_model_name, split_name, pending_qids)
            row_files = invalidated.get("row_files", [])
            aggregate_files = invalidated.get("aggregate_files", [])
            if row_files or aggregate_files:
                print(
                    f"[generate] invalidated evaluation artifacts for split={split_name}, model={model_name}: "
                    f"question_scoped_files={len(row_files)}, aggregate_files={len(aggregate_files)}, "
                    f"question_ids={len(pending_qids)}"
                )

        return {
            "action": "run",
            "input_file": input_file,
            "split_name": split_name,
            "output_file": output_file,
            "evaluation_file": evaluation_file,
            "all_rows": all_rows,
            "rows": rows,
            "existing_rows": existing_rows,
            "pending": pending,
            "resume": resume,
            "partial_overwrite": partial_overwrite,
            "max_retries": max_retries,
        }

    for input_file in input_files:
        ctx = _prepare_split_context(input_file)
        action = str(ctx.get("action", "run"))
        if action == "skip_existing":
            print(f"[generate] skip {ctx['split_name']}: existing output -> {ctx['output_file']}")
            generated_outputs.append(ctx["output_file"])
            continue
        if action == "skip_no_rows":
            print(f"[generate] skip {ctx['split_name']}: no rows matched question-id filter")
            continue
        split_contexts.append(ctx)

    if split_contexts and client is None:
        client = create_client(cfg, model_name=model_name)
        print(
            "[generate] reasoning solve -> "
            f"profile={solve_reasoning.get('profile_name') or 'none'}, "
            f"interface={solve_reasoning.get('reasoning_interface') or 'none'}, "
            f"mode={solve_reasoning.get('effective_mode') or 'off'}, "
            f"request={solve_reasoning.get('request_kwargs') or {}}"
        )
        print(
            "[generate] reasoning convert -> "
            f"profile={convert_reasoning.get('profile_name') or 'none'}, "
            f"interface={convert_reasoning.get('reasoning_interface') or 'none'}, "
            f"mode={convert_reasoning.get('effective_mode') or 'off'}, "
            f"request={convert_reasoning.get('request_kwargs') or {}}"
        )
        for ctx in split_contexts:
            provider = str(getattr(client, "_econ_provider", "") or "").strip().lower()
            ctx["max_retries"] = anthropic_retries if provider == "anthropic_foundry" else azure_retries

    def _solve_one(ctx: Dict[str, Any], idx: int, row: Dict[str, Any]) -> Tuple[str, int, str, Optional[Dict[str, Any]], str]:
        qid = str(row.get("id", "")).strip()
        input_file = ctx["input_file"]
        rows = ctx["rows"]
        if not qid:
            return ctx["split_name"], idx, "", None, f"[generate] {input_file.name} {idx}/{len(rows)} -> skip(empty id)"
        question_final = str(row.get("question_final", "")).strip()
        effective_question_final = resolve_effective_question(row) or question_final
        answer_kind = str(row.get("answer_kind", "sympy")).strip().lower()
        if answer_kind not in {"sympy", "bool", "text", "json"}:
            answer_kind = "sympy"
        comparison_mode = str(row.get("comparison_mode", "sympy")).strip().lower()
        if comparison_mode not in {"sympy", "exact", "json"}:
            if answer_kind == "sympy":
                comparison_mode = "sympy"
            elif answer_kind == "json":
                comparison_mode = "json"
            else:
                comparison_mode = "exact"
        allowed_symbols = _resolve_allowed_symbols(row)
        split_name = str(row.get("split", "")).strip() or ctx["split_name"]
        reference_answer_json = row.get("reference_answer_json", {})
        json_keys = sorted(reference_answer_json.keys()) if isinstance(reference_answer_json, dict) else []

        prompt = solve_user_prompt(
            effective_question_final,
            answer_kind,
            allowed_symbols=allowed_symbols,
            json_keys=json_keys,
        )
        response = azure_chat_call(
            client=client,
            model=model_name,
            system=SYSTEM_SOLVER_WITH_BOX,
            user=prompt,
            temperature=float(gen_cfg.get("solve_temperature", 0.0)),
            max_tokens=int(gen_cfg.get("max_solve_tokens", 4096)),
            max_retries=int(ctx["max_retries"]),
            reasoning_request=solve_reasoning_request,
            telemetry={
                "operation": "generate_solve",
                "split": split_name,
                "question_id": qid,
            },
        )

        boxed_answer = extract_last_boxed(response)
        boxed_answer = re.sub(r"\s+", " ", boxed_answer).strip() if boxed_answer != "N/A" else "N/A"
        answer_sympy = "N/A"
        answer_json: Dict[str, str] | None = None
        final_answer_for_compare = boxed_answer
        if answer_kind == "bool" or comparison_mode == "exact":
            final_answer_for_compare = normalize_bool_answer(boxed_answer) or boxed_answer
        elif comparison_mode == "json":
            answer_json, final_answer_for_compare = canonicalize_json_answer(boxed_answer)
        else:
            answer_sympy = maybe_to_sympy(
                client=client,
                model=model_name,
                max_retries=int(ctx["max_retries"]),
                temperature=float(gen_cfg.get("convert_temperature", 0.0)),
                max_tokens=int(gen_cfg.get("max_convert_tokens", 512)),
                boxed_answer=boxed_answer,
                split=split_name,
                question_id=qid,
                reasoning_request=convert_reasoning_request,
            )
            answer_sympy = normalize_sympy_expression(answer_sympy)
            final_answer_for_compare = answer_sympy if answer_sympy != "N/A" else boxed_answer

        mismatch_symbols: List[str] = []
        symbol_mismatch = False
        if comparison_mode == "sympy" and allowed_symbols:
            mismatch_symbols = detect_symbol_mismatch(str(final_answer_for_compare), allowed_symbols)
            symbol_mismatch = len(mismatch_symbols) > 0

        effective_reference_answer = resolve_effective_reference_answer(row) or row.get("reference_answer", "N/A")
        effective_reference_answer_sympy = resolve_effective_reference_answer_sympy(row) or row.get("reference_answer_sympy", "N/A")

        out_obj = {
            "id": qid,
            "split": split_name,
            "chapter": row.get("chapter"),
            "question_final": effective_question_final or question_final,
            "question_final_used": effective_question_final,
            "annotator_rewritten_question": row.get("annotator_rewritten_question", ""),
            "annotator_rewritten_solution": row.get("annotator_rewritten_solution", ""),
            "annotator_rewritten_answer": row.get("annotator_rewritten_answer", {}),
            "annotator_override_active": bool(row.get("annotator_override_active", False)),
            "answer_kind": answer_kind,
            "question_type": row.get("question_type", "value"),
            "comparison_mode": comparison_mode,
            "model_response": response,
            "answer_boxed": boxed_answer,
            "answer_sympy": answer_sympy,
            "answer_json": answer_json,
            "final_answer_for_compare": final_answer_for_compare,
            "reference_answer": effective_reference_answer,
            "reference_answer_sympy": effective_reference_answer_sympy,
            "reference_answer_json": row.get("reference_answer_json", {}),
            "reference_answer_json_modes": row.get("reference_answer_json_modes", {}),
            "symbol_contract_allowed_symbols": allowed_symbols,
            "symbol_mismatch": symbol_mismatch,
            "mismatch_symbols": mismatch_symbols,
            "points": row.get("points", 1.0),
            "meta": {
                "model": model_name,
                "solver_model_artifact": artifact_model_name,
                "solve_temperature": gen_cfg.get("solve_temperature", 0.0),
                "max_solve_tokens": gen_cfg.get("max_solve_tokens", 4096),
                "convert_temperature": gen_cfg.get("convert_temperature", 0.0),
                "max_convert_tokens": gen_cfg.get("max_convert_tokens", 512),
                "solve_reasoning": solve_reasoning or {},
                "convert_reasoning": convert_reasoning or {},
            },
        }
        out_obj.update(
            build_result_metadata(
                stage="generate",
                solver_model=artifact_model_name,
                split=split_name,
                question_id=qid,
            )
        )
        return split_name, idx, qid, out_obj, f"[generate] {input_file.name} {idx}/{len(rows)} -> {qid}"

    results_by_split: Dict[str, List[Tuple[int, str, Dict[str, Any], str]]] = {}
    failures_by_split: Dict[str, List[Dict[str, Any]]] = {}
    tasks: List[Tuple[Dict[str, Any], int, Dict[str, Any]]] = [
        (ctx, idx, row)
        for ctx in split_contexts
        for idx, row in ctx["pending"]
    ]

    if max_workers <= 1:
        for ctx, idx, row in tasks:
            try:
                split_name, i, qid, out_obj, msg = _solve_one(ctx, idx, row)
                print(msg)
                if out_obj is not None and qid:
                    results_by_split.setdefault(split_name, []).append((i, qid, out_obj, msg))
            except Exception as e:  # noqa: BLE001
                split_name = ctx["split_name"]
                qid = str(row.get("id", "")).strip()
                err_msg = f"[generate] {ctx['input_file'].name} {idx}/{len(ctx['rows'])} -> failed: {e}"
                print(err_msg)
                failures_by_split.setdefault(split_name, []).append(
                    {
                        "split": split_name,
                        "id": qid,
                        "row_index": idx,
                        "error": str(e),
                        "output_file": str(ctx["output_file"]),
                    }
                )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {
                ex.submit(_solve_one, ctx, idx, row): (ctx, idx, row)
                for ctx, idx, row in tasks
            }
            for fut in as_completed(future_map):
                ctx, idx, row = future_map[fut]
                try:
                    split_name, i, qid, out_obj, msg = fut.result()
                    print(msg)
                    if out_obj is not None and qid:
                        results_by_split.setdefault(split_name, []).append((i, qid, out_obj, msg))
                except Exception as e:  # noqa: BLE001
                    split_name = ctx["split_name"]
                    qid = str(row.get("id", "")).strip()
                    err_msg = f"[generate] {ctx['input_file'].name} {idx}/{len(ctx['rows'])} -> failed: {e}"
                    print(err_msg)
                    failures_by_split.setdefault(split_name, []).append(
                        {
                            "split": split_name,
                            "id": qid,
                            "row_index": idx,
                            "error": str(e),
                            "output_file": str(ctx["output_file"]),
                        }
                    )

    for ctx in split_contexts:
        split_name = ctx["split_name"]
        output_file = ctx["output_file"]
        rows = ctx["rows"]
        all_rows = ctx["all_rows"]
        existing_rows = ctx["existing_rows"]
        partial_overwrite = bool(ctx["partial_overwrite"])
        resume = bool(ctx["resume"])
        results = results_by_split.get(split_name, [])
        failures = failures_by_split.get(split_name, [])

        if failure_dir is not None:
            if failures:
                detail_path, qid_path = write_failure_artifacts(failure_dir, split_name, failures)
                print(
                    f"[generate] warning: {len(failures)} row(s) failed in split={split_name}. "
                    f"Failure detail -> {detail_path}; rerun ids -> {qid_path}"
                )
            else:
                clear_failure_artifacts(failure_dir, split_name)

        results.sort(key=lambda x: x[0])
        result_rows = [out_obj for _, _, out_obj, _ in results]
        if partial_overwrite:
            ordered_question_ids = [str(row.get("id", "")).strip() for row in all_rows if str(row.get("id", "")).strip()]
            merged_rows = merge_rows_by_question_id(
                existing_rows,
                result_rows,
                ordered_question_ids,
                keep_unordered_existing=False,
            )
            write_jsonl(output_file, merged_rows)
        elif result_rows:
            if resume and existing_rows:
                write_jsonl(output_file, existing_rows + result_rows)
            else:
                write_jsonl(output_file, result_rows)

        if failures:
            failure_summaries.append(
                (
                    split_name,
                    len(failures),
                    failure_artifact_paths(failure_dir, split_name)[1],
                )
            )
        if not output_file.exists():
            if failures:
                print(
                    f"[generate] warning: split={split_name} produced no successful generations. "
                    f"Use the rerun id file to retry failed rows: {failure_artifact_paths(failure_dir, split_name)[1]}"
                )
                continue
            raise RuntimeError(
                f"Generate produced no output file for split={split_name}, solver_model={model_name}, "
                f"artifact_label={artifact_model_name}: {output_file}. "
                "This usually means the input dataset rows are not in hf_json-compatible format or contain no usable ids."
            )
        generated_outputs.append(output_file)
        if rows:
            print(f"[generate] wrote -> {output_file}")

    register_run_outputs(generated_outputs)
    if failure_summaries:
        print("[generate] completed with failures:")
        for split_name, count, qid_path in failure_summaries:
            print(f"  - split={split_name}, failed_rows={count}, rerun_ids={qid_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate boxed final answers for econ HF splits.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model used for answering. Overrides config models.default_solver_model.",
    )
    parser.add_argument(
        "--hf-json-dir",
        default=None,
        help="Optional input dataset directory override. Example: data/review_annotations/post_step5_confirmed_dataset",
    )
    parser.add_argument(
        "--by-model-dir",
        default=None,
        help="Optional output root override to avoid collisions across datasets.",
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
        "--skip-existing",
        action="store_true",
        help="Skip split if generation output already exists and is non-empty.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate all rows (disable resume) for each processed split.",
    )
    parser.add_argument(
        "--max-solve-tokens",
        type=int,
        default=None,
        help="Optional override for generate.max_solve_tokens without editing config.yaml.",
    )
    args = parser.parse_args()
    run(
        config_path=args.config,
        solver_model=args.solver_model,
        hf_json_dir=args.hf_json_dir,
        by_model_dir=args.by_model_dir,
        question_ids=args.question_ids,
        question_ids_file=args.question_ids_file,
        skip_existing=args.skip_existing,
        force=args.force,
        max_solve_tokens=args.max_solve_tokens,
    )


if __name__ == "__main__":
    main()
