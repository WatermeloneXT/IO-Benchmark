#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import keyword
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cost_logging import get_run_context, register_run_outputs
from core.annotation_overrides import (
    apply_split_annotation_overrides,
    resolve_effective_reference_answer,
    resolve_effective_reference_answer_sympy,
)
from core.llm_utils import load_config, resolve_solver_model
from core.model_layout import resolve_generation_input, solver_rule_evaluation_file, solver_rule_summary_file
from core.path_overrides import apply_dataset_path_overrides
from core.question_filter import filter_rows_by_question_ids, load_question_id_filter, merge_rows_by_question_id
from core.result_metadata import build_result_metadata
from core.solver_variants import build_solver_artifact_label
from core.symbol_contract import detect_symbol_mismatch, extract_symbol_tokens, normalize_allowed_symbols, parse_symbol_contract
from core.sympy_format import KNOWN_CALLABLES, normalize_sympy_expression


_IDENT_PATTERN = re.compile(r"[A-Za-z_]\w*")
_KEEP_KEYWORDS = {"and", "or", "not", "True", "False"}
_RESERVED_PARSE_NAMES: Set[str] = set(KNOWN_CALLABLES) | {
    "true",
    "false",
    "True",
    "False",
    "pi",
    "E",
    "I",
    "oo",
    "nan",
    "zoo",
}


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


def normalize_exact(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def parse_points(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        try:
            return float(Fraction(str(value).strip()))
        except Exception:
            return 1.0


def resolve_assignment_group(row: Dict[str, Any], fallback_split: str = "") -> str:
    split_name = normalize_exact(str(row.get("split", "")).strip()) or fallback_split
    chapter_name = normalize_exact(str(row.get("chapter", "")).strip())
    if split_name != "chapter_assignment" and chapter_name != "assignment":
        return ""
    for key in ("original_id", "problem_number", "id"):
        raw = normalize_exact(str(row.get(key, "")).strip())
        if not raw:
            continue
        prefix = raw.split("#", 1)[0].split("/", 1)[0].strip()
        if prefix.isdigit():
            return prefix
    return ""


def summarize_assignment_groups(eval_rows: Sequence[Dict[str, Any]], split_name: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for row in eval_rows:
        group = resolve_assignment_group(row, split_name)
        if not group:
            continue
        stats = groups.setdefault(
            group,
            {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
                "total_points": 0.0,
                "earned_points": 0.0,
                "weighted_accuracy": 0.0,
            },
        )
        stats["total"] += 1
        if bool(row.get("is_correct", False)):
            stats["correct"] += 1
        stats["total_points"] += parse_points(row.get("points", 1.0))
        stats["earned_points"] += parse_points(row.get("points_earned", 0.0))

    ordered = sorted(groups.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0]))
    out: Dict[str, Dict[str, Any]] = {}
    for group, stats in ordered:
        stats["accuracy"] = (stats["correct"] / stats["total"]) if stats["total"] else 0.0
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
                "accuracy": 0.0,
                "total_points": 0.0,
                "earned_points": 0.0,
                "weighted_accuracy": 0.0,
            },
        )
        merged["total"] += int(stats.get("total", 0))
        merged["correct"] += int(stats.get("correct", 0))
        merged["total_points"] += parse_points(stats.get("total_points", 0.0))
        merged["earned_points"] += parse_points(stats.get("earned_points", 0.0))

    ordered = sorted(summary.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0]))
    out: Dict[str, Dict[str, Any]] = {}
    for group, merged in ordered:
        merged["accuracy"] = (merged["correct"] / merged["total"]) if merged["total"] else 0.0
        merged["weighted_accuracy"] = (
            merged["earned_points"] / merged["total_points"]
        ) if merged["total_points"] else 0.0
        out[group] = merged
    return out


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
                return None, normalize_exact(text)
    if not isinstance(raw_obj, dict):
        return None, normalize_exact(raw_obj)

    normalized: Dict[str, str] = {}
    for key, item in raw_obj.items():
        out_key = str(key).strip()
        if not out_key:
            continue
        if isinstance(item, (dict, list)):
            normalized[out_key] = json.dumps(item, ensure_ascii=False, sort_keys=True)
        else:
            normalized[out_key] = normalize_exact(item)
    return normalized, json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def looks_like_math_expression(text: str) -> bool:
    s = normalize_exact(text)
    if (not s) or s == "N/A":
        return False
    if normalize_bool_answer(s) is not None:
        return False
    if re.search(r"[+\-*/^=()]", s):
        return True
    if re.search(r"\d", s):
        return True
    if re.fullmatch(r"[A-Za-z_]\w*", s):
        return True
    return False


def _import_sympy() -> Any:
    import sympy as sp  # noqa: PLC0415

    return sp


def build_sympy_locals(expr: str, sp: Any) -> Dict[str, Any]:
    # Define unknown identifiers as symbols while keeping known parser names.
    tokens = set(_IDENT_PATTERN.findall(expr or ""))
    local_dict: Dict[str, Any] = {}
    for tok in tokens:
        if tok in _RESERVED_PARSE_NAMES:
            continue
        local_dict[tok] = sp.Symbol(tok)
    return local_dict


def _replace_ident_tokens(text: str, mapping: Dict[str, str]) -> str:
    out = text
    for src in sorted(mapping.keys(), key=lambda x: (-len(x), x)):
        dst = mapping[src]
        out = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(src)}(?![A-Za-z0-9_])", dst, out)
    return out


def _make_safe_alias(token: str, taken: Set[str]) -> str:
    base = f"sym_{re.sub(r'[^A-Za-z0-9_]', '_', token)}".strip("_")
    base = re.sub(r"_+", "_", base)
    if not base:
        base = "sym_token"
    if not re.fullmatch(r"[A-Za-z_]\w*", base):
        base = f"sym_{base}"
        base = re.sub(r"[^A-Za-z0-9_]", "_", base)
        base = re.sub(r"_+", "_", base)
    if base not in taken:
        return base
    idx = 2
    while True:
        cand = f"{base}_{idx}"
        if cand not in taken:
            return cand
        idx += 1


def _needs_symbol_remap(tok: str, allowed_set: Set[str], sp: Any) -> bool:
    if keyword.iskeyword(tok) and tok not in _KEEP_KEYWORDS:
        return True
    if allowed_set and tok not in allowed_set:
        return False
    if tok in allowed_set and (tok in _RESERVED_PARSE_NAMES or tok in KNOWN_CALLABLES or bool(hasattr(sp, tok))):
        return True
    if tok in _RESERVED_PARSE_NAMES:
        return False
    if tok in KNOWN_CALLABLES:
        return False
    return bool(hasattr(sp, tok))


def _build_symbol_remap(pred: str, gold: str, allowed_symbols: Sequence[str], sp: Any) -> Dict[str, str]:
    allowed_set = set(normalize_allowed_symbols(list(allowed_symbols or [])))
    tokens = set(_IDENT_PATTERN.findall(pred or "")) | set(_IDENT_PATTERN.findall(gold or ""))
    taken = set(tokens) | set(allowed_set)
    remap: Dict[str, str] = {}
    for tok in sorted(tokens):
        if not _needs_symbol_remap(tok, allowed_set, sp):
            continue
        alias = _make_safe_alias(tok, taken)
        remap[tok] = alias
        taken.add(alias)
    return remap


def parse_sympy_expr(expr: str, sp: Any, symbol_remap: Optional[Dict[str, str]] = None) -> Any:
    normalized = normalize_sympy_expression(normalize_exact(expr))
    if symbol_remap:
        normalized = _replace_ident_tokens(normalized, symbol_remap)
    local_dict = build_sympy_locals(normalized, sp)
    return sp.sympify(normalized, locals=local_dict, evaluate=True)


def is_zero_expr(expr: Any, sp: Any) -> bool:
    try:
        if bool(expr == 0):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        if getattr(expr, "is_zero", None) is True:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def equivalent_by_pipeline(a: Any, b: Any, sp: Any) -> bool:
    # Build symbolic difference once, then run type-aware simplification paths.
    try:
        diff = a - b
    except Exception:  # noqa: BLE001
        diff = None

    if diff is not None:
        # 1) Rational path: common denominator + reduction.
        try:
            d1 = sp.cancel(sp.ratsimp(diff))
            if is_zero_expr(d1, sp):
                return True
        except Exception:  # noqa: BLE001
            pass

        try:
            d2 = sp.cancel(sp.together(diff))
            num, _ = sp.fraction(d2)
            if is_zero_expr(sp.expand(num), sp):
                return True
        except Exception:  # noqa: BLE001
            pass

        # 2) Polynomial path: expand / factor.
        try:
            if is_zero_expr(sp.expand(diff), sp):
                return True
        except Exception:  # noqa: BLE001
            pass

        try:
            if is_zero_expr(sp.factor(diff), sp):
                return True
        except Exception:  # noqa: BLE001
            pass

        # 3) Trigonometric path.
        try:
            if is_zero_expr(sp.trigsimp(diff, method="combined"), sp):
                return True
        except Exception:  # noqa: BLE001
            pass

        # 4) Fallback path (existing behavior).
        try:
            if is_zero_expr(sp.simplify(diff), sp):
                return True
        except Exception:  # noqa: BLE001
            pass

    try:
        same = a.equals(b)
        return bool(same) if same is not None else False
    except Exception:  # noqa: BLE001
        return False


def expr_equivalent(a: Any, b: Any, sp: Any) -> bool:
    return equivalent_by_pipeline(a, b, sp)


def relational_equivalent(a: Any, b: Any, sp: Any) -> bool:
    if not getattr(a, "is_Relational", False) or not getattr(b, "is_Relational", False):
        return False

    op_a = str(getattr(a, "rel_op", ""))
    op_b = str(getattr(b, "rel_op", ""))

    if op_a == "==" and op_b == "==":
        left = sp.simplify(a.lhs - a.rhs)
        right = sp.simplify(b.lhs - b.rhs)
        return bool(sp.simplify(left - right) == 0 or sp.simplify(left + right) == 0)

    if op_a == op_b and expr_equivalent(a.lhs, b.lhs, sp) and expr_equivalent(a.rhs, b.rhs, sp):
        return True

    reverse_op = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}
    if reverse_op.get(op_a) == op_b and expr_equivalent(a.lhs, b.rhs, sp) and expr_equivalent(a.rhs, b.lhs, sp):
        return True

    return False


def _as_relation_list(expr: Any, sp: Any) -> Optional[List[Any]]:
    items: Optional[List[Any]] = None
    try:
        if isinstance(expr, tuple):
            items = list(expr)
        elif isinstance(expr, list):
            items = list(expr)
        elif isinstance(expr, sp.Tuple):
            items = list(expr)
        elif getattr(expr, "func", None) == sp.And:
            items = list(getattr(expr, "args", ()))
    except Exception:  # noqa: BLE001
        items = None
    if items is None:
        return None
    if not items:
        return []
    if not all(getattr(x, "is_Relational", False) for x in items):
        return None
    return items


def _relation_collections_equivalent(a_items: List[Any], b_items: List[Any], sp: Any) -> bool:
    if len(a_items) != len(b_items):
        return False
    used = [False] * len(b_items)
    for a_rel in a_items:
        matched = False
        for idx, b_rel in enumerate(b_items):
            if used[idx]:
                continue
            if relational_equivalent(a_rel, b_rel, sp):
                used[idx] = True
                matched = True
                break
        if not matched:
            return False
    return True


def _compare_sympy_exprs(pred_expr: Any, gold_expr: Any, sp: Any) -> bool:
    pred_rel_list = _as_relation_list(pred_expr, sp)
    gold_rel_list = _as_relation_list(gold_expr, sp)
    if pred_rel_list is not None or gold_rel_list is not None:
        if pred_rel_list is None or gold_rel_list is None:
            return False
        return _relation_collections_equivalent(pred_rel_list, gold_rel_list, sp)

    if getattr(pred_expr, "is_Relational", False) or getattr(gold_expr, "is_Relational", False):
        if getattr(pred_expr, "is_Relational", False) and str(getattr(pred_expr, "rel_op", "")) == "==" and not getattr(gold_expr, "is_Relational", False):
            return expr_equivalent(pred_expr.lhs, gold_expr, sp) or expr_equivalent(pred_expr.rhs, gold_expr, sp)
        if getattr(gold_expr, "is_Relational", False) and str(getattr(gold_expr, "rel_op", "")) == "==" and not getattr(pred_expr, "is_Relational", False):
            return expr_equivalent(pred_expr, gold_expr.lhs, sp) or expr_equivalent(pred_expr, gold_expr.rhs, sp)
        return relational_equivalent(pred_expr, gold_expr, sp)
    return expr_equivalent(pred_expr, gold_expr, sp)


def _attach_symbol_mismatch_detail(
    detail: str,
    mismatch_symbols: Sequence[str],
    reference_mismatch_symbols: Sequence[str],
) -> str:
    if not (mismatch_symbols or reference_mismatch_symbols):
        return detail
    if detail.endswith("_with_symbol_mismatch"):
        return detail
    return f"{detail}_with_symbol_mismatch"


def sympy_match(pred: str, gold: str, allowed_symbols: Optional[Sequence[str]] = None) -> Tuple[bool, str]:
    try:
        sp = _import_sympy()
    except ModuleNotFoundError:
        return False, "sympy_not_installed"

    remap_used = False
    try:
        pred_expr = parse_sympy_expr(pred, sp)
        gold_expr = parse_sympy_expr(gold, sp)
    except Exception as err:  # noqa: BLE001
        remap = _build_symbol_remap(pred, gold, allowed_symbols or [], sp)
        if not remap:
            return False, f"sympy_parse_failed: {err}"
        try:
            pred_expr = parse_sympy_expr(pred, sp, symbol_remap=remap)
            gold_expr = parse_sympy_expr(gold, sp, symbol_remap=remap)
            remap_used = True
        except Exception as remap_err:  # noqa: BLE001
            return False, f"sympy_parse_failed: {err}; remap_parse_failed: {remap_err}"

    try:
        ok = _compare_sympy_exprs(pred_expr, gold_expr, sp)
        if (not ok) and (not remap_used):
            remap = _build_symbol_remap(pred, gold, allowed_symbols or [], sp)
            if remap:
                try:
                    pred_expr2 = parse_sympy_expr(pred, sp, symbol_remap=remap)
                    gold_expr2 = parse_sympy_expr(gold, sp, symbol_remap=remap)
                    ok = _compare_sympy_exprs(pred_expr2, gold_expr2, sp)
                    remap_used = True
                except Exception:  # noqa: BLE001
                    pass
        if remap_used:
            return ok, "sympy_equal_after_symbol_remap" if ok else "sympy_not_equal_after_symbol_remap"
        return ok, "sympy_equal" if ok else "sympy_not_equal"
    except Exception as err:  # noqa: BLE001
        return False, f"sympy_compare_failed: {err}"


def compare_one(
    dataset_row: Dict[str, Any],
    pred_row: Dict[str, Any],
) -> Dict[str, Any]:
    answer_kind = str(dataset_row.get("answer_kind", "sympy")).strip().lower()
    comparison_mode = str(dataset_row.get("comparison_mode", "")).strip().lower()
    if comparison_mode not in {"exact", "sympy", "json"}:
        if answer_kind == "bool":
            comparison_mode = "exact"
        elif answer_kind == "json":
            comparison_mode = "json"
        elif answer_kind == "text":
            comparison_mode = "exact"
        else:
            comparison_mode = "sympy"

    question_type = str(dataset_row.get("question_type", "value")).strip().lower()
    symbol_contract_allowed_symbols = normalize_allowed_symbols(dataset_row.get("symbol_contract_allowed_symbols", []))
    if not symbol_contract_allowed_symbols:
        raw_contract = {
            "allowed_symbols": dataset_row.get("symbol_contract_allowed_symbols", []),
            "symbol_definitions": dataset_row.get("symbol_contract_definitions", {}),
        }
        if not raw_contract["allowed_symbols"]:
            raw_contract = dataset_row.get("symbol_contract", {})
        allowed2, _ = parse_symbol_contract(raw_contract)
        symbol_contract_allowed_symbols = allowed2
    symbol_mismatch = False
    mismatch_symbols: List[str] = []
    reference_symbol_issue = False
    reference_mismatch_symbols: List[str] = []
    json_items: List[Dict[str, Any]] = []
    json_missing_keys: List[str] = []
    json_extra_keys: List[str] = []
    points = parse_points(dataset_row.get("points", 1.0))
    assignment_group = resolve_assignment_group(dataset_row)

    if comparison_mode == "json":
        raw_gold = dataset_row.get("reference_answer_json", {})
        if not isinstance(raw_gold, dict) or not raw_gold:
            raw_gold = resolve_effective_reference_answer(dataset_row)
        gold_json, gold = canonicalize_json_answer(raw_gold)

        raw_pred = pred_row.get("answer_json", {})
        if not isinstance(raw_pred, dict) or not raw_pred:
            raw_pred = pred_row.get("final_answer_for_compare") or pred_row.get("answer_boxed") or ""
        pred_json, pred = canonicalize_json_answer(raw_pred)

        if gold_json is None:
            ok = False
            detail = "json_reference_invalid"
        elif pred_json is None:
            ok = False
            detail = "json_prediction_invalid"
        else:
            json_extra_keys = sorted(set(pred_json.keys()) - set(gold_json.keys()))
            item_results: List[Dict[str, Any]] = []
            json_modes = dataset_row.get("reference_answer_json_modes", {})
            if not isinstance(json_modes, dict):
                json_modes = {}

            for key in sorted(gold_json.keys()):
                gold_value = gold_json[key]
                pred_value = pred_json.get(key)
                if pred_value is None:
                    json_missing_keys.append(key)
                    item_results.append(
                        {
                            "key": key,
                            "reference": gold_value,
                            "prediction": "N/A",
                            "is_correct": False,
                            "detail": "missing_key",
                        }
                    )
                    continue

                field_mode = str(json_modes.get(key, "")).strip().lower()
                if field_mode not in {"sympy", "text"}:
                    field_mode = "sympy" if looks_like_math_expression(gold_value) else "text"

                if normalize_bool_answer(gold_value) is not None:
                    gold_item = normalize_bool_answer(gold_value)
                    pred_item = normalize_bool_answer(pred_value)
                    field_ok = bool(pred_item == gold_item)
                    field_detail = "exact_equal" if field_ok else "exact_not_equal"
                elif field_mode == "sympy":
                    field_allowed_symbols = symbol_contract_allowed_symbols or extract_symbol_tokens(gold_value)
                    field_mismatch = (
                        detect_symbol_mismatch(pred_value, field_allowed_symbols) if field_allowed_symbols else []
                    )
                    field_ref_mismatch = (
                        detect_symbol_mismatch(gold_value, field_allowed_symbols) if field_allowed_symbols else []
                    )
                    mismatch_symbols.extend(x for x in field_mismatch if x not in mismatch_symbols)
                    reference_mismatch_symbols.extend(
                        x for x in field_ref_mismatch if x not in reference_mismatch_symbols
                    )
                    field_ok, field_detail = sympy_match(
                        pred_value,
                        gold_value,
                        allowed_symbols=field_allowed_symbols,
                    )
                    field_detail = _attach_symbol_mismatch_detail(
                        field_detail,
                        field_mismatch,
                        field_ref_mismatch,
                    )
                else:
                    gold_item = normalize_exact(gold_value)
                    pred_item = normalize_exact(pred_value)
                    field_ok = bool(pred_item == gold_item)
                    field_detail = "exact_equal" if field_ok else "exact_not_equal"

                item_results.append(
                    {
                        "key": key,
                        "reference": gold_value,
                        "prediction": pred_value,
                        "is_correct": field_ok,
                        "detail": field_detail,
                    }
                )

            json_items = item_results
            symbol_mismatch = len(mismatch_symbols) > 0
            reference_symbol_issue = len(reference_mismatch_symbols) > 0
            ok = (
                len(json_missing_keys) == 0
                and len(json_extra_keys) == 0
                and all(bool(item.get("is_correct", False)) for item in json_items)
            )
            if ok:
                detail = "json_equal"
            elif json_missing_keys:
                detail = "json_missing_keys"
            elif json_extra_keys:
                detail = "json_extra_keys"
            else:
                detail = "json_not_equal"
            detail = _attach_symbol_mismatch_detail(
                detail,
                mismatch_symbols,
                reference_mismatch_symbols,
            )
    elif comparison_mode == "exact":
        effective_reference_answer = resolve_effective_reference_answer(dataset_row)
        gold = normalize_bool_answer(effective_reference_answer)
        if gold is None:
            gold = normalize_exact(effective_reference_answer)

        pred_candidate = str(pred_row.get("final_answer_for_compare") or pred_row.get("answer_boxed") or "")
        pred_bool = normalize_bool_answer(pred_candidate)
        pred = pred_bool if pred_bool is not None else normalize_exact(pred_candidate)

        ok = bool(pred == gold)
        detail = "exact_equal" if ok else "exact_not_equal"
    else:
        gold = normalize_exact(resolve_effective_reference_answer_sympy(dataset_row) or resolve_effective_reference_answer(dataset_row) or "N/A")
        pred = normalize_exact(
            str(pred_row.get("answer_sympy") or pred_row.get("final_answer_for_compare") or pred_row.get("answer_boxed") or "N/A")
        )
        symbol_contract_allowed_symbols = normalize_allowed_symbols(dataset_row.get("symbol_contract_allowed_symbols", []))
        if not symbol_contract_allowed_symbols:
            raw_contract = {
                "allowed_symbols": dataset_row.get("symbol_contract_allowed_symbols", []),
                "symbol_definitions": dataset_row.get("symbol_contract_definitions", {}),
            }
            if not raw_contract["allowed_symbols"]:
                raw_contract = dataset_row.get("symbol_contract", {})
            allowed2, _ = parse_symbol_contract(raw_contract)
            symbol_contract_allowed_symbols = allowed2
        if not symbol_contract_allowed_symbols:
            symbol_contract_allowed_symbols = extract_symbol_tokens(gold)

        if symbol_contract_allowed_symbols:
            mismatch_symbols = detect_symbol_mismatch(pred, symbol_contract_allowed_symbols)
            reference_mismatch_symbols = detect_symbol_mismatch(gold, symbol_contract_allowed_symbols)
            symbol_mismatch = len(mismatch_symbols) > 0
            reference_symbol_issue = len(reference_mismatch_symbols) > 0
        ok, detail = sympy_match(pred, gold, allowed_symbols=symbol_contract_allowed_symbols)
        detail = _attach_symbol_mismatch_detail(
            detail,
            mismatch_symbols,
            reference_mismatch_symbols,
        )

    points_earned = points if ok else 0.0

    return {
        "id": str(dataset_row.get("id", "")),
        "split": str(dataset_row.get("split", "")),
        "assignment_group": assignment_group,
        "question_type": question_type,
        "answer_kind": answer_kind,
        "comparison_mode": comparison_mode,
        "prediction": pred,
        "reference": gold,
        "is_correct": ok,
        "detail": detail,
        "symbol_contract_allowed_symbols": symbol_contract_allowed_symbols,
        "symbol_mismatch": symbol_mismatch,
        "mismatch_symbols": mismatch_symbols,
        "reference_symbol_issue": reference_symbol_issue,
        "reference_mismatch_symbols": reference_mismatch_symbols,
        "json_items": json_items,
        "json_missing_keys": json_missing_keys,
        "json_extra_keys": json_extra_keys,
        "points": points,
        "points_earned": points_earned,
    }


def run(
    config_path: str = "config.yaml",
    split: Optional[str] = None,
    include_missing: bool = True,
    force: bool = False,
    solver_model: Optional[str] = None,
    model: Optional[str] = None,
    hf_json_dir: Optional[str] = None,
    by_model_dir: Optional[str] = None,
    question_ids: Optional[str] = None,
    question_ids_file: Optional[str] = None,
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

    summary: Dict[str, Any] = {
        "model": model_name,
        "solver_model": base_model_name,
        "solver_model_artifact": model_name,
        "solver_reasoning_effort": str(solver_reasoning_effort or "").strip().lower(),
        "solver_max_solve_tokens": int(solver_max_solve_tokens or int((cfg.get("generate") or {}).get("max_solve_tokens", 4096) or 4096)),
        "force": bool(force),
        "total": 0,
        "correct": 0,
        "accuracy": 0.0,
        "total_points": 0.0,
        "earned_points": 0.0,
        "weighted_accuracy": 0.0,
        "splits": {},
    }
    overall_assignment_groups: Dict[str, Dict[str, Any]] = {}

    def record_split_summary(
        *,
        split_name: str,
        eval_rows: List[Dict[str, Any]],
        dataset_row_count: int,
        generation_row_count: Optional[int],
        evaluation_file: Path,
        skipped_existing: bool,
    ) -> None:
        total = len(eval_rows)
        correct = sum(1 for x in eval_rows if bool(x.get("is_correct", False)))
        accuracy = (correct / total) if total else 0.0
        total_points = sum(parse_points(x.get("points", 1.0)) for x in eval_rows)
        earned_points = sum(parse_points(x.get("points_earned", 0.0)) for x in eval_rows)
        weighted_accuracy = (earned_points / total_points) if total_points else 0.0
        assignment_groups = summarize_assignment_groups(eval_rows, split_name)
        summary["splits"][split_name] = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "total_points": total_points,
            "earned_points": earned_points,
            "weighted_accuracy": weighted_accuracy,
            "dataset_rows": dataset_row_count,
            "generation_rows": generation_row_count,
            "evaluation_file": evaluation_file.as_posix(),
            "skipped_existing": skipped_existing,
        }
        if assignment_groups:
            summary["splits"][split_name]["assignment_groups"] = assignment_groups
            nonlocal_overall = merge_assignment_group_summaries(overall_assignment_groups, assignment_groups)
            overall_assignment_groups.clear()
            overall_assignment_groups.update(nonlocal_overall)
        summary["total"] += total
        summary["correct"] += correct
        summary["total_points"] += total_points
        summary["earned_points"] += earned_points
        status = "skip existing" if skipped_existing else f"correct={correct}/{total} accuracy={accuracy:.4f}"
        print(f"[evaluate] {split_name}: {status} -> {evaluation_file}" if skipped_existing else f"[evaluate] {split_name}: {status}")
        for group, stats in assignment_groups.items():
            print(
                f"[evaluate] {split_name} assignment_{group}: "
                f"earned_points={stats['earned_points']:.4f}/{stats['total_points']:.4f} "
                f"weighted_accuracy={stats['weighted_accuracy']:.4f}"
            )

    for ds_file in dataset_files:
        split_name = ds_file.stem
        out_file = solver_rule_evaluation_file(paths, model_name, split_name)
        all_dataset_rows = apply_split_annotation_overrides(read_jsonl(ds_file), paths, split_name)
        dataset_rows = filter_rows_by_question_ids(all_dataset_rows, question_id_filter)
        ordered_question_ids = [str(row.get("id", "")).strip() for row in all_dataset_rows if str(row.get("id", "")).strip()]
        current_split_question_ids = set(ordered_question_ids)
        existing_eval_rows = read_jsonl(out_file) if out_file.exists() else []
        existing_eval_by_id = {
            str(row.get("id", "")).strip(): row
            for row in existing_eval_rows
            if str(row.get("id", "")).strip()
        }
        if not dataset_rows:
            if partial_overwrite and existing_eval_rows:
                record_split_summary(
                    split_name=split_name,
                    eval_rows=existing_eval_rows,
                    dataset_row_count=len(all_dataset_rows),
                    generation_row_count=None,
                    evaluation_file=out_file,
                    skipped_existing=True,
                )
            continue
        rows_to_evaluate = dataset_rows
        if out_file.exists() and (not force) and (not partial_overwrite):
            missing_dataset_rows = [
                row
                for row in dataset_rows
                if str(row.get("id", "")).strip() not in existing_eval_by_id
            ]
            stale_existing_ids = sorted(
                qid
                for qid in existing_eval_by_id
                if qid and qid not in current_split_question_ids
            )
            if not missing_dataset_rows and not stale_existing_ids:
                record_split_summary(
                    split_name=split_name,
                    eval_rows=existing_eval_rows,
                    dataset_row_count=len(dataset_rows),
                    generation_row_count=None,
                    evaluation_file=out_file,
                    skipped_existing=True,
                )
                continue
            rows_to_evaluate = missing_dataset_rows
            if stale_existing_ids:
                print(
                    f"[evaluate] {split_name}: dropping {len(stale_existing_ids)} stale existing rows "
                    "not present in current dataset"
                )
            if rows_to_evaluate:
                print(
                    f"[evaluate] {split_name}: evaluating {len(rows_to_evaluate)} missing rows "
                    f"(existing={len(existing_eval_rows)})"
                )
            else:
                final_eval_rows = merge_rows_by_question_id(
                    existing_eval_rows,
                    [],
                    ordered_question_ids,
                    keep_unordered_existing=False,
                )
                write_jsonl(out_file, final_eval_rows)
                written_outputs.append(out_file)
                record_split_summary(
                    split_name=split_name,
                    eval_rows=final_eval_rows,
                    dataset_row_count=len(all_dataset_rows),
                    generation_row_count=None,
                    evaluation_file=out_file,
                    skipped_existing=False,
                )
                continue
        gen_file = resolve_generation_input(paths, model_name, split_name)
        if not gen_file.exists():
            raise RuntimeError(
                f"Missing generation file for split={split_name}, solver_model={model_name}: {gen_file}. "
                "Run generate first with the same --solver-model."
            )

        all_gen_rows = read_jsonl(gen_file)
        gen_rows = filter_rows_by_question_ids(
            all_gen_rows,
            question_id_filter if partial_overwrite else {str(row.get("id", "")).strip() for row in rows_to_evaluate},
        )

        ds_by_id = {str(r.get("id", "")).strip(): r for r in rows_to_evaluate if str(r.get("id", "")).strip()}
        gen_by_id = {str(r.get("id", "")).strip(): r for r in gen_rows if str(r.get("id", "")).strip()}

        eval_rows: List[Dict[str, Any]] = []

        for qid, ds_row in ds_by_id.items():
            pred_row = gen_by_id.get(qid)
            if pred_row is None:
                if include_missing:
                    eval_rows.append(
                        {
                            "id": qid,
                            "split": split_name,
                            "assignment_group": resolve_assignment_group(ds_row, split_name),
                            "question_type": ds_row.get("question_type", "value"),
                            "answer_kind": ds_row.get("answer_kind", "sympy"),
                            "comparison_mode": ds_row.get("comparison_mode", "sympy"),
                            "prediction": "N/A",
                            "reference": resolve_effective_reference_answer(ds_row) or "N/A",
                            "is_correct": False,
                            "detail": "missing_prediction",
                            "symbol_contract_allowed_symbols": normalize_allowed_symbols(
                                ds_row.get("symbol_contract_allowed_symbols", [])
                            ),
                            "symbol_mismatch": False,
                            "mismatch_symbols": [],
                            "reference_symbol_issue": False,
                            "reference_mismatch_symbols": [],
                            "json_items": [],
                            "json_missing_keys": [],
                            "json_extra_keys": [],
                            "points": parse_points(ds_row.get("points", 1.0)),
                            "points_earned": 0.0,
                        }
                    )
                continue
            eval_rows.append(compare_one(ds_row, pred_row))

        for row in eval_rows:
            qid = str(row.get("id", "")).strip()
            row.update(
                build_result_metadata(
                    stage="evaluate",
                    solver_model=model_name,
                    split=split_name,
                    question_id=qid,
                )
            )

        final_eval_rows = eval_rows
        if partial_overwrite or (out_file.exists() and (not force)):
            merge_base_rows = existing_eval_rows
            if partial_overwrite:
                target_question_ids = {
                    str(row.get("id", "")).strip()
                    for row in rows_to_evaluate
                    if str(row.get("id", "")).strip()
                }
                merge_base_rows = [
                    row
                    for row in existing_eval_rows
                    if str(row.get("id", "")).strip() not in target_question_ids
                ]
            final_eval_rows = merge_rows_by_question_id(
                merge_base_rows,
                eval_rows,
                ordered_question_ids,
                keep_unordered_existing=False,
            )
        write_jsonl(out_file, final_eval_rows)
        written_outputs.append(out_file)

        record_split_summary(
            split_name=split_name,
            eval_rows=final_eval_rows,
            dataset_row_count=(len(all_dataset_rows) if (partial_overwrite or existing_eval_rows) else len(dataset_rows)),
            generation_row_count=(len(all_gen_rows) if (partial_overwrite or existing_eval_rows) else len(gen_rows)),
            evaluation_file=out_file,
            skipped_existing=False,
        )

    summary["accuracy"] = (summary["correct"] / summary["total"]) if summary["total"] else 0.0
    summary["weighted_accuracy"] = (
        summary["earned_points"] / summary["total_points"]
    ) if summary["total_points"] else 0.0
    if overall_assignment_groups:
        summary["assignment_groups"] = overall_assignment_groups
    summary["workflow_id"] = str(run_ctx.get("workflow_id", "") or "")
    summary["run_id"] = str(run_ctx.get("run_id", "") or "")
    summary["stage"] = "evaluate"
    summary_file = solver_rule_summary_file(paths, model_name)
    write_json(summary_file, summary)
    written_outputs.append(summary_file)
    register_run_outputs(written_outputs)
    print(f"[evaluate] overall: correct={summary['correct']}/{summary['total']} accuracy={summary['accuracy']:.4f}")
    print(f"[evaluate] summary -> {summary_file}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated answers using exact match (bool) or SymPy equivalence (value)."
    )
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--split", default=None, help="Optional split name, e.g. chapter_6")
    parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model used for generation/evaluation files. Overrides config models.default_solver_model.",
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
        "--no-include-missing",
        action="store_false",
        dest="include_missing",
        help="Do not count missing predictions as incorrect.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if split evaluation files already exist.",
    )
    parser.set_defaults(include_missing=True)
    args = parser.parse_args()
    run(
        config_path=args.config,
        split=args.split,
        include_missing=args.include_missing,
        force=args.force,
        solver_model=args.solver_model,
        hf_json_dir=args.hf_json_dir,
        by_model_dir=args.by_model_dir,
        question_ids=args.question_ids,
        question_ids_file=args.question_ids_file,
    )


if __name__ == "__main__":
    main()
