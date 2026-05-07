from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.llm_utils import azure_chat_call, azure_json_call
from core.prompts import (
    SYSTEM_CONVERT_TO_EVAL_QUESTION,
    SYSTEM_LATEX_TO_SYMPY,
    SYSTEM_REFERENCE_CORRECTION,
    SYSTEM_SPLIT_MULTI_QUESTION,
    convert_eval_user_prompt,
    reference_correction_user_prompt,
    split_questions_user_prompt,
)

from .common import (
    as_bool,
    compact_text,
    looks_like_sympy,
    normalize_bool_answer,
    normalize_confidence,
    parse_int_like,
    preserve_stem_context,
    read_json,
)
from core.sympy_format import normalize_sympy_expression
from core.symbol_contract import (
    detect_symbol_mismatch,
    ensure_symbol_contract_section,
    extract_symbol_tokens,
    parse_symbol_contract,
)


def maybe_to_sympy(
    client: Any,
    model: str,
    max_retries: int,
    temperature: float,
    text: str,
    split: str = "",
    question_id: str = "",
) -> str:
    raw = normalize_sympy_expression((text or "").strip())
    if not raw or raw == "N/A":
        return "N/A"
    if looks_like_sympy(raw):
        return compact_text(raw)
    try:
        converted = azure_chat_call(
            client=client,
            model=model,
            system=SYSTEM_LATEX_TO_SYMPY,
            user=f"Convert this expression to SymPy:\n{raw}",
            temperature=temperature,
            max_tokens=512,
            max_retries=max_retries,
            telemetry={
                "operation": "build_convert_to_sympy",
                "split": split,
                "question_id": question_id,
            },
        )
        converted = normalize_sympy_expression(compact_text(converted))
        return converted if converted else "N/A"
    except Exception:  # noqa: BLE001
        fallback = normalize_sympy_expression(raw)
        return fallback if fallback else "N/A"


def load_source_problem(
    chapter_dir: Path,
    problem_obj: Dict[str, Any],
    problem_number: str,
) -> Tuple[str, str, str]:
    by_problem = chapter_dir / "by_problem"
    if not by_problem.is_dir():
        return "", "", ""

    candidate_names: List[str] = []
    src = str(problem_obj.get("source_file", "")).strip()
    if src:
        candidate_names.append(src)
    if problem_number:
        candidate_names.append(f"{problem_number}.json")

    seen = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        p = by_problem / name
        if not p.exists():
            continue
        raw = read_json(p)
        obj = raw.get("problem", raw) if isinstance(raw, dict) else {}
        problem_latex = str(obj.get("problem_latex", "")).strip()
        answer_latex = str(obj.get("answer_latex", "")).strip()
        if problem_latex or answer_latex:
            return problem_latex, answer_latex, p.as_posix()
    return "", "", ""


def llm_correct_reference(
    client: Any,
    model: str,
    max_retries: int,
    temperature: float,
    question_latex: str,
    candidate_answer: str,
    split: str = "",
    question_id: str = "",
) -> Dict[str, Any]:
    fallback_answer = compact_text(candidate_answer)
    obj: Dict[str, Any] = {}
    try:
        obj = azure_json_call(
            client=client,
            model=model,
            system=SYSTEM_REFERENCE_CORRECTION,
            user=reference_correction_user_prompt(question_latex=question_latex, reference_answer=fallback_answer),
            temperature=temperature,
            max_retries=max_retries,
            telemetry={
                "operation": "build_step1_reference_correction",
                "split": split,
                "question_id": question_id,
            },
        )
    except Exception:  # noqa: BLE001
        obj = {}

    final_reference_answer = compact_text(str(obj.get("final_reference_answer", fallback_answer)), default=fallback_answer)
    reference_is_correct = as_bool(obj.get("reference_is_correct"), default=(final_reference_answer == fallback_answer))
    reference_generated_by_llm = as_bool(
        obj.get("reference_generated_by_llm"),
        default=(not reference_is_correct or final_reference_answer != fallback_answer),
    )
    confidence = normalize_confidence(str(obj.get("confidence", "low")))
    analysis = compact_text(str(obj.get("analysis", "")), default="")

    return {
        "raw_json": obj,
        "reference_is_correct": reference_is_correct,
        "final_reference_answer": final_reference_answer,
        "reference_generated_by_llm": reference_generated_by_llm,
        "analysis": analysis,
        "confidence": confidence,
    }


def llm_split_question_pairs(
    client: Any,
    model: str,
    max_retries: int,
    temperature: float,
    question_latex: str,
    correct_reference_answer: str,
    split: str = "",
    question_id: str = "",
) -> Dict[str, Any]:
    fallback_pair = {
        "sub_index": 1,
        "question": question_latex,
        "answer": correct_reference_answer,
        "notes": "fallback",
    }

    obj: Dict[str, Any] = {}
    try:
        obj = azure_json_call(
            client=client,
            model=model,
            system=SYSTEM_SPLIT_MULTI_QUESTION,
            user=split_questions_user_prompt(question_latex=question_latex, reference_answer=correct_reference_answer),
            temperature=temperature,
            max_retries=max_retries,
            telemetry={
                "operation": "build_step2_split_questions",
                "split": split,
                "question_id": question_id,
            },
        )
    except Exception:  # noqa: BLE001
        obj = {}

    raw_pairs = obj.get("pairs", [])
    pairs: List[Dict[str, Any]] = []
    if isinstance(raw_pairs, list):
        for idx, it in enumerate(raw_pairs, start=1):
            if not isinstance(it, dict):
                continue
            question = str(it.get("question", "")).strip()
            raw_answer = str(it.get("answer", ""))
            if not question:
                continue
            if not raw_answer.strip():
                answer = correct_reference_answer
            else:
                answer = compact_text(raw_answer, default=correct_reference_answer)
            notes = compact_text(str(it.get("notes", "")), default="")
            pairs.append(
                {
                    "sub_index": int(it.get("sub_index", idx) or idx),
                    "question": question,
                    "answer": answer,
                    "notes": notes,
                }
            )

    if not pairs:
        pairs = [fallback_pair]

    question_count = parse_int_like(obj.get("question_count", len(pairs)), len(pairs))
    if question_count <= 1 and len(pairs) == 1:
        pairs[0]["question"] = question_latex
        pairs[0]["answer"] = correct_reference_answer
    elif len(pairs) > 1:
        for pair in pairs:
            sub_index = int(pair.get("sub_index", 1) or 1)
            pair["question"] = preserve_stem_context(
                original_question=question_latex,
                split_question=str(pair.get("question", "")),
                sub_index=sub_index,
            )

    split_generated_by_llm = as_bool(obj.get("split_generated_by_llm"), default=(len(pairs) > 1))
    analysis = compact_text(str(obj.get("analysis", "")), default="")

    return {
        "raw_json": obj,
        "question_count": max(1, len(pairs)),
        "split_generated_by_llm": split_generated_by_llm,
        "pairs": pairs,
        "analysis": analysis,
    }


def llm_convert_to_eval_item(
    client: Any,
    model: str,
    max_retries: int,
    temperature: float,
    sympy_temperature: float,
    question_latex: str,
    reference_answer: str,
    repair_hint: str = "",
    split: str = "",
    question_id: str = "",
) -> Dict[str, Any]:
    user_prompt = convert_eval_user_prompt(question_latex=question_latex, reference_answer=reference_answer)
    if repair_hint.strip():
        user_prompt += (
            "\n\n[Repair requirements]\n"
            "- The previous output violated symbol-contract constraints.\n"
            f"- Fix this issue: {repair_hint.strip()}\n"
            "- Ensure the final SymPy answer uses ONLY allowed_symbols.\n"
            "- Ensure converted_question contains a 'Symbols (for final answer):' section.\n"
        )

    obj: Dict[str, Any] = {}
    try:
        obj = azure_json_call(
            client=client,
            model=model,
            system=SYSTEM_CONVERT_TO_EVAL_QUESTION,
            user=user_prompt,
            temperature=temperature,
            max_retries=max_retries,
            telemetry={
                "operation": "build_step3_convert_eval_item",
                "split": split,
                "question_id": question_id,
            },
        )
    except Exception:  # noqa: BLE001
        obj = {}

    question_type = str(obj.get("question_type", "value")).strip().lower()
    if question_type not in {"value", "judge"}:
        question_type = "value"

    converted_question = str(obj.get("converted_question", "")).strip() or question_latex
    reference_reasoning = str(obj.get("reference_reasoning", "")).strip()
    converted_answer = compact_text(str(obj.get("reference_answer", "")), default=reference_answer)

    raw_comparable = compact_text(str(obj.get("comparable_final_answer", "")), default=converted_answer)
    comparison_mode = str(obj.get("comparison_mode", "")).strip().lower()

    if question_type == "judge":
        comparison_mode = "exact"
        answer_kind = "bool"
        bool_answer = normalize_bool_answer(raw_comparable) or normalize_bool_answer(converted_answer)
        comparable_final_answer = bool_answer or "N/A"
        if bool_answer is None:
            converted_answer = "N/A"
        reference_answer_sympy = "N/A"
        allowed_symbols: List[str] = []
        symbol_definitions: Dict[str, str] = {}
        mismatch_symbols: List[str] = []
        symbol_contract_validation_passed = True
    else:
        comparison_mode = "sympy"
        answer_kind = "sympy"
        candidate_sympy = normalize_sympy_expression(str(obj.get("reference_answer_sympy", "")).strip())
        if not candidate_sympy or candidate_sympy == "N/A":
            candidate_sympy = normalize_sympy_expression(raw_comparable)
        if not candidate_sympy or candidate_sympy == "N/A":
            candidate_sympy = normalize_sympy_expression(converted_answer)
        comparable_final_answer = maybe_to_sympy(
            client=client,
            model=model,
            max_retries=max_retries,
            temperature=sympy_temperature,
            text=candidate_sympy,
            split=split,
            question_id=question_id,
        )
        if comparable_final_answer == "N/A":
            comparable_final_answer = normalize_sympy_expression(compact_text(candidate_sympy))
        reference_answer_sympy = normalize_sympy_expression(comparable_final_answer)
        comparable_final_answer = reference_answer_sympy
        allowed_symbols, symbol_definitions = parse_symbol_contract(obj.get("symbol_contract", {}))
        if not allowed_symbols:
            allowed_symbols = extract_symbol_tokens(reference_answer_sympy)
        mismatch_symbols = detect_symbol_mismatch(reference_answer_sympy, allowed_symbols)
        symbol_contract_validation_passed = len(mismatch_symbols) == 0
        converted_question = ensure_symbol_contract_section(converted_question, allowed_symbols, symbol_definitions)

    return {
        "raw_json": obj,
        "question_type": question_type,
        "answer_kind": answer_kind,
        "comparison_mode": comparison_mode,
        "converted_question": converted_question,
        "reference_reasoning": reference_reasoning,
        "converted_answer": converted_answer,
        "comparable_final_answer": comparable_final_answer,
        "reference_answer_sympy": reference_answer_sympy,
        "symbol_contract": {
            "allowed_symbols": allowed_symbols,
            "symbol_definitions": symbol_definitions,
        },
        "symbol_contract_validation_passed": symbol_contract_validation_passed,
        "symbol_contract_mismatch_symbols": mismatch_symbols,
    }
