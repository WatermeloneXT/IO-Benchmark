from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from core.llm_utils import azure_json_call
from core.symbol_contract import (
    build_symbol_contract_section,
    ensure_symbol_contract_section,
    extract_symbol_tokens,
)


SYSTEM_STAGE3_SYMBOL_SYNC = r"""
You are a post-processor for stage3 converted economics/math questions.

Goal:
- Check whether symbol usage in the converted question is consistent with the SymPy symbols in
  `allowed_symbols` and the provided symbol definitions.
- Focus ONLY on symbols that are actually used in the final answer (`used_symbols`).
- If inconsistent, rewrite only the relevant symbol mentions in the question body so they are
  consistent with `used_symbols` / `allowed_symbols` naming.
- Keep mathematical meaning unchanged.
- Keep non-relevant symbols/notation unchanged.
- Keep the question style and structure as unchanged as possible.
- Keep or preserve a "Symbols (for final answer):" section.

Return strict JSON only:
{
  "rewritten_question": "...",
  "changed": true,
  "notes": "short reason"
}
"""


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value or "").strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return default


def _unique_symbols(symbols: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in symbols:
        x = str(s or "").strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _canonicalize_symbols_section(
    question_text: str,
    allowed_symbols: Sequence[str],
    symbol_definitions: Dict[str, str],
) -> str:
    q = str(question_text or "").strip()
    section = build_symbol_contract_section(allowed_symbols, symbol_definitions)
    if not section:
        return q
    marker = "Symbols (for final answer):"
    idx = q.find(marker)
    if idx < 0:
        return ensure_symbol_contract_section(q, allowed_symbols, symbol_definitions)
    prefix = q[:idx].rstrip()
    if prefix:
        return f"{prefix}\n\n{section}"
    return section


def align_stage3_question_symbols(
    *,
    client: Any,
    model: str,
    max_retries: int,
    temperature: float,
    converted_question: str,
    reference_answer_sympy: str,
    allowed_symbols: Sequence[str],
    symbol_definitions: Dict[str, str],
    split: str = "",
    question_id: str = "",
) -> Dict[str, Any]:
    question_text = str(converted_question or "").strip()
    allowed = _unique_symbols(list(allowed_symbols or []))
    if (not question_text) or (not allowed):
        return {
            "converted_question_final": question_text,
            "applied": False,
            "changed_by_llm": False,
            "status": "skipped",
            "reason": "empty_question_or_allowed_symbols",
            "used_symbols": [],
            "raw_json": {},
        }

    used_symbols = _unique_symbols(extract_symbol_tokens(str(reference_answer_sympy or "")))
    if not used_symbols:
        used_symbols = list(allowed)

    canonical_question = _canonicalize_symbols_section(question_text, allowed, symbol_definitions)

    payload = {
        "converted_question": canonical_question,
        "reference_answer_sympy": str(reference_answer_sympy or "").strip(),
        "used_symbols": used_symbols,
        "allowed_symbols": allowed,
        "symbol_definitions": {k: str(v or "").strip() for k, v in (symbol_definitions or {}).items()},
        "requirements": {
            "rewrite_only_relevant_symbols": True,
            "keep_unrelated_notation": True,
            "preserve_question_meaning": True,
            "preserve_symbols_section": True,
        },
    }

    obj: Dict[str, Any] = {}
    try:
        obj = azure_json_call(
            client=client,
            model=model,
            system=SYSTEM_STAGE3_SYMBOL_SYNC,
            user="Check and rewrite symbol usage for this stage3 item:\n" + json.dumps(payload, ensure_ascii=False),
            temperature=temperature,
            max_retries=max_retries,
            telemetry={
                "operation": "build_step3_symbol_sync",
                "split": split,
                "question_id": question_id,
            },
        )
    except Exception as err:  # noqa: BLE001
        ensured = _canonicalize_symbols_section(question_text, allowed, symbol_definitions)
        return {
            "converted_question_final": ensured,
            "applied": ensured != question_text,
            "changed_by_llm": False,
            "status": "error",
            "reason": f"llm_call_failed: {err}",
            "used_symbols": used_symbols,
            "raw_json": {},
        }

    rewritten = str(obj.get("rewritten_question", "")).strip()
    if not rewritten:
        rewritten = canonical_question
    rewritten = _canonicalize_symbols_section(rewritten, allowed, symbol_definitions)

    # Guard against pathological truncation from model output.
    if question_text and (len(rewritten) < max(50, int(len(question_text) * 0.35))):
        rewritten = _canonicalize_symbols_section(question_text, allowed, symbol_definitions)
        return {
            "converted_question_final": rewritten,
            "applied": rewritten != question_text,
            "changed_by_llm": False,
            "status": "fallback",
            "reason": "rewritten_question_too_short",
            "used_symbols": used_symbols,
            "raw_json": obj,
        }

    return {
        "converted_question_final": rewritten,
        "applied": rewritten != question_text,
        "changed_by_llm": _as_bool(obj.get("changed"), default=(rewritten != question_text)),
        "status": "ok",
        "reason": str(obj.get("notes", "")).strip(),
        "used_symbols": used_symbols,
        "raw_json": obj,
    }
