from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from core.sympy_format import KNOWN_CALLABLES, normalize_sympy_expression

_IDENT_PATTERN = re.compile(r"[A-Za-z_]\w*")
_VALID_IDENT = re.compile(r"^[A-Za-z_]\w*$")

RESERVED_SYMBOLS: Set[str] = set(KNOWN_CALLABLES) | {
    "True",
    "False",
    "true",
    "false",
    "oo",
    "nan",
    "zoo",
    "pi",
    "E",
    "I",
    "and",
    "or",
    "not",
}


def _is_valid_symbol(text: str) -> bool:
    return bool(_VALID_IDENT.fullmatch((text or "").strip()))


def normalize_allowed_symbols(symbols: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(symbols, (list, tuple, set)):
        return out
    seen: Set[str] = set()
    for x in symbols:
        sym = str(x or "").strip()
        if not sym or sym in RESERVED_SYMBOLS or not _is_valid_symbol(sym):
            continue
        if sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    out.sort()
    return out


def normalize_symbol_definitions(defs: Any, allowed_symbols: Optional[Sequence[str]] = None) -> Dict[str, str]:
    if not isinstance(defs, dict):
        return {}
    allowed = set(allowed_symbols or [])
    out: Dict[str, str] = {}
    for k, v in defs.items():
        key = str(k or "").strip()
        if not key or key in RESERVED_SYMBOLS or not _is_valid_symbol(key):
            continue
        if allowed and key not in allowed:
            continue
        out[key] = str(v or "").strip()
    return out


def extract_symbol_tokens(expr: str) -> List[str]:
    s = (expr or "").strip()
    if not s or s == "N/A":
        return []
    s = normalize_sympy_expression(s)
    toks = sorted(set(_IDENT_PATTERN.findall(s)))
    return [t for t in toks if t not in RESERVED_SYMBOLS]


def detect_symbol_mismatch(expr: str, allowed_symbols: Sequence[str]) -> List[str]:
    allowed = set(allowed_symbols or [])
    if not allowed:
        return []
    used = set(extract_symbol_tokens(expr))
    mismatch = sorted([t for t in used if t not in allowed])
    return mismatch


def parse_symbol_contract(raw: Any) -> Tuple[List[str], Dict[str, str]]:
    if isinstance(raw, str):
        s = raw.strip()
        if s:
            try:
                raw = json.loads(s)
            except Exception:  # noqa: BLE001
                raw = {}
        else:
            raw = {}
    if not isinstance(raw, dict):
        return [], {}
    allowed_symbols = normalize_allowed_symbols(raw.get("allowed_symbols", []))
    symbol_definitions = normalize_symbol_definitions(raw.get("symbol_definitions", {}), allowed_symbols)
    return allowed_symbols, symbol_definitions


def build_symbol_contract_section(allowed_symbols: Sequence[str], symbol_definitions: Optional[Dict[str, str]] = None) -> str:
    allowed = list(allowed_symbols or [])
    defs = symbol_definitions or {}
    if not allowed:
        return ""
    lines = ["Symbols (for final answer):"]
    for sym in allowed:
        desc = str(defs.get(sym, "")).strip()
        if desc:
            lines.append(f"- `{sym}`: {desc}")
        else:
            lines.append(f"- `{sym}`")
    return "\n".join(lines)


def ensure_symbol_contract_section(
    question_text: str,
    allowed_symbols: Sequence[str],
    symbol_definitions: Optional[Dict[str, str]] = None,
) -> str:
    q = (question_text or "").strip()
    section = build_symbol_contract_section(allowed_symbols, symbol_definitions)
    if not section:
        return q
    if "Symbols (for final answer):" in q:
        return q
    if not q:
        return section
    return f"{q}\n\n{section}"
