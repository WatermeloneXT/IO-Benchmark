from __future__ import annotations

import re
from typing import Set

# Keep standard SymPy callable names untouched.
KNOWN_CALLABLES: Set[str] = {
    "Eq",
    "Ne",
    "Lt",
    "Le",
    "Gt",
    "Ge",
    "And",
    "Or",
    "Not",
    "sqrt",
    "log",
    "sin",
    "cos",
    "tan",
    "exp",
    "Abs",
    "Min",
    "Max",
    "Derivative",
    "Integral",
    "Sum",
    "Product",
    "Limit",
    "Piecewise",
    "Matrix",
}

_CALL_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\(([^(){}]*)\)")
_LATEX_IDENTIFIER_ALIASES = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "epsilon": "epsilon",
    "varepsilon": "epsilon",
    "theta": "theta",
    "vartheta": "theta",
    "kappa": "kappa",
    "lambda": "lambda",
    "mu": "mu",
    "nu": "nu",
    "xi": "xi",
    "pi": "pi",
    "rho": "rho",
    "sigma": "sigma",
    "tau": "tau",
    "phi": "phi",
    "varphi": "phi",
    "chi": "chi",
    "psi": "psi",
    "omega": "omega",
}


def _sanitize_fragment(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "arg"
    s = s.replace("**", "_pow_")
    s = s.replace("^", "_pow_")
    s = s.replace("*", "_mul_")
    s = s.replace("/", "_div_")
    s = s.replace("+", "_plus_")
    s = s.replace("-", "_minus_")
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "arg"
    if s[0].isdigit():
        s = f"n_{s}"
    return s


def normalize_sympy_expression(expr: str) -> str:
    """
    Normalize expression to evaluation-friendly symbol style.

    Unknown function-like calls are flattened:
    - D(p) -> D_p
    - u_i(D,R) -> u_i_D_R
    """
    s = (expr or "").strip()
    if not s or s == "N/A":
        return s or "N/A"

    for latex_name, plain_name in _LATEX_IDENTIFIER_ALIASES.items():
        s = re.sub(rf"\\{latex_name}\b", plain_name, s)

    for _ in range(12):
        changed = False

        def _repl(match: re.Match[str]) -> str:
            nonlocal changed
            name = match.group(1)
            args_raw = (match.group(2) or "").strip()
            if name in KNOWN_CALLABLES:
                return f"{name}({args_raw})"
            if not args_raw:
                changed = True
                return name
            parts = [_sanitize_fragment(x) for x in args_raw.split(",")]
            out = "_".join([name, *parts])
            out = re.sub(r"_+", "_", out).strip("_")
            changed = True
            return out or name

        new_s = _CALL_PATTERN.sub(_repl, s)
        s = new_s
        if not changed:
            break

    s = re.sub(r"\s+", " ", s).strip()
    return s or "N/A"
