from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def normalize_reasoning_effort(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip().lower()
    return text or None


def append_reasoning_effort_suffix(solver_label: str, reasoning_effort: Optional[str]) -> str:
    effort = normalize_reasoning_effort(reasoning_effort)
    if not effort:
        return solver_label
    suffix = f"__effort-{effort}"
    return solver_label if str(solver_label).endswith(suffix) else f"{solver_label}{suffix}"


def normalize_max_solve_tokens(value: Optional[Union[int, str]]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def append_max_solve_tokens_suffix(solver_label: str, max_solve_tokens: Optional[Union[int, str]]) -> str:
    max_tokens = normalize_max_solve_tokens(max_solve_tokens)
    if max_tokens is None:
        return solver_label
    suffix = f"__max-tokens-{max_tokens}"
    return solver_label if str(solver_label).endswith(suffix) else f"{solver_label}{suffix}"


def build_solver_artifact_label(
    solver_label: str,
    *,
    reasoning_effort: Optional[str] = None,
    max_solve_tokens: Optional[Union[int, str]] = None,
) -> str:
    label = append_reasoning_effort_suffix(solver_label, reasoning_effort)
    return append_max_solve_tokens_suffix(label, max_solve_tokens)


def _compact_model_name(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(model_name or "").lower())


def _resolve_model_profile(
    model_identifiers: Sequence[str],
    profiles: Dict[str, Any],
) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    for profile_name, raw_profile in (profiles or {}).items():
        profile = raw_profile if isinstance(raw_profile, dict) else {}
        regex_patterns = profile.get("match_regex", []) or []
        for identifier in model_identifiers:
            text = str(identifier or "").strip()
            if not text:
                continue
            for pattern in regex_patterns:
                try:
                    if re.search(str(pattern), text, flags=re.IGNORECASE):
                        return profile_name, profile, text
                except re.error:
                    continue

        substrings = profile.get("match_substrings", []) or []
        for identifier in model_identifiers:
            compact_identifier = _compact_model_name(identifier)
            if not compact_identifier:
                continue
            for needle in substrings:
                compact_needle = _compact_model_name(str(needle))
                if compact_needle and compact_needle in compact_identifier:
                    return profile_name, profile, str(identifier)
    return None, {}, None


def _native_reasoning_efforts(profile_cfg: Dict[str, Any]) -> List[str]:
    values = profile_cfg.get("native_reasoning_efforts", []) if isinstance(profile_cfg, dict) else []
    cleaned: List[str] = []
    for value in values or []:
        text = normalize_reasoning_effort(str(value or ""))
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _collect_native_reasoning_efforts(
    cfg: Dict[str, Any],
    model_identifiers: Sequence[str],
) -> List[str]:
    identifiers = [str(x or "").strip() for x in model_identifiers if str(x or "").strip()]
    if not identifiers:
        return []

    merged: List[str] = []
    for profiles in (
        ((cfg.get("generate") or {}).get("model_profiles", {}) or {}),
        ((cfg.get("generate_vllm") or {}).get("model_profiles", {}) or {}),
    ):
        _, profile_cfg, _ = _resolve_model_profile(identifiers, profiles)
        for effort in _native_reasoning_efforts(profile_cfg):
            if effort not in merged:
                merged.append(effort)
    return merged


def validate_native_reasoning_effort(
    cfg: Dict[str, Any],
    model_identifiers: Union[str, Sequence[str]],
    reasoning_effort: Optional[str],
) -> Optional[str]:
    effort = normalize_reasoning_effort(reasoning_effort)
    if effort is None:
        return None
    identifiers = [model_identifiers] if isinstance(model_identifiers, str) else list(model_identifiers)
    native_efforts = _collect_native_reasoning_efforts(cfg, identifiers)
    model_label = str(next((x for x in identifiers if str(x or "").strip()), "this model"))
    if not native_efforts:
        raise ValueError(
            f"Model '{model_label}' does not expose a native reasoning effort field. "
            "Do not pass a solver reasoning effort for it."
        )
    if effort not in native_efforts:
        raise ValueError(
            f"Model '{model_label}' supports reasoning effort values: "
            f"{', '.join(native_efforts)}. Received: {effort}"
        )
    return effort
