#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commands.generate_boxed_answers import (
    append_jsonl,
    canonicalize_json_answer,
    extract_last_boxed,
    load_done_ids,
    normalize_bool_answer,
    read_jsonl,
    split_input_files,
    write_jsonl,
)
from core.annotation_overrides import apply_split_annotation_overrides, resolve_effective_question
from core.cost_logging import finish_run, register_run_outputs, start_run
from core.file_naming import resolve_existing_model_jsonl
from core.llm_utils import get_default_solver_model, load_config
from core.model_layout import legacy_generations_dir, resolve_rule_evaluation_input, solver_generation_file
from core.path_overrides import apply_dataset_path_overrides
from core.prompts import SYSTEM_LATEX_TO_SYMPY, SYSTEM_SOLVER_WITH_BOX, solve_user_prompt
from core.question_filter import filter_rows_by_question_ids, load_question_id_filter, merge_rows_by_question_id
from core.result_metadata import build_result_metadata
from core.solver_variants import build_solver_artifact_label, validate_native_reasoning_effort
from core.symbol_contract import (
    detect_symbol_mismatch,
    extract_symbol_tokens,
    normalize_allowed_symbols,
    parse_symbol_contract,
)
from core.sympy_format import normalize_sympy_expression


SUPPORTED_FAMILIES = {"auto", "generic", "llama3", "qwen3", "deepseek", "kimi"}
SUPPORTED_REASONING_MODES = {"auto", "on", "off"}
_MISSING = object()


@dataclass
class SolverTask:
    idx: int
    row: Dict[str, Any]
    qid: str
    split_name: str
    question_final: str
    effective_question_final: str
    answer_kind: str
    comparison_mode: str
    allowed_symbols: List[str]
    json_keys: List[str]
    prompt: str


@dataclass
class SolverResult:
    idx: int
    qid: str
    row: Dict[str, Any]
    msg: str


@dataclass
class SplitRunContext:
    input_file: Path
    split_name: str
    output_file: Path
    evaluation_file: Path
    all_rows: List[Dict[str, Any]]
    rows: List[Dict[str, Any]]
    pending: List[SolverTask]
    partial_overwrite: bool
    existing_rows: List[Dict[str, Any]]


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


def _normalize_answer_kind(row: Dict[str, Any]) -> str:
    answer_kind = str(row.get("answer_kind", "sympy")).strip().lower()
    if answer_kind not in {"sympy", "bool", "text", "json"}:
        return "sympy"
    return answer_kind


def _normalize_comparison_mode(row: Dict[str, Any], answer_kind: str) -> str:
    comparison_mode = str(row.get("comparison_mode", "sympy")).strip().lower()
    if comparison_mode in {"sympy", "exact", "json"}:
        return comparison_mode
    if answer_kind == "sympy":
        return "sympy"
    if answer_kind == "json":
        return "json"
    return "exact"


def _prepare_task(input_file: Path, idx: int, row: Dict[str, Any]) -> Optional[SolverTask]:
    qid = str(row.get("id", "")).strip()
    if not qid:
        return None

    question_final = str(row.get("question_final", "")).strip()
    effective_question_final = resolve_effective_question(row) or question_final
    answer_kind = _normalize_answer_kind(row)
    comparison_mode = _normalize_comparison_mode(row, answer_kind)
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
    return SolverTask(
        idx=idx,
        row=row,
        qid=qid,
        split_name=split_name,
        question_final=question_final,
        effective_question_final=effective_question_final,
        answer_kind=answer_kind,
        comparison_mode=comparison_mode,
        allowed_symbols=allowed_symbols,
        json_keys=json_keys,
        prompt=prompt,
    )


def _detect_model_family(model_name: str) -> str:
    name = str(model_name or "").lower()
    compact = re.sub(r"[^a-z0-9]+", "", name)
    if "qwen3" in compact:
        return "qwen3"
    if "llama3" in compact or "llama31" in compact or "llama32" in compact or "llama33" in compact:
        return "llama3"
    if "metallama" in compact and "3" in compact:
        return "llama3"
    if "deepseek" in compact:
        return "deepseek"
    if "kimi" in compact or "moonshot" in compact:
        return "kimi"
    return "generic"


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


def _normalize_capability_type(value: Any) -> str:
    kind = str(value or "").strip().lower()
    if kind in {"base", "instruct", "reasoning"}:
        return kind
    return "instruct"


def _normalize_reasoning_interface(value: Any) -> str:
    interface = str(value or "").strip().lower()
    if interface in {"parser_only", "chat_template_enable_thinking", "chat_template_thinking_bool"}:
        return interface
    return "parser_only"


def _native_reasoning_efforts(profile_cfg: Dict[str, Any]) -> List[str]:
    values = profile_cfg.get("native_reasoning_efforts", []) if isinstance(profile_cfg, dict) else []
    cleaned: List[str] = []
    for value in values or []:
        text = str(value or "").strip().lower()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _pick_profile_value(
    key: str,
    *,
    user_value: Any = _MISSING,
    profile_defaults: Optional[Dict[str, Any]] = None,
    base_defaults: Optional[Dict[str, Any]] = None,
    fallback: Any = None,
) -> Any:
    if user_value is not _MISSING and user_value is not None:
        return user_value
    if isinstance(profile_defaults, dict) and key in profile_defaults:
        return profile_defaults.get(key)
    if isinstance(base_defaults, dict) and key in base_defaults:
        return base_defaults.get(key)
    return fallback


def _resolve_effective_qwen3_thinking_mode(family: str, requested_mode: str, *, default_mode: str = "on") -> str:
    mode = str(requested_mode or "auto").strip().lower()
    if family != "qwen3":
        return mode
    if mode == "auto":
        chosen = str(default_mode or "on").strip().lower()
        if chosen in {"on", "off"}:
            return chosen
        # Qwen3 model cards state thinking mode is enabled by default.
        return "on"
    return mode


def _resolve_effective_reasoning_mode(
    *,
    capability_type: str,
    reasoning_interface: str,
    profile_cfg: Dict[str, Any],
    requested_reasoning_mode: Optional[str],
    requested_qwen3_thinking: str,
    family: str,
    profile_name: Optional[str],
) -> Optional[str]:
    if capability_type != "reasoning":
        return None

    mode = str(requested_reasoning_mode or "auto").strip().lower()
    if mode not in SUPPORTED_REASONING_MODES:
        mode = "auto"

    legacy_qwen_mode = str(requested_qwen3_thinking or "auto").strip().lower()
    if legacy_qwen_mode not in SUPPORTED_REASONING_MODES:
        legacy_qwen_mode = "auto"
    if family == "qwen3" and legacy_qwen_mode != "auto":
        if mode == "auto":
            mode = legacy_qwen_mode
        elif mode != legacy_qwen_mode:
            raise ValueError(
                "--reasoning-mode and --qwen3-thinking disagree. "
                "Please set only one of them, or make them consistent."
            )

    default_mode = str(profile_cfg.get("reasoning_default_mode", "on")).strip().lower()
    if default_mode not in {"on", "off"}:
        default_mode = "on"
    effective_mode = default_mode if mode == "auto" else mode

    if reasoning_interface == "parser_only" and effective_mode == "off":
        profile_label = profile_name or family or "this reasoning model"
        raise ValueError(
            f"vLLM profile '{profile_label}' is classified as parser_only. "
            "Official vLLM docs do not expose a hard reasoning off switch for this family, "
            "so --reasoning-mode off would be misleading."
        )

    return effective_mode


def _build_reasoning_chat_template_kwargs(reasoning_interface: str, effective_reasoning_mode: Optional[str]) -> Dict[str, Any]:
    if effective_reasoning_mode not in {"on", "off"}:
        return {}
    if reasoning_interface == "chat_template_enable_thinking":
        return {"enable_thinking": effective_reasoning_mode == "on"}
    if reasoning_interface == "chat_template_thinking_bool":
        return {"thinking": effective_reasoning_mode == "on"}
    return {}


def _resolve_sampling_defaults(
    *,
    cfg: Dict[str, Any],
    model_identifiers: Sequence[str],
    family: str,
    reasoning_mode: Optional[str],
    reasoning_effort: Optional[str],
    qwen3_thinking: str,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    max_tokens: Optional[int],
    repetition_penalty: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[int], int, Optional[float], Dict[str, Any]]:
    gen_cfg = cfg.get("generate", {}) or {}
    vllm_cfg = cfg.get("generate_vllm", {}) or {}

    profile_name, profile_cfg, matched_identifier = _resolve_model_profile(
        model_identifiers,
        vllm_cfg.get("model_profiles", {}) or {},
    )
    capability_type = _normalize_capability_type((profile_cfg or {}).get("capability_type"))
    reasoning_interface = _normalize_reasoning_interface((profile_cfg or {}).get("reasoning_interface"))
    capability_defaults = ((vllm_cfg.get("capability_defaults", {}) or {}).get(capability_type, {}) or {}).copy()
    if "solve_temperature" not in capability_defaults:
        capability_defaults["solve_temperature"] = gen_cfg.get("solve_temperature", 0.0)
    if "max_solve_tokens" not in capability_defaults:
        capability_defaults["max_solve_tokens"] = gen_cfg.get("max_solve_tokens", 4096)
    if "solve_top_p" not in capability_defaults:
        capability_defaults["solve_top_p"] = 1.0
    if "solve_top_k" not in capability_defaults:
        capability_defaults["solve_top_k"] = None
    if "solve_repetition_penalty" not in capability_defaults:
        capability_defaults["solve_repetition_penalty"] = 1.0

    requested_reasoning_effort = str(reasoning_effort or "").strip().lower() or None
    native_reasoning_efforts = _native_reasoning_efforts(profile_cfg if isinstance(profile_cfg, dict) else {})
    if requested_reasoning_effort is not None:
        if capability_type != "reasoning":
            profile_label = profile_name or matched_identifier or family or "this model"
            raise ValueError(
                f"Model '{profile_label}' is not classified as a reasoning model, "
                "so --reasoning-effort is not valid for it."
            )
        if not native_reasoning_efforts:
            profile_label = profile_name or matched_identifier or family or "this reasoning model"
            raise ValueError(
                f"Model '{profile_label}' does not expose a native reasoning effort field. "
                "Use --reasoning-mode or --max-tokens instead."
            )
        if requested_reasoning_effort not in native_reasoning_efforts:
            profile_label = profile_name or matched_identifier or family or "this reasoning model"
            raise ValueError(
                f"Model '{profile_label}' supports reasoning effort values: "
                f"{', '.join(native_reasoning_efforts)}. Received: {requested_reasoning_effort}"
            )

    if requested_reasoning_effort is not None:
        effective_reasoning_effort = requested_reasoning_effort
    elif native_reasoning_efforts:
        default_native_reasoning_effort = str(
            (profile_cfg or {}).get("default_native_reasoning_effort", "")
        ).strip().lower()
        effective_reasoning_effort = (
            default_native_reasoning_effort
            if default_native_reasoning_effort in native_reasoning_efforts
            else native_reasoning_efforts[0]
        )
    else:
        effective_reasoning_effort = None

    profile_defaults: Dict[str, Any] = {}
    reasoning_backend = None
    reasoning_parser = None
    effective_reasoning_mode: Optional[str] = None
    effective_qwen3_thinking = "auto"
    chat_template_kwargs: Dict[str, Any] = {}
    conversion_chat_template_kwargs: Dict[str, Any] = {}
    if capability_type == "reasoning":
        effort_map = (profile_cfg or {}).get("reasoning_effort_map", {}) if isinstance(profile_cfg, dict) else {}
        if isinstance(effort_map, dict):
            selected = effort_map.get(effective_reasoning_effort, {})
            if isinstance(selected, dict):
                profile_defaults = selected
        if "max_tokens" not in profile_defaults and "max_solve_tokens" in profile_defaults:
            profile_defaults = dict(profile_defaults)
            profile_defaults["max_tokens"] = profile_defaults.get("max_solve_tokens")
        reasoning_backend = (profile_cfg or {}).get("reasoning_backend") if isinstance(profile_cfg, dict) else None
        reasoning_parser = (profile_cfg or {}).get("reasoning_parser") if isinstance(profile_cfg, dict) else None
        effective_reasoning_mode = _resolve_effective_reasoning_mode(
            capability_type=capability_type,
            reasoning_interface=reasoning_interface,
            profile_cfg=profile_cfg if isinstance(profile_cfg, dict) else {},
            requested_reasoning_mode=reasoning_mode,
            requested_qwen3_thinking=qwen3_thinking,
            family=family,
            profile_name=profile_name,
        )
        if family == "qwen3" and effective_reasoning_mode in {"on", "off"}:
            effective_qwen3_thinking = effective_reasoning_mode
        chat_template_kwargs = _build_reasoning_chat_template_kwargs(
            reasoning_interface,
            effective_reasoning_mode,
        )
        extra_chat_template_kwargs = profile_defaults.get("chat_template_kwargs")
        if isinstance(extra_chat_template_kwargs, dict) and extra_chat_template_kwargs:
            merged_chat_template_kwargs = dict(extra_chat_template_kwargs)
            merged_chat_template_kwargs.update(chat_template_kwargs)
            chat_template_kwargs = merged_chat_template_kwargs
        if reasoning_interface in {"chat_template_enable_thinking", "chat_template_thinking_bool"}:
            conversion_chat_template_kwargs = _build_reasoning_chat_template_kwargs(
                reasoning_interface,
                "off",
            )
        if reasoning_backend == "qwen3_thinking" and effective_reasoning_mode in {"on", "off"}:
            profile_defaults = dict(profile_defaults)
            profile_defaults["qwen3_thinking"] = effective_reasoning_mode

    sampling_profile_defaults = profile_defaults if capability_type != "reasoning" else {}
    if capability_type == "reasoning":
        base_defaults = {
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "max_tokens": capability_defaults.get("max_solve_tokens"),
            "repetition_penalty": None,
        }
    else:
        base_defaults = {
            "temperature": capability_defaults.get("solve_temperature"),
            "top_p": capability_defaults.get("solve_top_p"),
            "top_k": capability_defaults.get("solve_top_k"),
            "max_tokens": capability_defaults.get("max_solve_tokens"),
            "repetition_penalty": capability_defaults.get("solve_repetition_penalty"),
        }

    temperature_value = _pick_profile_value(
        "temperature",
        user_value=temperature,
        profile_defaults=sampling_profile_defaults,
        base_defaults=base_defaults,
        fallback=None,
    )
    resolved_temperature = float(temperature_value) if temperature_value is not None else None
    top_p_value = _pick_profile_value(
        "top_p",
        user_value=top_p,
        profile_defaults=sampling_profile_defaults,
        base_defaults=base_defaults,
        fallback=None,
    )
    resolved_top_p = float(top_p_value) if top_p_value is not None else None
    top_k_value = _pick_profile_value(
        "top_k",
        user_value=top_k,
        profile_defaults=sampling_profile_defaults,
        base_defaults=base_defaults,
        fallback=None,
    )
    resolved_top_k = int(top_k_value) if top_k_value is not None else None
    max_tokens_value = _pick_profile_value(
        "max_tokens",
        user_value=max_tokens,
        profile_defaults=profile_defaults,
        base_defaults=base_defaults,
        fallback=4096,
    )
    resolved_max_tokens = int(max_tokens_value)
    repetition_penalty_value = _pick_profile_value(
        "repetition_penalty",
        user_value=repetition_penalty,
        profile_defaults=sampling_profile_defaults,
        base_defaults=base_defaults,
        fallback=None,
    )
    resolved_repetition_penalty = (
        float(repetition_penalty_value) if repetition_penalty_value is not None else None
    )

    adaptation = {
        "family": family,
        "capability_type": capability_type,
        "profile_name": profile_name,
        "profile_doc_url": profile_cfg.get("doc_url") if isinstance(profile_cfg, dict) else None,
        "profile_note": profile_cfg.get("note") if isinstance(profile_cfg, dict) else None,
        "matched_identifier": matched_identifier,
        "reasoning_interface": (reasoning_interface if capability_type == "reasoning" else None),
        "reasoning_parser": reasoning_parser,
        "reasoning_backend": reasoning_backend,
        "requested_reasoning_mode": reasoning_mode,
        "effective_reasoning_mode": effective_reasoning_mode,
        "requested_reasoning_effort": reasoning_effort,
        "supports_native_reasoning_effort": bool(native_reasoning_efforts),
        "effective_reasoning_effort": (effective_reasoning_effort if capability_type == "reasoning" else None),
        "requested_qwen3_thinking": qwen3_thinking,
        "effective_qwen3_thinking": effective_qwen3_thinking,
        "chat_template_kwargs": dict(chat_template_kwargs),
        "conversion_chat_template_kwargs": dict(conversion_chat_template_kwargs),
        "temperature": resolved_temperature,
        "top_p": resolved_top_p,
        "top_k": resolved_top_k,
        "max_tokens": resolved_max_tokens,
        "repetition_penalty": resolved_repetition_penalty,
    }
    return (
        resolved_temperature,
        resolved_top_p,
        resolved_top_k,
        resolved_max_tokens,
        resolved_repetition_penalty,
        adaptation,
    )


def _manual_chat_template(messages: Sequence[Dict[str, str]], family: str) -> str:
    system = ""
    user_parts: List[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "") or "")
        if role == "system":
            system = content
        elif role == "user":
            user_parts.append(content)

    user = "\n\n".join([part for part in user_parts if part]).strip()
    if family == "llama3":
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    merged_user = user
    if system:
        merged_user = f"{system.strip()}\n\n{user}"
    return (
        "<|im_start|>user\n"
        f"{merged_user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _strip_reasoning_tags(text: str) -> str:
    candidate = str(text or "")
    candidate = re.sub(r"(?is)<think>.*?</think>", "", candidate)
    if "</think>" in candidate:
        candidate = candidate.rsplit("</think>", 1)[-1]
    return candidate.strip()


def _strip_code_fence(text: str) -> str:
    candidate = str(text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    return candidate.strip()


def _construct_with_supported_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Any:
    filtered = {key: value for key, value in kwargs.items() if value is not None}
    try:
        signature = inspect.signature(cls)
    except Exception:  # noqa: BLE001
        return cls(**filtered)

    params = signature.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        return cls(**filtered)
    return cls(**{key: value for key, value in filtered.items() if key in params})


class VLLMBatchRunner:
    def __init__(
        self,
        *,
        model: str,
        model_hint: Optional[str],
        tokenizer: Optional[str],
        family: str,
        trust_remote_code: bool,
        dtype: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
        max_num_seqs: Optional[int],
        quantization: Optional[str],
        load_format: Optional[str],
        download_dir: Optional[str],
        seed: Optional[int],
        enforce_eager: bool,
        use_tqdm: bool,
        qwen3_thinking: str,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Install it in the GPU environment, for example: "
                "pip install vllm"
            ) from exc

        self.model = model
        detection_text = " ".join(
            [
                piece
                for piece in [model_hint, tokenizer, model]
                if str(piece or "").strip()
            ]
        )
        self.family = _detect_model_family(detection_text) if family == "auto" else family
        self.SamplingParams = SamplingParams
        self.use_tqdm = use_tqdm
        self.qwen3_thinking = _resolve_effective_qwen3_thinking_mode(self.family, qwen3_thinking)
        self.default_chat_template_kwargs: Dict[str, Any] = {}

        llm_kwargs: Dict[str, Any] = {
            "model": model,
            "tokenizer": tokenizer or model,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "quantization": quantization,
            "load_format": load_format,
            "download_dir": download_dir,
            "seed": seed,
            "enforce_eager": enforce_eager,
        }
        print(
            f"[vllm] loading model={model} family={self.family} "
            f"dtype={dtype} tp={tensor_parallel_size}"
        )
        self.llm = LLM(**{key: value for key, value in llm_kwargs.items() if value is not None})
        self.tokenizer = self.llm.get_tokenizer()

    def sampling_params(
        self,
        *,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        max_tokens: int,
        repetition_penalty: Optional[float],
        stop: Optional[List[str]],
        seed: Optional[int],
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "stop": stop or None,
            "seed": seed,
        }
        return _construct_with_supported_kwargs(self.SamplingParams, kwargs)

    def render_prompt(
        self,
        system: str,
        user: str,
        *,
        qwen3_thinking: Optional[str] = None,
        chat_template_kwargs_override: Optional[Dict[str, Any]] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user},
        ]
        thinking = (qwen3_thinking or self.qwen3_thinking or "auto").strip().lower()

        attempts: List[Tuple[List[Dict[str, str]], Dict[str, Any]]] = []
        base_kwargs = {"tokenize": False, "add_generation_prompt": True}
        kwargs_with_thinking = dict(base_kwargs)
        kwargs_with_thinking.update(self.default_chat_template_kwargs)
        if chat_template_kwargs_override:
            kwargs_with_thinking.update(chat_template_kwargs_override)
        requested_reasoning_kwargs = {
            key: value
            for key, value in kwargs_with_thinking.items()
            if key not in base_kwargs
        }
        if (
            self.family == "qwen3"
            and thinking in {"on", "off"}
            and "enable_thinking" not in kwargs_with_thinking
        ):
            kwargs_with_thinking["enable_thinking"] = thinking == "on"
        attempts.append((messages, kwargs_with_thinking))
        if not requested_reasoning_kwargs:
            attempts.append((messages, base_kwargs))

        merged_user = f"{system.strip()}\n\n{user}" if system.strip() else user
        attempts.append(([{"role": "user", "content": merged_user}], kwargs_with_thinking))
        if not requested_reasoning_kwargs:
            attempts.append(([{"role": "user", "content": merged_user}], base_kwargs))

        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            for candidate_messages, kwargs in attempts:
                try:
                    rendered = apply_template(candidate_messages, **kwargs)
                    if isinstance(rendered, str) and rendered.strip():
                        return rendered
                except TypeError:
                    continue
                except Exception:
                    continue

        if requested_reasoning_kwargs:
            raise RuntimeError(
                "Tokenizer chat template did not accept the required reasoning switch kwargs "
                f"{requested_reasoning_kwargs} for family={self.family}. "
                "Refusing to silently fall back without applying the requested reasoning mode."
            )

        return _manual_chat_template(messages, self.family)

    def generate_texts(self, prompts: List[str], sampling_params: Any) -> List[str]:
        if not prompts:
            return []
        try:
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=self.use_tqdm)
        except TypeError:
            outputs = self.llm.generate(prompts, sampling_params)

        texts: List[str] = []
        for item in outputs:
            generations = getattr(item, "outputs", []) or []
            if not generations:
                texts.append("")
                continue
            texts.append(str(getattr(generations[0], "text", "") or ""))
        return texts


def _sympy_conversion_value(raw_text: str, fallback: str) -> str:
    cleaned = _strip_code_fence(_strip_reasoning_tags(raw_text))
    boxed = extract_last_boxed(cleaned)
    if boxed != "N/A":
        cleaned = boxed
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        cleaned = lines[-1]
    cleaned = cleaned.strip().strip("`")
    norm = normalize_sympy_expression(cleaned)
    if not norm or norm == "N/A":
        return fallback or "N/A"
    return norm


def _solve_chunk(
    *,
    runner: VLLMBatchRunner,
    tasks: List[SolverTask],
    model_label: str,
    vllm_model: str,
    solve_params: Any,
    convert_params: Any,
    convert_sympy: bool,
    sampling_adaptation: Optional[Dict[str, Any]] = None,
) -> List[SolverResult]:
    solve_prompts = [
        runner.render_prompt(SYSTEM_SOLVER_WITH_BOX, task.prompt)
        for task in tasks
    ]
    solve_texts = runner.generate_texts(solve_prompts, solve_params)
    if len(solve_texts) != len(tasks):
        raise RuntimeError(f"vLLM returned {len(solve_texts)} solve outputs for {len(tasks)} prompts.")

    boxed_answers: List[str] = []
    conversion_prompts: List[str] = []
    conversion_task_indexes: List[int] = []
    answer_sympy_by_index: Dict[int, str] = {}

    for task_index, (task, response) in enumerate(zip(tasks, solve_texts)):
        boxed_answer = extract_last_boxed(response)
        boxed_answer = re.sub(r"\s+", " ", boxed_answer).strip() if boxed_answer != "N/A" else "N/A"
        boxed_answers.append(boxed_answer)
        if task.comparison_mode == "sympy":
            fallback = normalize_sympy_expression((boxed_answer or "").strip())
            answer_sympy_by_index[task_index] = fallback if fallback else "N/A"
            if convert_sympy and fallback != "N/A":
                conversion_prompts.append(
                    runner.render_prompt(
                        SYSTEM_LATEX_TO_SYMPY,
                        f"Convert this LaTeX to SymPy:\n{fallback}",
                        qwen3_thinking="off",
                        chat_template_kwargs_override=(
                            (sampling_adaptation or {}).get("conversion_chat_template_kwargs") or {}
                        ),
                    )
                )
                conversion_task_indexes.append(task_index)

    if conversion_prompts:
        conversion_texts = runner.generate_texts(conversion_prompts, convert_params)
        if len(conversion_texts) != len(conversion_prompts):
            raise RuntimeError(
                f"vLLM returned {len(conversion_texts)} conversion outputs "
                f"for {len(conversion_prompts)} prompts."
            )
        for task_index, raw_text in zip(conversion_task_indexes, conversion_texts):
            answer_sympy_by_index[task_index] = _sympy_conversion_value(
                raw_text,
                answer_sympy_by_index.get(task_index, "N/A"),
            )

    results: List[SolverResult] = []
    for task_index, (task, response) in enumerate(zip(tasks, solve_texts)):
        boxed_answer = boxed_answers[task_index]
        answer_sympy = "N/A"
        answer_json: Dict[str, str] | None = None
        final_answer_for_compare = boxed_answer
        if task.answer_kind == "bool" or task.comparison_mode == "exact":
            final_answer_for_compare = normalize_bool_answer(boxed_answer) or boxed_answer
        elif task.comparison_mode == "json":
            answer_json, final_answer_for_compare = canonicalize_json_answer(boxed_answer)
        else:
            answer_sympy = normalize_sympy_expression(answer_sympy_by_index.get(task_index, "N/A"))
            final_answer_for_compare = answer_sympy if answer_sympy != "N/A" else boxed_answer

        mismatch_symbols: List[str] = []
        symbol_mismatch = False
        if task.comparison_mode == "sympy" and task.allowed_symbols:
            mismatch_symbols = detect_symbol_mismatch(str(final_answer_for_compare), task.allowed_symbols)
            symbol_mismatch = len(mismatch_symbols) > 0

        source_row = task.row
        out_obj: Dict[str, Any] = {
            "id": task.qid,
            "split": task.split_name,
            "chapter": source_row.get("chapter"),
            "question_final": task.question_final,
            "question_final_used": task.effective_question_final,
            "annotator_rewritten_question": source_row.get("annotator_rewritten_question", ""),
            "annotator_rewritten_solution": source_row.get("annotator_rewritten_solution", ""),
            "annotator_rewritten_answer": source_row.get("annotator_rewritten_answer", {}),
            "annotator_override_active": bool(source_row.get("annotator_override_active", False)),
            "answer_kind": task.answer_kind,
            "question_type": source_row.get("question_type", "value"),
            "comparison_mode": task.comparison_mode,
            "model_response": response,
            "answer_boxed": boxed_answer,
            "answer_sympy": answer_sympy,
            "answer_json": answer_json,
            "final_answer_for_compare": final_answer_for_compare,
            "reference_answer": source_row.get("reference_answer", "N/A"),
            "reference_answer_sympy": source_row.get("reference_answer_sympy", "N/A"),
            "reference_answer_json": source_row.get("reference_answer_json", {}),
            "reference_answer_json_modes": source_row.get("reference_answer_json_modes", {}),
            "symbol_contract_allowed_symbols": task.allowed_symbols,
            "symbol_mismatch": symbol_mismatch,
            "mismatch_symbols": mismatch_symbols,
            "points": source_row.get("points", 1.0),
            "meta": {
                "model": model_label,
                "provider": "vllm",
                "vllm_model": vllm_model,
                "model_family": runner.family,
                "qwen3_thinking": runner.qwen3_thinking,
                "reasoning_mode": (sampling_adaptation or {}).get("effective_reasoning_mode"),
                "reasoning_interface": (sampling_adaptation or {}).get("reasoning_interface"),
                "chat_template_kwargs": dict((sampling_adaptation or {}).get("chat_template_kwargs") or {}),
                "solve_temperature": getattr(solve_params, "temperature", None),
                "solve_top_p": getattr(solve_params, "top_p", None),
                "solve_top_k": getattr(solve_params, "top_k", None),
                "solve_repetition_penalty": getattr(solve_params, "repetition_penalty", None),
                "max_solve_tokens": getattr(solve_params, "max_tokens", None),
                "convert_sympy": convert_sympy,
                "sampling_adaptation": dict(sampling_adaptation or {}),
            },
        }
        out_obj.update(
            build_result_metadata(
                stage="generate",
                solver_model=model_label,
                split=task.split_name,
                question_id=task.qid,
            )
        )
        results.append(
            SolverResult(
                idx=task.idx,
                qid=task.qid,
                row=out_obj,
                msg=f"[generate-vllm] {task.split_name} {task.idx} -> {task.qid}",
            )
        )
    return results


def _collect_pending_tasks(
    *,
    input_file: Path,
    output_file: Path,
    evaluation_file: Path,
    cfg: Dict[str, Any],
    model_label: str,
    resume_override: Optional[bool],
    question_id_filter: Optional[Set[str]],
    mutate_output: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[SolverTask], bool]:
    all_rows = read_jsonl(input_file)
    all_rows = apply_split_annotation_overrides(all_rows, cfg["paths"], input_file.stem)
    rows = filter_rows_by_question_ids(all_rows, question_id_filter)
    if not rows:
        return all_rows, rows, [], bool(question_id_filter)

    gen_cfg = cfg["generate"]
    resume = bool(gen_cfg.get("resume", True)) if resume_override is None else bool(resume_override)
    partial_overwrite = bool(question_id_filter)

    if mutate_output and output_file.exists() and not resume and not partial_overwrite:
        output_file.unlink()

    effective_resume = resume
    if output_file.exists() and resume and not evaluation_file.exists() and not partial_overwrite:
        effective_resume = False
        if mutate_output:
            output_file.unlink()
        print(
            f"[generate-vllm] {input_file.name} -> found old generation but no evaluation "
            f"for model={model_label}; regenerate all rows."
        )

    done = load_done_ids(output_file) if effective_resume else set()
    if partial_overwrite:
        done = set()

    pending: List[SolverTask] = []
    for idx, row in enumerate(rows, start=1):
        qid = str(row.get("id", "")).strip()
        if not qid or qid in done:
            continue
        task = _prepare_task(input_file, idx, row)
        if task is not None:
            pending.append(task)
    return all_rows, rows, pending, partial_overwrite


def _chunks(items: Sequence[SolverTask], batch_size: int) -> Sequence[List[SolverTask]]:
    if batch_size <= 0:
        batch_size = len(items) or 1
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def _write_split_results(
    *,
    ctx: SplitRunContext,
    results: Sequence[SolverResult],
) -> None:
    if not results:
        return

    sorted_results = sorted(results, key=lambda item: item.idx)
    result_rows = [result.row for result in sorted_results]
    if ctx.partial_overwrite:
        ordered_question_ids = [
            str(row.get("id", "")).strip()
            for row in ctx.all_rows
            if str(row.get("id", "")).strip()
        ]
        merged_rows = merge_rows_by_question_id(
            ctx.existing_rows,
            result_rows,
            ordered_question_ids,
            keep_unordered_existing=False,
        )
        write_jsonl(ctx.output_file, merged_rows)
        return

    if ctx.existing_rows:
        write_jsonl(ctx.output_file, ctx.existing_rows + result_rows)
    else:
        write_jsonl(ctx.output_file, result_rows)


def _prepare_split_run_context(
    *,
    input_file: Path,
    output_file: Path,
    evaluation_file: Path,
    cfg: Dict[str, Any],
    model_label: str,
    resume_override: Optional[bool],
    question_id_filter: Optional[Set[str]],
) -> SplitRunContext:
    all_rows, rows, pending, partial_overwrite = _collect_pending_tasks(
        input_file=input_file,
        output_file=output_file,
        evaluation_file=evaluation_file,
        cfg=cfg,
        model_label=model_label,
        resume_override=resume_override,
        question_id_filter=question_id_filter,
        mutate_output=True,
    )
    existing_rows = read_jsonl(output_file) if output_file.exists() else []
    return SplitRunContext(
        input_file=input_file,
        split_name=input_file.stem,
        output_file=output_file,
        evaluation_file=evaluation_file,
        all_rows=all_rows,
        rows=rows,
        pending=pending,
        partial_overwrite=partial_overwrite,
        existing_rows=existing_rows,
    )


def run_for_split(
    *,
    input_file: Path,
    output_file: Path,
    evaluation_file: Path,
    cfg: Dict[str, Any],
    runner: VLLMBatchRunner,
    model_label: str,
    vllm_model: str,
    solve_params: Any,
    convert_params: Any,
    batch_size: int,
    convert_sympy: bool,
    sampling_adaptation: Optional[Dict[str, Any]],
    resume_override: Optional[bool],
    question_id_filter: Optional[Set[str]],
) -> int:
    ctx = _prepare_split_run_context(
        input_file=input_file,
        output_file=output_file,
        evaluation_file=evaluation_file,
        cfg=cfg,
        model_label=model_label,
        resume_override=resume_override,
        question_id_filter=question_id_filter,
    )
    rows = ctx.rows
    pending = ctx.pending
    if not rows:
        return 0
    if not pending:
        print(f"[generate-vllm] skip {input_file.stem}: no pending rows")
        return len(rows)

    all_results: List[SolverResult] = []
    for chunk_id, chunk in enumerate(_chunks(pending, batch_size), start=1):
        t0 = time.time()
        results = _solve_chunk(
            runner=runner,
            tasks=chunk,
            model_label=model_label,
            vllm_model=vllm_model,
            solve_params=solve_params,
            convert_params=convert_params,
            convert_sympy=convert_sympy,
            sampling_adaptation=sampling_adaptation,
        )
        results.sort(key=lambda item: item.idx)
        elapsed = time.time() - t0
        for result in results:
            print(result.msg)
        print(
            f"[generate-vllm] {input_file.stem} batch {chunk_id}: "
            f"{len(results)} rows in {elapsed:.1f}s"
        )

        if ctx.partial_overwrite:
            all_results.extend(results)
        else:
            for result in results:
                append_jsonl(output_file, result.row)

    if ctx.partial_overwrite:
        _write_split_results(ctx=ctx, results=all_results)
    return len(rows)


def run_global_batches(
    *,
    split_contexts: Sequence[SplitRunContext],
    runner: VLLMBatchRunner,
    model_label: str,
    vllm_model: str,
    solve_params: Any,
    convert_params: Any,
    batch_size: int,
    convert_sympy: bool,
    sampling_adaptation: Optional[Dict[str, Any]],
) -> None:
    pending: List[SolverTask] = []
    for ctx in split_contexts:
        pending.extend(ctx.pending)

    if not pending:
        return

    results_by_split: Dict[str, List[SolverResult]] = {}
    contexts_by_split = {ctx.split_name: ctx for ctx in split_contexts}
    total_chunks = len(_chunks(pending, batch_size))
    for chunk_id, chunk in enumerate(_chunks(pending, batch_size), start=1):
        t0 = time.time()
        results = _solve_chunk(
            runner=runner,
            tasks=chunk,
            model_label=model_label,
            vllm_model=vllm_model,
            solve_params=solve_params,
            convert_params=convert_params,
            convert_sympy=convert_sympy,
            sampling_adaptation=sampling_adaptation,
        )
        elapsed = time.time() - t0
        for result in results:
            split_name = str(result.row.get("split", "") or "")
            if not split_name:
                continue
            print(result.msg)
            results_by_split.setdefault(split_name, []).append(result)
        touched = sorted(
            {
                str(result.row.get("split", "") or "")
                for result in results
                if str(result.row.get("split", "") or "")
            }
        )
        print(
            f"[generate-vllm] global batch {chunk_id}/{total_chunks}: "
            f"{len(results)} rows in {elapsed:.1f}s splits={','.join(touched)}"
        )

        for split_name, split_results in sorted(results_by_split.items()):
            ctx = contexts_by_split.get(split_name)
            if ctx is None or ctx.partial_overwrite:
                continue
            for result in split_results:
                append_jsonl(ctx.output_file, result.row)
            results_by_split[split_name] = []

    for ctx in split_contexts:
        if not ctx.partial_overwrite:
            continue
        _write_split_results(
            ctx=ctx,
            results=results_by_split.get(ctx.split_name, []),
        )


def _default_solver_model_alias(model_name: Optional[str]) -> str:
    raw = str(model_name or "").strip()
    if not raw:
        return ""
    normalized = raw.rstrip("/\\")
    if not normalized:
        return ""
    alias = normalized.replace("\\", "/").split("/")[-1].strip()
    return alias or normalized


def _resolve_model_names(cfg: Dict[str, Any], model: Optional[str], solver_model: Optional[str]) -> Tuple[str, str]:
    requested_model = str(model or "").strip()
    requested_label = str(solver_model or "").strip()
    default_model = get_default_solver_model(cfg)
    vllm_model = requested_model or requested_label or default_model
    model_label = requested_label or _default_solver_model_alias(vllm_model) or default_model
    if not vllm_model:
        raise RuntimeError("No vLLM model specified. Use --model or --solver-model.")
    if not model_label:
        model_label = vllm_model
    return vllm_model, model_label


def _dry_run_split(
    *,
    input_file: Path,
    output_file: Path,
    evaluation_file: Path,
    cfg: Dict[str, Any],
    model_label: str,
    question_id_filter: Optional[Set[str]],
    force: bool,
    prompt_preview_chars: int,
) -> int:
    _, rows, pending, _ = _collect_pending_tasks(
        input_file=input_file,
        output_file=output_file,
        evaluation_file=evaluation_file,
        cfg=cfg,
        model_label=model_label,
        resume_override=(False if force else None),
        question_id_filter=question_id_filter,
        mutate_output=False,
    )
    print(
        f"[dry-run] {input_file.stem}: matched={len(rows)} pending={len(pending)} "
        f"output={output_file}"
    )
    if pending and prompt_preview_chars > 0:
        preview = pending[0].prompt[:prompt_preview_chars]
        print(f"[dry-run] first prompt for {pending[0].qid}:\n{preview}")
    return len(rows)


def run(
    config_path: str = "config.yaml",
    model: Optional[str] = None,
    solver_model: Optional[str] = None,
    tokenizer: Optional[str] = None,
    split: Optional[str] = None,
    hf_json_dir: Optional[str] = None,
    by_model_dir: Optional[str] = None,
    question_ids: Optional[str] = None,
    question_ids_file: Optional[str] = None,
    skip_existing: bool = False,
    force: bool = False,
    batch_size: int = 16,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    convert_sympy: bool = True,
    convert_temperature: Optional[float] = None,
    convert_max_tokens: Optional[int] = None,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.80,
    max_model_len: Optional[int] = None,
    max_num_seqs: Optional[int] = 8,
    quantization: Optional[str] = None,
    load_format: Optional[str] = None,
    download_dir: Optional[str] = None,
    family: str = "auto",
    trust_remote_code: bool = False,
    enforce_eager: bool = False,
    stop: Optional[List[str]] = None,
    seed: Optional[int] = None,
    reasoning_mode: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    qwen3_thinking: str = "auto",
    no_tqdm: bool = False,
    dry_run: bool = False,
    global_batch: bool = False,
    prompt_preview_chars: int = 1200,
    workflow_id: Optional[str] = None,
    run_id: Optional[str] = None,
    manage_run_lifecycle: bool = True,
) -> None:
    if skip_existing and force:
        raise ValueError("--skip-existing and --force cannot be used together.")
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported --family={family}. Expected one of: {', '.join(sorted(SUPPORTED_FAMILIES))}")
    if reasoning_mode is not None and str(reasoning_mode).strip().lower() not in SUPPORTED_REASONING_MODES:
        raise ValueError(f"--reasoning-mode must be one of: {', '.join(sorted(SUPPORTED_REASONING_MODES))}")
    if reasoning_effort is not None and not str(reasoning_effort).strip():
        raise ValueError("--reasoning-effort cannot be empty.")
    if qwen3_thinking not in {"auto", "on", "off"}:
        raise ValueError("--qwen3-thinking must be one of: auto, on, off")

    cfg = apply_dataset_path_overrides(
        load_config(config_path),
        hf_json_dir=hf_json_dir,
        by_model_dir=by_model_dir,
    )
    paths = cfg["paths"]
    input_dir = Path(paths["hf_json_dir"])
    vllm_model, model_label = _resolve_model_names(cfg, model=model, solver_model=solver_model)
    explicit_effort = str(reasoning_effort or "").strip()
    if explicit_effort:
        validate_native_reasoning_effort(cfg, [vllm_model, model_label], explicit_effort)
    question_id_filter = load_question_id_filter(question_ids=question_ids, question_ids_file=question_ids_file)
    legacy_gen_dir = legacy_generations_dir(paths)
    generated_outputs: List[Path] = []

    input_files = split_input_files(input_dir, split=split)
    if not input_files:
        if split:
            raise RuntimeError(f"Split not found under {input_dir}: {split}")
        raise RuntimeError(f"No split jsonl files found under: {input_dir}")

    gen_cfg = cfg.get("generate", {}) or {}
    vllm_cfg = cfg.get("generate_vllm", {}) or {}
    convert_cfg = vllm_cfg.get("convert", {}) or {}
    detection_text = " ".join(part for part in [vllm_model, model_label] if str(part or "").strip())
    effective_family = _detect_model_family(detection_text) if family == "auto" else family
    (
        solve_temperature,
        solve_top_p,
        solve_top_k,
        solve_max_tokens,
        solve_repetition_penalty,
        sampling_adaptation,
    ) = _resolve_sampling_defaults(
        cfg=cfg,
        model_identifiers=[vllm_model, model_label],
        family=effective_family,
        reasoning_mode=reasoning_mode,
        reasoning_effort=reasoning_effort,
        qwen3_thinking=qwen3_thinking,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )
    artifact_model_label = build_solver_artifact_label(
        model_label,
        reasoning_effort=explicit_effort,
        max_solve_tokens=solve_max_tokens,
    )

    run_started = False
    if (not dry_run) and manage_run_lifecycle:
        started_run_id = start_run(
            cfg,
            stage="generate-vllm",
            command=" ".join(sys.argv),
            config_path=config_path,
            solver_model=artifact_model_label,
            workflow_id=workflow_id,
            run_id=run_id,
        )
        run_started = bool(started_run_id)
        if started_run_id:
            print(f"[cost] run_id={started_run_id}")

    run_status = "success"
    run_error = ""
    try:
        if dry_run:
            for input_file in input_files:
                split_name = input_file.stem
                output_file = solver_generation_file(paths, artifact_model_label, split_name)
                evaluation_file = resolve_rule_evaluation_input(paths, artifact_model_label, split_name)
                _dry_run_split(
                    input_file=input_file,
                    output_file=output_file,
                    evaluation_file=evaluation_file,
                    cfg=cfg,
                    model_label=artifact_model_label,
                    question_id_filter=question_id_filter,
                    force=force,
                    prompt_preview_chars=prompt_preview_chars,
                )
            return

        sympy_temperature = float(
            convert_temperature
            if convert_temperature is not None
            else convert_cfg.get("temperature", gen_cfg.get("convert_temperature", 0.0))
        )
        resolved_convert_max_tokens = int(
            convert_max_tokens
            if convert_max_tokens is not None
            else convert_cfg.get("max_tokens", gen_cfg.get("max_convert_tokens", 512))
        )

        runner = VLLMBatchRunner(
            model=vllm_model,
            model_hint=model_label,
            tokenizer=tokenizer,
            family=family,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            quantization=quantization,
            load_format=load_format,
            download_dir=download_dir,
            seed=seed,
            enforce_eager=enforce_eager,
            use_tqdm=not no_tqdm,
            qwen3_thinking=qwen3_thinking,
        )
        runner.default_chat_template_kwargs = dict(sampling_adaptation.get("chat_template_kwargs") or {})
        runner.qwen3_thinking = str(sampling_adaptation.get("effective_qwen3_thinking") or runner.qwen3_thinking)
        supports_native_reasoning_effort = bool(sampling_adaptation.get("supports_native_reasoning_effort"))
        log_parts = [
            f"family={runner.family}",
            f"capability={sampling_adaptation.get('capability_type')}",
            f"profile={sampling_adaptation.get('profile_name')}",
        ]
        reasoning_interface = sampling_adaptation.get("reasoning_interface")
        if reasoning_interface:
            log_parts.append(f"reasoning_interface={reasoning_interface}")
        effective_reasoning_mode = sampling_adaptation.get("effective_reasoning_mode")
        if effective_reasoning_mode is not None:
            log_parts.append(f"reasoning_mode={effective_reasoning_mode}")
        effective_reasoning_effort = sampling_adaptation.get("effective_reasoning_effort")
        if supports_native_reasoning_effort and effective_reasoning_effort is not None:
            log_parts.append(f"reasoning_effort={effective_reasoning_effort}")
        if runner.default_chat_template_kwargs:
            log_parts.append(f"chat_template_kwargs={runner.default_chat_template_kwargs}")
        if solve_temperature is not None:
            log_parts.append(f"temperature={solve_temperature}")
        if solve_top_p is not None:
            log_parts.append(f"top_p={solve_top_p}")
        if solve_top_k is not None:
            log_parts.append(f"top_k={solve_top_k}")
        if solve_max_tokens is not None:
            log_parts.append(f"max_tokens={solve_max_tokens}")
        if solve_repetition_penalty is not None:
            log_parts.append(f"repetition_penalty={solve_repetition_penalty}")
        print("[generate-vllm] sampling: " + " ".join(log_parts))
        solve_params = runner.sampling_params(
            temperature=solve_temperature,
            top_p=solve_top_p,
            top_k=solve_top_k,
            max_tokens=solve_max_tokens,
            repetition_penalty=solve_repetition_penalty,
            stop=stop,
            seed=seed,
        )
        convert_params = runner.sampling_params(
            temperature=sympy_temperature,
            top_p=1.0,
            top_k=None,
            max_tokens=resolved_convert_max_tokens,
            repetition_penalty=1.0,
            stop=stop,
            seed=seed,
        )

        if global_batch:
            split_contexts: List[SplitRunContext] = []
            for input_file in input_files:
                split_name = input_file.stem
                output_file = solver_generation_file(paths, artifact_model_label, split_name)
                legacy_output_file = resolve_existing_model_jsonl(
                    legacy_gen_dir,
                    split_name,
                    artifact_model_label,
                    allow_legacy_model_suffix=True,
                    allow_legacy_plain_split=True,
                )
                if not output_file.exists() and legacy_output_file.exists() and legacy_output_file != output_file:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(legacy_output_file, output_file)
                    print(f"[generate-vllm] migrated legacy generation -> {output_file}")
                if skip_existing and output_file.exists() and output_file.stat().st_size > 0:
                    print(f"[generate-vllm] skip {split_name}: existing output -> {output_file}")
                    generated_outputs.append(output_file)
                    continue

                evaluation_file = resolve_rule_evaluation_input(paths, artifact_model_label, split_name)
                ctx = _prepare_split_run_context(
                    input_file=input_file,
                    output_file=output_file,
                    evaluation_file=evaluation_file,
                    cfg=cfg,
                    model_label=artifact_model_label,
                    resume_override=(False if force else None),
                    question_id_filter=question_id_filter,
                )
                if not ctx.rows:
                    print(f"[generate-vllm] skip {split_name}: no rows matched question-id filter")
                    continue
                if not ctx.pending:
                    print(f"[generate-vllm] skip {split_name}: no pending rows")
                    if output_file.exists():
                        generated_outputs.append(output_file)
                        print(f"[generate-vllm] wrote -> {output_file}")
                    continue
                split_contexts.append(ctx)

            total_pending = sum(len(ctx.pending) for ctx in split_contexts)
            print(
                f"[generate-vllm] global batching enabled: "
                f"splits={len(split_contexts)} pending_rows={total_pending} batch_size={batch_size}"
            )
            run_global_batches(
                split_contexts=split_contexts,
                runner=runner,
                model_label=artifact_model_label,
                vllm_model=vllm_model,
                solve_params=solve_params,
                convert_params=convert_params,
                batch_size=batch_size,
                convert_sympy=convert_sympy,
                sampling_adaptation=sampling_adaptation,
            )
            for ctx in split_contexts:
                if not ctx.output_file.exists():
                    raise RuntimeError(
                        f"Generate produced no output file for split={ctx.split_name}, "
                        f"solver_model={model_label}, artifact_label={artifact_model_label}: {ctx.output_file}"
                    )
                generated_outputs.append(ctx.output_file)
                print(f"[generate-vllm] wrote -> {ctx.output_file}")
        else:
            for input_file in input_files:
                split_name = input_file.stem
                output_file = solver_generation_file(paths, artifact_model_label, split_name)
                legacy_output_file = resolve_existing_model_jsonl(
                    legacy_gen_dir,
                    split_name,
                    artifact_model_label,
                    allow_legacy_model_suffix=True,
                    allow_legacy_plain_split=True,
                )
                if not output_file.exists() and legacy_output_file.exists() and legacy_output_file != output_file:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(legacy_output_file, output_file)
                    print(f"[generate-vllm] migrated legacy generation -> {output_file}")
                if skip_existing and output_file.exists() and output_file.stat().st_size > 0:
                    print(f"[generate-vllm] skip {split_name}: existing output -> {output_file}")
                    generated_outputs.append(output_file)
                    continue

                evaluation_file = resolve_rule_evaluation_input(paths, artifact_model_label, split_name)
                matched_rows = run_for_split(
                    input_file=input_file,
                    output_file=output_file,
                    evaluation_file=evaluation_file,
                    cfg=cfg,
                    runner=runner,
                    model_label=artifact_model_label,
                    vllm_model=vllm_model,
                    solve_params=solve_params,
                    convert_params=convert_params,
                    batch_size=batch_size,
                    convert_sympy=convert_sympy,
                    sampling_adaptation=sampling_adaptation,
                    resume_override=(False if force else None),
                    question_id_filter=question_id_filter,
                )
                if matched_rows == 0:
                    print(f"[generate-vllm] skip {split_name}: no rows matched question-id filter")
                    continue
                if not output_file.exists():
                    raise RuntimeError(
                        f"Generate produced no output file for split={split_name}, "
                        f"solver_model={model_label}, artifact_label={artifact_model_label}: {output_file}"
                    )
                generated_outputs.append(output_file)
                print(f"[generate-vllm] wrote -> {output_file}")

        register_run_outputs(generated_outputs)
    except Exception as exc:
        run_status = "error"
        run_error = f"{exc.__class__.__name__}: {exc}"
        raise
    finally:
        if run_started and manage_run_lifecycle:
            finish_run(status=run_status, error_message=run_error)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate boxed final answers with offline vLLM batch inference. "
            "This does not start an OpenAI-compatible vLLM server."
        )
    )
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model",
        default=None,
        help="Hugging Face model id or local model path loaded by vLLM.",
    )
    parser.add_argument(
        "--solver-model",
        default=None,
        help="Output/model label used under data/by_model. Defaults to a short alias derived from --model.",
    )
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer id/path. Defaults to --model.")
    parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    parser.add_argument("--hf-json-dir", default=None, help="Optional input dataset directory override.")
    parser.add_argument("--by-model-dir", default=None, help="Optional output root override.")
    parser.add_argument("--question-ids", default=None, help="Optional comma/whitespace separated question ids.")
    parser.add_argument("--question-ids-file", default=None, help="Optional text file containing question ids.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip split if output already exists.")
    parser.add_argument("--force", action="store_true", help="Force regenerate all rows for each split.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of prompts sent to vLLM per chunk.")
    parser.add_argument("--temperature", type=float, default=None, help="Solve sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Solve top-p sampling. Leave unset for model-specific auto defaults.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional solve top-k sampling.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Solve max generation tokens.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Solve repetition penalty. Leave unset for model-specific auto defaults.")
    parser.add_argument(
        "--no-convert-sympy",
        action="store_false",
        dest="convert_sympy",
        help="Do not run the second offline batch that normalizes boxed answers to SymPy.",
    )
    parser.add_argument("--convert-temperature", type=float, default=None, help="SymPy conversion temperature.")
    parser.add_argument("--convert-max-tokens", type=int, default=None, help="SymPy conversion max tokens.")
    parser.add_argument("--dtype", default="bfloat16", help="vLLM dtype, e.g. auto, bfloat16, float16.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80, help="vLLM GPU memory utilization.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Optional vLLM max model length.")
    parser.add_argument("--max-num-seqs", type=int, default=8, help="Optional vLLM max active sequences.")
    parser.add_argument("--quantization", default=None, help="Optional vLLM quantization, e.g. fp8, awq, gptq.")
    parser.add_argument("--load-format", default=None, help="Optional vLLM load format.")
    parser.add_argument("--download-dir", default=None, help="Optional Hugging Face download/cache directory.")
    parser.add_argument(
        "--family",
        default="auto",
        choices=sorted(SUPPORTED_FAMILIES),
        help="Chat-template family hint for Llama3/Qwen3/DeepSeek/Kimi checkpoints.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to vLLM for checkpoints that require custom code.",
    )
    parser.add_argument("--enforce-eager", action="store_true", help="Pass enforce_eager=True to vLLM.")
    parser.add_argument("--stop", action="append", default=None, help="Optional stop string. Can be repeated.")
    parser.add_argument("--seed", type=int, default=None, help="Optional vLLM/SamplingParams seed.")
    parser.add_argument(
        "--reasoning-mode",
        default=None,
        choices=sorted(SUPPORTED_REASONING_MODES),
        help=(
            "Unified reasoning switch for vLLM reasoning families. "
            "Some families support a hard on/off switch via chat template kwargs; "
            "parser-only families will reject --reasoning-mode off."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Model-native reasoning effort value. "
            "If the target model does not expose a native reasoning-effort field, the command will stop with an error."
        ),
    )
    parser.add_argument(
        "--qwen3-thinking",
        default="auto",
        choices=["auto", "on", "off"],
        help="Low-level override for Qwen3 thinking mode. Usually leave this on auto and use --reasoning-effort instead.",
    )
    parser.add_argument("--no-tqdm", action="store_true", help="Disable vLLM progress bars.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect pending rows and first prompt without loading vLLM.")
    parser.add_argument(
        "--global-batch",
        action="store_true",
        help="Batch pending rows across all selected splits while preserving per-split output files.",
    )
    parser.add_argument(
        "--prompt-preview-chars",
        type=int,
        default=1200,
        help="Characters of first dry-run prompt to print.",
    )
    parser.add_argument("--workflow-id", default=None, help="Optional workflow id for cost/workflow metadata.")
    parser.add_argument("--run-id", default=None, help="Optional run id for cost/workflow metadata.")

    args = parser.parse_args()
    run(
        config_path=args.config,
        model=args.model,
        solver_model=args.solver_model,
        tokenizer=args.tokenizer,
        split=args.split,
        hf_json_dir=args.hf_json_dir,
        by_model_dir=args.by_model_dir,
        question_ids=args.question_ids,
        question_ids_file=args.question_ids_file,
        skip_existing=args.skip_existing,
        force=args.force,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        convert_sympy=args.convert_sympy,
        convert_temperature=args.convert_temperature,
        convert_max_tokens=args.convert_max_tokens,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        quantization=args.quantization,
        load_format=args.load_format,
        download_dir=args.download_dir,
        family=args.family,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        stop=args.stop,
        seed=args.seed,
        reasoning_mode=args.reasoning_mode,
        reasoning_effort=args.reasoning_effort,
        qwen3_thinking=args.qwen3_thinking,
        no_tqdm=args.no_tqdm,
        dry_run=args.dry_run,
        global_batch=args.global_batch,
        prompt_preview_chars=args.prompt_preview_chars,
        workflow_id=args.workflow_id,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
