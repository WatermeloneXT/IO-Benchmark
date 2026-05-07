from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from core.cost_logging import get_run_context, log_api_call


def _is_claude_model(model: str) -> bool:
    name = str(model or "").strip().lower()
    return name.startswith("claude") or ("claude-" in name)


def _normalize_model_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for item in values:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _is_foundry_openai_model(cfg: Dict[str, Any], model_name: str) -> bool:
    foundry_cfg = cfg.get("foundry_openai", {}) or {}
    configured_models = _normalize_model_list(foundry_cfg.get("models", []))
    return str(model_name or "").strip() in configured_models


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
        text = str(value or "").strip().lower()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def resolve_generate_reasoning_request(
    cfg: Dict[str, Any],
    model: str,
    *,
    phase: str = "solve",
    requested_mode: Optional[str] = None,
    requested_effort: Optional[str] = None,
) -> Dict[str, Any]:
    gen_cfg = cfg.get("generate", {}) or {}
    defaults = gen_cfg.get("reasoning_defaults", {}) or {}
    profiles = gen_cfg.get("model_profiles", {}) or {}

    requested_mode_value = (
        requested_mode
        if requested_mode is not None
        else defaults.get(f"{phase}_mode", defaults.get("mode", "auto"))
    )
    requested_mode = str(requested_mode_value or "auto").strip().lower()
    if requested_mode not in {"auto", "on", "off"}:
        requested_mode = "auto"

    profile_name, profile_cfg, matched_identifier = _resolve_model_profile([model], profiles)
    if not profile_name:
        requested_effort_value = str(requested_effort or "").strip().lower() or None
        if requested_mode is not None and str(requested_mode).strip().lower() in {"on", "off"}:
            raise ValueError(
                f"Model '{model}' does not have a configured API reasoning profile, "
                "so explicit --reasoning-mode is not supported for it."
            )
        if requested_effort_value is not None:
            raise ValueError(
                f"Model '{model}' does not have a configured API reasoning profile, "
                "so explicit --reasoning-effort is not supported for it."
            )
        return {
            "phase": phase,
            "profile_name": None,
            "matched_identifier": None,
            "reasoning_interface": None,
            "requested_mode": requested_mode,
            "effective_mode": None,
            "request_kwargs": {},
            "omit_temperature": False,
            "doc_url": None,
            "note": None,
        }

    reasoning_interface = str(profile_cfg.get("reasoning_interface", "") or "").strip().lower()
    default_mode = str(profile_cfg.get("reasoning_default_mode", "auto") or "auto").strip().lower()
    if default_mode not in {"auto", "on", "off"}:
        default_mode = "auto"
    effective_mode = default_mode if requested_mode == "auto" else requested_mode
    if effective_mode == "auto":
        effective_mode = None

    request_kwargs: Dict[str, Any] = {}
    omit_temperature = False
    note = profile_cfg.get("note") if isinstance(profile_cfg, dict) else None
    requested_effort_value = str(requested_effort or "").strip().lower() or None
    native_reasoning_efforts = _native_reasoning_efforts(profile_cfg)

    if requested_effort_value is not None:
        if not native_reasoning_efforts:
            profile_label = profile_name or matched_identifier or model or "this model"
            raise ValueError(
                f"Model '{profile_label}' does not expose a native reasoning effort field. "
                "Remove the explicit reasoning effort for this API model."
            )
        if requested_effort_value not in native_reasoning_efforts:
            profile_label = profile_name or matched_identifier or model or "this model"
            raise ValueError(
                f"Model '{profile_label}' supports reasoning effort values: "
                f"{', '.join(native_reasoning_efforts)}. Received: {requested_effort_value}"
            )
        if effective_mode != "on":
            profile_label = profile_name or matched_identifier or model or "this model"
            raise ValueError(
                f"Model '{profile_label}' received reasoning_effort={requested_effort_value} "
                f"while reasoning mode is {effective_mode or 'provider-default'}. "
                "Explicit reasoning effort is only valid when reasoning mode is on."
            )

    if reasoning_interface == "openai_reasoning_effort":
        if effective_mode == "on":
            resolved_effort = str(
                requested_effort_value
                or profile_cfg.get("reasoning_on_effort", defaults.get("openai_effort", "medium"))
                or "medium"
            ).strip().lower()
            if native_reasoning_efforts and resolved_effort not in native_reasoning_efforts:
                profile_label = profile_name or matched_identifier or model or "this model"
                raise ValueError(
                    f"Configured reasoning effort for model '{profile_label}' must be one of: "
                    f"{', '.join(native_reasoning_efforts)}. Received: {resolved_effort}"
                )
            request_kwargs["reasoning_effort"] = resolved_effort
            omit_temperature = request_kwargs["reasoning_effort"] != "none"
        elif effective_mode == "off":
            off_effort = str(profile_cfg.get("reasoning_off_effort", "") or "").strip().lower()
            if not off_effort:
                effective_mode = None
                note = (
                    f"{note} Requested reasoning off is not documented for this model family; "
                    "falling back to the provider default behavior."
                ).strip() if note else (
                    "Requested reasoning off is not documented for this model family; "
                    "falling back to the provider default behavior."
                )
            else:
                if native_reasoning_efforts and off_effort not in native_reasoning_efforts:
                    profile_label = profile_name or matched_identifier or model or "this model"
                    raise ValueError(
                        f"Configured reasoning off effort for model '{profile_label}' must be one of: "
                        f"{', '.join(native_reasoning_efforts)}. Received: {off_effort}"
                    )
                request_kwargs["reasoning_effort"] = off_effort
                omit_temperature = off_effort != "none"
    elif reasoning_interface == "anthropic_manual_thinking":
        if effective_mode == "on":
            budget_tokens = int(profile_cfg.get("thinking_budget_tokens", defaults.get("claude_budget_tokens", 2048)) or 2048)
            thinking_obj: Dict[str, Any] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }
            thinking_display = str(
                profile_cfg.get("thinking_display", defaults.get("claude_thinking_display", "")) or ""
            ).strip().lower()
            if thinking_display in {"summarized", "omitted"}:
                thinking_obj["display"] = thinking_display
            request_kwargs["thinking"] = thinking_obj
            omit_temperature = True
        elif effective_mode == "off":
            request_kwargs = {}
            omit_temperature = bool(profile_cfg.get("omit_temperature_when_off", False))
    elif reasoning_interface == "anthropic_adaptive_thinking":
        omit_temperature = bool(profile_cfg.get("omit_temperature_when_off", False))
        if effective_mode == "on":
            resolved_effort = str(
                requested_effort_value
                or profile_cfg.get("adaptive_effort", defaults.get("claude_effort", "high"))
                or "high"
            ).strip().lower()
            if native_reasoning_efforts and resolved_effort not in native_reasoning_efforts:
                profile_label = profile_name or matched_identifier or model or "this model"
                raise ValueError(
                    f"Configured reasoning effort for model '{profile_label}' must be one of: "
                    f"{', '.join(native_reasoning_efforts)}. Received: {resolved_effort}"
                )
            thinking_obj = {"type": "adaptive"}
            thinking_display = str(
                profile_cfg.get("thinking_display", defaults.get("claude_thinking_display", "")) or ""
            ).strip().lower()
            if thinking_display in {"summarized", "omitted"}:
                thinking_obj["display"] = thinking_display
            request_kwargs["thinking"] = thinking_obj
            request_kwargs["output_config"] = {"effort": resolved_effort}
            omit_temperature = True

    return {
        "phase": phase,
        "profile_name": profile_name,
        "matched_identifier": matched_identifier,
        "reasoning_interface": reasoning_interface or None,
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "request_kwargs": request_kwargs,
        "omit_temperature": omit_temperature,
        "doc_url": profile_cfg.get("doc_url") if isinstance(profile_cfg, dict) else None,
        "note": note,
    }


def strip_control_chars(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text or "")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _obj_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    for attr in ("model_dump", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, dict):
                    return out
            except Exception:  # noqa: BLE001
                pass
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict):
        return dict(data)
    return {}


def _extract_usage(resp: Any) -> Dict[str, Any]:
    usage_obj = getattr(resp, "usage", None)
    usage_dict = _obj_to_dict(usage_obj)
    if not usage_dict and isinstance(resp, dict):
        usage_dict = _obj_to_dict(resp.get("usage"))

    input_tokens = int(
        usage_dict.get("input_tokens")
        or usage_dict.get("prompt_tokens")
        or 0
    )
    output_tokens = int(
        usage_dict.get("output_tokens")
        or usage_dict.get("completion_tokens")
        or 0
    )
    cache_creation_tokens = int(
        usage_dict.get("cache_creation_input_tokens")
        or ((usage_dict.get("cache_creation") or {}).get("ephemeral_5m_input_tokens") or 0)
        or 0
    )
    cache_read_tokens = int(
        usage_dict.get("cache_read_input_tokens")
        or 0
    )
    service_tier = str(
        usage_dict.get("service_tier")
        or getattr(resp, "service_tier", "")
        or ""
    )
    return {
        "usage_input_tokens": input_tokens,
        "usage_output_tokens": output_tokens,
        "usage_cache_creation_tokens": cache_creation_tokens,
        "usage_cache_read_tokens": cache_read_tokens,
        "service_tier": service_tier,
        "raw_usage_json": usage_dict,
    }


def _extract_error_fields(err: Exception) -> Dict[str, str]:
    code = str(getattr(err, "code", "") or "")
    message = str(getattr(err, "message", "") or str(err))
    status_code = str(getattr(err, "status_code", "") or "")
    if not code:
        body = getattr(err, "body", None)
        if isinstance(body, dict):
            code = str(((body.get("error") or {}).get("code")) or "")
            if not message:
                message = str(((body.get("error") or {}).get("message")) or message)
    return {
        "error_code": code,
        "error_message": message,
        "http_status": status_code,
    }


def _build_log_base(
    *,
    client: Any,
    model_requested: str,
    api_type: str,
    attempt: int,
    telemetry: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    t = dict(telemetry or {})
    run_ctx = get_run_context()
    return {
        "timestamp_utc": _now_utc_iso(),
        "run_id": run_ctx.get("run_id", ""),
        "workflow_id": run_ctx.get("workflow_id", ""),
        "stage": str(t.get("stage") or run_ctx.get("stage") or "unknown"),
        "operation": str(t.get("operation", "") or "unknown"),
        "split": str(t.get("split", "") or ""),
        "question_id": str(t.get("question_id", "") or t.get("id", "") or ""),
        "provider": str(
            t.get("provider", "")
            or getattr(client, "_econ_provider", "")
            or "azure_openai"
        ),
        "api_type": api_type,
        "endpoint_or_region": str(
            t.get("endpoint_or_region", "")
            or getattr(client, "_econ_endpoint", "")
            or ""
        ),
        "model_requested": model_requested,
        "attempt": attempt,
    }


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    azure = cfg.setdefault("azure", {})
    if not azure.get("subscription_key"):
        azure["subscription_key"] = os.environ.get("AZURE_OPENAI_KEY", "")
    if not azure.get("endpoint"):
        azure["endpoint"] = os.environ.get("AZURE_OPENAI_ENDPOINT", "")

    anthropic_cfg = cfg.setdefault("anthropic", {})
    if not anthropic_cfg.get("api_key"):
        anthropic_cfg["api_key"] = (
            os.environ.get("ANTHROPIC_FOUNDRY_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
    if not anthropic_cfg.get("base_url"):
        anthropic_cfg["base_url"] = (
            os.environ.get("ANTHROPIC_FOUNDRY_BASE_URL", "")
            or os.environ.get("ANTHROPIC_BASE_URL", "")
        )
    if "max_retries" not in anthropic_cfg:
        anthropic_cfg["max_retries"] = int(azure.get("max_retries", 3) or 3)

    foundry_openai_cfg = cfg.setdefault("foundry_openai", {})
    if not foundry_openai_cfg.get("api_key"):
        foundry_openai_cfg["api_key"] = (
            os.environ.get("FOUNDRY_OPENAI_API_KEY", "")
            or os.environ.get("AZURE_INFERENCE_CREDENTIAL", "")
            or azure.get("subscription_key", "")
        )
    if not foundry_openai_cfg.get("base_url"):
        foundry_openai_cfg["base_url"] = (
            os.environ.get("FOUNDRY_OPENAI_BASE_URL", "")
            or os.environ.get("AZURE_OPENAI_BASE_URL", "")
        )
    foundry_openai_cfg["models"] = _normalize_model_list(foundry_openai_cfg.get("models", []))

    paths = cfg.setdefault("paths", {})
    base = cfg_path.parent
    path_defaults = {
        "chapters_root": "datasets/chapters",
        "hf_json_dir": "data/hf_json",
        "hf_dataset_dir": "data/hf_dataset",
        "balanced_review_dir": "data/review_balanced",
        "review_annotations_dir": "data/review_annotations",
        "pipeline_dir": "data/pipeline",
        "generations_dir": "data/generations",  # legacy fallback only
        "evaluations_dir": "data/evaluations",  # legacy fallback only
        "evaluations_llm_dir": "data/evaluations_llm",  # legacy fallback only
        "by_model_dir": "data/by_model",
        "workflows_dir": "data/workflows",
    }
    for key, default_value in path_defaults.items():
        if not str(paths.get(key, "") or "").strip():
            paths[key] = default_value
    for key, value in list(paths.items()):
        p = Path(str(value))
        if not p.is_absolute():
            paths[key] = str((base / p).resolve())

    generate_cfg = cfg.setdefault("generate", {})
    evaluate_llm_cfg = cfg.setdefault("evaluate_llm", {})
    models_cfg = cfg.setdefault("models", {})

    default_solver = str(models_cfg.get("default_solver_model", "")).strip()
    if not default_solver:
        default_solver = (
            str(generate_cfg.get("deployment_name", "")).strip()
            or str(azure.get("deployment_name", "")).strip()
        )
    if default_solver:
        models_cfg["default_solver_model"] = default_solver

    allowed = models_cfg.get("allowed_solver_models", [])
    if not isinstance(allowed, list):
        allowed = []
    normalized_allowed: List[str] = []
    seen: set[str] = set()
    for item in allowed:
        candidate = str(item or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized_allowed.append(candidate)
    if not normalized_allowed and default_solver:
        normalized_allowed = [default_solver]
    models_cfg["allowed_solver_models"] = normalized_allowed

    if not evaluate_llm_cfg.get("judge_model"):
        evaluate_llm_cfg["judge_model"] = ""

    cost_cfg = cfg.setdefault("cost_logging", {})
    if "enabled" not in cost_cfg:
        cost_cfg["enabled"] = True
    if not cost_cfg.get("root_dir"):
        cost_cfg["root_dir"] = str(paths.get("by_model_dir", "data/by_model"))
    if "capture_raw_usage" not in cost_cfg:
        cost_cfg["capture_raw_usage"] = True
    root_dir = Path(str(cost_cfg.get("root_dir", paths.get("by_model_dir", "data/by_model"))))
    if not root_dir.is_absolute():
        cost_cfg["root_dir"] = str((base / root_dir).resolve())
    else:
        cost_cfg["root_dir"] = str(root_dir)

    cfg["_config_path"] = str(cfg_path)
    return cfg


def get_allowed_solver_models(cfg: Dict[str, Any]) -> List[str]:
    models_cfg = cfg.get("models", {}) or {}
    raw = models_cfg.get("allowed_solver_models", []) or []
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        candidate = str(item or "").strip()
        if candidate:
            out.append(candidate)
    return out


def get_default_solver_model(cfg: Dict[str, Any]) -> str:
    models_cfg = cfg.get("models", {}) or {}
    default_solver = str(models_cfg.get("default_solver_model", "")).strip()
    if default_solver:
        return default_solver
    return (
        str((cfg.get("generate") or {}).get("deployment_name", "")).strip()
        or str((cfg.get("azure") or {}).get("deployment_name", "")).strip()
    )


def validate_solver_model(cfg: Dict[str, Any], solver_model: str) -> None:
    allowed = get_allowed_solver_models(cfg)
    if not allowed:
        return
    if solver_model not in allowed:
        allowed_text = ", ".join(allowed)
        raise RuntimeError(
            f"Unsupported solver model: {solver_model}. Allowed solver models: {allowed_text}"
        )


def resolve_solver_model(cfg: Dict[str, Any], requested_model: Optional[str] = None) -> str:
    model_name = (requested_model or "").strip() or get_default_solver_model(cfg)
    if not model_name:
        raise RuntimeError(
            "No solver model specified. Set --solver-model or config models.default_solver_model."
        )
    validate_solver_model(cfg, model_name)
    return model_name


def resolve_judge_model(cfg: Dict[str, Any], requested_model: Optional[str] = None) -> str:
    model_name = (
        (requested_model or "").strip()
        or str((cfg.get("evaluate_llm") or {}).get("judge_model", "")).strip()
        or str((cfg.get("generate") or {}).get("deployment_name", "")).strip()
        or str((cfg.get("azure") or {}).get("deployment_name", "")).strip()
    )
    if not model_name:
        raise RuntimeError(
            "No judge model specified. Set --judge-model or config evaluate_llm.judge_model."
        )
    return model_name


def create_client(cfg: Dict[str, Any], model_name: Optional[str] = None) -> Any:
    requested_model = (
        str(model_name or "").strip()
        or str((cfg.get("generate") or {}).get("deployment_name", "")).strip()
        or str((cfg.get("azure") or {}).get("deployment_name", "")).strip()
    )
    if _is_claude_model(requested_model):
        from anthropic import AnthropicFoundry

        az = cfg.get("azure", {}) or {}
        anthropic_cfg = cfg.get("anthropic", {}) or {}
        base_url = str(
            anthropic_cfg.get("base_url", "")
            or az.get("endpoint", "")
            or ""
        ).strip()
        api_key = str(
            anthropic_cfg.get("api_key", "")
            or az.get("subscription_key", "")
            or ""
        ).strip()

        assert base_url, (
            "Anthropic base_url not set. Configure anthropic.base_url "
            "(or ANTHROPIC_FOUNDRY_BASE_URL), or reuse azure.endpoint."
        )
        assert api_key, (
            "Anthropic api_key not set. Configure anthropic.api_key "
            "(or ANTHROPIC_FOUNDRY_API_KEY), or reuse azure.subscription_key."
        )

        client = AnthropicFoundry(
            api_key=api_key,
            base_url=base_url,
        )
        setattr(client, "_econ_endpoint", base_url)
        setattr(client, "_econ_provider", "anthropic_foundry")
        return client

    if _is_foundry_openai_model(cfg, requested_model):
        from openai import OpenAI

        foundry_cfg = cfg.get("foundry_openai", {}) or {}
        base_url = str(foundry_cfg.get("base_url", "") or "").strip()
        api_key = str(foundry_cfg.get("api_key", "") or "").strip()

        assert base_url, (
            "Foundry OpenAI-compatible base_url not set. Configure foundry_openai.base_url "
            "(or FOUNDRY_OPENAI_BASE_URL), for example "
            "https://<resource>.services.ai.azure.com/openai/v1/."
        )
        assert api_key, (
            "Foundry OpenAI-compatible api_key not set. Configure foundry_openai.api_key "
            "(or FOUNDRY_OPENAI_API_KEY / AZURE_INFERENCE_CREDENTIAL)."
        )

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        setattr(client, "_econ_endpoint", base_url)
        setattr(client, "_econ_provider", "foundry_openai")
        return client

    from openai import AzureOpenAI

    az = cfg["azure"]
    assert az.get("endpoint"), "Azure endpoint not set (config.yaml or AZURE_OPENAI_ENDPOINT)."
    assert az.get("subscription_key"), "Azure key not set (config.yaml or AZURE_OPENAI_KEY)."
    client = AzureOpenAI(
        api_key=az["subscription_key"],
        azure_endpoint=az["endpoint"],
        api_version=az["api_version"],
    )
    setattr(client, "_econ_endpoint", str(az.get("endpoint", "")))
    setattr(client, "_econ_provider", "azure_openai")
    return client


def _anthropic_text_content(resp: Any) -> str:
    content = getattr(resp, "content", None)
    if not isinstance(content, list):
        return strip_control_chars(str(content or "")).strip()

    parts: List[str] = []
    for block in content:
        if isinstance(block, dict):
            if str(block.get("type", "")).strip() == "text":
                text = str(block.get("text", "") or "")
                if text:
                    parts.append(text)
            continue
        if str(getattr(block, "type", "") or "").strip() == "text":
            text = str(getattr(block, "text", "") or "")
            if text:
                parts.append(text)
    return strip_control_chars("\n".join(parts)).strip()


def _anthropic_messages_supports_stream(client: Any) -> bool:
    messages_api = getattr(client, "messages", None)
    return callable(getattr(messages_api, "stream", None))


def _anthropic_messages_text_call(
    client: Any,
    *,
    request_kwargs: Dict[str, Any],
) -> Tuple[str, Any, str]:
    """Return text content, final response/message object, and request id.

    Prefer streaming for compatibility with long-running Anthropic Foundry
    requests; fall back to messages.create() only when the SDK does not expose
    streaming helpers.
    """
    if _anthropic_messages_supports_stream(client):
        with client.messages.stream(**request_kwargs) as stream:
            final_message = stream.get_final_message()
            request_id = str(getattr(stream, "request_id", "") or "")
            return _anthropic_text_content(final_message), final_message, request_id

    resp = client.messages.create(**request_kwargs)
    request_id = str(getattr(resp, "_request_id", "") or "")
    return _anthropic_text_content(resp), resp, request_id


def _strip_code_fence(text: str) -> str:
    candidate = str(text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    return candidate.strip()


def azure_json_call(
    client: Any,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_retries: int = 4,
    telemetry: Optional[Dict[str, Any]] = None,
    reasoning_request: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    reasoning_request = dict(reasoning_request or {})
    omit_temperature = bool(reasoning_request.pop("omit_temperature", False))
    if str(getattr(client, "_econ_provider", "") or "") == "anthropic_foundry":
        system_prompt = system.rstrip() + "\n\nReturn valid JSON only."
        for attempt in range(1, max_retries + 1):
            t0 = time.time()
            base = _build_log_base(
                client=client,
                model_requested=model,
                api_type="messages.create.json",
                attempt=attempt,
                telemetry=telemetry,
            )
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": model,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": 2048,
                }
                if not omit_temperature:
                    request_kwargs["temperature"] = temperature
                request_kwargs.update(reasoning_request)
                text_content, final_resp, request_id = _anthropic_messages_text_call(
                    client,
                    request_kwargs=request_kwargs,
                )
                content = _strip_code_fence(text_content) or "{}"
                out = json.loads(content)
                usage_fields = _extract_usage(final_resp)
                base.update(
                    {
                        "status": "success",
                        "latency_ms": int((time.time() - t0) * 1000),
                        "request_id": request_id,
                        "model_returned": str(getattr(final_resp, "model", "") or model),
                        **usage_fields,
                    }
                )
                log_api_call(base)
                return out
            except Exception as err:  # noqa: BLE001
                last_err = err
                base.update(
                    {
                        "status": "error",
                        "latency_ms": int((time.time() - t0) * 1000),
                        "request_id": str(getattr(err, "request_id", "") or ""),
                        "model_returned": "",
                        "usage_input_tokens": 0,
                        "usage_output_tokens": 0,
                        "usage_cache_creation_tokens": 0,
                        "usage_cache_read_tokens": 0,
                        "service_tier": "",
                        "raw_usage_json": {},
                        **_extract_error_fields(err),
                    }
                )
                log_api_call(base)
                time.sleep(min(10.0, 1.7**attempt))
        raise RuntimeError(f"azure_json_call failed: {last_err}")

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        base = _build_log_base(
            client=client,
            model_requested=model,
            api_type="chat.completions.json",
            attempt=attempt,
            telemetry=telemetry,
        )
        try:
            request_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
            }
            if "reasoning_effort" in reasoning_request:
                request_kwargs["max_completion_tokens"] = max_tokens
            else:
                request_kwargs["max_tokens"] = max_tokens
            if not omit_temperature:
                request_kwargs["temperature"] = temperature
            request_kwargs.update(reasoning_request)
            resp = client.chat.completions.create(**request_kwargs)
            content = resp.choices[0].message.content or "{}"
            out = json.loads(content)
            usage_fields = _extract_usage(resp)
            base.update(
                {
                    "status": "success",
                    "latency_ms": int((time.time() - t0) * 1000),
                    "request_id": str(getattr(resp, "_request_id", "") or ""),
                    "model_returned": str(getattr(resp, "model", "") or model),
                    **usage_fields,
                }
            )
            log_api_call(base)
            return out
        except Exception as err:  # noqa: BLE001
            last_err = err
            base.update(
                {
                    "status": "error",
                    "latency_ms": int((time.time() - t0) * 1000),
                    "request_id": str(getattr(err, "request_id", "") or ""),
                    "model_returned": "",
                    "usage_input_tokens": 0,
                    "usage_output_tokens": 0,
                    "usage_cache_creation_tokens": 0,
                    "usage_cache_read_tokens": 0,
                    "service_tier": "",
                    "raw_usage_json": {},
                    **_extract_error_fields(err),
                }
            )
            log_api_call(base)
            time.sleep(min(10.0, 1.7**attempt))
    raise RuntimeError(f"azure_json_call failed: {last_err}")


def azure_chat_call(
    client: Any,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_retries: int = 4,
    telemetry: Optional[Dict[str, Any]] = None,
    reasoning_request: Optional[Dict[str, Any]] = None,
) -> str:
    last_err: Optional[Exception] = None
    reasoning_request = dict(reasoning_request or {})
    omit_temperature = bool(reasoning_request.pop("omit_temperature", False))
    if str(getattr(client, "_econ_provider", "") or "") == "anthropic_foundry":
        for attempt in range(1, max_retries + 1):
            t0 = time.time()
            base = _build_log_base(
                client=client,
                model_requested=model,
                api_type="messages.create",
                attempt=attempt,
                telemetry=telemetry,
            )
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": model,
                    "system": system,
                    "messages": [
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                }
                if not omit_temperature:
                    request_kwargs["temperature"] = temperature
                request_kwargs.update(reasoning_request)
                out, final_resp, request_id = _anthropic_messages_text_call(
                    client,
                    request_kwargs=request_kwargs,
                )
                usage_fields = _extract_usage(final_resp)
                base.update(
                    {
                        "status": "success",
                        "latency_ms": int((time.time() - t0) * 1000),
                        "request_id": request_id,
                        "model_returned": str(getattr(final_resp, "model", "") or model),
                        **usage_fields,
                    }
                )
                log_api_call(base)
                return out
            except Exception as err:  # noqa: BLE001
                last_err = err
                base.update(
                    {
                        "status": "error",
                        "latency_ms": int((time.time() - t0) * 1000),
                        "request_id": str(getattr(err, "request_id", "") or ""),
                        "model_returned": "",
                        "usage_input_tokens": 0,
                        "usage_output_tokens": 0,
                        "usage_cache_creation_tokens": 0,
                        "usage_cache_read_tokens": 0,
                        "service_tier": "",
                        "raw_usage_json": {},
                        **_extract_error_fields(err),
                    }
                )
                log_api_call(base)
                time.sleep(min(10.0, 1.7**attempt))
        raise RuntimeError(f"azure_chat_call failed: {last_err}")

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        base = _build_log_base(
            client=client,
            model_requested=model,
            api_type="chat.completions",
            attempt=attempt,
            telemetry=telemetry,
        )
        try:
            request_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if "reasoning_effort" in reasoning_request:
                request_kwargs["max_completion_tokens"] = max_tokens
            else:
                request_kwargs["max_tokens"] = max_tokens
            if not omit_temperature:
                request_kwargs["temperature"] = temperature
            request_kwargs.update(reasoning_request)
            resp = client.chat.completions.create(**request_kwargs)
            out = strip_control_chars(resp.choices[0].message.content or "").strip()
            usage_fields = _extract_usage(resp)
            base.update(
                {
                    "status": "success",
                    "latency_ms": int((time.time() - t0) * 1000),
                    "request_id": str(getattr(resp, "_request_id", "") or ""),
                    "model_returned": str(getattr(resp, "model", "") or model),
                    **usage_fields,
                }
            )
            log_api_call(base)
            return out
        except Exception as err:  # noqa: BLE001
            last_err = err
            base.update(
                {
                    "status": "error",
                    "latency_ms": int((time.time() - t0) * 1000),
                    "request_id": str(getattr(err, "request_id", "") or ""),
                    "model_returned": "",
                    "usage_input_tokens": 0,
                    "usage_output_tokens": 0,
                    "usage_cache_creation_tokens": 0,
                    "usage_cache_read_tokens": 0,
                    "service_tier": "",
                    "raw_usage_json": {},
                    **_extract_error_fields(err),
                }
            )
            log_api_call(base)
            time.sleep(min(10.0, 1.7**attempt))
    raise RuntimeError(f"azure_chat_call failed: {last_err}")
