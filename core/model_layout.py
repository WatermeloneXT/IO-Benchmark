from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from core.file_naming import resolve_existing_model_jsonl, sanitize_model_tag


def _path(value: Any, default: str) -> Path:
    text = str(value or "").strip()
    return Path(text or default)


def by_model_root(paths: Dict[str, Any]) -> Path:
    raw = str(paths.get("by_model_dir", "") or "").strip()
    if raw:
        return Path(raw)
    hf_json_dir = _path(paths.get("hf_json_dir"), "data/hf_json")
    return hf_json_dir.parent / "by_model"


def workflows_root(paths: Dict[str, Any]) -> Path:
    raw = str(paths.get("workflows_dir", "") or "").strip()
    if raw:
        return Path(raw)
    return by_model_root(paths).parent / "workflows"


def solver_root(paths: Dict[str, Any], solver_model: str) -> Path:
    return by_model_root(paths) / sanitize_model_tag(solver_model)


def solver_generations_dir(paths: Dict[str, Any], solver_model: str) -> Path:
    return solver_root(paths, solver_model) / "generations"


def solver_rule_evaluations_dir(paths: Dict[str, Any], solver_model: str) -> Path:
    return solver_root(paths, solver_model) / "evaluations" / "rule"


def solver_llm_evaluations_dir(paths: Dict[str, Any], solver_model: str, judge_model: str) -> Path:
    return solver_root(paths, solver_model) / "evaluations" / "llm" / sanitize_model_tag(judge_model)


def solver_reports_dir(paths: Dict[str, Any], solver_model: str) -> Path:
    return solver_root(paths, solver_model) / "reports"


def solver_compare_reports_dir(paths: Dict[str, Any], solver_model: str) -> Path:
    return solver_reports_dir(paths, solver_model) / "eval_error_compare"


def solver_cost_dir(paths: Dict[str, Any], solver_model: str) -> Path:
    return solver_root(paths, solver_model) / "cost"


def solver_generation_file(paths: Dict[str, Any], solver_model: str, split_name: str) -> Path:
    return solver_generations_dir(paths, solver_model) / f"{split_name}.jsonl"


def solver_rule_evaluation_file(paths: Dict[str, Any], solver_model: str, split_name: str) -> Path:
    return solver_rule_evaluations_dir(paths, solver_model) / f"{split_name}.jsonl"


def solver_rule_summary_file(paths: Dict[str, Any], solver_model: str) -> Path:
    return solver_rule_evaluations_dir(paths, solver_model) / "summary.json"


def solver_llm_evaluation_file(
    paths: Dict[str, Any],
    solver_model: str,
    judge_model: str,
    split_name: str,
) -> Path:
    return solver_llm_evaluations_dir(paths, solver_model, judge_model) / f"{split_name}.jsonl"


def solver_llm_summary_file(paths: Dict[str, Any], solver_model: str, judge_model: str) -> Path:
    return solver_llm_evaluations_dir(paths, solver_model, judge_model) / "summary.json"


def solver_compare_report_file(paths: Dict[str, Any], solver_model: str, judge_model: str) -> Path:
    return solver_compare_reports_dir(paths, solver_model) / f"{sanitize_model_tag(judge_model)}.jsonl"


def solver_compare_summary_file(paths: Dict[str, Any], solver_model: str, judge_model: str) -> Path:
    return solver_compare_reports_dir(paths, solver_model) / f"{sanitize_model_tag(judge_model)}__summary.json"


def legacy_generations_dir(paths: Dict[str, Any]) -> Path:
    return _path(paths.get("generations_dir"), "data/generations")


def legacy_rule_evaluations_dir(paths: Dict[str, Any]) -> Path:
    return _path(paths.get("evaluations_dir"), "data/evaluations")


def legacy_llm_evaluations_dir(paths: Dict[str, Any]) -> Path:
    raw = str(paths.get("evaluations_llm_dir", "") or "").strip()
    if raw:
        return Path(raw)
    return legacy_rule_evaluations_dir(paths).parent / "evaluations_llm"


def legacy_reports_dir(paths: Dict[str, Any]) -> Path:
    raw = str(paths.get("reports_dir", "") or "").strip()
    if raw:
        return Path(raw)
    hf_json_dir = _path(paths.get("hf_json_dir"), "data/hf_json")
    return hf_json_dir.parent / "reports"


def resolve_generation_input(paths: Dict[str, Any], solver_model: str, split_name: str) -> Path:
    preferred = solver_generation_file(paths, solver_model, split_name)
    if preferred.exists():
        return preferred
    return resolve_existing_model_jsonl(
        legacy_generations_dir(paths),
        split_name,
        solver_model,
        allow_legacy_model_suffix=True,
        allow_legacy_plain_split=True,
    )


def resolve_rule_evaluation_input(paths: Dict[str, Any], solver_model: str, split_name: str) -> Path:
    preferred = solver_rule_evaluation_file(paths, solver_model, split_name)
    if preferred.exists():
        return preferred
    return resolve_existing_model_jsonl(
        legacy_rule_evaluations_dir(paths),
        split_name,
        solver_model,
        allow_legacy_model_suffix=True,
        allow_legacy_plain_split=True,
    )


def resolve_llm_evaluation_input(
    paths: Dict[str, Any],
    solver_model: str,
    judge_model: str,
    split_name: str,
) -> Path:
    preferred = solver_llm_evaluation_file(paths, solver_model, judge_model, split_name)
    if preferred.exists():
        return preferred
    return resolve_existing_model_jsonl(
        legacy_llm_evaluations_dir(paths),
        split_name,
        solver_model,
        allow_legacy_model_suffix=True,
        allow_legacy_plain_split=True,
    )
