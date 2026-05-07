#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_utils import load_config, resolve_solver_model
from core.model_layout import solver_generations_dir, solver_root
from core.path_overrides import apply_dataset_path_overrides
from core.solver_variants import build_solver_artifact_label, validate_native_reasoning_effort


def _effective_max_solve_tokens(cfg: Dict[str, Any], value: Optional[int]) -> int:
    if value is not None:
        return int(value)
    return int((cfg.get("generate") or {}).get("max_solve_tokens", 4096) or 4096)


def _artifact_label(
    cfg: Dict[str, Any],
    *,
    base_model_name: str,
    reasoning_effort: Optional[str],
    max_solve_tokens: Optional[int],
) -> str:
    effort = str(reasoning_effort or "").strip()
    if effort:
        validate_native_reasoning_effort(cfg, [base_model_name], effort)
    return build_solver_artifact_label(
        base_model_name,
        reasoning_effort=effort or None,
        max_solve_tokens=_effective_max_solve_tokens(cfg, max_solve_tokens),
    )


def _copy_tree(src: Path, dst: Path) -> int:
    shutil.copytree(src, dst)
    return sum(1 for path in dst.rglob("*") if path.is_file())


def run(
    config_path: str = "config.yaml",
    solver_model: Optional[str] = None,
    model: Optional[str] = None,
    hf_json_dir: Optional[str] = None,
    by_model_dir: Optional[str] = None,
    source_solver_reasoning_effort: Optional[str] = None,
    source_solver_max_solve_tokens: Optional[int] = None,
    target_solver_reasoning_effort: Optional[str] = None,
    target_solver_max_solve_tokens: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    cfg = apply_dataset_path_overrides(
        load_config(config_path),
        hf_json_dir=hf_json_dir,
        by_model_dir=by_model_dir,
    )
    paths = cfg["paths"]
    base_model_name = resolve_solver_model(cfg, requested_model=solver_model or model)

    source_effort = str(source_solver_reasoning_effort or "").strip() or None
    source_max_tokens = _effective_max_solve_tokens(cfg, source_solver_max_solve_tokens)
    target_effort = str(target_solver_reasoning_effort or source_effort or "").strip() or None
    target_max_tokens = _effective_max_solve_tokens(cfg, target_solver_max_solve_tokens or source_max_tokens)

    source_label = _artifact_label(
        cfg,
        base_model_name=base_model_name,
        reasoning_effort=source_effort,
        max_solve_tokens=source_max_tokens,
    )
    target_label = _artifact_label(
        cfg,
        base_model_name=base_model_name,
        reasoning_effort=target_effort,
        max_solve_tokens=target_max_tokens,
    )
    if source_label == target_label:
        raise RuntimeError(
            "Source and target solver artifact labels are identical. "
            "Change the target reasoning effort or target max solve tokens."
        )

    source_root = solver_root(paths, source_label)
    source_generations = solver_generations_dir(paths, source_label)
    target_root_dir = solver_root(paths, target_label)
    target_generations = solver_generations_dir(paths, target_label)

    if not source_root.exists():
        raise RuntimeError(
            f"Source solver artifact does not exist: {source_root}. "
            "Run generate/evaluate first with the source artifact settings."
        )
    if not source_generations.exists():
        raise RuntimeError(
            f"Source generations directory does not exist: {source_generations}. "
            "This clone command expects a complete baseline generation artifact."
        )

    if target_root_dir.exists():
        if not force:
            raise RuntimeError(
                f"Target solver artifact already exists: {target_root_dir}. "
                "Pass --force to replace it."
            )
        shutil.rmtree(target_root_dir)

    target_root_dir.parent.mkdir(parents=True, exist_ok=True)
    copied_files = _copy_tree(source_generations, target_generations)

    summary = {
        "solver_model": base_model_name,
        "source_solver_artifact_label": source_label,
        "target_solver_artifact_label": target_label,
        "source_root": str(source_root),
        "target_root": str(target_root_dir),
        "copied_subdirs": ["generations"],
        "copied_generation_files": copied_files,
        "note": "Only generations/ was copied. Re-run evaluate/evaluate-llm after partial reruns to refresh derived outputs.",
    }

    print(
        "[clone-artifact-run] copied complete baseline generations:\n"
        f"  source={source_root}\n"
        f"  target={target_root_dir}\n"
        "  copied_subdirs=generations\n"
        f"  copied_generation_files={copied_files}\n"
        "  note=derived outputs were intentionally not copied"
    )
    return summary


if __name__ == "__main__":
    raise SystemExit("Use econ_cli.py clone-artifact-run ...")
