from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def apply_dataset_path_overrides(
    cfg: Dict[str, Any],
    hf_json_dir: Optional[str] = None,
    by_model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if not hf_json_dir and not by_model_dir:
        return cfg

    cfg_copy = dict(cfg)
    paths = dict(cfg_copy.get("paths", {}))
    if hf_json_dir:
        paths["hf_json_dir"] = str(Path(hf_json_dir).expanduser().resolve())
    if by_model_dir:
        paths["by_model_dir"] = str(Path(by_model_dir).expanduser().resolve())
    cfg_copy["paths"] = paths
    return cfg_copy
