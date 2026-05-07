from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple


def sanitize_model_tag(model_name: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", (model_name or "").strip())
    tag = re.sub(r"-{2,}", "-", tag).strip("._-")
    return tag or "model"


def model_split_jsonl_name(split_name: str, model_name: str) -> str:
    return f"{split_name}__{sanitize_model_tag(model_name)}.jsonl"


def split_jsonl_name(split_name: str) -> str:
    return f"{split_name}.jsonl"


def model_scoped_dir(base_dir: str | Path, model_name: str) -> Path:
    return Path(base_dir) / sanitize_model_tag(model_name)


def model_scoped_jsonl_path(base_dir: str | Path, split_name: str, model_name: str) -> Path:
    return model_scoped_dir(base_dir, model_name) / split_jsonl_name(split_name)


def model_scoped_summary_path(base_dir: str | Path, model_name: str) -> Path:
    return model_scoped_dir(base_dir, model_name) / "summary.json"


def legacy_model_split_jsonl_path(base_dir: str | Path, split_name: str, model_name: str) -> Path:
    return Path(base_dir) / model_split_jsonl_name(split_name, model_name)


def resolve_existing_model_jsonl(
    base_dir: str | Path,
    split_name: str,
    model_name: str,
    *,
    allow_legacy_model_suffix: bool = True,
    allow_legacy_plain_split: bool = True,
) -> Path:
    preferred = model_scoped_jsonl_path(base_dir, split_name, model_name)
    if preferred.exists():
        return preferred

    if allow_legacy_model_suffix:
        legacy_model = legacy_model_split_jsonl_path(base_dir, split_name, model_name)
        if legacy_model.exists():
            return legacy_model

    if allow_legacy_plain_split:
        legacy_plain = Path(base_dir) / split_jsonl_name(split_name)
        if legacy_plain.exists():
            return legacy_plain

    return preferred


def parse_split_and_model_tag(path_or_name: str | Path) -> Tuple[str, Optional[str]]:
    stem = Path(path_or_name).stem
    if "__" not in stem:
        return stem, None
    split_name, model_tag = stem.rsplit("__", 1)
    if not split_name or not model_tag:
        return stem, None
    return split_name, model_tag
