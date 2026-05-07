#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.llm_utils import load_config

from build_steps.common import (
    natural_key,
    read_jsonl,
    resolve_target_splits,
    stage_file,
    write_json,
)

try:
    from datasets import Dataset, DatasetDict, load_from_disk
except Exception:  # noqa: BLE001
    Dataset = None
    DatasetDict = None
    load_from_disk = None


def write_split_pipeline_aggregate(
    split: str,
    stage1_rows: List[Dict[str, Any]],
    stage2_rows: List[Dict[str, Any]],
    stage3_rows: List[Dict[str, Any]],
    final_rows: List[Dict[str, Any]],
    pipeline_dir: Path,
) -> Path:
    out = pipeline_dir / f"{split}_pipeline_aggregate.json"
    payload = {
        "split": split,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "stage1": len(stage1_rows),
            "stage2": len(stage2_rows),
            "stage3": len(stage3_rows),
            "final": len(final_rows),
        },
        "stage1_reference_check": stage1_rows,
        "stage2_split": stage2_rows,
        "stage3_transform": stage3_rows,
        "final_dataset_rows": final_rows,
    }
    write_json(out, payload)
    return out


def finalize_hf_dataset(hf_json_dir: Path, hf_dataset_dir: Path, target_splits: List[str]) -> None:
    if Dataset is None or DatasetDict is None:
        print("[step5] datasets package not installed; skipped DatasetDict.save_to_disk.")
        return

    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    for split in target_splits:
        f = hf_json_dir / f"{split}.jsonl"
        if not f.exists():
            continue
        rows = read_jsonl(f)
        if rows:
            split_rows[split] = rows

    if not split_rows:
        print("[step5] no final split jsonl found for target splits; skipped hf_dataset build.")
        return

    merged_splits: Dict[str, Any] = {}
    if load_from_disk is not None and hf_dataset_dir.exists():
        try:
            existing = load_from_disk(str(hf_dataset_dir))
            if isinstance(existing, DatasetDict):
                # Detach existing splits from on-disk backing files to avoid
                # "dataset can't overwrite itself" when saving to the same root.
                for split_name, ds in existing.items():
                    merged_splits[split_name] = Dataset.from_list(ds.to_list())
        except Exception:  # noqa: BLE001
            merged_splits = {}

    for split, rows in split_rows.items():
        merged_splits[split] = Dataset.from_list(rows)

    ds_dict = DatasetDict(merged_splits)
    hf_dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    if hf_dataset_dir.exists():
        shutil.rmtree(hf_dataset_dir)
    ds_dict.save_to_disk(str(hf_dataset_dir))
    print(f"[step5] HuggingFace DatasetDict saved to: {hf_dataset_dir}")


def run(
    config_path: str = "config.yaml",
    chapter: Optional[str] = None,
    target_splits: Optional[List[str]] = None,
) -> List[str]:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    chapters_root = Path(paths["chapters_root"])
    hf_json_dir = Path(paths["hf_json_dir"])
    hf_dataset_dir = Path(paths["hf_dataset_dir"])
    pipeline_dir = Path(paths.get("pipeline_dir") or (hf_json_dir.parent / "pipeline"))
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    splits = resolve_target_splits(chapters_root, chapter=chapter, target_splits=target_splits)
    if not splits:
        splits = sorted(
            [p.stem for p in hf_json_dir.glob("*.jsonl") if p.is_file() and not p.stem.endswith("_balanced")],
            key=natural_key,
        )

    if not splits:
        print("[step5] no target splits found.")
        return []

    aggregate_paths: List[Path] = []
    processed: List[str] = []

    for split in splits:
        stage1_rows = read_jsonl(stage_file(pipeline_dir, split, "stage1_reference"))
        stage2_rows = read_jsonl(stage_file(pipeline_dir, split, "stage2_split"))
        stage3_rows = read_jsonl(stage_file(pipeline_dir, split, "stage3_transform"))
        final_rows = read_jsonl(hf_json_dir / f"{split}.jsonl")

        if not final_rows:
            print(f"[step5] skip {split}: missing/empty final split jsonl")
            continue

        aggregate_paths.append(
            write_split_pipeline_aggregate(
                split=split,
                stage1_rows=stage1_rows,
                stage2_rows=stage2_rows,
                stage3_rows=stage3_rows,
                final_rows=final_rows,
                pipeline_dir=pipeline_dir,
            )
        )
        processed.append(split)

    global_aggregate = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
        "aggregate_files": [p.as_posix() for p in aggregate_paths],
    }
    for split in processed:
        s1 = read_jsonl(stage_file(pipeline_dir, split, "stage1_reference"))
        s2 = read_jsonl(stage_file(pipeline_dir, split, "stage2_split"))
        s3 = read_jsonl(stage_file(pipeline_dir, split, "stage3_transform"))
        final_rows = read_jsonl(hf_json_dir / f"{split}.jsonl")
        global_aggregate["splits"][split] = {
            "stage1": len(s1),
            "stage2": len(s2),
            "stage3": len(s3),
            "final": len(final_rows),
        }

    global_aggregate_file = pipeline_dir / "pipeline_aggregate_all.json"
    write_json(global_aggregate_file, global_aggregate)
    print(f"[step5] wrote global aggregate -> {global_aggregate_file}")

    finalize_hf_dataset(hf_json_dir=hf_json_dir, hf_dataset_dir=hf_dataset_dir, target_splits=processed)
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage5: write aggregate reports and hf_dataset.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--chapter", default=None, help="Optional chapter selector, e.g. 6 or chapter_6")
    args = parser.parse_args()
    run(config_path=args.config, chapter=args.chapter)


if __name__ == "__main__":
    main()
