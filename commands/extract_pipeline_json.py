#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_utils import load_config


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def collect_splits(hf_json_dir: Path, selected_split: Optional[str]) -> List[str]:
    splits = sorted([p.stem for p in hf_json_dir.glob("*.jsonl") if p.is_file()])
    if selected_split:
        return [s for s in splits if s == selected_split]
    return [s for s in splits if not s.endswith("_balanced")]


def run(config_path: str = "config.yaml", split: Optional[str] = None, out: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    hf_json_dir = Path(paths["hf_json_dir"])
    pipeline_dir = Path(paths.get("pipeline_dir") or (hf_json_dir.parent / "pipeline"))

    splits = collect_splits(hf_json_dir, split)
    if not splits:
        raise RuntimeError(f"No split jsonl files found under {hf_json_dir} (split={split}).")

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(config_path).resolve()),
        "pipeline_dir": pipeline_dir.as_posix(),
        "splits": {},
    }

    for split_name in splits:
        stage1_file = pipeline_dir / f"{split_name}_stage1_reference.jsonl"
        stage2_file = pipeline_dir / f"{split_name}_stage2_split.jsonl"
        stage3_file = pipeline_dir / f"{split_name}_stage3_transform.jsonl"
        final_file = hf_json_dir / f"{split_name}.jsonl"

        stage1_rows = read_jsonl(stage1_file)
        stage2_rows = read_jsonl(stage2_file)
        stage3_rows = read_jsonl(stage3_file)
        final_rows = read_jsonl(final_file)

        payload["splits"][split_name] = {
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

    if out is None:
        out_path = pipeline_dir / "pipeline_extracted_all.json"
    else:
        out_path = Path(out)
    write_json(out_path, payload)
    print(f"[extract] wrote -> {out_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract stage JSONL data into one aggregated JSON file.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    args = parser.parse_args()
    run(config_path=args.config, split=args.split, out=args.out)


if __name__ == "__main__":
    main()
