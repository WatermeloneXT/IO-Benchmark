#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_utils import load_config
from build_steps.common import natural_key, resolve_target_splits, write_json


def _parse_split_arg(split: Optional[str]) -> List[str]:
    if not isinstance(split, str) or not split.strip():
        return []
    return [x.strip() for x in split.split(",") if x.strip()]


def _candidate_splits(
    *,
    chapters_root: Path,
    balanced_review_dir: Path,
    chapter: Optional[str],
    split: Optional[str],
) -> List[str]:
    explicit = _parse_split_arg(split)
    if explicit:
        return sorted(list(dict.fromkeys(explicit)), key=natural_key)

    splits = resolve_target_splits(chapters_root, chapter=chapter, target_splits=None)
    if splits:
        return sorted(list(dict.fromkeys(splits)), key=natural_key)

    discovered = sorted(
        [
            p.name.replace("_balanced.jsonl", "")
            for p in balanced_review_dir.glob("*_balanced.jsonl")
            if p.is_file() and p.name.endswith("_balanced.jsonl")
        ],
        key=natural_key,
    )
    return discovered


def run(
    config_path: str = "config.yaml",
    chapter: Optional[str] = None,
    split: Optional[str] = None,
    rebuild_step5: bool = False,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    chapters_root = Path(paths["chapters_root"])
    hf_json_dir = Path(paths["hf_json_dir"])
    balanced_review_dir = Path(paths.get("balanced_review_dir") or (hf_json_dir.parent / "review_balanced"))
    reports_dir = hf_json_dir.parent / "reports"
    backup_dir = balanced_review_dir / "backups"
    reports_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    splits = _candidate_splits(
        chapters_root=chapters_root,
        balanced_review_dir=balanced_review_dir,
        chapter=chapter,
        split=split,
    )
    if not splits:
        raise RuntimeError("No target splits found for apply-balanced.")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "balanced_review_dir": str(balanced_review_dir),
        "hf_json_dir": str(hf_json_dir),
        "requested": {
            "chapter": chapter,
            "split": split,
            "rebuild_step5": rebuild_step5,
        },
        "applied": [],
        "skipped": [],
        "failed": [],
    }

    applied_splits: List[str] = []
    for s in splits:
        base_file = hf_json_dir / f"{s}.jsonl"
        balanced_file = balanced_review_dir / f"{s}_balanced.jsonl"
        if not balanced_file.exists():
            report["skipped"].append({"split": s, "reason": f"missing balanced file: {balanced_file}"})
            print(f"[apply-balanced] skip {s}: missing {balanced_file.name}")
            continue
        if not base_file.exists():
            report["skipped"].append({"split": s, "reason": f"missing base file: {base_file}"})
            print(f"[apply-balanced] skip {s}: missing {base_file.name}")
            continue

        backup_file = backup_dir / f"{s}.before_apply_{ts}.jsonl"
        try:
            shutil.copy2(base_file, backup_file)
            shutil.copy2(balanced_file, base_file)
            applied_splits.append(s)
            report["applied"].append(
                {
                    "split": s,
                    "base_file": str(base_file),
                    "balanced_file": str(balanced_file),
                    "backup_file": str(backup_file),
                }
            )
            print(f"[apply-balanced] applied {s}: {balanced_file.name} -> {base_file.name}")
        except Exception as err:  # noqa: BLE001
            report["failed"].append(
                {
                    "split": s,
                    "base_file": str(base_file),
                    "balanced_file": str(balanced_file),
                    "backup_file": str(backup_file),
                    "error": str(err),
                }
            )
            print(f"[apply-balanced] failed {s}: {err}")

    if rebuild_step5 and applied_splits:
        from build_steps.step4_finalize import run as run_step5_finalize  # local import

        run_step5_finalize(
            config_path=config_path,
            chapter=chapter,
            target_splits=applied_splits,
        )
        report["rebuild_step5"] = {"ran": True, "splits": applied_splits}
    else:
        report["rebuild_step5"] = {"ran": False, "splits": []}

    report_path = reports_dir / "apply_balanced_report.json"
    write_json(report_path, report)
    print(f"[apply-balanced] report -> {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply reviewed balanced files into hf_json split files.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--chapter", default=None, help="Optional chapter selector, e.g. 6 or chapter_6")
    parser.add_argument("--split", default=None, help="Optional split selector(s), comma-separated, e.g. chapter_6")
    parser.add_argument(
        "--rebuild-step5",
        action="store_true",
        help="After apply, run build-step5 (finalize aggregate + hf_dataset) for applied splits.",
    )
    args = parser.parse_args()
    run(
        config_path=args.config,
        chapter=args.chapter,
        split=args.split,
        rebuild_step5=args.rebuild_step5,
    )


if __name__ == "__main__":
    main()
