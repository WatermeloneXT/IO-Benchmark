#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_utils import load_config

# Reuse evaluation-time SymPy parsing rules for consistency.
from commands.evaluate_generations import _import_sympy, normalize_exact, parse_sympy_expr


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


def resolve_stage3_file(config_path: str, split: str, stage3_file: Optional[str]) -> Path:
    if stage3_file:
        return Path(stage3_file).expanduser().resolve()
    cfg = load_config(config_path)
    paths = cfg["paths"]
    hf_json_dir = Path(paths["hf_json_dir"])
    pipeline_dir = Path(paths.get("pipeline_dir") or (hf_json_dir.parent / "pipeline"))
    return (pipeline_dir / f"{split}_stage3_transform.jsonl").resolve()


def try_parse_sympy(sp: Any, text: str) -> Optional[str]:
    src = normalize_exact(text or "")
    if not src or src == "N/A":
        return "empty_or_na"
    try:
        parse_sympy_expr(src, sp)
        return None
    except Exception as err:  # noqa: BLE001
        return str(err)


def run(
    config_path: str = "config.yaml",
    split: str = "chapter_6",
    stage3_file: Optional[str] = None,
    output: Optional[str] = None,
    fail_limit: int = 200,
) -> Dict[str, Any]:
    path = resolve_stage3_file(config_path=config_path, split=split, stage3_file=stage3_file)
    if not path.exists():
        raise FileNotFoundError(f"stage3 file not found: {path}")

    rows = read_jsonl(path)
    try:
        sp = _import_sympy()
    except ModuleNotFoundError as err:
        raise RuntimeError("sympy is not installed. Please install requirements first.") from err

    summary: Dict[str, Any] = {
        "split": split,
        "stage3_file": path.as_posix(),
        "total_rows": len(rows),
        "sympy_rows": 0,
        "parse_targets_total": 0,
        "parse_targets_ok": 0,
        "parse_targets_failed": 0,
        "all_sympy_targets_parseable": True,
        "failed_cases": [],
    }

    for row in rows:
        qid = str(row.get("id", "")).strip()
        comparison_mode = str(row.get("comparison_mode", "")).strip().lower()
        if comparison_mode != "sympy":
            continue

        summary["sympy_rows"] += 1
        checks = [
            ("comparable_final_answer", str(row.get("comparable_final_answer", ""))),
            ("reference_answer_sympy", str(row.get("reference_answer_sympy", ""))),
        ]
        for field_name, value in checks:
            summary["parse_targets_total"] += 1
            err = try_parse_sympy(sp=sp, text=value)
            if err is None:
                summary["parse_targets_ok"] += 1
                continue
            summary["parse_targets_failed"] += 1
            summary["all_sympy_targets_parseable"] = False
            if len(summary["failed_cases"]) < fail_limit:
                summary["failed_cases"].append(
                    {
                        "id": qid,
                        "field": field_name,
                        "value": value,
                        "error": err,
                    }
                )

    if output:
        out_path = Path(output).expanduser().resolve()
    else:
        out_path = path.with_name(path.stem + "_sympy_parse_check.json")
    write_json(out_path, summary)
    summary["output_file"] = out_path.as_posix()

    print(
        "[stage3-sympy-check]",
        f"split={summary['split']}",
        f"rows={summary['total_rows']}",
        f"sympy_rows={summary['sympy_rows']}",
        f"ok={summary['parse_targets_ok']}/{summary['parse_targets_total']}",
        f"failed={summary['parse_targets_failed']}",
        f"all_ok={summary['all_sympy_targets_parseable']}",
    )
    print(f"[stage3-sympy-check] report -> {summary['output_file']}")
    if summary["failed_cases"]:
        first = summary["failed_cases"][0]
        print(
            "[stage3-sympy-check] first failure:",
            f"id={first['id']}",
            f"field={first['field']}",
            f"error={first['error']}",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether stage3 sympy-type answers can be parsed by SymPy."
    )
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--split", default="chapter_6", help="Split name, e.g. chapter_6")
    parser.add_argument("--stage3-file", default=None, help="Optional path to *_stage3_transform.jsonl")
    parser.add_argument("--output", default=None, help="Optional output report path (json)")
    parser.add_argument("--fail-limit", type=int, default=200, help="Max failed cases written to report")
    args = parser.parse_args()
    run(
        config_path=args.config,
        split=args.split,
        stage3_file=args.stage3_file,
        output=args.output,
        fail_limit=max(1, int(args.fail_limit)),
    )


if __name__ == "__main__":
    main()
