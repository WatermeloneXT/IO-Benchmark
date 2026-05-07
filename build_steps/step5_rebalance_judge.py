#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.llm_utils import azure_json_call, create_client, load_config
from core.prompts import SYSTEM_REBALANCE_JUDGE_FLIP, rebalance_judge_user_prompt

from build_steps.common import (
    as_bool,
    natural_key,
    normalize_bool_answer,
    read_jsonl,
    resolve_target_splits,
    write_json,
    write_jsonl,
)

QUESTION_FIELDS = [
    "question_original",
    "split_question",
    "question_standalone",
    "question_final",
]

REFERENCE_FIELDS = [
    "reference_answer_original",
    "reference_answer_corrected",
    "split_answer",
    "reference_answer",
    "final_answer_for_compare",
]


def _base_split_files(hf_json_dir: Path) -> List[Path]:
    files = [p for p in hf_json_dir.glob("*.jsonl") if p.is_file() and not p.stem.endswith("_balanced")]
    return sorted(files, key=lambda p: natural_key(p.stem))


def _pick_question_text(row: Dict[str, Any]) -> str:
    for key in ["question_final", "question_standalone", "split_question", "question_original"]:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _detect_judge_label(row: Dict[str, Any]) -> Optional[str]:
    for key in [
        "reference_answer",
        "final_answer_for_compare",
        "reference_answer_corrected",
        "reference_answer_original",
        "split_answer",
    ]:
        label = normalize_bool_answer(str(row.get(key, "")))
        if label in {"True", "False"}:
            return label
    return None


def _flip_label(label: str) -> str:
    return "False" if label == "True" else "True"


def _rewrite_one(
    *,
    client: Any,
    model: str,
    max_retries: int,
    temperature: float,
    question_text: str,
    old_label: str,
    max_attempts: int,
    split: str = "",
    question_id: str = "",
) -> Tuple[bool, str, str, Dict[str, Any], str]:
    errors: List[str] = []
    last_obj: Dict[str, Any] = {}
    for _ in range(max(1, max_attempts)):
        user_prompt = rebalance_judge_user_prompt(question_text=question_text, old_label=old_label)
        if errors:
            user_prompt += "\nPrevious output issues to fix:\n"
            for err in errors[-3:]:
                user_prompt += f"- {err}\n"
        try:
            obj = azure_json_call(
                client=client,
                model=model,
                system=SYSTEM_REBALANCE_JUDGE_FLIP,
                user=user_prompt,
                temperature=temperature,
                max_retries=max_retries,
                telemetry={
                    "operation": "rebalance_judge_flip",
                    "split": split,
                    "question_id": question_id,
                },
            )
            last_obj = obj
        except Exception as err:  # noqa: BLE001
            errors.append(f"api_error: {err}")
            continue

        rewritten_question = str(obj.get("rewritten_question", "")).strip()
        new_label = normalize_bool_answer(str(obj.get("new_label", "")))
        flipped = as_bool(obj.get("flipped"), default=False)
        natural = as_bool(obj.get("natural"), default=False)

        if not rewritten_question:
            errors.append("empty rewritten_question")
            continue
        if new_label not in {"True", "False"}:
            errors.append("new_label is not True/False")
            continue
        if new_label == old_label:
            errors.append("new_label did not flip")
            continue
        if not flipped:
            errors.append("flipped is not true")
            continue
        if not natural:
            errors.append("natural is not true")
            continue
        return True, rewritten_question, new_label, obj, ""

    reason = "; ".join(errors[-3:]) if errors else "unknown_error"
    return False, question_text, old_label, last_obj, reason


def _apply_updates(row: Dict[str, Any], rewritten_question: str, new_label: str) -> Dict[str, Any]:
    out = copy.deepcopy(row)
    for key in QUESTION_FIELDS:
        if key in out:
            out[key] = rewritten_question
    for key in REFERENCE_FIELDS:
        if key in out:
            out[key] = new_label
    if "reference_answer_sympy" in out:
        out["reference_answer_sympy"] = "N/A"
    return out


def _judge_stats(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    true_count = 0
    false_count = 0
    unknown = 0
    for row in rows:
        if str(row.get("question_type", "")).strip().lower() != "judge":
            continue
        label = _detect_judge_label(row)
        if label == "True":
            true_count += 1
        elif label == "False":
            false_count += 1
        else:
            unknown += 1
    return {
        "judge_total": true_count + false_count + unknown,
        "true": true_count,
        "false": false_count,
        "unknown": unknown,
    }


def run(
    config_path: str = "config.yaml",
    chapter: Optional[str] = None,
    target_splits: Optional[List[str]] = None,
    skip_existing: bool = False,
) -> List[str]:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    build_cfg = cfg.get("build", {})
    chapters_root = Path(paths["chapters_root"])
    hf_json_dir = Path(paths["hf_json_dir"])
    balanced_review_dir = Path(paths.get("balanced_review_dir") or (hf_json_dir.parent / "review_balanced"))
    balanced_review_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = hf_json_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    splits = resolve_target_splits(chapters_root, chapter=chapter, target_splits=target_splits)
    if not splits:
        splits = [p.stem for p in _base_split_files(hf_json_dir)]
    splits = sorted(list(dict.fromkeys(splits)), key=natural_key)

    if not splits:
        print("[step4] no target splits found.")
        return []

    source_rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    balanced_rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    judge_entries: List[Dict[str, Any]] = []
    skipped_splits: List[str] = []

    for split in splits:
        out_file = balanced_review_dir / f"{split}_balanced.jsonl"
        if skip_existing and out_file.exists() and out_file.stat().st_size > 0:
            print(f"[step4] skip {split}: existing output -> {out_file}")
            skipped_splits.append(split)
            continue
        src_file = hf_json_dir / f"{split}.jsonl"
        if not src_file.exists():
            print(f"[step4] skip {split}: missing {src_file.name}")
            continue
        rows = read_jsonl(src_file)
        if not rows:
            print(f"[step4] skip {split}: empty {src_file.name}")
            continue
        source_rows_by_split[split] = rows
        balanced_rows_by_split[split] = copy.deepcopy(rows)
        for idx, row in enumerate(rows):
            if str(row.get("question_type", "")).strip().lower() != "judge":
                continue
            label = _detect_judge_label(row)
            if label not in {"True", "False"}:
                continue
            qid = str(row.get("id", "")).strip()
            judge_entries.append(
                {
                    "split": split,
                    "index": idx,
                    "id": qid,
                    "label": label,
                }
            )

    if not source_rows_by_split:
        if skipped_splits:
            return sorted(list(dict.fromkeys(skipped_splits)), key=natural_key)
        print("[step4] no available splits to process.")
        return []

    true_count = sum(1 for x in judge_entries if x["label"] == "True")
    false_count = sum(1 for x in judge_entries if x["label"] == "False")
    diff = abs(true_count - false_count)
    flips_needed = diff // 2
    majority = "True" if true_count >= false_count else "False"

    candidates = [x for x in judge_entries if x["label"] == majority]
    candidates.sort(key=lambda x: natural_key(x["id"]))
    seed = int(build_cfg.get("rebalance_seed", 42))
    rng = random.Random(seed)
    if flips_needed > len(candidates):
        flips_needed = len(candidates)
    selected = rng.sample(candidates, flips_needed) if flips_needed > 0 else []
    selected.sort(key=lambda x: (natural_key(x["split"]), x["index"]))
    selected_keys = {(x["split"], x["index"]) for x in selected}

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "splits": sorted(list(source_rows_by_split.keys()), key=natural_key),
        "judge_counts_before": {
            "true": true_count,
            "false": false_count,
            "total": true_count + false_count,
        },
        "majority_label": majority,
        "flips_needed": flips_needed,
        "selected_count": len(selected),
        "selected_ids": [x["id"] for x in selected],
        "flipped_ids": [],
        "failed": [],
        "per_split": {},
    }

    for split, rows in source_rows_by_split.items():
        report["per_split"][split] = {
            "row_count": len(rows),
            "judge_before": _judge_stats(rows),
            "judge_after": {},
            "flipped_count": 0,
        }

    if selected:
        client = create_client(cfg)
        model = cfg["azure"]["deployment_name"]
        max_retries = int(cfg["azure"].get("max_retries", 4))
        max_attempts = int(build_cfg.get("rebalance_max_attempts", 3))
        temperature = float(build_cfg.get("rebalance_temperature", 0.0))

        for entry in selected:
            split = str(entry["split"])
            idx = int(entry["index"])
            row = source_rows_by_split[split][idx]
            old_label = str(entry["label"])
            question_text = _pick_question_text(row)
            if not question_text:
                report["failed"].append(
                    {
                        "split": split,
                        "id": entry["id"],
                        "reason": "empty_question_text",
                    }
                )
                continue

            ok, rewritten_question, new_label, llm_json, reason = _rewrite_one(
                client=client,
                model=model,
                max_retries=max_retries,
                temperature=temperature,
                question_text=question_text,
                old_label=old_label,
                max_attempts=max_attempts,
                split=split,
                question_id=str(entry.get("id", "")),
            )
            if not ok:
                report["failed"].append(
                    {
                        "split": split,
                        "id": entry["id"],
                        "reason": reason,
                        "llm_json": llm_json,
                    }
                )
                continue

            if new_label != _flip_label(old_label):
                report["failed"].append(
                    {
                        "split": split,
                        "id": entry["id"],
                        "reason": f"local_check_failed: expected {_flip_label(old_label)} got {new_label}",
                        "llm_json": llm_json,
                    }
                )
                continue

            balanced_rows_by_split[split][idx] = _apply_updates(row, rewritten_question, new_label)
            report["flipped_ids"].append(entry["id"])
            report["per_split"][split]["flipped_count"] += 1

    processed: List[str] = list(skipped_splits)
    for split, rows in balanced_rows_by_split.items():
        out_file = balanced_review_dir / f"{split}_balanced.jsonl"
        write_jsonl(out_file, rows)
        processed.append(split)
        report["per_split"][split]["judge_after"] = _judge_stats(rows)
        changed = report["per_split"][split]["flipped_count"]
        print(f"[step4] wrote {len(rows)} rows -> {out_file} (flipped={changed})")

    report_path = reports_dir / "judge_rebalance_report.json"
    write_json(report_path, report)
    print(f"[step4] report -> {report_path}")

    untouched_selected = len(selected_keys) - len(report["flipped_ids"])
    if untouched_selected > 0:
        print(f"[step4] warning: {untouched_selected} selected judge rows kept unchanged due to rewrite failures.")

    return sorted(list(dict.fromkeys(processed)), key=natural_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build step4: create balanced judge split copies.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--chapter", default=None, help="Optional chapter selector, e.g. 6 or chapter_6")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip split if balanced output file already exists and is non-empty.",
    )
    args = parser.parse_args()
    run(config_path=args.config, chapter=args.chapter, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
