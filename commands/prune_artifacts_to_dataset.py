#!/usr/bin/env python3
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cost_logging import register_run_outputs
from core.llm_utils import load_config
from core.model_layout import by_model_root
from core.path_overrides import apply_dataset_path_overrides


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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def chapter_sort_key(split_name: str) -> Tuple[int, str]:
    if split_name == "chapter_intro":
        return (-1, split_name)
    if split_name.startswith("chapter_"):
        suffix = split_name[len("chapter_") :]
        if suffix.isdigit():
            return (int(suffix), split_name)
    return (9999, split_name)


def parse_points(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        try:
            return float(Fraction(str(value).strip()))
        except Exception:
            return 1.0


def normalize_exact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def resolve_assignment_group(row: Dict[str, Any], fallback_split: str = "") -> str:
    split_name = normalize_exact(str(row.get("split", "")).strip()) or fallback_split
    chapter_name = normalize_exact(str(row.get("chapter", "")).strip())
    if split_name != "chapter_assignment" and chapter_name != "assignment":
        return ""
    for key in ("original_id", "problem_number", "id"):
        raw = normalize_exact(str(row.get(key, "")).strip())
        if not raw:
            continue
        prefix = raw.split("#", 1)[0].split("/", 1)[0].strip()
        if prefix.isdigit():
            return prefix
    return ""


def summarize_assignment_groups_rule(eval_rows: Iterable[Dict[str, Any]], split_name: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for row in eval_rows:
        group = resolve_assignment_group(row, split_name)
        if not group:
            continue
        stats = groups.setdefault(
            group,
            {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
                "total_points": 0.0,
                "earned_points": 0.0,
                "weighted_accuracy": 0.0,
            },
        )
        stats["total"] += 1
        if bool(row.get("is_correct", False)):
            stats["correct"] += 1
        stats["total_points"] += parse_points(row.get("points", 1.0))
        stats["earned_points"] += parse_points(row.get("points_earned", 0.0))
    ordered = sorted(groups.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0]))
    out: Dict[str, Dict[str, Any]] = {}
    for group, stats in ordered:
        stats["accuracy"] = (stats["correct"] / stats["total"]) if stats["total"] else 0.0
        stats["weighted_accuracy"] = (
            stats["earned_points"] / stats["total_points"]
        ) if stats["total_points"] else 0.0
        out[group] = stats
    return out


def merge_assignment_group_summaries_rule(
    summary: Dict[str, Dict[str, Any]],
    group_summary: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    for group, stats in group_summary.items():
        merged = summary.setdefault(
            group,
            {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
                "total_points": 0.0,
                "earned_points": 0.0,
                "weighted_accuracy": 0.0,
            },
        )
        merged["total"] += int(stats.get("total", 0))
        merged["correct"] += int(stats.get("correct", 0))
        merged["total_points"] += parse_points(stats.get("total_points", 0.0))
        merged["earned_points"] += parse_points(stats.get("earned_points", 0.0))
    ordered = sorted(summary.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0]))
    out: Dict[str, Dict[str, Any]] = {}
    for group, merged in ordered:
        merged["accuracy"] = (merged["correct"] / merged["total"]) if merged["total"] else 0.0
        merged["weighted_accuracy"] = (
            merged["earned_points"] / merged["total_points"]
        ) if merged["total_points"] else 0.0
        out[group] = merged
    return out


def summarize_assignment_groups_llm(split_rows: Iterable[Dict[str, Any]], split_name: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for row in split_rows:
        group = resolve_assignment_group(row, split_name)
        if not group:
            continue
        stats = groups.setdefault(
            group,
            {
                "total": 0,
                "correct": 0,
                "reasoning_correct": 0,
                "accuracy": 0.0,
                "reasoning_accuracy": 0.0,
                "total_points": 0.0,
                "earned_points": 0.0,
                "weighted_accuracy": 0.0,
            },
        )
        stats["total"] += 1
        if bool(row.get("answer_correct", False)):
            stats["correct"] += 1
        if bool(row.get("reasoning_correct", False)):
            stats["reasoning_correct"] += 1
        stats["total_points"] += parse_points(row.get("points", 1.0))
        stats["earned_points"] += parse_points(row.get("points_earned", 0.0))
    ordered = sorted(groups.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0]))
    out: Dict[str, Dict[str, Any]] = {}
    for group, stats in ordered:
        stats["accuracy"] = (stats["correct"] / stats["total"]) if stats["total"] else 0.0
        stats["reasoning_accuracy"] = (stats["reasoning_correct"] / stats["total"]) if stats["total"] else 0.0
        stats["weighted_accuracy"] = (
            stats["earned_points"] / stats["total_points"]
        ) if stats["total_points"] else 0.0
        out[group] = stats
    return out


def merge_assignment_group_summaries_llm(
    summary: Dict[str, Dict[str, Any]],
    group_summary: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    for group, stats in group_summary.items():
        merged = summary.setdefault(
            group,
            {
                "total": 0,
                "correct": 0,
                "reasoning_correct": 0,
                "accuracy": 0.0,
                "reasoning_accuracy": 0.0,
                "total_points": 0.0,
                "earned_points": 0.0,
                "weighted_accuracy": 0.0,
            },
        )
        merged["total"] += int(stats.get("total", 0))
        merged["correct"] += int(stats.get("correct", 0))
        merged["reasoning_correct"] += int(stats.get("reasoning_correct", 0))
        merged["total_points"] += parse_points(stats.get("total_points", 0.0))
        merged["earned_points"] += parse_points(stats.get("earned_points", 0.0))
    ordered = sorted(summary.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0]))
    out: Dict[str, Dict[str, Any]] = {}
    for group, merged in ordered:
        merged["accuracy"] = (merged["correct"] / merged["total"]) if merged["total"] else 0.0
        merged["reasoning_accuracy"] = (
            merged["reasoning_correct"] / merged["total"]
        ) if merged["total"] else 0.0
        merged["weighted_accuracy"] = (
            merged["earned_points"] / merged["total_points"]
        ) if merged["total_points"] else 0.0
        out[group] = merged
    return out


def load_dataset_question_ids(hf_json_dir: Path) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    ordered_ids_by_split: Dict[str, List[str]] = {}
    counts_by_split: Dict[str, int] = {}
    for path in sorted(hf_json_dir.glob("*.jsonl"), key=lambda p: chapter_sort_key(p.stem)):
        rows = read_jsonl(path)
        ordered_ids = [str(row.get("id", "")).strip() for row in rows if str(row.get("id", "")).strip()]
        ordered_ids_by_split[path.stem] = ordered_ids
        counts_by_split[path.stem] = len(ordered_ids)
    return ordered_ids_by_split, counts_by_split


def prune_rows_for_split(rows: List[Dict[str, Any]], allowed_ids: set[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    kept: List[Dict[str, Any]] = []
    removed: List[str] = []
    for row in rows:
        qid = str(row.get("id", "")).strip()
        if qid and qid not in allowed_ids:
            removed.append(qid)
            continue
        kept.append(row)
    return kept, removed


def rebuild_rule_summary(
    rule_dir: Path,
    generation_dir: Path,
    ordered_ids_by_split: Dict[str, List[str]],
    counts_by_split: Dict[str, int],
) -> Path | None:
    summary_path = rule_dir / "summary.json"
    base_summary = read_json(summary_path)
    summary: Dict[str, Any] = {
        "model": base_summary.get("model", ""),
        "solver_model": base_summary.get("solver_model", ""),
        "force": base_summary.get("force", False),
        "total": 0,
        "correct": 0,
        "accuracy": 0.0,
        "total_points": 0.0,
        "earned_points": 0.0,
        "weighted_accuracy": 0.0,
        "splits": {},
    }
    for key in ("workflow_id", "run_id", "stage", "solver_model_artifact", "solver_reasoning_effort", "solver_max_solve_tokens"):
        if key in base_summary:
            summary[key] = base_summary.get(key)
    overall_assignment_groups: Dict[str, Dict[str, Any]] = {}
    split_files = sorted(rule_dir.glob("chapter_*.jsonl"), key=lambda p: chapter_sort_key(p.stem))
    for path in split_files:
        split_name = path.stem
        rows = read_jsonl(path)
        total = len(rows)
        correct = sum(1 for row in rows if bool(row.get("is_correct", False)))
        accuracy = (correct / total) if total else 0.0
        total_points = sum(parse_points(row.get("points", 1.0)) for row in rows)
        earned_points = sum(parse_points(row.get("points_earned", 0.0)) for row in rows)
        weighted_accuracy = (earned_points / total_points) if total_points else 0.0
        assignment_groups = summarize_assignment_groups_rule(rows, split_name)
        generation_file = generation_dir / f"{split_name}.jsonl"
        generation_rows = len(read_jsonl(generation_file)) if generation_file.exists() else None
        summary["splits"][split_name] = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "total_points": total_points,
            "earned_points": earned_points,
            "weighted_accuracy": weighted_accuracy,
            "dataset_rows": counts_by_split.get(split_name),
            "generation_rows": generation_rows,
            "evaluation_file": path.as_posix(),
            "skipped_existing": False,
        }
        if assignment_groups:
            summary["splits"][split_name]["assignment_groups"] = assignment_groups
            overall_assignment_groups = merge_assignment_group_summaries_rule(overall_assignment_groups, assignment_groups)
        summary["total"] += total
        summary["correct"] += correct
        summary["total_points"] += total_points
        summary["earned_points"] += earned_points
    summary["accuracy"] = (summary["correct"] / summary["total"]) if summary["total"] else 0.0
    summary["weighted_accuracy"] = (
        summary["earned_points"] / summary["total_points"]
    ) if summary["total_points"] else 0.0
    if overall_assignment_groups:
        summary["assignment_groups"] = overall_assignment_groups
    write_json(summary_path, summary)
    return summary_path


def rebuild_llm_summary(judge_dir: Path) -> Path | None:
    summary_path = judge_dir / "summary.json"
    base_summary = read_json(summary_path)
    summary: Dict[str, Any] = {
        "solver_model": base_summary.get("solver_model", ""),
        "judge_model": base_summary.get("judge_model", ""),
        "force": base_summary.get("force", False),
        "total": 0,
        "correct": 0,
        "reasoning_correct": 0,
        "accuracy": 0.0,
        "reasoning_accuracy": 0.0,
        "total_points": 0.0,
        "earned_points": 0.0,
        "weighted_accuracy": 0.0,
        "splits": {},
    }
    for key in ("workflow_id", "run_id", "stage", "solver_model_artifact", "solver_reasoning_effort", "solver_max_solve_tokens"):
        if key in base_summary:
            summary[key] = base_summary.get(key)
    overall_assignment_groups: Dict[str, Dict[str, Any]] = {}
    split_files = sorted(judge_dir.glob("chapter_*.jsonl"), key=lambda p: chapter_sort_key(p.stem))
    for path in split_files:
        split_name = path.stem
        rows = read_jsonl(path)
        total = len(rows)
        correct = sum(1 for row in rows if bool(row.get("answer_correct", False)))
        reasoning_correct = sum(1 for row in rows if bool(row.get("reasoning_correct", False)))
        accuracy = (correct / total) if total else 0.0
        reasoning_accuracy = (reasoning_correct / total) if total else 0.0
        total_points = sum(parse_points(row.get("points", 1.0)) for row in rows)
        earned_points = sum(parse_points(row.get("points_earned", 0.0)) for row in rows)
        weighted_accuracy = (earned_points / total_points) if total_points else 0.0
        assignment_groups = summarize_assignment_groups_llm(rows, split_name)
        summary["splits"][split_name] = {
            "total": total,
            "correct": correct,
            "reasoning_correct": reasoning_correct,
            "accuracy": accuracy,
            "reasoning_accuracy": reasoning_accuracy,
            "total_points": total_points,
            "earned_points": earned_points,
            "weighted_accuracy": weighted_accuracy,
            "skipped_existing": False,
        }
        if assignment_groups:
            summary["splits"][split_name]["assignment_groups"] = assignment_groups
            overall_assignment_groups = merge_assignment_group_summaries_llm(overall_assignment_groups, assignment_groups)
        summary["total"] += total
        summary["correct"] += correct
        summary["reasoning_correct"] += reasoning_correct
        summary["total_points"] += total_points
        summary["earned_points"] += earned_points
    summary["accuracy"] = (summary["correct"] / summary["total"]) if summary["total"] else 0.0
    summary["reasoning_accuracy"] = (
        summary["reasoning_correct"] / summary["total"]
    ) if summary["total"] else 0.0
    summary["weighted_accuracy"] = (
        summary["earned_points"] / summary["total_points"]
    ) if summary["total_points"] else 0.0
    if overall_assignment_groups:
        summary["assignment_groups"] = overall_assignment_groups
    write_json(summary_path, summary)
    return summary_path


def run(
    *,
    config_path: str,
    hf_json_dir: str | None = None,
    by_model_dir: str | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    cfg = apply_dataset_path_overrides(cfg, hf_json_dir=hf_json_dir, by_model_dir=by_model_dir)
    paths = cfg["paths"]
    dataset_dir = Path(paths["hf_json_dir"]).resolve()
    model_root = by_model_root(paths).resolve()
    ordered_ids_by_split, counts_by_split = load_dataset_question_ids(dataset_dir)
    allowed_ids_by_split = {split: set(ids) for split, ids in ordered_ids_by_split.items()}

    report: Dict[str, Any] = {
        "dataset_dir": dataset_dir.as_posix(),
        "by_model_dir": model_root.as_posix(),
        "artifacts": {},
    }
    written_outputs: List[Path] = []

    if not model_root.exists():
        raise FileNotFoundError(f"by_model_dir not found: {model_root}")

    for artifact_dir in sorted([p for p in model_root.iterdir() if p.is_dir()]):
        artifact_report = {
            "generations_removed": 0,
            "rule_removed": 0,
            "llm_removed": 0,
            "splits": {},
        }

        generation_dir = artifact_dir / "generations"
        if generation_dir.exists():
            for path in sorted(generation_dir.glob("chapter_*.jsonl"), key=lambda p: chapter_sort_key(p.stem)):
                split_name = path.stem
                allowed_ids = allowed_ids_by_split.get(split_name)
                if allowed_ids is None:
                    continue
                rows = read_jsonl(path)
                kept_rows, removed_ids = prune_rows_for_split(rows, allowed_ids)
                if removed_ids:
                    artifact_report["generations_removed"] += len(removed_ids)
                    artifact_report["splits"].setdefault(split_name, {})["generation_removed_ids"] = removed_ids
                    if not dry_run:
                        write_jsonl(path, kept_rows)
                        written_outputs.append(path)

        rule_dir = artifact_dir / "evaluations" / "rule"
        if rule_dir.exists():
            changed_rule = False
            for path in sorted(rule_dir.glob("chapter_*.jsonl"), key=lambda p: chapter_sort_key(p.stem)):
                split_name = path.stem
                allowed_ids = allowed_ids_by_split.get(split_name)
                if allowed_ids is None:
                    continue
                rows = read_jsonl(path)
                kept_rows, removed_ids = prune_rows_for_split(rows, allowed_ids)
                if removed_ids:
                    changed_rule = True
                    artifact_report["rule_removed"] += len(removed_ids)
                    artifact_report["splits"].setdefault(split_name, {})["rule_removed_ids"] = removed_ids
                    if not dry_run:
                        write_jsonl(path, kept_rows)
                        written_outputs.append(path)
            if changed_rule and not dry_run:
                summary_path = rebuild_rule_summary(rule_dir, generation_dir, ordered_ids_by_split, counts_by_split)
                if summary_path is not None:
                    written_outputs.append(summary_path)

        llm_root = artifact_dir / "evaluations" / "llm"
        if llm_root.exists():
            for judge_dir in sorted([p for p in llm_root.iterdir() if p.is_dir()]):
                changed_llm = False
                for path in sorted(judge_dir.glob("chapter_*.jsonl"), key=lambda p: chapter_sort_key(p.stem)):
                    split_name = path.stem
                    allowed_ids = allowed_ids_by_split.get(split_name)
                    if allowed_ids is None:
                        continue
                    rows = read_jsonl(path)
                    kept_rows, removed_ids = prune_rows_for_split(rows, allowed_ids)
                    if removed_ids:
                        changed_llm = True
                        artifact_report["llm_removed"] += len(removed_ids)
                        judge_report = artifact_report["splits"].setdefault(split_name, {}).setdefault("llm_removed_ids", {})
                        judge_report[judge_dir.name] = removed_ids
                        if not dry_run:
                            write_jsonl(path, kept_rows)
                            written_outputs.append(path)
                if changed_llm and not dry_run:
                    summary_path = rebuild_llm_summary(judge_dir)
                    if summary_path is not None:
                        written_outputs.append(summary_path)

        if artifact_report["generations_removed"] or artifact_report["rule_removed"] or artifact_report["llm_removed"]:
            report["artifacts"][artifact_dir.name] = artifact_report

    if not dry_run:
        register_run_outputs(written_outputs)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prune generation/evaluation artifact rows that no longer belong to the current dataset. "
            "Useful after shrinking the benchmark without rerunning generation."
        )
    )
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--hf-json-dir",
        default=None,
        help="Optional dataset directory override. Stale rows are defined relative to this dataset.",
    )
    parser.add_argument(
        "--by-model-dir",
        default=None,
        help="Optional artifact root override. Defaults to paths.by_model_dir or data/by_model.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report removals without writing any files.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run(
        config_path=args.config,
        hf_json_dir=args.hf_json_dir,
        by_model_dir=args.by_model_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
