#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_utils import load_config


WORD_RE = re.compile(r"\b\w+\b")
WORD_COUNT_METHOD = r"len(re.findall(r'\b\w+\b', question_final))"

BIN_ORDER = ["Short", "Medium", "Long"]

PAPER_MODEL_ORDER: Tuple[Tuple[str, str], ...] = (
    ("GPT", "gpt-5.4__effort-xhigh__max-tokens-32768"),
    ("Opus", "claude-opus-4-6__effort-max__max-tokens-32768"),
    ("Sonnet", "claude-sonnet-4-6__effort-max__max-tokens-32768"),
    ("Qwen32B", "Qwen3-32B__max-tokens-32768"),
    ("DS32B", "DeepSeek-R1-Distill-Qwen-32B__max-tokens-32768"),
    ("Llama70B", "Llama-3.3-70B-Instruct__max-tokens-32768"),
)

DISPLAY_NAMES: Dict[str, str] = {
    "gpt-5.4__effort-medium__max-tokens-32768": "GPT-5.4 (medium)",
    "gpt-5.4__effort-high__max-tokens-32768": "GPT-5.4 (high)",
    "gpt-5.4__effort-xhigh__max-tokens-32768": "GPT-5.4 (xhigh)",
    "claude-opus-4-6__effort-medium__max-tokens-32768": "Claude Opus 4.6 (medium)",
    "claude-opus-4-6__effort-high__max-tokens-32768": "Claude Opus 4.6 (high)",
    "claude-opus-4-6__effort-max__max-tokens-32768": "Claude Opus 4.6 (max)",
    "claude-sonnet-4-6__effort-medium__max-tokens-32768": "Claude Sonnet 4.6 (medium)",
    "claude-sonnet-4-6__effort-high__max-tokens-32768": "Claude Sonnet 4.6 (high)",
    "claude-sonnet-4-6__effort-max__max-tokens-32768": "Claude Sonnet 4.6 (max)",
    "Qwen3-8B__max-tokens-32768": "Qwen3-8B",
    "Qwen3-32B__max-tokens-32768": "Qwen3-32B",
    "DeepSeek-R1-Distill-Qwen-7B__max-tokens-32768": "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Qwen-14B__max-tokens-32768": "DeepSeek-R1-Distill-Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-32B__max-tokens-32768": "DeepSeek-R1-Distill-Qwen-32B",
    "Llama-3.1-8B-Instruct__max-tokens-32768": "Llama-3.1-8B-Instruct",
    "Llama-3.3-70B-Instruct__max-tokens-32768": "Llama-3.3-70B-Instruct",
}


def natural_key(text: str) -> List[Any]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(text))]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def preview(text: str, max_chars: int = 180) -> str:
    text = normalize_ws(text)
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def resolve_path(config_dir: Path, value: Any, default: str) -> Path:
    raw = str(value or "").strip() or default
    path = Path(raw)
    return path if path.is_absolute() else config_dir / path


def load_paths(config_path: str, hf_json_dir: Optional[str], by_model_dir: Optional[str]) -> Tuple[Path, Path, Path]:
    config_file = Path(config_path).resolve()
    cfg = load_config(str(config_file))
    config_dir = config_file.parent
    paths = cfg.get("paths", {}) or {}

    hf_dir = resolve_path(config_dir, hf_json_dir or paths.get("hf_json_dir"), "data/hf_json")
    if by_model_dir:
        model_dir = resolve_path(config_dir, by_model_dir, "data/by_model")
    else:
        configured = str(paths.get("by_model_dir", "") or "").strip()
        model_dir = resolve_path(config_dir, configured, str(hf_dir.parent / "by_model")) if configured else hf_dir.parent / "by_model"
    return config_dir, hf_dir, model_dir


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def compute_tertile_thresholds(counts: Sequence[int]) -> Tuple[int, int]:
    if len(counts) < 3:
        raise ValueError("Need at least three questions to compute tertile thresholds.")
    values = sorted(counts)
    first_n = len(values) // 3
    second_n = len(values) // 3
    short_max = values[first_n - 1]
    medium_max = values[first_n + second_n - 1]
    if medium_max < short_max:
        raise ValueError("Invalid tertile thresholds computed from word counts.")
    return short_max, medium_max


def length_bin(count: int, short_max: int, medium_max: int) -> str:
    if count <= short_max:
        return "Short"
    if count <= medium_max:
        return "Medium"
    return "Long"


def bin_label(bin_name: str, short_max: int, medium_max: int) -> str:
    if bin_name == "Short":
        return rf"Short ($\le {short_max}$ words)"
    if bin_name == "Medium":
        return rf"Medium ({short_max + 1}--{medium_max} words)"
    return rf"Long ($>{medium_max}$ words)"


def load_dataset(hf_dir: Path, short_max: Optional[int], medium_max: Optional[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], int, int]:
    dataset_files = sorted([p for p in hf_dir.glob("*.jsonl") if p.is_file()], key=lambda p: natural_key(p.stem))
    if not dataset_files:
        raise RuntimeError(f"No dataset JSONL files found under {hf_dir}")

    rows: List[Dict[str, Any]] = []
    for dataset_file in dataset_files:
        for row in read_jsonl(dataset_file):
            qid = str(row.get("id", "")).strip()
            if not qid:
                continue
            question_text = str(row.get("question_final") or row.get("question_standalone") or "")
            count = word_count(question_text)
            rows.append(
                {
                    "split": str(row.get("split") or dataset_file.stem),
                    "id": qid,
                    "chapter": str(row.get("chapter", "")),
                    "problem_number": str(row.get("problem_number", "")),
                    "sub_id": str(row.get("sub_id", "")),
                    "question_type": str(row.get("question_type", "")),
                    "answer_kind": str(row.get("answer_kind", "")),
                    "comparison_mode": str(row.get("comparison_mode", "")),
                    "word_count": count,
                    "question_preview": preview(question_text),
                }
            )

    if short_max is None or medium_max is None:
        short_max, medium_max = compute_tertile_thresholds([int(row["word_count"]) for row in rows])
    if short_max >= medium_max:
        raise ValueError(f"Expected short_max < medium_max, got {short_max=} {medium_max=}")

    by_id: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    for row in rows:
        qid = str(row["id"])
        row["length_bin"] = length_bin(int(row["word_count"]), short_max, medium_max)
        row["bin_order"] = BIN_ORDER.index(str(row["length_bin"])) + 1
        if qid in by_id:
            duplicates.append(qid)
        by_id[qid] = row
    if duplicates:
        dup_text = ", ".join(sorted(set(duplicates), key=natural_key)[:10])
        raise RuntimeError(f"Duplicate dataset ids found: {dup_text}")

    rows.sort(key=lambda row: (int(row["bin_order"]), int(row["word_count"]), natural_key(str(row["split"])), natural_key(str(row["id"]))))
    return rows, by_id, short_max, medium_max


def discover_model_dirs(by_model_dir: Path, models: Optional[str]) -> List[Path]:
    if models:
        names = [name.strip() for name in models.split(",") if name.strip()]
        model_dirs = [by_model_dir / name for name in names]
    else:
        model_dirs = [
            path
            for path in by_model_dir.iterdir()
            if path.is_dir() and (path / "evaluations" / "rule" / "summary.json").exists()
        ] if by_model_dir.exists() else []
    model_dirs = sorted(model_dirs, key=lambda p: natural_key(p.name))
    missing = [path for path in model_dirs if not (path / "evaluations" / "rule" / "summary.json").exists()]
    if missing:
        raise FileNotFoundError("Missing rule-evaluation summary for: " + ", ".join(str(path) for path in missing))
    if not model_dirs:
        raise RuntimeError(f"No model rule-evaluation summaries found under {by_model_dir}")
    return model_dirs


def load_rule_results(model_dir: Path) -> Dict[str, Dict[str, Any]]:
    eval_dir = model_dir / "evaluations" / "rule"
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    for path in sorted(eval_dir.glob("*.jsonl"), key=lambda p: natural_key(p.stem)):
        for row in read_jsonl(path):
            qid = str(row.get("id") or row.get("question_id") or "").strip()
            if qid:
                rows_by_id[qid] = row
    return rows_by_id


def percent(correct: int, total: int) -> Optional[float]:
    return (100.0 * correct / total) if total else None


def fmt_pct(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.1f}"


def collect_model_breakdowns(
    model_dirs: Sequence[Path],
    dataset_rows: Sequence[Dict[str, Any]],
    dataset_by_id: Dict[str, Dict[str, Any]],
    short_max: int,
    medium_max: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    breakdown_rows: List[Dict[str, Any]] = []
    item_result_rows: List[Dict[str, Any]] = []
    model_summaries: List[Dict[str, Any]] = []

    dataset_ids = set(dataset_by_id)
    for model_dir in model_dirs:
        model = model_dir.name
        display_name = DISPLAY_NAMES.get(model, model)
        eval_by_id = load_rule_results(model_dir)
        eval_ids = set(eval_by_id)
        missing_ids = sorted(dataset_ids - eval_ids, key=natural_key)
        extra_ids = sorted(eval_ids - dataset_ids, key=natural_key)

        totals = {bin_name: {"total": 0, "correct": 0} for bin_name in BIN_ORDER}
        for dataset_row in dataset_rows:
            qid = str(dataset_row["id"])
            bin_name = str(dataset_row["length_bin"])
            eval_row = eval_by_id.get(qid)
            if eval_row is None:
                item_result_rows.append(
                    {
                        "model": model,
                        "display_name": display_name,
                        "split": dataset_row["split"],
                        "id": qid,
                        "word_count": dataset_row["word_count"],
                        "length_bin": bin_name,
                        "is_correct": "",
                        "detail": "missing_rule_evaluation",
                        "question_type": dataset_row["question_type"],
                        "comparison_mode": dataset_row["comparison_mode"],
                    }
                )
                continue
            is_correct = bool(eval_row.get("is_correct", False))
            totals[bin_name]["total"] += 1
            totals[bin_name]["correct"] += int(is_correct)
            item_result_rows.append(
                {
                    "model": model,
                    "display_name": display_name,
                    "split": dataset_row["split"],
                    "id": qid,
                    "word_count": dataset_row["word_count"],
                    "length_bin": bin_name,
                    "is_correct": str(is_correct),
                    "detail": str(eval_row.get("detail", "")),
                    "question_type": dataset_row["question_type"],
                    "comparison_mode": dataset_row["comparison_mode"],
                }
            )

        for bin_name in BIN_ORDER:
            total = int(totals[bin_name]["total"])
            correct = int(totals[bin_name]["correct"])
            pct = percent(correct, total)
            breakdown_rows.append(
                {
                    "model": model,
                    "display_name": display_name,
                    "length_bin": bin_name,
                    "bin_order": BIN_ORDER.index(bin_name) + 1,
                    "bin_label": bin_label(bin_name, short_max, medium_max),
                    "total": total,
                    "correct": correct,
                    "accuracy": "" if pct is None else pct / 100.0,
                    "accuracy_pct": fmt_pct(pct),
                    "missing_eval_count": len(missing_ids),
                    "extra_eval_count": len(extra_ids),
                }
            )

        model_summaries.append(
            {
                "model": model,
                "display_name": display_name,
                "evaluation_dir": str(model_dir / "evaluations" / "rule"),
                "dataset_rows": len(dataset_rows),
                "evaluation_rows_matched": len(dataset_ids & eval_ids),
                "missing_eval_count": len(missing_ids),
                "extra_eval_count": len(extra_ids),
                "missing_eval_ids": missing_ids,
                "extra_eval_ids": extra_ids,
            }
        )

    breakdown_rows.sort(key=lambda row: (natural_key(str(row["model"])), int(row["bin_order"])))
    item_result_rows.sort(key=lambda row: (natural_key(str(row["model"])), int(BIN_ORDER.index(str(row["length_bin"]))), natural_key(str(row["split"])), natural_key(str(row["id"]))))
    return breakdown_rows, item_result_rows, model_summaries


def write_paper_table(path: Path, breakdown_rows: Sequence[Dict[str, Any]], short_max: int, medium_max: int) -> List[str]:
    by_model_bin = {
        (str(row["model"]), str(row["length_bin"])): row
        for row in breakdown_rows
    }
    available_models = [(label, model) for label, model in PAPER_MODEL_ORDER if (model, "Short") in by_model_bin]
    if not available_models:
        return []

    bin_counts = {
        bin_name: int(by_model_bin[(available_models[0][1], bin_name)]["total"])
        for bin_name in BIN_ORDER
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Breakdown by context length for the representative configurations, using word-count tertiles from the final question text.}",
        r"\label{tab:breakdown_context_length}",
        r"\small",
        r"\begin{tabular}{" + "l" + ("c" * (1 + len(available_models))) + r"}",
        r"\toprule",
        r"\textbf{Length bin} & \textbf{Count} & " + " & ".join(rf"\textbf{{{label}}}" for label, _ in available_models) + r" \\",
        r"\midrule",
    ]
    for bin_name in BIN_ORDER:
        values = [
            str(bin_counts[bin_name]),
            *[str(by_model_bin[(model, bin_name)]["accuracy_pct"]) for _, model in available_models],
        ]
        lines.append(f"{bin_label(bin_name, short_max, medium_max)} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return [model for _, model in available_models]


def write_markdown(
    path: Path,
    bin_counts: Dict[str, int],
    breakdown_rows: Sequence[Dict[str, Any]],
    paper_models: Sequence[str],
) -> None:
    by_model_bin = {(str(row["model"]), str(row["length_bin"])): row for row in breakdown_rows}
    lines = [
        "# Context-Length Breakdown",
        "",
        f"Word-count method: `{WORD_COUNT_METHOD}`.",
        "",
        "## Bin Counts",
        "",
        "| Bin | Count |",
        "| --- | ---: |",
    ]
    for bin_name in BIN_ORDER:
        lines.append(f"| {bin_name} | {bin_counts.get(bin_name, 0)} |")
    if paper_models:
        lines.extend(["", "## Paper Representative Models", "", "| Length bin | Count | " + " | ".join(DISPLAY_NAMES.get(model, model) for model in paper_models) + " |"])
        lines.append("| --- | ---: | " + " | ".join("---:" for _ in paper_models) + " |")
        for bin_name in BIN_ORDER:
            values = [str(bin_counts.get(bin_name, 0))]
            values.extend(str(by_model_bin[(model, bin_name)]["accuracy_pct"]) for model in paper_models)
            lines.append(f"| {bin_name} | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: str = "config.yaml",
    hf_json_dir: Optional[str] = None,
    by_model_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
    models: Optional[str] = None,
    binning: str = "tertiles",
    short_max: Optional[int] = None,
    medium_max: Optional[int] = None,
) -> Dict[str, Any]:
    config_dir, hf_dir, model_dir = load_paths(config_path, hf_json_dir, by_model_dir)
    if out_dir:
        out_path = Path(out_dir)
        reports_dir = out_path if out_path.is_absolute() else config_dir / out_path
    else:
        reports_dir = hf_dir.parent / "reports" / "context_length_breakdown"

    if binning not in {"tertiles", "fixed"}:
        raise ValueError("binning must be either 'tertiles' or 'fixed'")
    if binning == "fixed":
        short_max = 121 if short_max is None else short_max
        medium_max = 214 if medium_max is None else medium_max
    else:
        short_max = None
        medium_max = None

    dataset_rows, dataset_by_id, effective_short_max, effective_medium_max = load_dataset(
        hf_dir,
        short_max=short_max,
        medium_max=medium_max,
    )
    model_dirs = discover_model_dirs(model_dir, models=models)
    breakdown_rows, item_result_rows, model_summaries = collect_model_breakdowns(
        model_dirs,
        dataset_rows=dataset_rows,
        dataset_by_id=dataset_by_id,
        short_max=effective_short_max,
        medium_max=effective_medium_max,
    )

    bin_counts = {bin_name: 0 for bin_name in BIN_ORDER}
    for row in dataset_rows:
        bin_counts[str(row["length_bin"])] += 1

    question_lengths_csv = reports_dir / "question_lengths.csv"
    breakdown_csv = reports_dir / "model_context_length_breakdown.csv"
    item_results_csv = reports_dir / "model_context_length_item_results.csv"
    summary_json = reports_dir / "summary.json"
    paper_table_tex = reports_dir / "paper_context_length_table.tex"
    markdown = reports_dir / "context_length_breakdown.md"

    write_csv(
        question_lengths_csv,
        dataset_rows,
        [
            "split",
            "id",
            "chapter",
            "problem_number",
            "sub_id",
            "question_type",
            "answer_kind",
            "comparison_mode",
            "word_count",
            "length_bin",
            "bin_order",
            "question_preview",
        ],
    )
    write_csv(
        breakdown_csv,
        breakdown_rows,
        [
            "model",
            "display_name",
            "length_bin",
            "bin_order",
            "bin_label",
            "total",
            "correct",
            "accuracy",
            "accuracy_pct",
            "missing_eval_count",
            "extra_eval_count",
        ],
    )
    write_csv(
        item_results_csv,
        item_result_rows,
        [
            "model",
            "display_name",
            "split",
            "id",
            "word_count",
            "length_bin",
            "is_correct",
            "detail",
            "question_type",
            "comparison_mode",
        ],
    )
    paper_models = write_paper_table(
        paper_table_tex,
        breakdown_rows,
        short_max=effective_short_max,
        medium_max=effective_medium_max,
    )
    write_markdown(markdown, bin_counts, breakdown_rows, paper_models)

    outputs = {
        "summary_json": str(summary_json),
        "question_lengths_csv": str(question_lengths_csv),
        "model_context_length_breakdown_csv": str(breakdown_csv),
        "model_context_length_item_results_csv": str(item_results_csv),
        "paper_context_length_table_tex": str(paper_table_tex),
        "markdown": str(markdown),
    }
    summary = {
        "hf_json_dir": str(hf_dir),
        "by_model_dir": str(model_dir),
        "out_dir": str(reports_dir),
        "word_count_method": WORD_COUNT_METHOD,
        "question_text_field": "question_final",
        "binning": binning,
        "thresholds": {
            "short_max": effective_short_max,
            "medium_min": effective_short_max + 1,
            "medium_max": effective_medium_max,
            "long_min": effective_medium_max + 1,
        },
        "dataset_question_count": len(dataset_rows),
        "bin_counts": bin_counts,
        "models": model_summaries,
        "paper_table_models": paper_models,
        "outputs": outputs,
    }
    write_json(summary_json, summary)

    print(f"[context-length] wrote reports -> {reports_dir}")
    print(
        "[context-length] bins "
        + ", ".join(f"{name}={bin_counts[name]}" for name in BIN_ORDER)
        + f" thresholds=({effective_short_max}, {effective_medium_max})"
    )
    print(f"[context-length] models={len(model_dirs)} dataset_questions={len(dataset_rows)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and save context-length question bins and rule-evaluation breakdowns."
    )
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--hf-json-dir", default=None, help="Optional dataset JSONL directory override.")
    parser.add_argument("--by-model-dir", default=None, help="Optional by_model directory override.")
    parser.add_argument("--out-dir", default=None, help="Output directory for context-length reports.")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model artifact labels. Defaults to all models with evaluations/rule/summary.json.",
    )
    parser.add_argument(
        "--binning",
        default="tertiles",
        choices=["tertiles", "fixed"],
        help="Use current dataset tertiles or fixed thresholds. Fixed defaults to 121 and 214.",
    )
    parser.add_argument("--short-max", type=int, default=None, help="Fixed short-bin maximum word count.")
    parser.add_argument("--medium-max", type=int, default=None, help="Fixed medium-bin maximum word count.")
    args = parser.parse_args()
    run(
        config_path=args.config,
        hf_json_dir=args.hf_json_dir,
        by_model_dir=args.by_model_dir,
        out_dir=args.out_dir,
        models=args.models,
        binning=args.binning,
        short_max=args.short_max,
        medium_max=args.medium_max,
    )


if __name__ == "__main__":
    main()
