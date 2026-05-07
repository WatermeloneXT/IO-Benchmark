#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.llm_utils import create_client, load_config

from build_steps.common import (
    natural_key,
    read_jsonl,
    resolve_target_splits,
    stage_file,
    write_jsonl,
)
from build_steps.llm_ops import llm_split_question_pairs


def build_stage2_for_split(
    split: str,
    stage1_rows: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    client: Any,
) -> List[Dict[str, Any]]:
    model = cfg["azure"]["deployment_name"]
    max_retries = int(cfg["azure"].get("max_retries", 4))
    build_cfg = cfg.get("build", {})
    max_workers = int(build_cfg.get("max_workers", 1) or 1)

    if not stage1_rows:
        return []

    def _process_one(idx: int, row: Dict[str, Any]) -> Tuple[int, Dict[str, Any], str]:
        qid = str(row.get("id", "")).strip()
        question_latex = str(row.get("question_original", "")).strip()
        corrected_answer = str(row.get("reference_answer_corrected", "N/A")).strip() or "N/A"

        stage2 = llm_split_question_pairs(
            client=client,
            model=model,
            max_retries=max_retries,
            temperature=float(build_cfg.get("split_temperature", 0.0)),
            question_latex=question_latex,
            correct_reference_answer=corrected_answer,
            split=split,
            question_id=qid,
        )

        out = {
            "id": qid,
            "chapter": row.get("chapter"),
            "split": row.get("split", split),
            "problem_number": row.get("problem_number"),
            "sub_id": row.get("sub_id"),
            "source_path": row.get("source_path"),
            "source_problem_path": row.get("source_problem_path"),
            "question_original": question_latex,
            "source_problem_latex": row.get("source_problem_latex", ""),
            "source_answer_latex": row.get("source_answer_latex", ""),
            "reference_answer_latex_from_by_problem": row.get("reference_answer_latex_from_by_problem", ""),
            "reference_answer_original": row.get("reference_answer_original", "N/A"),
            "reference_answer_corrected": corrected_answer,
            "reference_answer_generated_by_llm": row.get("reference_answer_generated_by_llm", False),
            "stage1_confidence": row.get("confidence", "low"),
            "question_count": stage2["question_count"],
            "split_generated_by_llm": stage2["split_generated_by_llm"],
            "pairs": stage2["pairs"],
            "analysis": stage2["analysis"],
            "stage1_json": row.get("llm_json", {}),
            "stage2_json": stage2["raw_json"],
        }
        return idx, out, f"[step2] {split} {idx}/{len(stage1_rows)} -> {qid}"

    rows_indexed: List[Tuple[int, Dict[str, Any]]] = []
    if max_workers <= 1:
        for idx, row in enumerate(stage1_rows, start=1):
            i, out, msg = _process_one(idx, row)
            print(msg)
            rows_indexed.append((i, out))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(_process_one, idx, row): idx for idx, row in enumerate(stage1_rows, start=1)}
            for fut in as_completed(future_map):
                try:
                    i, out, msg = fut.result()
                    print(msg)
                    rows_indexed.append((i, out))
                except Exception as err:  # noqa: BLE001
                    i = future_map[fut]
                    print(f"[step2] {split} {i}/{len(stage1_rows)} -> failed: {err}")

    rows_indexed.sort(key=lambda x: (x[0], natural_key(str(x[1].get("id", "")))))
    return [r for _, r in rows_indexed]


def run(
    config_path: str = "config.yaml",
    chapter: Optional[str] = None,
    target_splits: Optional[List[str]] = None,
    skip_existing: bool = False,
) -> List[str]:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    chapters_root = Path(paths["chapters_root"])
    pipeline_dir = Path(paths.get("pipeline_dir") or (Path(paths["hf_json_dir"]).parent / "pipeline"))
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    splits = resolve_target_splits(chapters_root, chapter=chapter, target_splits=target_splits)
    if not splits:
        # Fallback to all available stage1 files.
        splits = sorted(
            [p.name.replace("_stage1_reference.jsonl", "") for p in pipeline_dir.glob("*_stage1_reference.jsonl")],
            key=natural_key,
        )

    if not splits:
        print("[step2] no target splits found.")
        return []

    client: Any = None
    processed: List[str] = []

    for split in splits:
        in_file = stage_file(pipeline_dir, split, "stage1_reference")
        if not in_file.exists():
            print(f"[step2] skip {split}: missing {in_file.name}")
            continue
        out_file = stage_file(pipeline_dir, split, "stage2_split")
        if skip_existing and out_file.exists() and out_file.stat().st_size > 0:
            print(f"[step2] skip {split}: existing output -> {out_file}")
            processed.append(split)
            continue

        stage1_rows = read_jsonl(in_file)
        if client is None:
            client = create_client(cfg)
        stage2_rows = build_stage2_for_split(split, stage1_rows, cfg, client)
        if not stage2_rows:
            continue

        write_jsonl(out_file, stage2_rows)
        print(f"[step2] wrote {len(stage2_rows)} rows -> {out_file}")
        processed.append(split)

    return sorted(list(dict.fromkeys(processed)), key=natural_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage2: multi-question splitting.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--chapter", default=None, help="Optional chapter selector, e.g. 6 or chapter_6")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip split if stage2 output file already exists and is non-empty.",
    )
    args = parser.parse_args()
    run(config_path=args.config, chapter=args.chapter, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
