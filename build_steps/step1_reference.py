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
    compact_text,
    discover_chapters,
    filter_chapters,
    infer_sub_id,
    iter_problem_files,
    natural_key,
    read_json,
    read_jsonl,
    sanitize_split_name,
    split_already_exists,
    stage_file,
    write_jsonl,
)
from build_steps.llm_ops import llm_correct_reference, load_source_problem


def build_stage1_for_chapter(
    chapter_id: str,
    chapter_dir: Path,
    cfg: Dict[str, Any],
    client: Optional[Any],
    existing_rows_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    model = cfg["azure"]["deployment_name"]
    max_retries = int(cfg["azure"].get("max_retries", 4))
    build_cfg = cfg.get("build", {})
    limit = build_cfg.get("max_items_per_split")
    max_workers = int(build_cfg.get("max_workers", 1) or 1)

    files = iter_problem_files(chapter_dir)
    if not files:
        print(f"[step1] skip {chapter_id}: no out_split json files found.")
        return []
    if limit is not None:
        files = files[: int(limit)]

    client_ref = client

    def _process_one(idx: int, path: Path) -> Tuple[int, Optional[Dict[str, Any]], str]:
        nonlocal client_ref
        raw = read_json(path)
        prob = raw.get("problem", raw) if isinstance(raw, dict) else {}
        problem_latex = str(prob.get("problem_latex", "")).strip()
        answer_latex = str(prob.get("answer_latex", "")).strip()
        if not problem_latex:
            return idx, None, f"[step1] {chapter_id} {idx}/{len(files)} -> skip(empty problem_latex)"

        problem_number = str(prob.get("original_number") or prob.get("number") or path.parent.name).strip()
        sub_id = infer_sub_id(prob, path.stem)
        base_id = f"{problem_number}/{sub_id or 'main'}"
        existing_row = (existing_rows_by_id or {}).get(base_id)
        if existing_row and existing_row.get("llm_json"):
            return idx, existing_row, f"[step1] {chapter_id} {idx}/{len(files)} -> {base_id} (reuse cached)"
        if existing_row and not existing_row.get("llm_json"):
            retry_note = "retry empty llm_json"
        else:
            retry_note = "fresh"

        source_problem_latex, source_answer_latex, source_problem_path = load_source_problem(
            chapter_dir=chapter_dir,
            problem_obj=prob,
            problem_number=problem_number,
        )
        source_reference_answer = compact_text(answer_latex or source_answer_latex)
        if client_ref is None:
            client_ref = create_client(cfg)

        stage1 = llm_correct_reference(
            client=client_ref,
            model=model,
            max_retries=max_retries,
            temperature=float(build_cfg.get("reference_fix_temperature", 0.0)),
            question_latex=problem_latex,
            candidate_answer=source_reference_answer,
            split=sanitize_split_name(chapter_id),
            question_id=base_id,
        )

        row = {
            "id": base_id,
            "chapter": chapter_id,
            "split": sanitize_split_name(chapter_id),
            "problem_number": problem_number,
            "sub_id": sub_id,
            "source_path": path.as_posix(),
            "source_problem_path": source_problem_path,
            "question_original": problem_latex,
            "source_problem_latex": source_problem_latex,
            "source_answer_latex": answer_latex,
            "reference_answer_latex_from_by_problem": source_answer_latex,
            "reference_answer_original": source_reference_answer,
            "reference_answer_corrected": stage1["final_reference_answer"],
            "reference_is_correct": stage1["reference_is_correct"],
            "reference_answer_generated_by_llm": stage1["reference_generated_by_llm"],
            "confidence": stage1["confidence"],
            "analysis": stage1["analysis"],
            "llm_json": stage1["raw_json"],
        }
        return idx, row, f"[step1] {chapter_id} {idx}/{len(files)} -> {base_id} ({retry_note})"

    rows_indexed: List[Tuple[int, Dict[str, Any]]] = []
    if max_workers <= 1:
        for idx, path in enumerate(files, start=1):
            i, row, msg = _process_one(idx, path)
            print(msg)
            if row is not None:
                rows_indexed.append((i, row))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(_process_one, idx, path): idx for idx, path in enumerate(files, start=1)}
            for fut in as_completed(future_map):
                try:
                    i, row, msg = fut.result()
                    print(msg)
                    if row is not None:
                        rows_indexed.append((i, row))
                except Exception as err:  # noqa: BLE001
                    i = future_map[fut]
                    print(f"[step1] {chapter_id} {i}/{len(files)} -> failed: {err}")

    rows_indexed.sort(key=lambda x: (x[0], natural_key(str(x[1].get("id", "")))))
    return [r for _, r in rows_indexed]


def run(
    config_path: str = "config.yaml",
    chapter: Optional[str] = None,
    skip_existing_split: Optional[bool] = None,
    target_splits: Optional[List[str]] = None,
    force_all: bool = False,
    skip_existing: bool = False,
) -> List[str]:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    build_cfg = cfg.get("build", {})

    chapters_root = Path(paths["chapters_root"])
    hf_json_dir = Path(paths["hf_json_dir"])
    hf_dataset_dir = Path(paths["hf_dataset_dir"])
    pipeline_dir = Path(paths.get("pipeline_dir") or (hf_json_dir.parent / "pipeline"))
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    chapters = filter_chapters(discover_chapters(chapters_root), chapter)
    if target_splits is not None:
        target = set(target_splits)
        chapters = [x for x in chapters if sanitize_split_name(x[0]) in target]

    skip_existing_final = (
        bool(build_cfg.get("skip_existing_split", False))
        if skip_existing_split is None
        else bool(skip_existing_split)
    )

    if not chapters:
        print("[step1] no chapters selected.")
        return []

    client: Optional[Any] = None
    processed_splits: List[str] = []

    for chapter_id, chapter_dir in chapters:
        split = sanitize_split_name(chapter_id)
        out_file = stage_file(pipeline_dir, split, "stage1_reference")
        if skip_existing and out_file.exists() and out_file.stat().st_size > 0:
            print(f"[step1] skip {chapter_id} ({split}): existing output -> {out_file}")
            processed_splits.append(split)
            continue
        if skip_existing_final and split_already_exists(hf_json_dir, hf_dataset_dir, split):
            print(f"[step1] skip {chapter_id} ({split}): final split already exists")
            continue

        existing_rows_by_id: Dict[str, Dict[str, Any]] = {}
        if (not force_all) and out_file.exists():
            for row in read_jsonl(out_file):
                qid = str(row.get("id", "")).strip()
                if qid:
                    existing_rows_by_id[qid] = row

        rows = build_stage1_for_chapter(
            chapter_id=chapter_id,
            chapter_dir=chapter_dir,
            cfg=cfg,
            client=client,
            existing_rows_by_id=(None if force_all else existing_rows_by_id),
        )
        if not rows:
            continue
        write_jsonl(out_file, rows)
        print(f"[step1] wrote {len(rows)} rows -> {out_file}")
        processed_splits.append(split)

    return sorted(list(dict.fromkeys(processed_splits)), key=natural_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage1: reference-answer correction.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--chapter", default=None, help="Optional chapter selector, e.g. 6 or chapter_6")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--skip-existing-split",
        action="store_true",
        dest="skip_existing_split",
        help="Skip stage1 for splits whose final output already exists.",
    )
    group.add_argument(
        "--no-skip-existing-split",
        action="store_false",
        dest="skip_existing_split",
        help="Always regenerate stage1 even if final split already exists.",
    )
    parser.set_defaults(skip_existing_split=None)
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Re-run all items in this split/chapter (ignore cached non-empty llm_json rows).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip split if stage1 output file already exists and is non-empty.",
    )
    args = parser.parse_args()
    run(
        config_path=args.config,
        chapter=args.chapter,
        skip_existing_split=args.skip_existing_split,
        force_all=args.force_all,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
