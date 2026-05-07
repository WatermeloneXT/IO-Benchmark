#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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


def collect_splits(hf_json_dir: Path, selected_split: str | None) -> List[Tuple[str, List[Dict[str, Any]]]]:
    files = sorted([p for p in hf_json_dir.glob("*.jsonl") if p.is_file()], key=lambda p: p.name)
    out: List[Tuple[str, List[Dict[str, Any]]]] = []
    for path in files:
        split = path.stem
        if selected_split and split != selected_split:
            continue
        if not selected_split and split.endswith("_balanced"):
            continue
        out.append((split, read_jsonl(path)))
    return out


def shorten(text: str, max_chars: int) -> str:
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def print_one_example(example: Dict[str, Any], idx: int, max_chars: int) -> None:
    print(f"[{idx}] id={example.get('id', '')}")
    print(f"    chapter={example.get('chapter', '')}  sub_id={example.get('sub_id', '')}")
    print(
        "    question_type="
        f"{example.get('question_type', '')}  comparison_mode={example.get('comparison_mode', '')}"
    )
    question = str(
        example.get("question_standalone")
        or example.get("question_final")
        or example.get("question_original")
        or ""
    )
    print("    question:")
    print("    " + shorten(question, max_chars).replace("\n", "\n    "))
    print(f"    reference_answer={example.get('reference_answer', 'N/A')}")
    print(f"    reference_answer_sympy={example.get('reference_answer_sympy', 'N/A')}")


def run(
    config_path: str = "config.yaml",
    split: str | None = None,
    n: int = 3,
    random_pick: bool = False,
    seed: int = 42,
    max_chars: int = 600,
) -> None:
    cfg = load_config(config_path)
    hf_json_dir = Path(cfg["paths"]["hf_json_dir"])

    split_rows = collect_splits(hf_json_dir, split)
    if not split_rows:
        raise RuntimeError(f"No split jsonl found under {hf_json_dir} (split={split}).")

    rng = random.Random(seed)
    for split_name, rows in split_rows:
        if not rows:
            print(f"\n=== split: {split_name} ===")
            print("(empty)")
            continue

        k = min(max(1, n), len(rows))
        chosen = rng.sample(rows, k) if random_pick else rows[:k]

        print(f"\n=== split: {split_name} | total={len(rows)} | showing={k} ===")
        for i, ex in enumerate(chosen, start=1):
            print_one_example(ex, i, max_chars=max_chars)
            print("-" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show examples from generated econ dataset.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--split", default=None, help="Split name like chapter_6 (default: all splits)")
    parser.add_argument("--num", "-n", type=int, default=3, help="Number of examples per split")
    parser.add_argument("--random", action="store_true", help="Randomly sample examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --random")
    parser.add_argument("--max-chars", type=int, default=600, help="Max chars shown for question text")
    args = parser.parse_args()
    run(
        config_path=args.config,
        split=args.split,
        n=args.num,
        random_pick=args.random,
        seed=args.seed,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
