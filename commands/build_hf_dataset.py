#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_steps.pipeline_runner import run_all


def run(
    config_path: str = "config.yaml",
    chapter: Optional[str] = None,
    skip_existing_split: Optional[bool] = None,
    skip_existing_llm: bool = False,
) -> None:
    run_all(
        config_path=config_path,
        chapter=chapter,
        skip_existing_split=skip_existing_split,
        skip_existing_llm=skip_existing_llm,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chapter-split HuggingFace dataset for econ benchmark.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--chapter",
        default=None,
        help="Only build specific chapter id(s), comma-separated. Examples: 6 or 6/7 or chapter_6",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--skip-existing-split",
        action="store_true",
        dest="skip_existing_split",
        help="Skip generation when target chapter split data already exists.",
    )
    group.add_argument(
        "--no-skip-existing-split",
        action="store_false",
        dest="skip_existing_split",
        help="Always regenerate even if split already exists.",
    )
    parser.add_argument(
        "--skip-existing-llm",
        action="store_true",
        help="For step1/step2/step3/step4, skip split when target output files already exist.",
    )
    parser.set_defaults(skip_existing_split=None)
    args = parser.parse_args()
    run(
        config_path=args.config,
        chapter=args.chapter,
        skip_existing_split=args.skip_existing_split,
        skip_existing_llm=args.skip_existing_llm,
    )


if __name__ == "__main__":
    main()
