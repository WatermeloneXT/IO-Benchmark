from __future__ import annotations

from typing import List, Optional

from .common import natural_key
from .step1_reference import run as run_step1
from .step2_split import run as run_step2
from .step3_transform import run as run_step3
from .step5_rebalance_judge import run as run_step4
from .step4_finalize import run as run_step5


def run_all(
    config_path: str = "config.yaml",
    chapter: Optional[str] = None,
    skip_existing_split: Optional[bool] = None,
    skip_existing_llm: bool = False,
) -> List[str]:
    splits = run_step1(
        config_path=config_path,
        chapter=chapter,
        skip_existing_split=skip_existing_split,
        skip_existing=skip_existing_llm,
    )
    if not splits:
        print("[build] no splits generated in step1. nothing to do.")
        return []

    splits = run_step2(
        config_path=config_path,
        chapter=chapter,
        target_splits=splits,
        skip_existing=skip_existing_llm,
    )
    if not splits:
        print("[build] no splits generated in step2. nothing to do.")
        return []

    splits = run_step3(
        config_path=config_path,
        chapter=chapter,
        target_splits=splits,
        skip_existing=skip_existing_llm,
    )
    if not splits:
        print("[build] no splits generated in step3. nothing to do.")
        return []

    splits = run_step4(
        config_path=config_path,
        chapter=chapter,
        target_splits=splits,
        skip_existing=skip_existing_llm,
    )
    if not splits:
        print("[build] no splits generated in step4. nothing to do.")
        return []

    splits = run_step5(config_path=config_path, chapter=chapter, target_splits=splits)
    return sorted(list(dict.fromkeys(splits)), key=natural_key)
