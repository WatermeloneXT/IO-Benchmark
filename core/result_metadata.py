from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.cost_logging import get_run_context


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_result_metadata(
    *,
    stage: str,
    solver_model: str,
    split: str,
    question_id: str,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    run_ctx = get_run_context()
    return {
        "workflow_id": str(run_ctx.get("workflow_id", "") or ""),
        "run_id": str(run_ctx.get("run_id", "") or ""),
        "stage": stage,
        "solver_model": solver_model,
        "judge_model": str(judge_model or ""),
        "split": split,
        "question_id": question_id,
        "created_at": now_utc_iso(),
    }
