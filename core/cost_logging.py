from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

from core.file_naming import sanitize_model_tag
from core.model_layout import workflows_root


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:  # noqa: BLE001
        return 0


def _safe_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


@dataclass
class _RunContext:
    run_id: str
    workflow_id: str
    stage: str
    command: str
    config_path: str
    started_at: str
    root_dir: Path
    workflows_dir: Path
    call_log_file: Path
    summary_file: Path
    enabled: bool
    capture_raw_usage: bool
    solver_model: str
    output_paths: List[str]


class _CallLogWriter:
    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file
        self.queue: Queue[Optional[Dict[str, Any]]] = Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self._summary: Dict[str, Any] = {
            "api_call_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_creation_tokens": 0,
            "total_cache_read_tokens": 0,
            "by_model": {},
            "by_operation": {},
        }

    def start(self) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def enqueue(self, row: Dict[str, Any]) -> None:
        self.queue.put(row)

    def close(self) -> Dict[str, Any]:
        self.queue.put(None)
        self.thread.join()
        return self._summary

    def _inc_group(self, key: str, name: str, row: Dict[str, Any]) -> None:
        bucket = self._summary[key].setdefault(
            name,
            {
                "api_call_count": 0,
                "success_count": 0,
                "error_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
        )
        bucket["api_call_count"] += 1
        if row.get("status") == "success":
            bucket["success_count"] += 1
        else:
            bucket["error_count"] += 1
        bucket["input_tokens"] += _to_int(row.get("usage_input_tokens"))
        bucket["output_tokens"] += _to_int(row.get("usage_output_tokens"))

    def _accumulate(self, row: Dict[str, Any]) -> None:
        self._summary["api_call_count"] += 1
        if row.get("status") == "success":
            self._summary["success_count"] += 1
        else:
            self._summary["error_count"] += 1
        self._summary["total_input_tokens"] += _to_int(row.get("usage_input_tokens"))
        self._summary["total_output_tokens"] += _to_int(row.get("usage_output_tokens"))
        self._summary["total_cache_creation_tokens"] += _to_int(row.get("usage_cache_creation_tokens"))
        self._summary["total_cache_read_tokens"] += _to_int(row.get("usage_cache_read_tokens"))
        self._inc_group("by_model", str(row.get("model_requested", "") or "unknown"), row)
        self._inc_group("by_operation", str(row.get("operation", "") or "unknown"), row)

    def _worker(self) -> None:
        with self.output_file.open("a", encoding="utf-8") as f:
            while True:
                try:
                    row = self.queue.get(timeout=0.5)
                except Empty:
                    continue
                if row is None:
                    break
                f.write(_safe_json(row) + "\n")
                self._accumulate(row)


_LOCK = threading.Lock()
_ACTIVE_RUN: Optional[_RunContext] = None
_WRITER: Optional[_CallLogWriter] = None


def _resolve_root_dir(cfg: Dict[str, Any]) -> Path:
    cost_cfg = cfg.get("cost_logging", {}) or {}
    root = str(cost_cfg.get("root_dir", "data/by_model")).strip() or "data/by_model"
    root_path = Path(root)
    if root_path.is_absolute():
        return root_path
    base = Path(str(cfg.get("_config_path", "config.yaml"))).resolve().parent
    return (base / root_path).resolve()


def _update_workflow_manifest(workflows_dir: Path, run_summary: Dict[str, Any]) -> None:
    workflow_id = str(run_summary.get("workflow_id", "") or "").strip()
    if not workflow_id:
        return
    manifest_file = workflows_dir / workflow_id / "manifest.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {"workflow_id": workflow_id, "updated_at": _now_utc_iso(), "runs": []}
    if manifest_file.exists():
        try:
            with manifest_file.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                manifest.update(loaded)
                if not isinstance(manifest.get("runs"), list):
                    manifest["runs"] = []
        except Exception:  # noqa: BLE001
            manifest = {"workflow_id": workflow_id, "updated_at": _now_utc_iso(), "runs": []}

    run_entry = {
        "run_id": run_summary.get("run_id", ""),
        "stage": run_summary.get("stage", ""),
        "status": run_summary.get("status", ""),
        "solver_model": run_summary.get("solver_model", ""),
        "started_at": run_summary.get("started_at", ""),
        "ended_at": run_summary.get("ended_at", ""),
        "summary_file": run_summary.get("summary_file", ""),
        "call_log_file": run_summary.get("call_log_file", ""),
        "output_paths": run_summary.get("output_paths", []),
    }

    runs: List[Dict[str, Any]] = [x for x in manifest.get("runs", []) if isinstance(x, dict)]
    filtered_runs = [x for x in runs if str(x.get("run_id", "")) != str(run_entry["run_id"])]
    filtered_runs.append(run_entry)
    filtered_runs.sort(key=lambda x: str(x.get("started_at", "")))
    manifest["runs"] = filtered_runs
    manifest["updated_at"] = _now_utc_iso()

    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def start_run(
    cfg: Dict[str, Any],
    *,
    stage: str,
    command: str,
    config_path: str,
    solver_model: Optional[str] = None,
    workflow_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Optional[str]:
    global _ACTIVE_RUN, _WRITER
    with _LOCK:
        if _ACTIVE_RUN is not None:
            return _ACTIVE_RUN.run_id

        cost_cfg = cfg.get("cost_logging", {}) or {}
        enabled = bool(cost_cfg.get("enabled", True))
        capture_raw_usage = bool(cost_cfg.get("capture_raw_usage", True))
        root_dir = _resolve_root_dir(cfg)
        workflows_dir = workflows_root(cfg.get("paths", {}) or {})
        solver_model_name = str(solver_model or "").strip()
        model_folder = sanitize_model_tag(solver_model_name) if solver_model_name else "_global"
        model_root_dir = root_dir / model_folder
        cost_dir = model_root_dir / "cost"
        started_at = _now_utc_iso()
        date_part = started_at[:10]
        resolved_run_id = (run_id or "").strip() or uuid.uuid4().hex[:16]
        resolved_workflow_id = (workflow_id or "").strip() or resolved_run_id
        if not enabled:
            _ACTIVE_RUN = None
            _WRITER = None
            return None
        call_log_file = cost_dir / "call_logs" / date_part / f"{resolved_run_id}.jsonl"
        summary_file = cost_dir / "run_summaries" / f"{resolved_run_id}.json"

        _ACTIVE_RUN = _RunContext(
            run_id=resolved_run_id,
            workflow_id=resolved_workflow_id,
            stage=(stage or "").strip() or "unknown",
            command=(command or "").strip(),
            config_path=str(Path(config_path).resolve()),
            started_at=started_at,
            root_dir=root_dir,
            workflows_dir=workflows_dir,
            call_log_file=call_log_file,
            summary_file=summary_file,
            enabled=enabled,
            capture_raw_usage=capture_raw_usage,
            solver_model=solver_model_name,
            output_paths=[],
        )

        if enabled:
            _WRITER = _CallLogWriter(call_log_file)
            _WRITER.start()
        else:
            _WRITER = None
        return resolved_run_id


def get_run_context() -> Dict[str, Any]:
    with _LOCK:
        if _ACTIVE_RUN is None:
            return {}
        return {
            "run_id": _ACTIVE_RUN.run_id,
            "workflow_id": _ACTIVE_RUN.workflow_id,
            "stage": _ACTIVE_RUN.stage,
            "command": _ACTIVE_RUN.command,
            "config_path": _ACTIVE_RUN.config_path,
            "started_at": _ACTIVE_RUN.started_at,
            "enabled": _ACTIVE_RUN.enabled,
            "capture_raw_usage": _ACTIVE_RUN.capture_raw_usage,
            "solver_model": _ACTIVE_RUN.solver_model,
            "output_paths": list(_ACTIVE_RUN.output_paths),
        }


def log_api_call(row: Dict[str, Any]) -> None:
    with _LOCK:
        if _ACTIVE_RUN is None or _WRITER is None or not _ACTIVE_RUN.enabled:
            return
        payload = dict(row)
        payload.setdefault("timestamp_utc", _now_utc_iso())
        payload.setdefault("run_id", _ACTIVE_RUN.run_id)
        payload.setdefault("workflow_id", _ACTIVE_RUN.workflow_id)
        payload.setdefault("stage", _ACTIVE_RUN.stage)
        payload.setdefault("solver_model", _ACTIVE_RUN.solver_model)
        if not _ACTIVE_RUN.capture_raw_usage:
            payload.pop("raw_usage_json", None)
        _WRITER.enqueue(payload)


def register_run_outputs(paths: List[str | Path]) -> None:
    with _LOCK:
        if _ACTIVE_RUN is None:
            return
        seen = set(_ACTIVE_RUN.output_paths)
        for p in paths:
            value = str(Path(p).resolve())
            if value in seen:
                continue
            _ACTIVE_RUN.output_paths.append(value)
            seen.add(value)


def register_run_output(path: str | Path) -> None:
    register_run_outputs([path])


def finish_run(status: str = "success", error_message: str = "") -> None:
    global _ACTIVE_RUN, _WRITER
    with _LOCK:
        if _ACTIVE_RUN is None:
            return
        active = _ACTIVE_RUN
        writer = _WRITER
        _ACTIVE_RUN = None
        _WRITER = None

    summary_counts: Dict[str, Any] = {
        "api_call_count": 0,
        "success_count": 0,
        "error_count": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cache_creation_tokens": 0,
        "total_cache_read_tokens": 0,
        "by_model": {},
        "by_operation": {},
    }
    if writer is not None:
        summary_counts = writer.close()

    ended_at = _now_utc_iso()
    run_summary = {
        "run_id": active.run_id,
        "workflow_id": active.workflow_id,
        "stage": active.stage,
        "solver_model": active.solver_model,
        "command": active.command,
        "config_path": active.config_path,
        "status": status,
        "error_message": error_message,
        "started_at": active.started_at,
        "ended_at": ended_at,
        "call_log_file": str(active.call_log_file),
        "summary_file": str(active.summary_file),
        "output_paths": list(active.output_paths),
        "summary_generated_at": ended_at,
        **summary_counts,
    }
    active.summary_file.parent.mkdir(parents=True, exist_ok=True)
    with active.summary_file.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)
    _update_workflow_manifest(active.workflows_dir, run_summary)
