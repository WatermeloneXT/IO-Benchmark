#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_utils import azure_json_call, create_client, load_config


SYSTEM_LLM_QA = r"""
You are an auditor for dataset conversion quality.

You must evaluate ONE converted item against the original out_split data.
Do semantic checking, not string matching.

Check two stages:
1) conversion quality:
   - objective fidelity (same mathematical target as original sub-question)
   - self-containedness (can solve this item alone without external context)
   - final-answer-only style (asks for final answer, not full proof/discussion)
2) extraction quality:
   - whether extracted reference_answer is correct given source answer
   - whether answer format follows rules:
     * if cannot extract -> N/A
     * number/letter -> only number/letter token
     * relation -> complete relation
   - whether reference_answer_sympy is consistent with reference_answer

Return JSON only. Schema:
{
  "conversion": {
    "status": "pass|warn|fail",
    "reason": "...",
    "issues": [
      {"severity":"info|warn|error","code":"...","message":"...","details":"..."}
    ]
  },
  "extraction": {
    "status": "pass|warn|fail",
    "reason": "...",
    "issues": [
      {"severity":"info|warn|error","code":"...","message":"...","details":"..."}
    ]
  },
  "overall": {
    "verdict": "pass|warn|fail",
    "score": 0,
    "notes": "..."
  }
}
"""


@dataclass
class Issue:
    id: str
    split: str
    stage: str
    severity: str
    code: str
    message: str
    details: Dict[str, Any]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_code(text: str, fallback: str = "llm_issue") -> str:
    x = (text or "").strip().lower()
    if not x:
        return fallback
    x = re.sub(r"[^a-z0-9_]+", "_", x).strip("_")
    return x or fallback


def normalize_severity(text: str) -> str:
    s = (text or "").strip().lower()
    if s in {"error", "warn", "info"}:
        return s
    if s in {"warning"}:
        return "warn"
    return "warn"


def remap_source_path(raw_source_path: str, chapters_root: Path) -> Optional[Path]:
    s = (raw_source_path or "").strip()
    if not s:
        return None
    p = Path(s)
    if p.exists() and p.is_file():
        return p
    marker = "datasets/chapters/"
    if marker in s:
        rel = s.split(marker, 1)[1]
        p2 = chapters_root / rel
        if p2.exists() and p2.is_file():
            return p2
    return None


def resolve_out_split_path(row: Dict[str, Any], chapters_root: Path) -> Optional[Path]:
    p = remap_source_path(str(row.get("source_path", "")), chapters_root)
    if p:
        return p

    chapter = str(row.get("chapter", "")).strip()
    problem_number = str(row.get("problem_number", "")).strip()
    sub_id = str(row.get("sub_id", "")).strip() or "main"
    sub_file = f"{sub_id}.json" if not sub_id.endswith(".json") else sub_id
    candidates = [
        chapters_root / chapter / "out_split" / problem_number / sub_file,
        chapters_root / chapter / "out_split" / problem_number / "main.json",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def mk_issue(
    row: Dict[str, Any],
    stage: str,
    severity: str,
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Issue:
    return Issue(
        id=str(row.get("id", "")),
        split=str(row.get("split", "")),
        stage=stage,
        severity=severity,
        code=normalize_code(code),
        message=message,
        details=details or {},
    )


def llm_eval_one(
    row: Dict[str, Any],
    out_problem_latex: str,
    out_answer_latex: str,
    client: Any,
    model: str,
    max_retries: int,
) -> Dict[str, Any]:
    payload = {
        "row": {
            "id": row.get("id"),
            "split": row.get("split"),
            "chapter": row.get("chapter"),
            "problem_number": row.get("problem_number"),
            "sub_id": row.get("sub_id"),
            "question_original": row.get("question_original"),
            "question_standalone": row.get("question_standalone") or row.get("question_final"),
            "reference_answer": row.get("reference_answer"),
            "reference_answer_sympy": row.get("reference_answer_sympy"),
            "answer_kind": row.get("answer_kind"),
            "extract_confidence": row.get("extract_confidence"),
        },
        "out_split_source": {
            "problem_latex": out_problem_latex,
            "answer_latex": out_answer_latex,
        },
        "rules": {
            "na_rule": "if cannot extract, use N/A",
            "number_or_letter_rule": "if answer is number/letter, extract only that token",
            "relation_rule": "if answer is a relation, keep the complete relation",
        },
    }
    user = "Audit this converted item. Input JSON:\n" + json.dumps(payload, ensure_ascii=False)
    obj = azure_json_call(
        client=client,
        model=model,
        system=SYSTEM_LLM_QA,
        user=user,
        temperature=0.0,
        max_retries=max_retries,
        telemetry={
            "operation": "analyze_quality_judge",
            "split": str(row.get("split", "")).strip(),
            "question_id": str(row.get("id", "")).strip(),
        },
    )
    if not isinstance(obj, dict):
        return {}
    return obj


def issues_from_assessment(row: Dict[str, Any], assessment: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    for stage in ("conversion", "extraction"):
        block = assessment.get(stage, {})
        if not isinstance(block, dict):
            continue
        raw_issues = block.get("issues", [])
        if not isinstance(raw_issues, list):
            raw_issues = []
        for it in raw_issues:
            if not isinstance(it, dict):
                continue
            severity = normalize_severity(str(it.get("severity", "warn")))
            code = str(it.get("code", "llm_issue"))
            message = str(it.get("message", "")).strip() or f"{stage} issue flagged by LLM"
            details = it.get("details")
            if isinstance(details, str):
                details = {"text": details}
            if not isinstance(details, dict):
                details = {}
            issues.append(mk_issue(row, stage, severity, code, message, details))
    return issues


def run(
    config_path: str = "config.yaml",
    split: Optional[str] = None,
    max_items: Optional[int] = None,
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    hf_json_dir = Path(cfg["paths"]["hf_json_dir"])
    chapters_root = Path(cfg["paths"]["chapters_root"])
    model = cfg["azure"]["deployment_name"]
    max_retries = int(cfg["azure"].get("max_retries", 4))
    max_workers = int(cfg.get("analyze", {}).get("max_workers", 1) or 1)

    if out_json is None:
        out_json = str((Path(cfg["paths"]["hf_json_dir"]).parent / "reports" / "conversion_quality_report.json").resolve())
    if out_md is None:
        out_md = str((Path(cfg["paths"]["hf_json_dir"]).parent / "reports" / "conversion_quality_report.md").resolve())
    out_json_path = Path(out_json)
    if skip_existing and out_json_path.exists() and out_json_path.stat().st_size > 0:
        print(f"[analyze] skip existing report -> {out_json_path}")
        return read_json(out_json_path)

    files = sorted([p for p in hf_json_dir.glob("*.jsonl") if p.is_file()], key=lambda p: p.name)
    if split:
        files = [p for p in files if p.stem == split]

    rows: List[Dict[str, Any]] = []
    for fp in files:
        for row in read_jsonl(fp):
            row = dict(row)
            row["split"] = row.get("split") or fp.stem
            rows.append(row)

    if max_items is not None and max_items > 0:
        rows = rows[:max_items]

    try:
        client = create_client(cfg)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependency: openai. Install project deps first: pip install -r requirements.txt"
        ) from e

    issues: List[Issue] = []
    assessments: Dict[str, Any] = {}
    missing_source_rows = 0

    def _analyze_one(idx: int, row: Dict[str, Any]) -> Tuple[int, str, Optional[Dict[str, Any]], List[Issue], int, str]:
        rid = str(row.get("id", ""))
        src = resolve_out_split_path(row, chapters_root)
        if src is None:
            miss_issue = mk_issue(
                row, "source", "error", "out_split_source_missing",
                "Cannot locate corresponding out_split source JSON.",
                {
                    "source_path": row.get("source_path"),
                    "chapter": row.get("chapter"),
                    "problem_number": row.get("problem_number"),
                    "sub_id": row.get("sub_id"),
                },
            )
            return idx, rid, None, [miss_issue], 1, f"[analyze-llm] {idx}/{len(rows)} -> {rid} (missing source)"

        raw = read_json(src)
        prob = raw.get("problem", raw) if isinstance(raw, dict) else {}
        out_problem = str(prob.get("problem_latex", "")).strip()
        out_answer = str(prob.get("answer_latex", "")).strip()

        try:
            assessment = llm_eval_one(
                row=row,
                out_problem_latex=out_problem,
                out_answer_latex=out_answer,
                client=client,
                model=model,
                max_retries=max_retries,
            )
            local_issues = issues_from_assessment(row, assessment)
            return idx, rid, assessment, local_issues, 0, f"[analyze-llm] {idx}/{len(rows)} -> {rid}"
        except Exception as e:  # noqa: BLE001
            fail_issue = mk_issue(
                row, "llm_eval", "error", "llm_eval_failed",
                f"LLM audit failed for this item: {e}",
            )
            return idx, rid, None, [fail_issue], 0, f"[analyze-llm] {idx}/{len(rows)} -> {rid} (failed)"

    pending = list(enumerate(rows, start=1))
    results: List[Tuple[int, str, Optional[Dict[str, Any]], List[Issue], int, str]] = []
    if max_workers <= 1:
        for idx, row in pending:
            results.append(_analyze_one(idx, row))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(_analyze_one, idx, row): idx for idx, row in pending}
            for fut in as_completed(fut_map):
                try:
                    results.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    idx = fut_map[fut]
                    row = rows[idx - 1]
                    rid = str(row.get("id", ""))
                    fail_issue = mk_issue(
                        row, "llm_eval", "error", "llm_eval_failed",
                        f"LLM audit future failed: {e}",
                    )
                    results.append((idx, rid, None, [fail_issue], 0, f"[analyze-llm] {idx}/{len(rows)} -> {rid} (future failed)"))

    results.sort(key=lambda x: x[0])
    for _, rid, assessment, local_issues, miss_cnt, log_msg in results:
        print(log_msg)
        missing_source_rows += miss_cnt
        if assessment is not None:
            assessments[rid] = assessment
        issues.extend(local_issues)

    sev_count = Counter(i.severity for i in issues)
    code_count = Counter(i.code for i in issues)
    stage_count = Counter(i.stage for i in issues)

    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for i in issues:
        by_id[i.id].append(
            {
                "stage": i.stage,
                "severity": i.severity,
                "code": i.code,
                "message": i.message,
                "details": i.details,
            }
        )

    scores: List[float] = []
    for rid, a in assessments.items():
        if not isinstance(a, dict):
            continue
        ov = a.get("overall", {})
        if isinstance(ov, dict):
            s = ov.get("score")
            try:
                scores.append(float(s))
            except Exception:  # noqa: BLE001
                pass

    report = {
        "summary": {
            "mode": "llm_audit",
            "total_rows": len(rows),
            "rows_with_issues": len(by_id),
            "total_issues": len(issues),
            "missing_source_rows": missing_source_rows,
            "severity_counts": dict(sev_count),
            "stage_counts": dict(stage_count),
            "top_issue_codes": dict(code_count.most_common(30)),
            "avg_llm_score": (sum(scores) / len(scores)) if scores else None,
        },
        "issues": [
            {
                "id": i.id,
                "split": i.split,
                "stage": i.stage,
                "severity": i.severity,
                "code": i.code,
                "message": i.message,
                "details": i.details,
            }
            for i in issues
        ],
        "issues_by_id": dict(by_id),
        "llm_assessments_by_id": assessments,
    }
    write_json(Path(out_json), report)

    md_lines: List[str] = []
    md_lines.append("# Conversion/Extraction Quality Report (LLM Audit)")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append(f"- mode: llm_audit")
    md_lines.append(f"- total rows: {len(rows)}")
    md_lines.append(f"- rows with issues: {len(by_id)}")
    md_lines.append(f"- total issues: {len(issues)}")
    md_lines.append(f"- missing out_split source rows: {missing_source_rows}")
    if scores:
        md_lines.append(f"- avg llm score: {sum(scores) / len(scores):.2f}")
    md_lines.append("")
    md_lines.append("### Severity Counts")
    md_lines.append("")
    for k, v in sev_count.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.append("")
    md_lines.append("### Top Issue Codes")
    md_lines.append("")
    for code, cnt in code_count.most_common(20):
        md_lines.append(f"- {code}: {cnt}")
    md_lines.append("")
    md_lines.append("## Sample Issues (first 100)")
    md_lines.append("")
    for i in issues[:100]:
        md_lines.append(f"- [{i.severity}] `{i.id}` `{i.stage}` `{i.code}`: {i.message}")
    write_markdown(Path(out_md), md_lines)

    print(f"[analyze] rows={len(rows)} issues={len(issues)} rows_with_issues={len(by_id)}")
    print(f"[analyze] report json: {out_json}")
    print(f"[analyze] report md:   {out_md}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze conversion/extraction quality with LLM audit.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--split", default=None, help="Optional split filter, e.g., chapter_6")
    parser.add_argument("--max-items", type=int, default=None, help="Only analyze first N rows")
    parser.add_argument("--out-json", default=None, help="Output JSON report path")
    parser.add_argument("--out-md", default=None, help="Output markdown report path")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output JSON report already exists.")
    args = parser.parse_args()
    run(
        config_path=args.config,
        split=args.split,
        max_items=args.max_items,
        out_json=args.out_json,
        out_md=args.out_md,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
