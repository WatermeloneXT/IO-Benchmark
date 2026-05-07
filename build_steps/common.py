from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def natural_key(text: str) -> List[Any]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize_split_name(chapter_id: str) -> str:
    body = re.sub(r"[^A-Za-z0-9]+", "_", chapter_id).strip("_")
    return f"chapter_{body}" if body else "chapter_unknown"


def normalize_chapter_selector(selector: str) -> str:
    s = (selector or "").strip().lower()
    if s.startswith("chapter_"):
        s = s[len("chapter_") :]
        s = s.replace("_", "/")
    return s


def chapter_matches(chapter_id: str, selectors: List[str]) -> bool:
    if not selectors:
        return True
    cid = chapter_id.strip().lower()
    for raw in selectors:
        sel = normalize_chapter_selector(raw)
        if sel and cid == sel:
            return True
    return False


def split_already_exists(hf_json_dir: Path, hf_dataset_dir: Path, split: str) -> bool:
    jsonl_file = hf_json_dir / f"{split}.jsonl"
    if jsonl_file.exists() and jsonl_file.stat().st_size > 0:
        return True
    dataset_split_dir = hf_dataset_dir / split
    if dataset_split_dir.exists() and dataset_split_dir.is_dir():
        return True
    return False


def discover_chapters(chapters_root: Path) -> List[Tuple[str, Path]]:
    chapter_dirs: List[Tuple[str, Path]] = []
    for d in chapters_root.rglob("*"):
        if not d.is_dir():
            continue
        if (d / "out_split").is_dir():
            chapter_id = d.relative_to(chapters_root).as_posix()
            chapter_dirs.append((chapter_id, d))
    chapter_dirs.sort(key=lambda x: natural_key(x[0]))
    return chapter_dirs


def parse_chapter_selectors(chapter: Optional[str]) -> List[str]:
    if isinstance(chapter, str) and chapter.strip():
        return [x.strip() for x in chapter.split(",") if x.strip()]
    return []


def filter_chapters(chapters: List[Tuple[str, Path]], chapter: Optional[str]) -> List[Tuple[str, Path]]:
    selectors = parse_chapter_selectors(chapter)
    if not selectors:
        return chapters
    return [x for x in chapters if chapter_matches(x[0], selectors)]


def iter_problem_files(chapter_dir: Path) -> List[Path]:
    out_split = chapter_dir / "out_split"
    files: List[Path] = []
    if out_split.is_dir():
        for pdir in sorted([x for x in out_split.iterdir() if x.is_dir()], key=lambda x: natural_key(x.name)):
            files.extend(sorted([x for x in pdir.glob("*.json") if x.is_file()], key=lambda x: natural_key(x.name)))
    return files


def infer_sub_id(problem_obj: Dict[str, Any], file_stem: str) -> str:
    sub_path = problem_obj.get("subquestion_path")
    if isinstance(sub_path, list) and sub_path:
        return ".".join(str(x).strip() for x in sub_path if str(x).strip())
    return file_stem


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value or "").strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return default


def normalize_confidence(text: str) -> str:
    s = (text or "").strip().lower()
    return s if s in {"high", "medium", "low"} else "low"


def parse_int_like(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    s = str(value or "").strip()
    if not s:
        return default
    m = re.search(r"[-+]?\d+", s)
    if not m:
        return default
    try:
        return int(m.group(0))
    except Exception:  # noqa: BLE001
        return default


def normalize_bool_answer(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    s = s.replace("$", "")
    s = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"[`\s]", "", s).lower()
    if s in {"true", "t", "yes", "y", "1"}:
        return "True"
    if s in {"false", "f", "no", "n", "0"}:
        return "False"
    return None


def compact_text(text: str, default: str = "N/A") -> str:
    out = re.sub(r"\s+", " ", (text or "").strip())
    return out if out else default


def preserve_stem_context(original_question: str, split_question: str, sub_index: int) -> str:
    src = (original_question or "").strip()
    q = (split_question or "").strip()
    if not src:
        return q
    if not q:
        return src
    if src in q:
        return q
    src_head = src[: min(160, len(src))]
    if src_head and src_head in q:
        return q
    return f"{src}\n\n[Split target {sub_index}] {q}"


def looks_like_sympy(expr: str) -> bool:
    s = (expr or "").strip()
    if not s or s == "N/A":
        return False
    if "\\" in s or "$" in s:
        return False
    return True


def stage_file(path_dir: Path, split: str, stage_tag: str) -> Path:
    return path_dir / f"{split}_{stage_tag}.jsonl"


def resolve_target_splits(
    chapters_root: Path,
    chapter: Optional[str] = None,
    target_splits: Optional[List[str]] = None,
) -> List[str]:
    if target_splits is not None:
        return sorted(list(dict.fromkeys(target_splits)), key=natural_key)
    chapters = filter_chapters(discover_chapters(chapters_root), chapter)
    return sorted(list(dict.fromkeys([sanitize_split_name(cid) for cid, _ in chapters])), key=natural_key)
