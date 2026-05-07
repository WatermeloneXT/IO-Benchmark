#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_utils import load_config


DatasetKey = Tuple[str, str]

DEFAULT_SELECTED_KEYWORDS: Tuple[str, ...] = (
    "equilibrium",
    "monopolist",
    "marginal",
    "discount factor",
)
SELECTED_KEYWORD_REPORT_NAME = "core_error_keywords"
SELECTED_KEYWORD_REPORT_TITLE = "Core Error Keywords"

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "gpt-5.4__effort-high__max-tokens-32768": "GPT-5.4 high",
    "gpt-5.4__effort-medium__max-tokens-32768": "GPT-5.4 medium",
    "gpt-5.4__effort-xhigh__max-tokens-32768": "GPT-5.4 xhigh",
    "claude-opus-4-6__effort-high__max-tokens-32768": "Opus 4.6 high",
    "claude-opus-4-6__effort-medium__max-tokens-32768": "Opus 4.6 medium",
    "claude-opus-4-6__effort-max__max-tokens-32768": "Opus 4.6 max",
    "claude-sonnet-4-6__effort-high__max-tokens-32768": "Sonnet 4.6 high",
    "claude-sonnet-4-6__effort-medium__max-tokens-32768": "Sonnet 4.6 medium",
    "claude-sonnet-4-6__effort-max__max-tokens-32768": "Sonnet 4.6 max",
    "Qwen3-32B__max-tokens-32768": "Qwen3-32B",
    "Qwen3-8B__max-tokens-32768": "Qwen3-8B",
    "DeepSeek-R1-Distill-Qwen-32B__max-tokens-32768": "DS-R1-Qwen-32B",
    "DeepSeek-R1-Distill-Qwen-14B__max-tokens-32768": "DS-R1-Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-7B__max-tokens-32768": "DS-R1-Qwen-7B",
    "Llama-3.3-70B-Instruct__max-tokens-32768": "Llama-3.3-70B",
    "Llama-3.1-8B-Instruct__max-tokens-32768": "Llama-3.1-8B",
}

FAMILY_CHART_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "id": "gpt54",
        "title": "GPT-5.4",
        "models": (
            ("gpt-5.4__effort-medium__max-tokens-32768", (147, 197, 253)),
            ("gpt-5.4__effort-high__max-tokens-32768", (59, 130, 246)),
            ("gpt-5.4__effort-xhigh__max-tokens-32768", (30, 64, 175)),
        ),
    },
    {
        "id": "opus46",
        "title": "Claude Opus 4.6",
        "models": (
            ("claude-opus-4-6__effort-medium__max-tokens-32768", (196, 181, 253)),
            ("claude-opus-4-6__effort-high__max-tokens-32768", (139, 92, 246)),
            ("claude-opus-4-6__effort-max__max-tokens-32768", (91, 33, 182)),
        ),
    },
    {
        "id": "sonnet46",
        "title": "Claude Sonnet 4.6",
        "models": (
            ("claude-sonnet-4-6__effort-medium__max-tokens-32768", (253, 186, 116)),
            ("claude-sonnet-4-6__effort-high__max-tokens-32768", (249, 115, 22)),
            ("claude-sonnet-4-6__effort-max__max-tokens-32768", (194, 65, 12)),
        ),
    },
    {
        "id": "qwen3",
        "title": "Qwen3",
        "models": (
            ("Qwen3-8B__max-tokens-32768", (134, 239, 172)),
            ("Qwen3-32B__max-tokens-32768", (22, 101, 52)),
        ),
    },
    {
        "id": "deepseek_r1_qwen",
        "title": "DeepSeek-R1-Distill-Qwen",
        "models": (
            ("DeepSeek-R1-Distill-Qwen-7B__max-tokens-32768", (252, 165, 165)),
            ("DeepSeek-R1-Distill-Qwen-14B__max-tokens-32768", (239, 68, 68)),
            ("DeepSeek-R1-Distill-Qwen-32B__max-tokens-32768", (153, 27, 27)),
        ),
    },
    {
        "id": "llama",
        "title": "Llama",
        "models": (
            ("Llama-3.1-8B-Instruct__max-tokens-32768", (94, 234, 212)),
            ("Llama-3.3-70B-Instruct__max-tokens-32768", (13, 148, 136)),
        ),
    },
)


DEFAULT_STOPWORDS: Set[str] = {
    "a",
    "an",
    "all",
    "and",
    "are",
    "as",
    "at",
    "be",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "does",
    "do",
    "each",
    "for",
    "first",
    "from",
    "given",
    "has",
    "have",
    "his",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "let",
    "least",
    "make",
    "may",
    "more",
    "must",
    "not",
    "now",
    "of",
    "only",
    "on",
    "one",
    "or",
    "other",
    "over",
    "respectively",
    "show",
    "shown",
    "since",
    "so",
    "some",
    "such",
    "suppose",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "total",
    "two",
    "under",
    "use",
    "using",
    "what",
    "when",
    "where",
    "whether",
    "which",
    "while",
    "who",
    "with",
    "without",
    "would",
}

BENCHMARK_STOPWORDS: Set[str] = {
    "according",
    "along",
    "alpha",
    "analyze",
    "answer",
    "assume",
    "assumption",
    "bar",
    "calculate",
    "chapter",
    "choose",
    "chooses",
    "choice",
    "compute",
    "conclude",
    "consider",
    "constant",
    "consumer",
    "consumers",
    "converted",
    "date",
    "defined",
    "derive",
    "determine",
    "delta",
    "denote",
    "denotes",
    "drawn",
    "equation",
    "exceed",
    "exactly",
    "everything",
    "exercise",
    "expression",
    "faced",
    "false",
    "final",
    "find",
    "following",
    "function",
    "functions",
    "given",
    "inspired",
    "item",
    "judge",
    "large",
    "legally",
    "maximize",
    "maximizes",
    "model",
    "models",
    "number",
    "parameter",
    "parameters",
    "problem",
    "prove",
    "question",
    "receiving",
    "same",
    "second",
    "small",
    "sign",
    "statement",
    "subject",
    "study",
    "symbol",
    "symbols",
    "true",
    "unit",
    "verify",
}

# These terms are often too broad in industrial-organization items to be
# diagnostic as top keywords. Use --keep-generic-econ-terms to retain them.
GENERIC_ECON_STOPWORDS: Set[str] = {
    "cost",
    "costs",
    "demand",
    "firm",
    "firms",
    "good",
    "market",
    "payoff",
    "payoffs",
    "price",
    "prices",
    "profit",
    "profits",
    "quantity",
}

# These are non-topic standalone labels after inspecting keyword_stats.csv.
# Topic words such as equilibrium, marginal, and discount factor are retained.
CANDIDATE_STOPWORDS: Set[str] = {
    "after",
    "any",
    "before",
    "between",
    "case",
    "common",
    "consumer",
    "converted",
    "date",
    "delta",
    "equal",
    "faces",
    "high",
    "iii",
    "instead",
    "level",
    "lower",
    "low",
    "only",
    "per",
    "perfect",
    "pure",
    "rate",
    "split",
    "target",
    "they",
    "thus",
    "who",
    "zero",
}

TOKEN_ALIASES: Dict[str, str] = {
    "buyers": "buyer",
    "buys": "buy",
    "charges": "charge",
    "consumers": "consumer",
    "firms": "firm",
    "markets": "market",
    "monopolists": "monopolist",
    "prices": "price",
    "products": "product",
    "produces": "produce",
    "profits": "profit",
    "sellers": "seller",
    "strategies": "strategy",
    "types": "type",
}

PHRASE_ALIASES: Dict[str, str] = {
    "discount": "discount factor",
    "elasticity epsilon": "epsilon elasticity",
    "factor": "discount factor",
    "payments lump": "lump sum",
    "payments lump sum": "lump sum",
    "quality high": "high quality",
    "quality low": "low quality",
    "rate interest": "interest rate",
    "side payments lump": "side payments",
}


def normalize_candidate_keyword(candidate: str) -> str:
    return PHRASE_ALIASES.get(candidate, candidate)


def parse_selected_keywords(raw_keywords: Optional[str]) -> List[str]:
    if raw_keywords is None:
        return list(DEFAULT_SELECTED_KEYWORDS)
    keywords: List[str] = []
    for part in raw_keywords.split(","):
        keyword = normalize_candidate_keyword(part.strip().lower())
        if keyword and keyword not in keywords:
            keywords.append(keyword)
    return keywords


def natural_key(text: str) -> List[Any]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
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


def resolve_path(base: Path, override: Optional[str], config_value: str) -> Path:
    raw = str(override or config_value or "").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def dataset_question_text(row: Dict[str, Any]) -> str:
    for key in ("question_final", "question_standalone", "split_question", "question_original"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def strip_latex(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"\\(?:texttt|textbf|textit|emph|mathrm|mathbf|mathit)\s*\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[_^{}$&%#~]", " ", s)
    s = s.replace("\\", " ")
    return s


def normalize_token(token: str) -> str:
    token = token.lower().strip("_")
    token = re.sub(r"^[0-9]+|[0-9]+$", "", token)
    return token


def tokenize(text: str, stopwords: Set[str], min_token_len: int) -> List[Optional[str]]:
    cleaned = strip_latex(text)
    cleaned = cleaned.replace("-", " ")
    tokens: List[Optional[str]] = []
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9_]*", cleaned):
        tok = normalize_token(match.group(0))
        if len(tok) < min_token_len:
            tokens.append(None)
            continue
        if tok in stopwords:
            tokens.append(None)
            continue
        tok = TOKEN_ALIASES.get(tok, tok)
        tokens.append(tok)
    return tokens


def ngram_candidates(
    tokens: Sequence[Optional[str]],
    ngram_min: int,
    ngram_max: int,
    candidate_stopwords: Set[str],
) -> Set[str]:
    out: Set[str] = set()
    n_tokens = len(tokens)
    for n in range(max(1, ngram_min), max(ngram_min, ngram_max) + 1):
        if n > n_tokens:
            continue
        for idx in range(0, n_tokens - n + 1):
            span = tokens[idx : idx + n]
            if any(tok is None for tok in span):
                continue
            gram = tuple(str(tok) for tok in span)
            if len(set(gram)) < len(gram):
                continue
            candidate = " ".join(gram)
            candidate = normalize_candidate_keyword(candidate)
            if candidate in candidate_stopwords:
                continue
            out.add(candidate)
    return out


def load_dataset(hf_json_dir: Path) -> Tuple[Dict[DatasetKey, Dict[str, Any]], List[Dict[str, Any]]]:
    dataset_by_key: Dict[DatasetKey, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    files = sorted(
        [p for p in hf_json_dir.glob("*.jsonl") if p.is_file() and not p.stem.endswith("_balanced")],
        key=lambda p: natural_key(p.stem),
    )
    for path in files:
        split = path.stem
        for row in read_jsonl(path):
            qid = str(row.get("id", "") or "").strip()
            if not qid:
                continue
            out = dict(row)
            out["_split_file"] = split
            key = (split, qid)
            dataset_by_key[key] = out
            rows.append(out)
    return dataset_by_key, rows


def discover_rule_model_dirs(by_model_dir: Path) -> List[Path]:
    dirs: List[Path] = []
    if not by_model_dir.exists():
        return dirs
    for summary in by_model_dir.glob("*/evaluations/rule/summary.json"):
        model_dir = summary.parents[2]
        dirs.append(model_dir)
    return sorted(dirs, key=lambda p: natural_key(p.name))


def load_wrong_rows(model_dir: Path) -> List[Dict[str, Any]]:
    eval_dir = model_dir / "evaluations" / "rule"
    rows: List[Dict[str, Any]] = []
    for path in sorted(eval_dir.glob("*.jsonl"), key=lambda p: natural_key(p.stem)):
        if path.name == "summary.json":
            continue
        split = path.stem
        for row in read_jsonl(path):
            if bool(row.get("is_correct", False)):
                continue
            qid = str(row.get("id", "") or "").strip()
            if not qid:
                continue
            out = dict(row)
            out["_eval_split"] = split
            rows.append(out)
    return rows


def candidate_doc_sets(
    dataset_rows: Sequence[Dict[str, Any]],
    stopwords: Set[str],
    min_token_len: int,
    ngram_min: int,
    ngram_max: int,
    candidate_stopwords: Set[str],
) -> Dict[DatasetKey, Set[str]]:
    out: Dict[DatasetKey, Set[str]] = {}
    for row in dataset_rows:
        split = str(row.get("_split_file") or row.get("split") or "").strip()
        qid = str(row.get("id", "") or "").strip()
        if not split or not qid:
            continue
        tokens = tokenize(dataset_question_text(row), stopwords, min_token_len)
        out[(split, qid)] = ngram_candidates(tokens, ngram_min, ngram_max, candidate_stopwords)
    return out


def coverage_counter(doc_candidates: Iterable[Set[str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for candidates in doc_candidates:
        counter.update(candidates)
    return counter


def split_chapter(split: str) -> str:
    s = str(split or "").strip()
    return s[len("chapter_") :] if s.startswith("chapter_") else s


def sort_keyword_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            -int(r.get("wrong_question_count", 0)),
            -float(r.get("wrong_coverage", 0.0)),
            -len(str(r.get("keyword", "")).split()),
            -float(r.get("lift", 0.0)),
            str(r.get("keyword", "")),
        ),
    )


def is_contiguous_subphrase(shorter: Sequence[str], longer: Sequence[str]) -> bool:
    if len(shorter) >= len(longer):
        return False
    n = len(shorter)
    return any(tuple(longer[idx : idx + n]) == tuple(shorter) for idx in range(0, len(longer) - n + 1))


def prune_nested_keyword_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for row in sort_keyword_rows(rows):
        keyword_tokens = str(row.get("keyword", "")).split()
        key_set = set(row.get("_question_keys", set()))
        redundant = False
        for kept_row in kept:
            kept_tokens = str(kept_row.get("keyword", "")).split()
            if not is_contiguous_subphrase(keyword_tokens, kept_tokens):
                continue
            kept_key_set = set(kept_row.get("_question_keys", set()))
            if key_set and kept_key_set and key_set.issubset(kept_key_set):
                redundant = True
                break
        if not redundant:
            kept.append(row)
    return kept


def analyze_model(
    model_dir: Path,
    dataset_by_key: Dict[DatasetKey, Dict[str, Any]],
    all_doc_candidates: Dict[DatasetKey, Set[str]],
    all_counter: Counter[str],
    all_question_count: int,
    top_k: int,
    min_wrong_question_count: int,
    selected_keywords: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    wrong_rows = load_wrong_rows(model_dir)
    wrong_keys: List[DatasetKey] = []
    wrong_by_keyword: Dict[str, List[DatasetKey]] = defaultdict(list)
    detail_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    chapter_counts: Counter[str] = Counter()

    for row in wrong_rows:
        split = str(row.get("_eval_split") or row.get("split") or "").strip()
        qid = str(row.get("id", "") or "").strip()
        key = (split, qid)
        if key not in dataset_by_key:
            continue
        wrong_keys.append(key)
        ds_row = dataset_by_key[key]
        detail_counts[str(row.get("detail", "") or "unknown")] += 1
        type_counts[str(row.get("question_type") or ds_row.get("question_type") or "unknown")] += 1
        chapter_counts[split_chapter(split)] += 1
        for keyword in all_doc_candidates.get(key, set()):
            wrong_by_keyword[keyword].append(key)

    wrong_question_count = len(wrong_keys)
    selected_keyword_list = list(selected_keywords or [])
    selected_keyword_set = set(selected_keyword_list)
    keyword_items: Iterable[Tuple[str, List[DatasetKey]]]
    if selected_keyword_list:
        keyword_items = [(keyword, wrong_by_keyword.get(keyword, [])) for keyword in selected_keyword_list]
    else:
        keyword_items = wrong_by_keyword.items()

    keyword_rows: List[Dict[str, Any]] = []
    for keyword, keys in keyword_items:
        wrong_count = len(set(keys))
        if not selected_keyword_list and wrong_count < min_wrong_question_count:
            continue
        all_count = int(all_counter.get(keyword, 0))
        if not selected_keyword_list and all_count <= 0:
            continue
        wrong_coverage = wrong_count / wrong_question_count if wrong_question_count else 0.0
        all_coverage = all_count / all_question_count if all_question_count else 0.0
        lift = wrong_coverage / all_coverage if all_coverage else math.inf
        unique_keys = sorted(set(keys), key=lambda x: (natural_key(x[0]), natural_key(x[1])))
        keyword_rows.append(
            {
                "model": model_dir.name,
                "keyword": keyword,
                "wrong_question_count": wrong_count,
                "total_wrong": wrong_question_count,
                "wrong_coverage": round(wrong_coverage, 6),
                "all_question_count": all_count,
                "all_total": all_question_count,
                "all_coverage": round(all_coverage, 6),
                "lift": round(lift, 6) if math.isfinite(lift) else "inf",
                "question_ids": ",".join(qid for _, qid in unique_keys),
                "splits": ",".join(split for split, _ in unique_keys),
                "_question_keys": set(unique_keys),
            }
        )

    if selected_keyword_list:
        keyword_rows = [
            row for row in keyword_rows if str(row.get("keyword", "")) in selected_keyword_set
        ]
        top_rows = keyword_rows[: len(selected_keyword_list)]
    else:
        keyword_rows = prune_nested_keyword_rows(keyword_rows)
        top_rows = keyword_rows[:top_k]

    link_rows: List[Dict[str, Any]] = []
    for keyword_row in top_rows:
        keyword = str(keyword_row["keyword"])
        for split, qid in sorted(set(wrong_by_keyword.get(keyword, [])), key=lambda x: (natural_key(x[0]), natural_key(x[1]))):
            ds_row = dataset_by_key.get((split, qid), {})
            link_rows.append(
                {
                    "model": model_dir.name,
                    "keyword": keyword,
                    "split": split,
                    "id": qid,
                    "chapter": split_chapter(split),
                    "question_type": str(ds_row.get("question_type", "")),
                    "question_preview": re.sub(r"\s+", " ", dataset_question_text(ds_row)).strip()[:500],
                }
            )

    summary = {
        "model": model_dir.name,
        "wrong_count": wrong_question_count,
        "keywords": [row["keyword"] for row in top_rows],
        "detail_counts": dict(detail_counts.most_common()),
        "question_type_counts": dict(type_counts.most_common()),
        "chapter_counts": dict(chapter_counts.most_common()),
    }
    return summary, keyword_rows, link_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(
    path: Path,
    summaries: Sequence[Dict[str, Any]],
    top_rows_by_model: Dict[str, List[Dict[str, Any]]],
    title: str = "Model Wrong-Keyword Summary",
) -> None:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("| Model | Wrong count | Keywords |")
    lines.append("| --- | ---: | --- |")
    for summary in summaries:
        model = str(summary["model"])
        top = ", ".join(str(row["keyword"]) for row in top_rows_by_model.get(model, []))
        lines.append(f"| `{model}` | {summary['wrong_count']} | {top} |")
    lines.append("")

    for summary in summaries:
        model = str(summary["model"])
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| Keyword | Wrong questions | Wrong coverage | All coverage | Lift |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in top_rows_by_model.get(model, []):
            lines.append(
                "| "
                f"{row['keyword']} | "
                f"{row['wrong_question_count']} | "
                f"{float(row['wrong_coverage']) * 100:.1f}% | "
                f"{float(row['all_coverage']) * 100:.1f}% | "
                f"{row['lift']} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_filename(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return name.strip("._") or "model"


def compact_label(text: str, max_chars: int = 34) -> str:
    s = str(text)
    return s if len(s) <= max_chars else s[: max_chars - 1].rstrip() + "..."


def display_model_name(model: str) -> str:
    if model in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model]
    name = model
    name = re.sub(r"__max-tokens-\d+", "", name)
    name = name.replace("__effort-", " ")
    name = name.replace("claude-opus-4-6", "Opus 4.6")
    name = name.replace("claude-sonnet-4-6", "Sonnet 4.6")
    name = name.replace("DeepSeek-R1-Distill-Qwen-", "DS-R1-Qwen-")
    name = name.replace("-Instruct", "")
    return name


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_keyword_chart_svg(
    path: Path,
    model: str,
    rows: Sequence[Dict[str, Any]],
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1020
    header_h = 110
    row_h = 78
    footer_h = 56
    height = header_h + max(1, len(rows)) * row_h + footer_h
    label_x = 28
    bar_x = 300
    bar_w = 340
    metric_x = 810
    max_cov = max(
        [as_float(row.get("wrong_coverage")) for row in rows]
        + [as_float(row.get("all_coverage")) for row in rows]
        + [0.05]
    )
    axis_max = min(1.0, max(0.10, math.ceil(max_cov * 20.0) / 20.0))

    def pct(value: Any) -> str:
        return f"{as_float(value) * 100:.1f}%"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{label_x}" y="30" font-family="Arial, Helvetica, sans-serif" font-size="17" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="{label_x}" y="60" font-family="Arial, Helvetica, sans-serif" font-size="17" font-weight="700" fill="#0f172a">{html.escape(model)}</text>',
        f'<text x="{bar_x}" y="92" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#6b7280">scale: 0-{axis_max * 100:.0f}% coverage</text>',
    ]
    if not rows:
        lines.append(f'<text x="{label_x}" y="{header_h + 28}" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#6b7280">No keywords found.</text>')
    for idx, row in enumerate(rows):
        y = header_h + idx * row_h
        wrong_cov = as_float(row.get("wrong_coverage"))
        all_cov = as_float(row.get("all_coverage"))
        wrong_w = min(bar_w, wrong_cov / axis_max * bar_w if axis_max else 0.0)
        all_w = min(bar_w, all_cov / axis_max * bar_w if axis_max else 0.0)
        keyword = html.escape(compact_label(str(row.get("keyword", ""))))
        wrong_count = int(row.get("wrong_question_count", 0) or 0)
        total_wrong = int(row.get("total_wrong", 0) or 0)
        lift = as_float(row.get("lift"))
        row_bg = "#f9fafb" if idx % 2 == 0 else "#ffffff"
        lines.extend(
            [
                f'<rect x="0" y="{y}" width="{width}" height="{row_h}" fill="{row_bg}"/>',
                f'<text x="{label_x}" y="{y + 28}" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#111827">{keyword}</text>',
                f'<text x="{label_x}" y="{y + 52}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#6b7280">{wrong_count}/{total_wrong} wrong questions</text>',
                f'<rect x="{bar_x}" y="{y + 16}" width="{bar_w}" height="14" rx="3" fill="#e5e7eb"/>',
                f'<rect x="{bar_x}" y="{y + 16}" width="{all_w:.1f}" height="14" rx="3" fill="#9ca3af"/>',
                f'<rect x="{bar_x}" y="{y + 42}" width="{bar_w}" height="14" rx="3" fill="#e0f2fe"/>',
                f'<rect x="{bar_x}" y="{y + 42}" width="{wrong_w:.1f}" height="14" rx="3" fill="#0284c7"/>',
                f'<text x="{bar_x + bar_w + 10}" y="{y + 28}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#374151">all {pct(all_cov)}</text>',
                f'<text x="{bar_x + bar_w + 10}" y="{y + 54}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#075985">wrong {pct(wrong_cov)}</text>',
                f'<text x="{metric_x}" y="{y + 28}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#374151">lift {lift:.2f}</text>',
            ]
        )

    legend_y = height - 26
    lines.extend(
        [
            f'<rect x="{label_x}" y="{legend_y - 11}" width="14" height="10" rx="2" fill="#9ca3af"/>',
            f'<text x="{label_x + 22}" y="{legend_y - 2}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#4b5563">coverage in all benchmark questions</text>',
            f'<rect x="{label_x + 260}" y="{legend_y - 11}" width="14" height="10" rx="2" fill="#0284c7"/>',
            f'<text x="{label_x + 282}" y="{legend_y - 2}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#4b5563">coverage in this model&apos;s wrong questions</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_keyword_chart_tex(
    path: Path,
    model: str,
    rows: Sequence[Dict[str, Any]],
    axis_max_pct: float = 50.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label = tex_escape(display_model_name(model))
    wrong_total = int(rows[0].get("total_wrong", 0) or 0) if rows else 0
    axis_max_pct = max(axis_max_pct, 10.0)
    x0 = 2.10
    x1 = 9.05
    bar_span = x1 - x0
    tick_values = [0, 20, 40, int(axis_max_pct)]
    tick_values = sorted(set(v for v in tick_values if 0 <= v <= axis_max_pct))

    lines = [
        r"\documentclass{article}",
        r"\usepackage[paperwidth=4.55in,paperheight=1.62in,margin=0.06in]{geometry}",
        r"\usepackage{tikz}",
        r"\pagestyle{empty}",
        r"\setlength{\parindent}{0pt}",
        r"\definecolor{modelblue}{RGB}{37,99,235}",
        r"\definecolor{benchgray}{RGB}{107,114,128}",
        r"\definecolor{lightgray}{RGB}{229,231,235}",
        r"\begin{document}",
        r"\scriptsize",
        r"\begin{tikzpicture}[x=1cm,y=1cm, font=\sffamily]",
        rf"\fill[modelblue, rounded corners=0.45pt] ({x0:.2f},0.08) rectangle ({x0 + 0.30:.2f},0.18);",
        rf"\node[anchor=west, text=benchgray, font=\sffamily\tiny] at ({x0 + 0.40:.2f},0.13) {{{label}}};",
        rf"\node[anchor=east, text=benchgray, font=\sffamily\tiny] at ({x1:.2f},0.13) {{{wrong_total} wrong questions}};",
        "",
        rf"\draw[lightgray] ({x0:.2f},-0.42) -- ({x1:.2f},-0.42);",
    ]
    for tick in tick_values:
        tx = x0 + (tick / axis_max_pct) * bar_span
        lines.extend(
            [
                rf"\draw[lightgray] ({tx:.2f},-0.48) -- ({tx:.2f},-0.36);",
                rf"\node[anchor=north, text=benchgray, font=\sffamily\tiny] at ({tx:.2f},-0.49) {{{tick}}};",
            ]
        )
    lines.append("")

    for idx, row in enumerate(rows):
        y = -0.96 - idx * 0.46
        keyword = tex_escape(str(row.get("keyword", "")))
        coverage_pct = as_float(row.get("wrong_coverage")) * 100.0
        clamped_pct = min(max(coverage_pct, 0.0), axis_max_pct)
        bar_end = x0 + (clamped_pct / axis_max_pct) * bar_span
        value_x = min(bar_end + 0.09, x1 + 0.10)
        lines.extend(
            [
                rf"\node[anchor=east, align=right] at (1.92,{y:.2f}) {{{keyword}}};",
                rf"\fill[modelblue, rounded corners=0.45pt] ({x0:.2f},{y + 0.035:.3f}) rectangle ({bar_end:.2f},{y + 0.145:.3f});",
                rf"\node[anchor=west, font=\sffamily\tiny] at ({value_x:.2f},{y + 0.09:.2f}) {{{coverage_pct:.1f}\%}};",
                "",
            ]
        )

    lines.extend(
        [
            rf"\node[anchor=north, text=benchgray, font=\sffamily\tiny] at ({(x0 + x1) / 2:.2f},-2.72) {{Share of wrong-question set (\%)}};",
            r"\end{tikzpicture}",
            r"\end{document}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_latex_pdf(tex_path: Path) -> Optional[Path]:
    if shutil.which("pdflatex") is None:
        return None
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=str(tex_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdflatex failed for {tex_path}:\n{result.stdout[-4000:]}")
    pdf_path = tex_path.with_suffix(".pdf")
    return pdf_path if pdf_path.exists() else None


def keyword_order_from_rows(top_rows_by_model: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    keywords: List[str] = []
    for rows in top_rows_by_model.values():
        for row in rows:
            keyword = str(row.get("keyword", "") or "")
            if keyword and keyword not in keywords:
                keywords.append(keyword)
    return keywords


def rows_by_keyword(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("keyword", "")): row for row in rows}


def write_family_keyword_chart_tex(
    path: Path,
    family: Dict[str, Any],
    top_rows_by_model: Dict[str, List[Dict[str, Any]]],
    keyword_order: Sequence[str],
    axis_max_pct: float = 50.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    models: List[Tuple[str, Tuple[int, int, int]]] = [
        (str(model), tuple(color)) for model, color in family.get("models", ())
        if str(model) in top_rows_by_model
    ]
    if not models:
        return

    family_title = tex_escape(str(family.get("title", "Model family")))
    axis_max_pct = max(axis_max_pct, 10.0)
    x0 = 2.25
    x1 = 10.20
    bar_span = x1 - x0
    row_gap = 0.58
    title_y = 0.42
    legend_y = 0.02
    axis_y = -0.58
    last_y = -1.08 - (max(1, len(keyword_order)) - 1) * row_gap
    footer_y = last_y - 0.48
    paper_height = max(2.05, abs(footer_y) / 2.54 + 0.55)
    tick_values = [0, 20, 40, int(axis_max_pct)]
    tick_values = sorted(set(v for v in tick_values if 0 <= v <= axis_max_pct))

    lines = [
        r"\documentclass{article}",
        rf"\usepackage[paperwidth=5.35in,paperheight={paper_height:.2f}in,margin=0.06in]{{geometry}}",
        r"\usepackage{tikz}",
        r"\pagestyle{empty}",
        r"\setlength{\parindent}{0pt}",
        r"\definecolor{benchgray}{RGB}{107,114,128}",
        r"\definecolor{lightgray}{RGB}{229,231,235}",
    ]
    for idx, (_, color) in enumerate(models):
        lines.append(rf"\definecolor{{family{idx}}}{{RGB}}{{{color[0]},{color[1]},{color[2]}}}")

    lines.extend(
        [
            r"\begin{document}",
            r"\scriptsize",
            r"\begin{tikzpicture}[x=1cm,y=1cm, font=\sffamily]",
            rf"\node[anchor=west, font=\sffamily\scriptsize\bfseries] at (0.10,{title_y:.2f}) {{{family_title}}};",
        ]
    )

    legend_x = 2.25
    for idx, (model, _) in enumerate(models):
        x = legend_x + idx * 2.55
        label = tex_escape(display_model_name(model))
        wrong_total = int(top_rows_by_model.get(model, [{}])[0].get("total_wrong", 0) or 0)
        lines.extend(
            [
                rf"\fill[family{idx}, rounded corners=0.45pt] ({x:.2f},{legend_y:.2f}) rectangle ({x + 0.28:.2f},{legend_y + 0.10:.2f});",
                rf"\node[anchor=west, text=benchgray, font=\sffamily\tiny] at ({x + 0.36:.2f},{legend_y + 0.05:.2f}) {{{label} ({wrong_total})}};",
            ]
        )

    lines.append("")
    lines.append(rf"\draw[lightgray] ({x0:.2f},{axis_y:.2f}) -- ({x1:.2f},{axis_y:.2f});")
    for tick in tick_values:
        tx = x0 + (tick / axis_max_pct) * bar_span
        lines.extend(
            [
                rf"\draw[lightgray] ({tx:.2f},{axis_y - 0.06:.2f}) -- ({tx:.2f},{axis_y + 0.06:.2f});",
                rf"\node[anchor=north, text=benchgray, font=\sffamily\tiny] at ({tx:.2f},{axis_y - 0.07:.2f}) {{{tick}}};",
            ]
        )
    lines.append("")

    bar_h = 0.085
    if len(models) == 1:
        offsets = [0.0]
    elif len(models) == 2:
        offsets = [0.075, -0.075]
    else:
        offsets = [0.13, 0.0, -0.13]

    model_rows = {model: rows_by_keyword(top_rows_by_model.get(model, [])) for model, _ in models}
    for row_idx, keyword in enumerate(keyword_order):
        y = -1.08 - row_idx * row_gap
        lines.append(rf"\node[anchor=east, align=right] at (2.03,{y:.2f}) {{{tex_escape(keyword)}}};")
        for idx, (model, _) in enumerate(models):
            row = model_rows.get(model, {}).get(keyword, {})
            coverage_pct = as_float(row.get("wrong_coverage")) * 100.0
            clamped_pct = min(max(coverage_pct, 0.0), axis_max_pct)
            bar_end = x0 + (clamped_pct / axis_max_pct) * bar_span
            ybar = y + offsets[idx]
            value_x = min(bar_end + 0.08, x1 + 0.08)
            lines.extend(
                [
                    rf"\fill[family{idx}, rounded corners=0.45pt] ({x0:.2f},{ybar - bar_h / 2:.3f}) rectangle ({bar_end:.2f},{ybar + bar_h / 2:.3f});",
                    rf"\node[anchor=west, font=\sffamily\tiny] at ({value_x:.2f},{ybar:.2f}) {{{coverage_pct:.1f}\%}};",
                ]
            )
        lines.append("")

    lines.extend(
        [
            rf"\node[anchor=north, text=benchgray, font=\sffamily\tiny] at ({(x0 + x1) / 2:.2f},{footer_y:.2f}) {{Share of wrong-question set (\%)}};",
            r"\end{tikzpicture}",
            r"\end{document}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_family_chart_index(path: Path, family_charts: Sequence[Tuple[str, str]]) -> None:
    rows = "\n".join(
        f'<li><a href="{html.escape(filename)}">{html.escape(title)}</a></li>'
        for title, filename in family_charts
    )
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Core Error Keywords by Family</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 28px; color: #111827; }}
    a {{ color: #0369a1; text-decoration: none; }}
    li {{ margin: 8px 0; }}
  </style>
</head>
<body>
  <h1>Core Error Keywords by Family</h1>
  <ul>
    {rows}
  </ul>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def write_family_keyword_charts(
    chart_dir: Path,
    top_rows_by_model: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    family_dir = chart_dir / "families"
    family_dir.mkdir(parents=True, exist_ok=True)
    keyword_order = keyword_order_from_rows(top_rows_by_model)
    chart_paths: List[str] = []
    index_rows: List[Tuple[str, str]] = []
    max_pct = 50.0
    for rows in top_rows_by_model.values():
        for row in rows:
            max_pct = max(max_pct, math.ceil(as_float(row.get("wrong_coverage")) * 100.0 / 10.0) * 10.0)

    for family in FAMILY_CHART_SPECS:
        family_id = safe_filename(str(family.get("id", "family")))
        tex_path = family_dir / f"paper_family_{family_id}.tex"
        write_family_keyword_chart_tex(
            tex_path,
            family=family,
            top_rows_by_model=top_rows_by_model,
            keyword_order=keyword_order,
            axis_max_pct=max_pct,
        )
        if not tex_path.exists():
            continue
        pdf_path = compile_latex_pdf(tex_path)
        if pdf_path is not None:
            chart_paths.append(str(pdf_path))
            index_rows.append((str(family.get("title", family_id)), pdf_path.name))

    write_family_chart_index(family_dir / "index.html", index_rows)
    return chart_paths


def write_paper_keyword_charts(
    chart_dir: Path,
    summaries: Sequence[Dict[str, Any]],
    top_rows_by_model: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    chart_paths: List[str] = []
    max_pct = 50.0
    for rows in top_rows_by_model.values():
        for row in rows:
            max_pct = max(max_pct, math.ceil(as_float(row.get("wrong_coverage")) * 100.0 / 10.0) * 10.0)

    for summary in summaries:
        model = str(summary["model"])
        tex_path = chart_dir / f"paper_{safe_filename(model)}.tex"
        write_paper_keyword_chart_tex(tex_path, model, top_rows_by_model.get(model, []), axis_max_pct=max_pct)
        pdf_path = compile_latex_pdf(tex_path)
        if pdf_path is not None:
            chart_paths.append(str(pdf_path))
    return chart_paths


def write_chart_index(
    path: Path,
    summaries: Sequence[Dict[str, Any]],
    top_rows_by_model: Dict[str, List[Dict[str, Any]]],
    chart_files: Dict[str, str],
    title: str,
) -> None:
    rows = []
    for summary in summaries:
        model = str(summary["model"])
        top = ", ".join(str(row["keyword"]) for row in top_rows_by_model.get(model, []))
        filename = chart_files.get(model, "")
        rows.append(
            "<tr>"
            f"<td><a href=\"{html.escape(filename)}\">{html.escape(model)}</a></td>"
            f"<td>{int(summary.get('wrong_count', 0))}</td>"
            f"<td>{html.escape(top)}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 28px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    a {{ color: #0369a1; text-decoration: none; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <table>
    <thead><tr><th>Model</th><th>Wrong count</th><th>Keywords</th></tr></thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def write_keyword_charts(
    reports_dir: Path,
    summaries: Sequence[Dict[str, Any]],
    top_rows_by_model: Dict[str, List[Dict[str, Any]]],
    dirname: str,
    title: str,
) -> Dict[str, Any]:
    chart_dir = reports_dir / "charts" / dirname
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_files: Dict[str, str] = {}
    chart_paths: List[str] = []
    for summary in summaries:
        model = str(summary["model"])
        filename = f"{safe_filename(model)}.svg"
        chart_path = chart_dir / filename
        write_keyword_chart_svg(chart_path, model, top_rows_by_model.get(model, []), title)
        chart_files[model] = filename
        chart_paths.append(str(chart_path))
    index_path = chart_dir / "index.html"
    write_chart_index(index_path, summaries, top_rows_by_model, chart_files, title)
    paper_chart_paths: List[str] = []
    family_chart_paths: List[str] = []
    if dirname == SELECTED_KEYWORD_REPORT_NAME:
        paper_chart_paths = write_paper_keyword_charts(chart_dir, summaries, top_rows_by_model)
        family_chart_paths = write_family_keyword_charts(chart_dir, top_rows_by_model)
    return {
        "index": str(index_path),
        "charts": chart_paths,
        "paper_charts": paper_chart_paths,
        "family_charts": family_chart_paths,
    }


def collect_analysis(
    model_dirs: Sequence[Path],
    dataset_by_key: Dict[DatasetKey, Dict[str, Any]],
    doc_candidates: Dict[DatasetKey, Set[str]],
    counter: Counter[str],
    all_question_count: int,
    top_k: int,
    min_wrong_question_count: int,
    selected_keywords: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    summaries: List[Dict[str, Any]] = []
    keyword_rows_all: List[Dict[str, Any]] = []
    link_rows_all: List[Dict[str, Any]] = []
    top_rows_by_model: Dict[str, List[Dict[str, Any]]] = {}

    for one_model_dir in model_dirs:
        summary, keyword_rows, link_rows = analyze_model(
            one_model_dir,
            dataset_by_key=dataset_by_key,
            all_doc_candidates=doc_candidates,
            all_counter=counter,
            all_question_count=all_question_count,
            top_k=top_k,
            min_wrong_question_count=min_wrong_question_count,
            selected_keywords=selected_keywords,
        )
        summaries.append(summary)
        keyword_rows_all.extend(keyword_rows)
        link_rows_all.extend(link_rows)
        top_rows_by_model[one_model_dir.name] = keyword_rows[:top_k]

    summaries.sort(key=lambda row: natural_key(str(row["model"])))
    if selected_keywords:
        keyword_order = {keyword: idx for idx, keyword in enumerate(selected_keywords)}
        keyword_rows_all.sort(
            key=lambda row: (
                natural_key(str(row["model"])),
                keyword_order.get(str(row["keyword"]), len(keyword_order)),
                str(row["keyword"]),
            )
        )
    else:
        keyword_rows_all.sort(key=lambda row: (natural_key(str(row["model"])), -float(row["wrong_coverage"]), str(row["keyword"])))
    link_rows_all.sort(key=lambda row: (natural_key(str(row["model"])), str(row["keyword"]), natural_key(str(row["split"])), natural_key(str(row["id"]))))
    return summaries, keyword_rows_all, link_rows_all, top_rows_by_model


def run(
    config_path: str = "config.yaml",
    hf_json_dir: Optional[str] = None,
    by_model_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
    top_k: int = 3,
    ngram_min: int = 1,
    ngram_max: int = 3,
    min_token_len: int = 3,
    min_wrong_question_count: int = 1,
    keep_generic_econ_terms: bool = False,
    keywords: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    config_base = Path(config_path).resolve().parent
    paths = cfg.get("paths", {}) or {}
    hf_dir = resolve_path(config_base, hf_json_dir, str(paths.get("hf_json_dir", "data/hf_json")))
    model_dir = resolve_path(config_base, by_model_dir, str(paths.get("by_model_dir", "data/by_model")))
    reports_dir = (
        Path(out_dir).resolve()
        if out_dir
        else (hf_dir.parent / "reports" / "model_wrong_keywords").resolve()
    )

    stopwords = set(DEFAULT_STOPWORDS) | set(BENCHMARK_STOPWORDS)
    if not keep_generic_econ_terms:
        stopwords |= set(GENERIC_ECON_STOPWORDS)
    candidate_stopwords = set(CANDIDATE_STOPWORDS)
    selected_keywords = parse_selected_keywords(keywords)
    selected_keyword_set = set(selected_keywords)
    effective_top_k = len(selected_keywords) if selected_keywords else top_k

    dataset_by_key, dataset_rows = load_dataset(hf_dir)
    if not dataset_rows:
        raise RuntimeError(f"No dataset rows found under: {hf_dir}")
    all_doc_candidates = candidate_doc_sets(
        dataset_rows,
        stopwords=stopwords,
        min_token_len=min_token_len,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        candidate_stopwords=candidate_stopwords,
    )
    all_counter = coverage_counter(all_doc_candidates.values())
    if selected_keywords:
        all_doc_candidates = {
            key: {keyword for keyword in candidates if keyword in selected_keyword_set}
            for key, candidates in all_doc_candidates.items()
        }
        all_counter = coverage_counter(all_doc_candidates.values())
    model_dirs = discover_rule_model_dirs(model_dir)
    if not model_dirs:
        raise RuntimeError(f"No rule evaluation summaries found under: {model_dir}")

    summaries, keyword_rows_all, link_rows_all, top_rows_by_model = collect_analysis(
        model_dirs,
        dataset_by_key=dataset_by_key,
        doc_candidates=all_doc_candidates,
        counter=all_counter,
        all_question_count=len(dataset_rows),
        top_k=effective_top_k,
        min_wrong_question_count=min_wrong_question_count,
        selected_keywords=selected_keywords,
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_json = reports_dir / "summary.json"
    keywords_csv = reports_dir / "keyword_stats.csv"
    links_csv = reports_dir / "top_keyword_questions.csv"
    links_jsonl = reports_dir / "top_keyword_questions.jsonl"
    markdown = (
        reports_dir / f"{SELECTED_KEYWORD_REPORT_NAME}.md"
        if selected_keywords
        else reports_dir / f"top{effective_top_k}.md"
    )
    write_json(
        summary_json,
        {
            "hf_json_dir": str(hf_dir),
            "by_model_dir": str(model_dir),
            "out_dir": str(reports_dir),
            "dataset_question_count": len(dataset_rows),
            "selection_name": SELECTED_KEYWORD_REPORT_NAME if selected_keywords else "top_keywords",
            "keyword_count": effective_top_k,
            "ngram_min": ngram_min,
            "ngram_max": ngram_max,
            "min_token_len": min_token_len,
            "min_wrong_question_count": min_wrong_question_count,
            "keep_generic_econ_terms": keep_generic_econ_terms,
            "selected_keywords": selected_keywords,
            "models": summaries,
        },
    )
    write_csv(
        keywords_csv,
        keyword_rows_all,
        [
            "model",
            "keyword",
            "wrong_question_count",
            "total_wrong",
            "wrong_coverage",
            "all_question_count",
            "all_total",
            "all_coverage",
            "lift",
            "question_ids",
            "splits",
        ],
    )
    write_csv(
        links_csv,
        link_rows_all,
        ["model", "keyword", "split", "id", "chapter", "question_type", "question_preview"],
    )
    write_jsonl(links_jsonl, link_rows_all)
    write_markdown(
        markdown,
        summaries,
        top_rows_by_model,
        title=SELECTED_KEYWORD_REPORT_TITLE if selected_keywords else "Model Wrong-Keyword Summary",
    )
    keyword_charts = write_keyword_charts(
        reports_dir,
        summaries,
        top_rows_by_model,
        dirname=SELECTED_KEYWORD_REPORT_NAME if selected_keywords else f"top{effective_top_k}_keywords",
        title=SELECTED_KEYWORD_REPORT_TITLE if selected_keywords else f"Top {effective_top_k} Wrong Keywords",
    )
    result = {
        "summary_json": str(summary_json),
        "keyword_stats_csv": str(keywords_csv),
        "top_keyword_questions_csv": str(links_csv),
        "top_keyword_questions_jsonl": str(links_jsonl),
        "top_markdown": str(markdown),
        "keyword_charts_index": keyword_charts["index"],
        "paper_keyword_charts": keyword_charts.get("paper_charts", []),
        "family_keyword_charts": keyword_charts.get("family_charts", []),
        "model_count": len(summaries),
        "dataset_question_count": len(dataset_rows),
    }
    print(f"[wrong-keywords] wrote reports -> {reports_dir}")
    print(f"[wrong-keywords] models={len(summaries)} dataset_questions={len(dataset_rows)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze each model's wrong rule-eval questions and extract diagnostic top keywords."
    )
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--hf-json-dir", default=None, help="Optional dataset JSONL directory override.")
    parser.add_argument("--by-model-dir", default=None, help="Optional by_model directory override.")
    parser.add_argument("--out-dir", default=None, help="Output directory for reports.")
    parser.add_argument("--top-k", type=int, default=3, help="Top keywords to list per model.")
    parser.add_argument(
        "--keywords",
        default=None,
        help=(
            "Comma-separated keywords to compute instead of automatic top keywords. "
            "Defaults to equilibrium, monopolist, marginal, discount factor."
        ),
    )
    parser.add_argument("--ngram-min", type=int, default=1, help="Minimum n-gram length.")
    parser.add_argument("--ngram-max", type=int, default=3, help="Maximum n-gram length.")
    parser.add_argument("--min-token-len", type=int, default=3, help="Minimum token length before n-gram extraction.")
    parser.add_argument(
        "--min-wrong-question-count",
        type=int,
        default=1,
        help="Drop keywords that appear in fewer than this many wrong questions for a model.",
    )
    parser.add_argument(
        "--keep-generic-econ-terms",
        action="store_true",
        help="Keep broad IO terms such as firm, price, profit, demand, and cost.",
    )
    args = parser.parse_args()
    run(
        config_path=args.config,
        hf_json_dir=args.hf_json_dir,
        by_model_dir=args.by_model_dir,
        out_dir=args.out_dir,
        top_k=args.top_k,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_token_len=args.min_token_len,
        min_wrong_question_count=args.min_wrong_question_count,
        keep_generic_econ_terms=args.keep_generic_econ_terms,
        keywords=args.keywords,
    )


if __name__ == "__main__":
    main()
