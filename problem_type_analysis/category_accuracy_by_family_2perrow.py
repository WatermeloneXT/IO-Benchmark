#!/usr/bin/env python3

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
BOOK_DATA_DIR = REPO_ROOT / "data" / "book_final_155"
BY_MODEL_DIR = REPO_ROOT / "data" / "by_model"
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_PDF = OUTPUT_DIR / "plot_category_accuracy_by_family_2perrow.pdf"

CATEGORY_ORDER = [
    "Pure Math / Basic Calculation",
    "Simple Math but Requires Economics Concepts",
    "Model-Based, Medium Difficulty",
    "Advanced / Hardest Model Problems",
]

CATEGORY_SHORT_LABELS = {
    "Pure Math / Basic Calculation": "Basic Math",
    "Simple Math but Requires Economics Concepts": "Conceptual Econ Math",
    "Model-Based, Medium Difficulty": "Model Based Medium",
    "Advanced / Hardest Model Problems": "Model Based Advanced",
}

FAMILY_ORDER = [
    "GPT-5.4",
    "Claude Sonnet 4.6",
    "Claude Opus 4.6",
    "DeepSeek R1 Distill Qwen",
    "Llama",
    "Qwen3",
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
PALETTE = [
    "#4F84B2",
    "#C86C75",
    "#78AC65",
    "#BF9032",
    "#9178C2",
    "#438F8A",
    "#CC7748",
    "#5F819C",
]

API_FAMILIES = {"GPT-5.4", "Claude Sonnet 4.6", "Claude Opus 4.6"}
EFFORT_RANK = {
    "default": 0,
    "medium": 1,
    "high": 2,
    "max": 3,
    "xhigh": 4,
}


def parse_effort_and_max_tokens(dirname: str, effort: str) -> tuple[str, int]:
    effort_clean = str(effort or "").strip() or "default"
    match = re.search(r"__effort-([a-z0-9]+)__", dirname)
    if effort_clean == "default" and match:
        effort_clean = match.group(1)
    token_match = re.search(r"__max-tokens-(\d+)", dirname)
    max_tokens = int(token_match.group(1)) if token_match else 0
    return effort_clean, max_tokens


def format_model_label(summary: dict, dirname: str) -> tuple[str, str, str, int]:
    solver_model = str(summary.get("solver_model") or summary.get("model") or dirname)
    effort = str(summary.get("solver_reasoning_effort") or "").strip()
    effort, max_tokens = parse_effort_and_max_tokens(dirname, effort)

    if solver_model == "gpt-5.4":
        family = "GPT-5.4"
        label = effort
    elif solver_model == "claude-sonnet-4-6":
        family = "Claude Sonnet 4.6"
        label = effort
    elif solver_model == "claude-opus-4-6":
        family = "Claude Opus 4.6"
        label = effort
    elif "DeepSeek-R1-Distill-Qwen" in solver_model:
        family = "DeepSeek R1 Distill Qwen"
        label = solver_model.replace("DeepSeek-R1-Distill-Qwen-", "")
    elif solver_model.startswith("Llama-"):
        family = "Llama"
        label = solver_model.replace("Llama-", "")
    elif solver_model.startswith("Qwen3-"):
        family = "Qwen3"
        label = solver_model.replace("Qwen3-", "")
    else:
        family = solver_model
        label = effort or solver_model

    return family, label, effort, max_tokens


def load_labels() -> pd.DataFrame:
    frames = []
    for jsonl_path in sorted(BOOK_DATA_DIR.glob("*.jsonl")):
        part = pd.read_json(jsonl_path, lines=True, dtype={"id": str})
        frames.append(part[["id", "problem_type_label"]].copy())

    if not frames:
        raise ValueError(f"No dataset JSONL files found in {BOOK_DATA_DIR}")

    labels_df = pd.concat(frames, ignore_index=True).rename(
        columns={"problem_type_label": "final_label"}
    )
    labels_df = labels_df[labels_df["final_label"].isin(CATEGORY_ORDER)].copy()
    if labels_df["id"].duplicated().any():
        duplicates = labels_df.loc[labels_df["id"].duplicated(), "id"].tolist()
        raise ValueError(f"Duplicate labeled ids in {BOOK_DATA_DIR}: {duplicates[:5]}")
    return labels_df


def load_model_eval_rows() -> tuple[pd.DataFrame, set[str]]:
    frames = []
    id_sets = []

    for model_dir in sorted(BY_MODEL_DIR.iterdir()):
        summary_path = model_dir / "evaluations" / "rule" / "summary.json"
        if not summary_path.exists():
            continue

        summary = pd.read_json(summary_path, typ="series").to_dict()
        if int(summary.get("total", 0) or 0) <= 0:
            continue

        family, variant, effort, max_tokens = format_model_label(summary, model_dir.name)
        if family not in FAMILY_ORDER:
            continue

        eval_parts = []
        for jsonl_path in sorted((model_dir / "evaluations" / "rule").glob("chapter*.jsonl")):
            part = pd.read_json(jsonl_path, lines=True)
            eval_parts.append(part[["id", "is_correct"]].copy())

        if not eval_parts:
            continue

        model_df = pd.concat(eval_parts, ignore_index=True)
        model_df["id"] = model_df["id"].astype(str)
        id_sets.append(set(model_df["id"].tolist()))
        model_df["family"] = family
        model_df["variant"] = variant
        model_df["effort"] = effort
        model_df["max_tokens"] = max_tokens
        model_df["model_dir"] = model_dir.name
        frames.append(model_df)

    if not frames:
        raise ValueError(f"No evaluation rows found in {BY_MODEL_DIR}")

    common_ids = set.intersection(*id_sets) if id_sets else set()
    return pd.concat(frames, ignore_index=True), common_ids


def keep_representative_configs(eval_df: pd.DataFrame) -> pd.DataFrame:
    keep_dirs = set()

    for family, family_df in eval_df.groupby("family", sort=False):
        if family not in API_FAMILIES:
            keep_dirs.update(family_df["model_dir"].unique().tolist())
            continue

        candidates = family_df[["model_dir", "effort", "max_tokens"]].drop_duplicates().copy()
        candidates["effort_rank"] = candidates["effort"].map(
            lambda x: EFFORT_RANK.get(str(x), -1)
        )
        candidates = candidates.sort_values(
            ["effort_rank", "max_tokens", "model_dir"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        for _, effort_df in candidates.groupby("effort", sort=False):
            best = effort_df.sort_values(
                ["max_tokens", "model_dir"], ascending=[False, True]
            ).iloc[0]
            keep_dirs.add(str(best["model_dir"]))

    return eval_df[eval_df["model_dir"].isin(keep_dirs)].copy()


def build_summary(eval_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    merged = eval_df.merge(labels_df[["id", "final_label"]], on="id", how="inner")
    grouped = (
        merged.groupby(["family", "variant", "model_dir", "final_label"], as_index=False)
        .agg(total=("id", "count"), correct=("is_correct", "sum"))
    )
    grouped["accuracy_pct"] = grouped["correct"] / grouped["total"] * 100.0
    grouped["final_label"] = pd.Categorical(
        grouped["final_label"], categories=CATEGORY_ORDER, ordered=True
    )
    return grouped.sort_values(["family", "variant", "final_label"]).reset_index(drop=True)


def plot_facets(summary_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
    counts = labels_df.groupby("final_label").size().reindex(CATEGORY_ORDER).fillna(0)
    x_labels = [
        f"{CATEGORY_SHORT_LABELS[cat]}\nn={int(counts[cat])}" for cat in CATEGORY_ORDER
    ]

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        }
    )

    n_panels = len(FAMILY_ORDER)
    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols
    fig_width = 6.2 * ncols
    fig_height = 3.35 * nrows + 0.95

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharey=True)
    fig.suptitle("Accuracy by Problem Type Across Configurations", fontsize=22, y=0.985)
    axes_flat = list(axes.flatten())

    for panel_idx, (ax, family) in enumerate(zip(axes_flat, FAMILY_ORDER)):
        family_df = summary_df[summary_df["family"] == family]
        ax.set_facecolor("white")
        ax.set_box_aspect(0.72)

        annotation_points = {idx: [] for idx in range(len(CATEGORY_ORDER))}

        for idx, (variant, variant_df) in enumerate(family_df.groupby("variant", sort=True)):
            variant_df = (
                variant_df.groupby("final_label", as_index=False)
                .agg(accuracy_pct=("accuracy_pct", "mean"))
            )
            variant_df["final_label"] = pd.Categorical(
                variant_df["final_label"], categories=CATEGORY_ORDER, ordered=True
            )
            variant_df = variant_df.sort_values("final_label")

            y = variant_df["accuracy_pct"].tolist()
            marker = MARKERS[idx % len(MARKERS)]
            color = PALETTE[idx % len(PALETTE)]

            ax.plot(
                range(len(CATEGORY_ORDER)),
                y,
                marker=marker,
                linewidth=2.2,
                markersize=7,
                label=variant,
                color=color,
            )

            for x, value in enumerate(y):
                annotation_points[x].append((value, color))

        for x, points in annotation_points.items():
            if not points:
                continue

            points_sorted = sorted(points, key=lambda item: item[0])
            for value, color in points_sorted:
                ax.annotate(
                    f"{value:.1f}",
                    (x, value),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color=color,
                )

        ax.set_title(family, fontsize=15)
        ax.set_xticks(range(len(CATEGORY_ORDER)))
        if panel_idx // ncols == nrows - 1:
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.tick_params(axis="x", bottom=True, labelbottom=True)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", bottom=False, labelbottom=False)

        if panel_idx % ncols == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.tick_params(axis="y", left=True, labelleft=True, labelsize=10)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

        if panel_idx // ncols == nrows - 1:
            ax.set_xlabel("Problem Type", fontsize=12)

        ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=9, loc="lower left")

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.92,
        bottom=0.14,
        wspace=0.04,
        hspace=0.18,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    labels_df = load_labels()
    eval_df, common_ids = load_model_eval_rows()
    eval_df = eval_df[eval_df["id"].isin(common_ids)].copy()
    eval_df = keep_representative_configs(eval_df)
    summary_df = build_summary(eval_df, labels_df)
    plot_facets(summary_df, labels_df)
    print(f"Saved: {OUTPUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
