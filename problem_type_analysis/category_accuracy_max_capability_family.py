#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path("econ_ai/problem_type_analysis")
BY_MODEL_DIR = Path("econ_ai/data/by_model")
LABELS_PATH = BASE_DIR / "problem_type_label.csv"
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_PDF = OUTPUT_DIR / "plot_category_accuracy_max_capability_family.pdf"

CATEGORY_ORDER = [
    "Pure Math / Basic Calculation",
    "Simple Math but Requires Economics Concepts",
    "Model-Based, Medium Difficulty",
    "Advanced / Hardest Model Problems",
]

CATEGORY_SHORT = {
    "Pure Math / Basic Calculation": "Basic Math",
    "Simple Math but Requires Economics Concepts": "Conceptual Econ Math",
    "Model-Based, Medium Difficulty": "Model-Based Medium",
    "Advanced / Hardest Model Problems": "Model-Based Advanced",
}

FAMILY_ORDER = [
    "GPT-5.4",
    "Claude Sonnet 4.6",
    "Claude Opus 4.6",
    "DeepSeek R1 Distill Qwen",
    "Llama",
    "Qwen3",
]

# Manually select the highest-capability configuration for each family.
SELECTED_MODELS = {
    "GPT-5.4": "gpt-5.4__effort-xhigh__max-tokens-65536",
    "Claude Sonnet 4.6": "claude-sonnet-4-6__effort-max__max-tokens-65536",
    "Claude Opus 4.6": "claude-opus-4-6__effort-max__max-tokens-65536",
    "DeepSeek R1 Distill Qwen": "DeepSeek-R1-Distill-Qwen-32B__max-tokens-32768",
    "Llama": "Llama-3.3-70B-Instruct__max-tokens-32768",
    "Qwen3": "Qwen3-32B__max-tokens-32768",
}

LEGEND_LABELS = {
    "GPT-5.4": "GPT-5.4 (xhigh)",
    "Claude Sonnet 4.6": "Claude Sonnet 4.6 (max)",
    "Claude Opus 4.6": "Claude Opus 4.6 (max)",
    "DeepSeek R1 Distill Qwen": "DeepSeek R1 Distill Qwen (32B)",
    "Llama": "Llama (70B)",
    "Qwen3": "Qwen3 (32B)",
}

PALETTE = ["#4F84B2", "#C86C75", "#78AC65", "#BF9032", "#9178C2", "#438F8A"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def load_labels() -> pd.DataFrame:
    labels_df = pd.read_csv(LABELS_PATH, dtype=str)[["id", "final_label"]]
    labels_df = labels_df[labels_df["final_label"].isin(CATEGORY_ORDER)].copy()
    return labels_df


def build_plot_df(labels_df: pd.DataFrame) -> pd.DataFrame:
    label_ids = set(labels_df["id"])
    rows = []

    for family in FAMILY_ORDER:
        model_dir_name = SELECTED_MODELS[family]
        model_dir = BY_MODEL_DIR / model_dir_name
        eval_dir = model_dir / "evaluations" / "rule"

        eval_parts = []
        for jsonl_path in sorted(eval_dir.glob("chapter*.jsonl")):
            part = pd.read_json(jsonl_path, lines=True)
            eval_parts.append(part[["id", "is_correct"]].copy())

        if not eval_parts:
            raise ValueError(f"No evaluation files found for {model_dir_name}")

        df = pd.concat(eval_parts, ignore_index=True)
        df["id"] = df["id"].astype(str)
        df = df[df["id"].isin(label_ids)].copy()

        merged = df.merge(labels_df, on="id", how="inner")
        overall_accuracy = merged["is_correct"].mean() * 100.0

        grouped = (
            merged.groupby("final_label", as_index=False)
            .agg(total=("id", "count"), correct=("is_correct", "sum"))
        )
        grouped["accuracy_pct"] = grouped["correct"] / grouped["total"] * 100.0

        for _, row in grouped.iterrows():
            rows.append(
                {
                    "family": family,
                    "model_dir": model_dir_name,
                    "overall_accuracy_pct": overall_accuracy,
                    "final_label": row["final_label"],
                    "total": int(row["total"]),
                    "correct": int(row["correct"]),
                    "accuracy_pct": float(row["accuracy_pct"]),
                }
            )

    return pd.DataFrame(rows)


def plot(plot_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
    counts = labels_df.groupby("final_label").size().reindex(CATEGORY_ORDER)
    x_labels = [f"{CATEGORY_SHORT[c]}\nn={int(counts[c])}" for c in CATEGORY_ORDER]

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        }
    )

    fig, ax = plt.subplots(figsize=(14, 8.8))

    for i, family in enumerate(FAMILY_ORDER):
        fam = plot_df[plot_df["family"] == family].copy()
        fam["final_label"] = pd.Categorical(
            fam["final_label"], categories=CATEGORY_ORDER, ordered=True
        )
        fam = fam.sort_values("final_label")
        y = fam["accuracy_pct"].tolist()

        line = ax.plot(
            range(len(CATEGORY_ORDER)),
            y,
            marker=MARKERS[i],
            linewidth=3.0,
            markersize=10,
            color=PALETTE[i],
            label=LEGEND_LABELS[family],
        )[0]

        for x, value in enumerate(y):
            ax.annotate(
                f"{value:.1f}",
                (x, value),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=13,
                color=line.get_color(),
            )

    ax.set_title("Accuracy by Problem Type", fontsize=22, pad=14, fontweight="semibold")
    ax.set_xlabel("Problem Type", fontsize=18)
    ax.set_ylabel("Accuracy (%)", fontsize=18)
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels(x_labels, fontsize=15)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(0, 100)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=15,
    )

    fig.tight_layout(rect=(0, 0, 0.84, 1))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    labels_df = load_labels()
    plot_df = build_plot_df(labels_df)
    plot(plot_df, labels_df)
    print(f"Saved: {OUTPUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
