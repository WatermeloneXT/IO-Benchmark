# Problem Type Analysis

This folder contains the scripts for the two problem-type accuracy figures used in the paper.

## Input

Both scripts use:

- problem-type labels embedded in `../data/book_final_155/*.jsonl` as `problem_type_label`
- model evaluation outputs under `../data/by_model/`

## Main-text figure

This is the single-line plot used in the main paper figure.

Run:

```bash
python3 problem_type_analysis/category_accuracy_max_capability_family.py
```

Output:

```bash
problem_type_analysis/figures/plot_category_accuracy_max_capability_family.pdf
```

## Appendix figure

This is the `2 per row` family-level plot used in the appendix.

Run:

```bash
python3 problem_type_analysis/category_accuracy_by_family_2perrow.py
```

Output:

```bash
problem_type_analysis/figures/plot_category_accuracy_by_family_2perrow.pdf
```
