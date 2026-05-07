# IO-Bench

This repository contains code and data for running economics textbook benchmark experiments.

## Repository Contents

- Pipeline and evaluation code: `econ_cli.py`, `commands/`, `core/`, and `build_steps/`
- Final textbook benchmark data: `data/book_final_155`
- Configuration and packaging files: `config.yaml`, `config.example.yaml`, `environment.yml`, `requirements.txt`, and `pyproject.toml`


## Setup

Use the provided conda environment for a reproducible setup.

```bash
cd <repo-root>
conda env create -f environment.yml
conda activate io-bench
```

If you prefer to create the environment manually:

```bash
conda create -n io-bench python=3.10 pip
conda activate io-bench
pip install -r requirements.txt
```

You can optionally install the package in editable mode after activating the conda environment:

```bash
pip install -e .
```

Editable installation provides the `io-bench` command-line entry point. The legacy `econ-bench` command is also kept as an alias.

For offline GPU generation with `generate-vllm`, install `vllm` separately in a CUDA-compatible environment:

```bash
pip install vllm
```

## Evaluation Environment

Rule-based evaluation only requires the packages in `requirements.txt`. The most important runtime dependencies are:

- `sympy` for symbolic equivalence checks
- `pyyaml` for reading `config.yaml`
- `openai` and `anthropic` for API-based generation and LLM-judge evaluation
- `datasets` for dataset-oriented utilities


To run rule-based evaluation, you first need generation outputs under `data/by_model/<solver_artifact_label>/generations/`. These can be produced with `generate` for API models or `generate-vllm` for local GPU models. LLM-judge evaluation additionally requires valid API credentials in `config.yaml` or environment variables.

## Configuration

The committed `config.yaml` file contains empty credential fields. Fill `config.yaml` directly or provide credentials through environment variables such as:

- `AZURE_OPENAI_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_FOUNDRY_API_KEY`
- `FOUNDRY_OPENAI_API_KEY`

The default dataset path is:

```yaml
paths:
  hf_json_dir: "data/book_final_155"
```

## Inspect the Dataset

Show one example from a split:

```bash
python3 econ_cli.py --config config.yaml show-examples --split chapter_1 --n 1
```

The included dataset contains 155 examples across the textbook chapter splits listed in `data/book_final_155/manifest.json`.

## Reproduce API-Based Generation and Rule Evaluation

Example API generation command with an explicit split, reasoning setting, and solve token budget:

```bash
python3 econ_cli.py --config config.yaml generate \
  --solver-model gpt-5.4 \
  --split chapter_1 \
  --reasoning-mode on \
  --reasoning-effort xhigh \
  --max-solve-tokens 32768 \
  --skip-existing
```

Run rule-based evaluation on the matching generated artifact:

```bash
python3 econ_cli.py --config config.yaml evaluate \
  --solver-model gpt-5.4 \
  --split chapter_1 \
  --solver-reasoning-effort xhigh \
  --solver-max-solve-tokens 32768
```

Generated outputs and evaluation files are written under:

```text
data/by_model/<solver_artifact_label>/
```

This directory is ignored by Git.

## Reproduce LLM-Judge Evaluation

After generation, run the LLM judge:

```bash
python3 econ_cli.py --config config.yaml evaluate-llm --solver-model gpt-5.4
```

Compare rule-based and LLM-judge error cases:

```bash
python3 econ_cli.py --config config.yaml compare-eval-errors --solver-model gpt-5.4
```

## Targeted Reruns

Build a list of question ids to rerun from current evaluation artifacts:

```bash
python3 econ_cli.py --config config.yaml build-rerun-question-ids --solver-model gpt-5.4
```

Rerun only selected questions:

```bash
python3 econ_cli.py --config config.yaml generate \
  --solver-model gpt-5.4 \
  --question-ids-file data/by_model/<solver_artifact_label>/reports/rerun_question_ids.txt
```

## Offline vLLM Generation

Example local GPU/vLLM generation command with explicit inference parameters:

```bash
python3 econ_cli.py --config config.yaml generate-vllm \
  --model /path/to/local/model \
  --solver-model local_model \
  --split chapter_1 \
  --batch-size 8 \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-tokens 32768 \
  --convert-temperature 0.0 \
  --convert-max-tokens 512 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 65536 \
  --family auto \
  --reasoning-mode auto \
  --skip-existing
```

Then evaluate the resulting artifact:

```bash
python3 econ_cli.py --config config.yaml evaluate \
  --solver-model local_model \
  --split chapter_1 \
  --solver-max-solve-tokens 32768
```
