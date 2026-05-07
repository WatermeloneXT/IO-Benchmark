#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="io-bench",
        description="Economics textbook benchmark pipeline: build data, generate answers, and evaluate.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to econ config.yaml",
    )
    parser.add_argument(
        "--workflow-id",
        default=None,
        help="Optional workflow id to group multiple commands/runs for cost logging aggregation.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id override for this command execution.",
    )
    def add_dataset_override_args(target_parser: argparse.ArgumentParser) -> None:
        target_parser.add_argument(
            "--hf-json-dir",
            default=None,
            help="Optional input dataset directory override. Example: data/book_final_155",
        )
        target_parser.add_argument(
            "--by-model-dir",
            default=None,
            help="Optional output root override to keep results for different datasets separate.",
        )

    def add_question_filter_args(target_parser: argparse.ArgumentParser) -> None:
        target_parser.add_argument(
            "--question-ids",
            default=None,
            help="Optional question id filter, comma-separated. Example: 1.7/i,3.1/i",
        )
        target_parser.add_argument(
            "--question-ids-file",
            default=None,
            help="Optional text file containing question ids to run, separated by commas or whitespace.",
        )

    sub = parser.add_subparsers(dest="stage", required=True)
    build_parser = sub.add_parser("build-dataset", help="Build chapter-split HF dataset with step1~step5 pipeline.")
    build_parser.add_argument(
        "--chapter",
        default=None,
        help="Only build specific chapter id(s), comma-separated. Examples: 6 or 6/7 or chapter_6",
    )
    build_group = build_parser.add_mutually_exclusive_group()
    build_group.add_argument(
        "--skip-existing-split",
        action="store_true",
        dest="skip_existing_split",
        help="Skip generation when target chapter split data already exists.",
    )
    build_group.add_argument(
        "--no-skip-existing-split",
        action="store_false",
        dest="skip_existing_split",
        help="Always regenerate even if split already exists.",
    )
    build_parser.add_argument(
        "--skip-existing-llm",
        action="store_true",
        help="For step1/step2/step3/step4, skip split when target output files already exist.",
    )
    build_parser.set_defaults(skip_existing_split=None)
    build_step1_parser = sub.add_parser("build-step1", help="Run stage1 only: reference-answer correction.")
    build_step1_parser.add_argument(
        "--chapter",
        default=None,
        help="Only build specific chapter id(s), comma-separated. Examples: 6 or 6/7 or chapter_6",
    )
    step1_group = build_step1_parser.add_mutually_exclusive_group()
    step1_group.add_argument(
        "--skip-existing-split",
        action="store_true",
        dest="skip_existing_split",
        help="Skip generation when target chapter split data already exists.",
    )
    step1_group.add_argument(
        "--no-skip-existing-split",
        action="store_false",
        dest="skip_existing_split",
        help="Always regenerate even if split already exists.",
    )
    build_step1_parser.add_argument(
        "--force-all",
        action="store_true",
        help="Re-run all items in stage1, not only rows with empty llm_json.",
    )
    build_step1_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip split if stage1 output file already exists and is non-empty.",
    )
    build_step1_parser.set_defaults(skip_existing_split=None)
    build_step2_parser = sub.add_parser("build-step2", help="Run stage2 only: question splitting.")
    build_step2_parser.add_argument(
        "--chapter",
        default=None,
        help="Only run specific chapter id(s), comma-separated. Examples: 6 or 6/7 or chapter_6",
    )
    build_step2_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip split if stage2 output file already exists and is non-empty.",
    )
    build_step3_parser = sub.add_parser("build-step3", help="Run stage3 only: transform to eval items.")
    build_step3_parser.add_argument(
        "--chapter",
        default=None,
        help="Only run specific chapter id(s), comma-separated. Examples: 6 or 6/7 or chapter_6",
    )
    build_step3_parser.add_argument(
        "--problem",
        default=None,
        help="Only run specific problem number(s), comma-separated. Examples: 6.8 or 6.8,6.9",
    )
    build_step3_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip split if stage3/final outputs already exist (only when --problem is not used).",
    )
    build_step3_sync_parser = sub.add_parser(
        "build-step3-symbol-sync",
        help="Run only stage3 symbol text sync on existing stage3/hf_json outputs.",
    )
    build_step3_sync_parser.add_argument(
        "--chapter",
        default=None,
        help="Only run specific chapter id(s), comma-separated. Examples: 6 or 6/7 or chapter_6",
    )
    build_step3_sync_parser.add_argument(
        "--problem",
        default=None,
        help="Only run specific problem number(s), comma-separated. Examples: 6.8 or 6.8,6.9",
    )
    build_step4_parser = sub.add_parser(
        "build-step4",
        help="Run stage4 only: judge rebalance review-file generation.",
    )
    build_step4_parser.add_argument(
        "--chapter",
        default=None,
        help="Only run specific chapter id(s), comma-separated. Examples: 6 or 6/7 or chapter_6",
    )
    build_step4_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip split if balanced review output already exists and is non-empty.",
    )
    build_step5_parser = sub.add_parser(
        "build-step5",
        help="Run stage5 only: aggregate pipeline json and build hf_dataset.",
    )
    build_step5_parser.add_argument(
        "--chapter",
        default=None,
        help="Only run specific chapter id(s), comma-separated. Examples: 6 or 6/7 or chapter_6",
    )
    build_step4_legacy_parser = sub.add_parser(
        "build-step4-finalize",
        help="[Deprecated] Legacy alias of old step4 finalize. Use build-step5.",
    )
    build_step4_legacy_parser.add_argument(
        "--chapter",
        default=None,
        help="Only run specific chapter id(s), comma-separated. Examples: 6 or 6/7 or chapter_6",
    )
    build_step5_legacy_parser = sub.add_parser(
        "build-step5-rebalance",
        help="[Deprecated] Legacy alias of old step5 rebalance. Use build-step4.",
    )
    build_step5_legacy_parser.add_argument(
        "--chapter",
        default=None,
        help="Only run specific chapter id(s), comma-separated. Examples: 6 or 6/7 or chapter_6",
    )
    build_step5_legacy_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip split if balanced review output already exists and is non-empty.",
    )
    gen_parser = sub.add_parser("generate", help="Generate boxed final answers from rewritten questions.")
    gen_parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model used for answering. Overrides config models.default_solver_model.",
    )
    add_dataset_override_args(gen_parser)
    add_question_filter_args(gen_parser)
    gen_parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    gen_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip split if generation output already exists and is non-empty.",
    )
    gen_parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate all rows (disable resume) for each processed split.",
    )
    gen_parser.add_argument(
        "--reasoning-mode",
        default=None,
        choices=["auto", "off", "on"],
        help="API reasoning mode override for the solve stage.",
    )
    gen_parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Model-native API reasoning effort value for the solve stage. "
            "If the target API model does not expose this field, generation will stop with an error."
        ),
    )
    gen_parser.add_argument(
        "--max-solve-tokens",
        type=int,
        default=None,
        help="Optional override for generate.max_solve_tokens without editing config.yaml.",
    )
    gen_vllm_parser = sub.add_parser(
        "generate-vllm",
        help="Generate boxed final answers with offline vLLM batch inference (GPU/local model).",
    )
    gen_vllm_parser.add_argument(
        "--model",
        "--model-path",
        dest="model",
        default=None,
        help="Hugging Face model id or local model path loaded by vLLM.",
    )
    gen_vllm_parser.add_argument(
        "--solver-model",
        default=None,
        help="Output/model label used under data/by_model. Defaults to a short alias derived from --model.",
    )
    gen_vllm_parser.add_argument("--tokenizer", default=None, help="Optional tokenizer id/path. Defaults to --model.")
    gen_vllm_parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    add_dataset_override_args(gen_vllm_parser)
    add_question_filter_args(gen_vllm_parser)
    gen_vllm_parser.add_argument("--skip-existing", action="store_true", help="Skip split if output already exists.")
    gen_vllm_parser.add_argument("--force", action="store_true", help="Force regenerate all rows for each split.")
    gen_vllm_parser.add_argument("--batch-size", type=int, default=16, help="Number of prompts sent to vLLM per chunk.")
    gen_vllm_parser.add_argument("--temperature", type=float, default=None, help="Solve sampling temperature.")
    gen_vllm_parser.add_argument("--top-p", type=float, default=None, help="Solve top-p sampling. Leave unset for model-specific auto defaults.")
    gen_vllm_parser.add_argument("--top-k", type=int, default=None, help="Optional solve top-k sampling.")
    gen_vllm_parser.add_argument("--max-tokens", type=int, default=None, help="Solve max generation tokens.")
    gen_vllm_parser.add_argument("--repetition-penalty", type=float, default=None, help="Solve repetition penalty. Leave unset for model-specific auto defaults.")
    gen_vllm_parser.add_argument(
        "--no-convert-sympy",
        action="store_false",
        dest="convert_sympy",
        help="Do not run the second offline batch that normalizes boxed answers to SymPy.",
    )
    gen_vllm_parser.add_argument("--convert-temperature", type=float, default=None, help="SymPy conversion temperature.")
    gen_vllm_parser.add_argument("--convert-max-tokens", type=int, default=None, help="SymPy conversion max tokens.")
    gen_vllm_parser.add_argument("--dtype", default="bfloat16", help="vLLM dtype, e.g. auto, bfloat16, float16.")
    gen_vllm_parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    gen_vllm_parser.add_argument("--gpu-memory-utilization", type=float, default=0.80, help="vLLM GPU memory utilization.")
    gen_vllm_parser.add_argument("--max-model-len", type=int, default=None, help="Optional vLLM max model length.")
    gen_vllm_parser.add_argument("--max-num-seqs", type=int, default=8, help="Optional vLLM max active sequences.")
    gen_vllm_parser.add_argument("--quantization", default=None, help="Optional vLLM quantization, e.g. fp8, awq, gptq.")
    gen_vllm_parser.add_argument("--load-format", default=None, help="Optional vLLM load format.")
    gen_vllm_parser.add_argument("--download-dir", default=None, help="Optional Hugging Face download/cache directory.")
    gen_vllm_parser.add_argument(
        "--family",
        default="auto",
        choices=["auto", "generic", "llama3", "qwen3", "deepseek", "kimi"],
        help="Chat-template family hint for Llama3/Qwen3/DeepSeek/Kimi checkpoints.",
    )
    gen_vllm_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to vLLM for checkpoints that require custom code.",
    )
    gen_vllm_parser.add_argument("--enforce-eager", action="store_true", help="Pass enforce_eager=True to vLLM.")
    gen_vllm_parser.add_argument("--stop", action="append", default=None, help="Optional stop string. Can be repeated.")
    gen_vllm_parser.add_argument("--seed", type=int, default=None, help="Optional vLLM/SamplingParams seed.")
    gen_vllm_parser.add_argument(
        "--reasoning-mode",
        default=None,
        choices=["auto", "off", "on"],
        help=(
            "Unified reasoning switch for vLLM reasoning models. "
            "Families with a hard chat-template switch will honor on/off; parser-only families reject off."
        ),
    )
    gen_vllm_parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Model-native reasoning effort value. "
            "If the target model does not expose a native reasoning-effort field, generation will stop with an error."
        ),
    )
    gen_vllm_parser.add_argument(
        "--qwen3-thinking",
        default="auto",
        choices=["auto", "on", "off"],
        help="Low-level override for Qwen3 thinking mode. Usually leave this on auto and use --reasoning-effort instead.",
    )
    gen_vllm_parser.add_argument("--no-tqdm", action="store_true", help="Disable vLLM progress bars.")
    gen_vllm_parser.add_argument("--dry-run", action="store_true", help="Inspect pending rows and first prompt without loading vLLM.")
    gen_vllm_parser.add_argument(
        "--global-batch",
        action="store_true",
        help="Batch pending rows across all selected splits while preserving per-split output files.",
    )
    gen_vllm_parser.add_argument(
        "--prompt-preview-chars",
        type=int,
        default=1200,
        help="Characters of first dry-run prompt to print.",
    )
    eval_parser = sub.add_parser(
        "evaluate",
        help="Evaluate generated answers (bool->exact match, value->sympy equivalence).",
    )
    eval_parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    eval_parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model used for generation/evaluation file naming. Overrides config models.default_solver_model.",
    )
    add_dataset_override_args(eval_parser)
    add_question_filter_args(eval_parser)
    eval_parser.add_argument(
        "--no-include-missing",
        action="store_false",
        dest="include_missing",
        help="Do not count missing predictions as incorrect.",
    )
    eval_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if split evaluation files already exist.",
    )
    eval_parser.add_argument(
        "--solver-reasoning-effort",
        default=None,
        help="Artifact label override for solver outputs generated with an explicit native reasoning effort.",
    )
    eval_parser.add_argument(
        "--solver-max-solve-tokens",
        type=int,
        default=None,
        help="Artifact label override for solver outputs generated with a non-default max solve token budget.",
    )
    eval_parser.set_defaults(include_missing=True)
    eval_llm_parser = sub.add_parser(
        "evaluate-llm",
        help="Evaluate generated answers with LLM judge (answer correctness + reasoning correctness).",
    )
    eval_llm_parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    eval_llm_parser.add_argument(
        "--solver-model",
        default=None,
        help="Solver model whose generated answers will be judged.",
    )
    eval_llm_parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model for LLM evaluation. Defaults to config evaluate_llm.judge_model.",
    )
    eval_llm_parser.add_argument(
        "--model",
        dest="legacy_model",
        default=None,
        help="[Legacy alias] If set, applies to both solver and judge model unless overridden.",
    )
    add_dataset_override_args(eval_llm_parser)
    add_question_filter_args(eval_llm_parser)
    eval_llm_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if split evaluation files already exist.",
    )
    eval_llm_parser.add_argument(
        "--reasoning-mode",
        default=None,
        choices=["auto", "off", "on"],
        help=(
            "Reasoning mode for the judge model. "
            "Uses the model's native API interface and will stop with an error if unsupported."
        ),
    )
    eval_llm_parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Model-native reasoning effort value for the judge model. "
            "No cross-provider mapping is applied; invalid values will stop evaluation."
        ),
    )
    eval_llm_parser.add_argument(
        "--solver-reasoning-effort",
        default=None,
        help="Artifact label override for solver outputs generated with an explicit native reasoning effort.",
    )
    eval_llm_parser.add_argument(
        "--solver-max-solve-tokens",
        type=int,
        default=None,
        help="Artifact label override for solver outputs generated with a non-default max solve token budget.",
    )
    reconvert_parser = sub.add_parser(
        "reconvert",
        help="Re-run SymPy conversion on existing generation files without re-solving.",
    )
    reconvert_parser.add_argument("--split", default=None, help="Optional split name, e.g. chapter_6")
    add_dataset_override_args(reconvert_parser)
    add_question_filter_args(reconvert_parser)
    reconvert_parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model for generation/evaluation file naming.",
    )
    reconvert_parser.add_argument("--only-bad", action="store_true", help="Only reconvert likely problematic rows.")
    reconvert_parser.add_argument("--dry-run", action="store_true", help="Compute changes but do not write files.")
    reconvert_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Do not call conversion API; only normalize current boxed answers.",
    )
    reconvert_parser.add_argument(
        "--solver-reasoning-effort",
        default=None,
        help="Artifact label override for solver outputs generated with an explicit native reasoning effort.",
    )
    reconvert_parser.add_argument(
        "--solver-max-solve-tokens",
        type=int,
        default=None,
        help="Artifact label override for solver outputs generated with a non-default max solve token budget.",
    )
    extract_parser = sub.add_parser(
        "extract-json",
        help="Extract saved stage JSON data into one aggregated JSON file.",
    )
    extract_parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    extract_parser.add_argument("--out", default=None, help="Optional output JSON path")
    show_parser = sub.add_parser("show-examples", help="Show dataset examples from generated splits.")
    show_parser.add_argument("--split", default=None, help="Split name like chapter_6")
    show_parser.add_argument("--num", "-n", type=int, default=3, help="Number of examples per split")
    show_parser.add_argument("--random", action="store_true", help="Randomly sample examples")
    show_parser.add_argument("--seed", type=int, default=42, help="Random seed for --random")
    show_parser.add_argument("--max-chars", type=int, default=600, help="Max chars for displayed question")
    analyze_parser = sub.add_parser(
        "analyze-quality",
        help="Analyze conversion/extraction quality with LLM audit vs out_split source data.",
    )
    analyze_parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    analyze_parser.add_argument("--max-items", type=int, default=None, help="Analyze first N rows only")
    analyze_parser.add_argument("--out-json", default=None, help="Output report JSON path")
    analyze_parser.add_argument("--out-md", default=None, help="Output report markdown path")
    analyze_parser.add_argument("--skip-existing", action="store_true", help="Skip if output JSON report already exists.")
    compare_parser = sub.add_parser(
        "compare-eval-errors",
        help="Compare wrong cases between rule-based evaluate and evaluate-llm outputs.",
    )
    compare_parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    compare_parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model used by evaluation files. Overrides config models.default_solver_model.",
    )
    compare_parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model used by evaluate-llm files. Defaults to config evaluate_llm.judge_model.",
    )
    add_dataset_override_args(compare_parser)
    compare_parser.add_argument("--out", default=None, help="Optional output JSONL path")
    compare_parser.add_argument(
        "--solver-reasoning-effort",
        default=None,
        help="Artifact label override for solver outputs generated with an explicit native reasoning effort.",
    )
    compare_parser.add_argument(
        "--solver-max-solve-tokens",
        type=int,
        default=None,
        help="Artifact label override for solver outputs generated with a non-default max solve token budget.",
    )
    export_rule_errors_parser = sub.add_parser(
        "export-rule-errors",
        help="Export current wrong rule-eval questions into a JSONL for manual error categorization.",
    )
    export_rule_errors_parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    export_rule_errors_parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model used for generation/evaluation file lookup. Overrides config models.default_solver_model.",
    )
    add_dataset_override_args(export_rule_errors_parser)
    add_question_filter_args(export_rule_errors_parser)
    export_rule_errors_parser.add_argument("--out", default=None, help="Optional output JSONL path")
    export_rule_errors_parser.add_argument(
        "--solver-reasoning-effort",
        default=None,
        help="Artifact label override for solver outputs generated with an explicit native reasoning effort.",
    )
    export_rule_errors_parser.add_argument(
        "--solver-max-solve-tokens",
        type=int,
        default=None,
        help="Artifact label override for solver outputs generated with a non-default max solve token budget.",
    )
    wrong_keywords_parser = sub.add_parser(
        "analyze-wrong-keywords",
        help="Analyze each model's wrong rule-eval questions and extract diagnostic top keywords.",
    )
    add_dataset_override_args(wrong_keywords_parser)
    wrong_keywords_parser.add_argument("--out-dir", default=None, help="Output directory for wrong-keyword reports.")
    wrong_keywords_parser.add_argument("--top-k", type=int, default=3, help="Top keywords to list per model.")
    wrong_keywords_parser.add_argument(
        "--keywords",
        default=None,
        help=(
            "Comma-separated keywords to compute instead of automatic top keywords. "
            "Defaults to equilibrium, monopolist, marginal, discount factor."
        ),
    )
    wrong_keywords_parser.add_argument("--ngram-min", type=int, default=1, help="Minimum n-gram length.")
    wrong_keywords_parser.add_argument("--ngram-max", type=int, default=3, help="Maximum n-gram length.")
    wrong_keywords_parser.add_argument("--min-token-len", type=int, default=3, help="Minimum token length before n-gram extraction.")
    wrong_keywords_parser.add_argument(
        "--min-wrong-question-count",
        type=int,
        default=1,
        help="Drop keywords that appear in fewer than this many wrong questions for a model.",
    )
    wrong_keywords_parser.add_argument(
        "--keep-generic-econ-terms",
        action="store_true",
        help="Keep broad IO terms such as firm, price, profit, demand, and cost.",
    )
    context_length_parser = sub.add_parser(
        "analyze-context-length",
        help="Compute context-length bins and rule-evaluation breakdowns for model results.",
    )
    add_dataset_override_args(context_length_parser)
    context_length_parser.add_argument("--out-dir", default=None, help="Output directory for context-length reports.")
    context_length_parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model artifact labels. Defaults to all models with rule-evaluation summaries.",
    )
    context_length_parser.add_argument(
        "--binning",
        default="tertiles",
        choices=["tertiles", "fixed"],
        help="Use current dataset tertiles or fixed thresholds. Fixed defaults to 121 and 214.",
    )
    context_length_parser.add_argument("--short-max", type=int, default=None, help="Fixed short-bin maximum word count.")
    context_length_parser.add_argument("--medium-max", type=int, default=None, help="Fixed medium-bin maximum word count.")
    rerun_ids_parser = sub.add_parser(
        "build-rerun-question-ids",
        help="Build a question-id txt file for targeted reruns from current rule evaluation results.",
    )
    rerun_ids_parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    rerun_ids_parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model used for evaluation file lookup. Overrides config models.default_solver_model.",
    )
    add_dataset_override_args(rerun_ids_parser)
    rerun_ids_parser.add_argument("--out", default=None, help="Optional output txt path for question ids.")
    rerun_ids_parser.add_argument(
        "--detail-pattern",
        dest="detail_patterns",
        action="append",
        default=None,
        help="Substring filter for rule eval detail. Can be passed multiple times. Defaults to parse/conversion-style failures.",
    )
    rerun_ids_parser.add_argument(
        "--solver-reasoning-effort",
        default=None,
        help="Artifact label override for solver outputs generated with an explicit native reasoning effort.",
    )
    rerun_ids_parser.add_argument(
        "--solver-max-solve-tokens",
        type=int,
        default=None,
        help="Artifact label override for solver outputs generated with a non-default max solve token budget.",
    )
    token_rerun_ids_parser = sub.add_parser(
        "build-token-limit-rerun-question-ids",
        help="Build a question-id txt file for rows that produced no final answer and likely exhausted the solve token budget.",
    )
    token_rerun_ids_parser.add_argument("--split", default=None, help="Optional split name like chapter_6")
    token_rerun_ids_parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model used for generation file lookup. Overrides config models.default_solver_model.",
    )
    add_dataset_override_args(token_rerun_ids_parser)
    token_rerun_ids_parser.add_argument("--out", default=None, help="Optional output txt path for question ids.")
    token_rerun_ids_parser.add_argument("--out-json", default=None, help="Optional output JSONL path with selection details.")
    token_rerun_ids_parser.add_argument(
        "--min-token-ratio",
        type=float,
        default=0.98,
        help="Treat a row as token-limited when usage_output_tokens >= max_solve_tokens * ratio. Default: 0.98",
    )
    token_rerun_ids_parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=None,
        help="Optional absolute token threshold override. If set, ignores --min-token-ratio.",
    )
    token_rerun_ids_parser.add_argument(
        "--solver-reasoning-effort",
        default=None,
        help="Artifact label override for solver outputs generated with an explicit native reasoning effort.",
    )
    token_rerun_ids_parser.add_argument(
        "--solver-max-solve-tokens",
        type=int,
        default=None,
        help="Artifact label override for solver outputs generated with a non-default max solve token budget.",
    )
    clone_artifact_parser = sub.add_parser(
        "clone-artifact-run",
        help=(
            "Clone a complete baseline generation artifact into a new solver artifact label. "
            "Useful before rerunning only a subset of rows under a larger token budget."
        ),
    )
    clone_artifact_parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Base solver model name for both source and target artifact labels.",
    )
    add_dataset_override_args(clone_artifact_parser)
    clone_artifact_parser.add_argument(
        "--source-solver-reasoning-effort",
        default=None,
        help="Optional native reasoning effort encoded in the source artifact label.",
    )
    clone_artifact_parser.add_argument(
        "--source-solver-max-solve-tokens",
        type=int,
        default=None,
        help="Optional source artifact max solve token budget override. Defaults to config generate.max_solve_tokens.",
    )
    clone_artifact_parser.add_argument(
        "--target-solver-reasoning-effort",
        default=None,
        help="Optional native reasoning effort encoded in the target artifact label.",
    )
    clone_artifact_parser.add_argument(
        "--target-solver-max-solve-tokens",
        type=int,
        default=None,
        help="Optional target artifact max solve token budget override.",
    )
    clone_artifact_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the target artifact directory if it already exists.",
    )
    prune_artifacts_parser = sub.add_parser(
        "prune-artifacts-to-dataset",
        help=(
            "Prune generation/evaluation rows that no longer belong to the current dataset, "
            "without rerunning generation."
        ),
    )
    add_dataset_override_args(prune_artifacts_parser)
    prune_artifacts_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report removals without writing any files.",
    )
    apply_balanced_parser = sub.add_parser(
        "apply-balanced",
        help="Apply reviewed balanced split files into hf_json split files.",
    )
    apply_balanced_parser.add_argument(
        "--chapter",
        default=None,
        help="Optional chapter selector, e.g. 6 or chapter_6",
    )
    apply_balanced_parser.add_argument(
        "--split",
        default=None,
        help="Optional split selector(s), comma-separated, e.g. chapter_6",
    )
    apply_balanced_parser.add_argument(
        "--rebuild-step5",
        action="store_true",
        help="After apply, run build-step5 (finalize aggregate + hf_dataset) for applied splits.",
    )
    all_parser = sub.add_parser("all", help="Run build-dataset, generate, and evaluate.")
    all_parser.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model used for generation/evaluation in `all`.",
    )

    args = parser.parse_args()

    from core.cost_logging import finish_run, start_run
    from core.llm_utils import load_config, resolve_solver_model
    from core.solver_variants import build_solver_artifact_label, validate_native_reasoning_effort

    cfg_for_cost = load_config(args.config)
    solver_model_for_cost = None

    def _artifact_label_for_solver(
        *,
        base_label: Optional[str],
        reasoning_effort: Optional[str] = None,
        max_solve_tokens: Optional[int] = None,
        validation_identifiers: Optional[list[str]] = None,
    ) -> Optional[str]:
        label = str(base_label or "").strip()
        if not label:
            return None
        effort = str(reasoning_effort or "").strip()
        if effort:
            validate_native_reasoning_effort(cfg_for_cost, validation_identifiers or [label], effort)
        effective_max_solve_tokens = (
            int(max_solve_tokens)
            if max_solve_tokens is not None
            else int((cfg_for_cost.get("generate") or {}).get("max_solve_tokens", 4096) or 4096)
        )
        return build_solver_artifact_label(
            label,
            reasoning_effort=effort,
            max_solve_tokens=effective_max_solve_tokens,
        )

    if args.stage == "generate-vllm":
        requested_label = str(getattr(args, "solver_model", None) or "").strip()
        requested_model = str(getattr(args, "model", None) or "").strip()
        base_label = requested_label or requested_model.rstrip("/\\").replace("\\", "/").split("/")[-1].strip() or requested_model or None
        solver_model_for_cost = _artifact_label_for_solver(
            base_label=base_label,
            reasoning_effort=getattr(args, "reasoning_effort", None),
            max_solve_tokens=getattr(args, "max_tokens", None),
            validation_identifiers=[
                requested_model,
                str(base_label or "").strip(),
            ],
        )
    elif args.stage in {"generate", "evaluate", "compare-eval-errors", "export-rule-errors", "reconvert", "build-rerun-question-ids", "build-token-limit-rerun-question-ids", "all"}:
        base_solver_model = resolve_solver_model(
            cfg_for_cost,
            requested_model=getattr(args, "solver_model", None),
        )
        solver_effort = getattr(args, "solver_reasoning_effort", None)
        if args.stage == "generate":
            solver_effort = getattr(args, "reasoning_effort", None)
        solver_model_for_cost = _artifact_label_for_solver(
            base_label=base_solver_model,
            reasoning_effort=solver_effort,
            max_solve_tokens=getattr(args, "solver_max_solve_tokens", None) if args.stage != "generate" else getattr(args, "max_solve_tokens", None),
        )
    elif args.stage == "evaluate-llm":
        base_solver_model = resolve_solver_model(
            cfg_for_cost,
            requested_model=(getattr(args, "solver_model", None) or getattr(args, "legacy_model", None)),
        )
        solver_model_for_cost = _artifact_label_for_solver(
            base_label=base_solver_model,
            reasoning_effort=getattr(args, "solver_reasoning_effort", None),
            max_solve_tokens=getattr(args, "solver_max_solve_tokens", None),
        )

    skip_cost_run = (
        args.stage == "show-examples"
        or (args.stage == "generate-vllm" and bool(getattr(args, "dry_run", False)))
    )
    run_id = None
    if not skip_cost_run:
        run_id = start_run(
            cfg_for_cost,
            stage=args.stage,
            command=" ".join(sys.argv),
            config_path=args.config,
            solver_model=solver_model_for_cost,
            workflow_id=args.workflow_id,
            run_id=args.run_id,
        )
    if run_id:
        print(f"[cost] run_id={run_id}")

    run_status = "success"
    run_error = ""
    try:
        if args.stage == "build-dataset":
            from commands.build_hf_dataset import run as build_run
            build_run(
                config_path=args.config,
                chapter=args.chapter,
                skip_existing_split=args.skip_existing_split,
                skip_existing_llm=args.skip_existing_llm,
            )
        elif args.stage == "build-step1":
            from build_steps.step1_reference import run as step1_run
            step1_run(
                config_path=args.config,
                chapter=args.chapter,
                skip_existing_split=args.skip_existing_split,
                force_all=args.force_all,
                skip_existing=args.skip_existing,
            )
        elif args.stage == "build-step2":
            from build_steps.step2_split import run as step2_run
            step2_run(
                config_path=args.config,
                chapter=args.chapter,
                skip_existing=args.skip_existing,
            )
        elif args.stage == "build-step3":
            from build_steps.step3_transform import run as step3_run
            step3_run(
                config_path=args.config,
                chapter=args.chapter,
                problem=args.problem,
                skip_existing=args.skip_existing,
            )
        elif args.stage == "build-step3-symbol-sync":
            from build_steps.step3_symbol_sync_only import run as step3_sync_run
            step3_sync_run(
                config_path=args.config,
                chapter=args.chapter,
                problem=args.problem,
            )
        elif args.stage == "build-step4":
            from build_steps.step5_rebalance_judge import run as step4_run
            step4_run(
                config_path=args.config,
                chapter=args.chapter,
                skip_existing=args.skip_existing,
            )
        elif args.stage == "build-step5":
            from build_steps.step4_finalize import run as step5_run
            step5_run(
                config_path=args.config,
                chapter=args.chapter,
            )
        elif args.stage == "build-step4-finalize":
            print("[deprecated] `build-step4-finalize` is deprecated. Use `build-step5`.")
            from build_steps.step4_finalize import run as step5_run
            step5_run(
                config_path=args.config,
                chapter=args.chapter,
            )
        elif args.stage == "build-step5-rebalance":
            print("[deprecated] `build-step5-rebalance` is deprecated. Use `build-step4`.")
            from build_steps.step5_rebalance_judge import run as step4_run
            step4_run(
                config_path=args.config,
                chapter=args.chapter,
                skip_existing=args.skip_existing,
            )
        elif args.stage == "generate":
            from commands.generate_boxed_answers import run as gen_run
            gen_run(
                config_path=args.config,
                solver_model=args.solver_model,
                split=args.split,
                hf_json_dir=args.hf_json_dir,
                by_model_dir=args.by_model_dir,
                question_ids=args.question_ids,
                question_ids_file=args.question_ids_file,
                skip_existing=args.skip_existing,
                force=args.force,
                reasoning_mode=args.reasoning_mode,
                reasoning_effort=args.reasoning_effort,
                max_solve_tokens=args.max_solve_tokens,
            )
        elif args.stage == "generate-vllm":
            from commands.generate_vllm_batch import run as gen_vllm_run
            gen_vllm_run(
                config_path=args.config,
                model=args.model,
                solver_model=args.solver_model,
                tokenizer=args.tokenizer,
                split=args.split,
                hf_json_dir=args.hf_json_dir,
                by_model_dir=args.by_model_dir,
                question_ids=args.question_ids,
                question_ids_file=args.question_ids_file,
                skip_existing=args.skip_existing,
                force=args.force,
                batch_size=args.batch_size,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty,
                convert_sympy=args.convert_sympy,
                convert_temperature=args.convert_temperature,
                convert_max_tokens=args.convert_max_tokens,
                dtype=args.dtype,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                quantization=args.quantization,
                load_format=args.load_format,
                download_dir=args.download_dir,
                family=args.family,
                trust_remote_code=args.trust_remote_code,
                enforce_eager=args.enforce_eager,
                stop=args.stop,
                seed=args.seed,
                reasoning_mode=args.reasoning_mode,
                reasoning_effort=args.reasoning_effort,
                qwen3_thinking=args.qwen3_thinking,
                no_tqdm=args.no_tqdm,
                dry_run=args.dry_run,
                global_batch=args.global_batch,
                prompt_preview_chars=args.prompt_preview_chars,
                workflow_id=args.workflow_id,
                run_id=args.run_id,
                manage_run_lifecycle=False,
            )
        elif args.stage == "evaluate":
            from commands.evaluate_generations import run as eval_run
            eval_run(
                config_path=args.config,
                split=args.split,
                include_missing=args.include_missing,
                force=args.force,
                solver_model=args.solver_model,
                hf_json_dir=args.hf_json_dir,
                by_model_dir=args.by_model_dir,
                question_ids=args.question_ids,
                question_ids_file=args.question_ids_file,
                solver_reasoning_effort=args.solver_reasoning_effort,
                solver_max_solve_tokens=args.solver_max_solve_tokens,
            )
        elif args.stage == "evaluate-llm":
            from commands.evaluate_llm_generations import run as eval_llm_run
            eval_llm_run(
                config_path=args.config,
                split=args.split,
                force=args.force,
                solver_model=args.solver_model or args.legacy_model,
                judge_model=args.judge_model or args.legacy_model,
                hf_json_dir=args.hf_json_dir,
                by_model_dir=args.by_model_dir,
                question_ids=args.question_ids,
                question_ids_file=args.question_ids_file,
                reasoning_mode=args.reasoning_mode,
                reasoning_effort=args.reasoning_effort,
                solver_reasoning_effort=args.solver_reasoning_effort,
                solver_max_solve_tokens=args.solver_max_solve_tokens,
            )
        elif args.stage == "reconvert":
            from commands.reconvert_generations import run as reconvert_run
            reconvert_run(
                config_path=args.config,
                split=args.split,
                solver_model=args.solver_model,
                hf_json_dir=args.hf_json_dir,
                by_model_dir=args.by_model_dir,
                question_ids=args.question_ids,
                question_ids_file=args.question_ids_file,
                only_bad=args.only_bad,
                dry_run=args.dry_run,
                no_llm=args.no_llm,
                solver_reasoning_effort=args.solver_reasoning_effort,
                solver_max_solve_tokens=args.solver_max_solve_tokens,
            )
        elif args.stage == "extract-json":
            from commands.extract_pipeline_json import run as extract_run
            extract_run(
                config_path=args.config,
                split=args.split,
                out=args.out,
            )
        elif args.stage == "show-examples":
            from commands.show_dataset_examples import run as show_run
            show_run(
                config_path=args.config,
                split=args.split,
                n=args.num,
                random_pick=args.random,
                seed=args.seed,
                max_chars=args.max_chars,
            )
        elif args.stage == "analyze-quality":
            from commands.analyze_conversion_quality import run as analyze_run
            analyze_run(
                config_path=args.config,
                split=args.split,
                max_items=args.max_items,
                out_json=args.out_json,
                out_md=args.out_md,
                skip_existing=args.skip_existing,
            )
        elif args.stage == "compare-eval-errors":
            from commands.compare_eval_errors import run as compare_run
            compare_run(
                config_path=args.config,
                solver_model=args.solver_model,
                judge_model=args.judge_model,
                split=args.split,
                out=args.out,
                hf_json_dir=args.hf_json_dir,
                by_model_dir=args.by_model_dir,
                solver_reasoning_effort=args.solver_reasoning_effort,
                solver_max_solve_tokens=args.solver_max_solve_tokens,
            )
        elif args.stage == "export-rule-errors":
            from commands.export_rule_errors import run as export_rule_errors_run
            export_rule_errors_run(
                config_path=args.config,
                solver_model=args.solver_model,
                split=args.split,
                hf_json_dir=args.hf_json_dir,
                by_model_dir=args.by_model_dir,
                question_ids=args.question_ids,
                question_ids_file=args.question_ids_file,
                out=args.out,
                solver_reasoning_effort=args.solver_reasoning_effort,
                solver_max_solve_tokens=args.solver_max_solve_tokens,
            )
        elif args.stage == "analyze-wrong-keywords":
            from commands.analyze_model_wrong_keywords import run as wrong_keywords_run
            wrong_keywords_run(
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
        elif args.stage == "analyze-context-length":
            from commands.analyze_context_length import run as context_length_run
            context_length_run(
                config_path=args.config,
                hf_json_dir=args.hf_json_dir,
                by_model_dir=args.by_model_dir,
                out_dir=args.out_dir,
                models=args.models,
                binning=args.binning,
                short_max=args.short_max,
                medium_max=args.medium_max,
            )
        elif args.stage == "build-rerun-question-ids":
            from commands.build_rerun_question_ids import run as build_rerun_ids_run
            build_rerun_ids_run(
                config_path=args.config,
                split=args.split,
                solver_model=args.solver_model,
                hf_json_dir=args.hf_json_dir,
                by_model_dir=args.by_model_dir,
                out=args.out,
                detail_patterns=args.detail_patterns,
                solver_reasoning_effort=args.solver_reasoning_effort,
                solver_max_solve_tokens=args.solver_max_solve_tokens,
            )
        elif args.stage == "build-token-limit-rerun-question-ids":
            from commands.build_token_limit_rerun_question_ids import run as build_token_rerun_ids_run
            build_token_rerun_ids_run(
                config_path=args.config,
                split=args.split,
                solver_model=args.solver_model,
                hf_json_dir=args.hf_json_dir,
                by_model_dir=args.by_model_dir,
                out=args.out,
                out_json=args.out_json,
                min_token_ratio=args.min_token_ratio,
                min_output_tokens=args.min_output_tokens,
                solver_reasoning_effort=args.solver_reasoning_effort,
                solver_max_solve_tokens=args.solver_max_solve_tokens,
            )
        elif args.stage == "clone-artifact-run":
            from commands.clone_artifact_run import run as clone_artifact_run
            clone_artifact_run(
                config_path=args.config,
                solver_model=args.solver_model,
                hf_json_dir=args.hf_json_dir,
                by_model_dir=args.by_model_dir,
                source_solver_reasoning_effort=args.source_solver_reasoning_effort,
                source_solver_max_solve_tokens=args.source_solver_max_solve_tokens,
                target_solver_reasoning_effort=args.target_solver_reasoning_effort,
                target_solver_max_solve_tokens=args.target_solver_max_solve_tokens,
                force=args.force,
            )
        elif args.stage == "prune-artifacts-to-dataset":
            from commands.prune_artifacts_to_dataset import run as prune_artifacts_to_dataset_run
            prune_artifacts_to_dataset_run(
                config_path=args.config,
                hf_json_dir=args.hf_json_dir,
                by_model_dir=args.by_model_dir,
                dry_run=args.dry_run,
            )
        elif args.stage == "apply-balanced":
            from commands.apply_balanced import run as apply_balanced_run
            apply_balanced_run(
                config_path=args.config,
                chapter=args.chapter,
                split=args.split,
                rebuild_step5=args.rebuild_step5,
            )
        elif args.stage == "all":
            from commands.build_hf_dataset import run as build_run
            from commands.generate_boxed_answers import run as gen_run
            from commands.evaluate_generations import run as eval_run
            build_run(args.config)
            gen_run(config_path=args.config, solver_model=args.solver_model)
            eval_run(config_path=args.config, solver_model=args.solver_model)
    except Exception as exc:
        run_status = "error"
        run_error = f"{exc.__class__.__name__}: {exc}"
        raise
    finally:
        finish_run(status=run_status, error_message=run_error)


if __name__ == "__main__":
    main()
