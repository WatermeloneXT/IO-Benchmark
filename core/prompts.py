from __future__ import annotations

SYSTEM_REFERENCE_CORRECTION = r"""
You are verifying and correcting textbook reference answers.

Given one question and a candidate reference answer:
1) Judge whether the candidate answer is fully correct and complete for the asked question.
2) If correct and complete, keep it unchanged.
3) If incorrect or incomplete, you need to provide a corrected and complete reference answer that can be used as a gold standard for evaluation.
4) Keep symbols/notation consistent with the question.
5) Citation-only or placeholder answers are NOT complete:
   - examples: \"See X\", \"See section Y\", \"This is straightforward\", \"as above\".
   - when only such content is available, write your own corrected full answer.
6) If candidate answer is incomplete, ignore it and solve from the question to produce a complete final answer.
7) Never return \"N/A\". Always provide the best complete answer you can generate.
8) Do not add irrelevant explanation in the final reference answer string.
9) Output JSON only.

Return JSON:
{
  "reference_is_correct": true,
  "final_reference_answer": "...",
  "reference_generated_by_llm": false,
  "analysis": "short reason",
  "confidence": "high|medium|low"
}
"""


SYSTEM_SPLIT_MULTI_QUESTION = r"""
You are splitting a question into standalone QA pairs when needed.

Input:
- one question
- one correct reference answer

Task:
1) Determine how many distinct asked questions are inside this question.
2) If there is only one asked question, return it as-is (do not rewrite content).
3) If there are multiple asked targets, split into multiple standalone question-answer pairs.
4) Use atomic split policy:
   - Treat enumerated items (i)/(ii)/(iii)... as baseline split units.
   - Within one enumerated item, if there are multiple independently gradable asked targets,
     split them into separate pairs.
   - Imperative cues include: obtain/derive/show/prove/conclude/find/compute/verify/determine.
   - Example: "(i) obtain FOC. Show that A<=p<=B. Show quasi-concavity."
     should be split into 3 pairs, not 1.
5) Every split pair must contain exactly one asked target and be complete/solvable without external references.
6) Keep mathematical/economic objective fidelity.
7) Preserve the original stem/context as much as possible:
   - Keep background/assumptions/notation text unchanged whenever feasible.
   - Prefer copying the stem verbatim and only changing the asked-target part.
   - Avoid unnecessary paraphrasing.
8) Output JSON only.

Return JSON:
{
  "question_count": 1,
  "split_generated_by_llm": false,
  "pairs": [
    {
      "sub_index": 1,
      "question": "...",
      "answer": "...",
      "notes": "short"
    }
  ],
  "analysis": "short reason"
}
"""


SYSTEM_CONVERT_TO_EVAL_QUESTION = r"""
You convert a standalone QA pair into an evaluation-ready item.

Hard constraints:
1) Keep the stem/context as unchanged as possible.
2) Modify only the asked target so the item becomes:
   - preferably a value-computation question ("value"), or
   - a true/false judgement question ("judge") when value form is not suitable.
3) If type is "value", comparable_final_answer MUST be a valid SymPy expression string.
4) If type is "judge", comparable_final_answer MUST be exactly "True" or "False".
5) Update reference reasoning and answer to match the converted question.
6) Output JSON only.
7) For "value" type, enforce parser-friendly symbol style:
   - Do NOT use unknown function-call syntax like D(p), u_i(D,R), f(x,y).
   - Flatten unknown function-like forms into symbols with underscores.
   - Examples:
     * D(p) -> D_p
     * D(c2) -> D_c2
     * u_i(D,R) -> u_i_D_R
   - Keep standard SymPy built-ins unchanged: sqrt(), log(), sin(), cos(), tan(), exp(), Abs(), Min(), Max(), Eq(), Ne(), Le(), Ge(), Lt(), Gt(), Derivative().
8) reference_answer_sympy and comparable_final_answer must follow the same naming convention.
9) For "value" type, you MUST provide symbol_contract.allowed_symbols that covers every symbol used in reference_answer_sympy and comparable_final_answer.
10) For "value" type, converted_question MUST include a "Symbols (for final answer):" section.
11) Self-check before output:
   - If reference_answer_sympy or comparable_final_answer uses symbol X, then X MUST appear in symbol_contract.allowed_symbols.
   - If not, revise symbol_contract or revise the answer so they are consistent.

Return JSON:
{
  "question_type": "value|judge",
  "converted_question": "...",
  "reference_reasoning": "...",
  "reference_answer": "...",
  "comparable_final_answer": "...",
  "comparison_mode": "sympy|exact",
  "reference_answer_sympy": "...",
  "symbol_contract": {
    "allowed_symbols": ["delta", "alpha_star", "Pi_q_plus", "Pi_q_c", "T"],
    "symbol_definitions": {
      "delta": "discount factor",
      "alpha_star": "trigger probability parameter"
    }
  },
  "notes": "short"
}
"""


SYSTEM_LATEX_TO_SYMPY = r"""
Convert a LaTeX math expression to a valid SymPy expression.

Output ONLY the SymPy expression string and nothing else.

Rules:
- \frac{a}{b} -> (a)/(b)
- x^{n} -> x**n
- \sqrt{x} -> sqrt(x)
- \ln or \log -> log
- a=b -> Eq(a, b)
- a\le b or a\leq b -> Le(a, b)
- a\ge b or a\geq b -> Ge(a, b)
- a<b -> Lt(a, b)
- a>b -> Gt(a, b)
- a\neq b -> Ne(a, b)
- Keep plain text answers unchanged.
- Do NOT use unknown function-call syntax like D(p), u_i(D,R), f(x,y).
- Flatten such forms into underscore symbols:
  * D(p) -> D_p
  * D(c2) -> D_c2
  * u_i(D,R) -> u_i_D_R
"""


SYSTEM_SOLVER_WITH_BOX = r"""
You are solving advanced economics/math problems.

You must:
1) Solve the problem step by step.
2) End with exactly one final answer in \boxed{...}.
3) Respect expected answer mode:
   - if mode is sympy: the box must contain one SymPy-style final expression
   - if mode is bool: the box must be exactly True or False
4) If the final answer cannot be determined, output \boxed{N/A}.
5) Do not put extra prose inside \boxed{}.
6) When mode is sympy, follow parser-friendly symbol naming:
   - Do NOT use unknown function-call syntax like D(p), u_i(D,R), f(x,y).
   - Flatten to underscore symbols instead:
     * D(p) -> D_p
     * D(c2) -> D_c2
     * u_i(D,R) -> u_i_D_R
   - Keep standard SymPy built-ins unchanged: sqrt(), log(), sin(), cos(), tan(), exp(), Abs(), Min(), Max(), Eq(), Ne(), Le(), Ge(), Lt(), Gt(), Derivative().
"""


SYSTEM_REBALANCE_JUDGE_FLIP = r"""
You rewrite a True/False question so the correct label flips.

Input:
- one existing judge question
- old label (True/False)

Task:
1) Keep the stem/context as unchanged as possible.
2) Prefer editing only the asked-target sentence/phrase.
3) The new question must be natural, self-consistent, and solvable from itself.
4) The new correct label MUST be the opposite of old_label.
5) Return JSON only.
6) Self-check before output:
   - flipped must be true
   - natural must be true
   - new_label is exactly True or False
   - new_label != old_label

Return JSON:
{
  "rewritten_question": "...",
  "new_label": "True|False",
  "flipped": true,
  "natural": true,
  "analysis": "short reason"
}
"""


SYSTEM_LLM_EVAL = r"""
You are a strict evaluator for economics/math QA.

You are given:
- question
- reference_reasoning
- reference_answer
- predict_reasoning
- predict_answer

Evaluate two dimensions independently:

1) is_correct (answer only)
- True iff predict_answer is correct/equivalent to reference_answer.
- Ignore reasoning quality completely.
- Ignore formatting differences that do not change meaning.
- If predict_answer is missing, ambiguous, or not a real answer -> false.

2) reasoning_correct (reasoning only)
- True iff predict_reasoning is logically valid and supports the predicted conclusion.
- Ignore whether the final answer matches reference_answer.
- False if reasoning is missing, irrelevant, self-contradictory, circular, or has decisive math/econ errors.

Output constraints:
- Return strict JSON only (no markdown, no extra keys, no comments).
- Use lowercase JSON booleans true/false.
- judge_reason must be exactly one sentence.

JSON schema:
{
  "is_correct": true,
  "reasoning_correct": true,
  "judge_reason": "..."
}
"""



def reference_correction_user_prompt(question_latex: str, reference_answer: str) -> str:
    return (
        f"Question (LaTeX):\n{question_latex}\n\n"
        f"Candidate reference answer:\n{reference_answer}\n"
    )


def split_questions_user_prompt(question_latex: str, reference_answer: str) -> str:
    return (
        f"Question (LaTeX):\n{question_latex}\n\n"
        f"Correct reference answer:\n{reference_answer}\n\n"
        "Hard constraint: keep the original stem/context text unchanged as much as possible.\n"
        "Atomic split policy: one asked-target per pair.\n"
        "If one item contains multiple independent asks (e.g., obtain... show... show...), split them separately.\n"
        "Do not merge two independent show/prove/derive targets into one pair.\n"
    )


def convert_eval_user_prompt(question_latex: str, reference_answer: str) -> str:
    return (
        f"Standalone question (LaTeX/text):\n{question_latex}\n\n"
        f"Reference answer:\n{reference_answer}\n"
    )


def solve_user_prompt(
    question_final: str,
    answer_kind: str,
    allowed_symbols: list[str] | None = None,
    json_keys: list[str] | None = None,
) -> str:
    expected = (answer_kind or "sympy").strip().lower()
    allowed = [str(x).strip() for x in (allowed_symbols or []) if str(x).strip()]
    expected_json_keys = [str(x).strip() for x in (json_keys or []) if str(x).strip()]
    symbols_rule = ""
    if expected == "sympy" and allowed:
        joined = ", ".join(allowed)
        symbols_rule = (
            "- You MUST only use the allowed symbols listed below in the final boxed answer.\n"
            f"- Allowed symbols: {joined}\n"
            "- Do NOT invent new symbols or aliases.\n"
        )
    json_rule = ""
    if expected == "json":
        json_rule = (
            "- If expected answer type is json, put exactly one JSON object in the box.\n"
            "- Use strict JSON with double-quoted keys and string values when appropriate.\n"
            "- Do NOT add explanation outside the JSON object inside the box.\n"
        )
        if expected_json_keys:
            joined = ", ".join(expected_json_keys)
            json_rule += (
                f"- The JSON object should use these keys: {joined}\n"
                "- Preserve the key names exactly.\n"
            )
    text_rule = ""
    if expected == "text":
        text_rule = "- If expected answer type is text, put only the final short answer in the box.\n"
    return (
        f"Question:\n{question_final}\n\n"
        f"Expected answer type: {expected}\n\n"
        "[Final answer formatting]\n"
        "- Put your final answer in exactly one \\boxed{...}.\n"
        "- If expected answer type is bool, the box must be exactly True or False.\n"
        "- If expected answer type is sympy, put only one final SymPy expression in the box.\n"
        "- If expected answer type is sympy, do NOT use unknown function-call syntax (e.g., D(p), u_i(D,R)).\n"
        "- Use flattened symbol names with underscores instead (e.g., D_p, u_i_D_R, D_c2).\n"
        f"{symbols_rule}"
        f"{json_rule}"
        f"{text_rule}"
    )


def rebalance_judge_user_prompt(question_text: str, old_label: str) -> str:
    return (
        f"Original judge question:\n{question_text}\n\n"
        f"Old label: {old_label}\n\n"
        "Rewrite the question so the new label is the opposite of old label.\n"
    )
