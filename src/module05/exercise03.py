import json
import re
from dataclasses import dataclass
from typing import Callable, List, Dict, Any

from langchain_ollama import ChatOllama


@dataclass
class GEvalResult:
    score: int
    reason: str
    evaluation_steps: List[str]


def extract_json(text: str) -> Dict[str, Any]:
    """
    Extracts a JSON object from an LLM response.

    This is useful because local LLMs may sometimes return extra explanatory text
    before or after the JSON payload.
    """

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(f"No JSON object found in model output:\n{text}")

    return json.loads(match.group(0))


def build_ollama_judge(model_name: str = "gemma4") -> Callable[[str], str]:
    """
    Creates a local Ollama judge callable.

    The judge callable receives a prompt and returns the model response as text.
    """

    llm = ChatOllama(
        model=model_name,
        temperature=0,
    )

    def judge_callable(prompt: str) -> str:
        response = llm.invoke(prompt)
        return response.content

    return judge_callable


def generate_evaluation_steps(
    criteria_definition: str,
    judge_callable: Callable[[str], str],
) -> List[str]:
    """
    Stage 1 of G-Eval:
    Ask the judge LLM to create step-by-step evaluation instructions.
    """

    rubric_prompt = f"""
You are designing an evaluation rubric for an LLM-generated answer.

Evaluation criterion:
{criteria_definition}

Create a concise step-by-step evaluation plan.

Return only valid JSON in this format:

{{
  "steps": [
    "Step 1...",
    "Step 2...",
    "Step 3..."
  ]
}}
"""

    raw_output = judge_callable(rubric_prompt)
    parsed_output = extract_json(raw_output)

    return parsed_output["steps"]


def apply_geval_rubric(
    target_text: str,
    criteria_definition: str,
    evaluation_steps: List[str],
    judge_callable: Callable[[str], str],
) -> GEvalResult:
    """
    Stage 2 of G-Eval:
    Ask the judge LLM to apply the generated rubric to the target text.
    """

    steps_text = "\n".join(
        f"{index + 1}. {step}" for index, step in enumerate(evaluation_steps)
    )

    evaluation_prompt = f"""
You are an impartial evaluator.

Evaluation criterion:
{criteria_definition}

Evaluation steps:
{steps_text}

Text to evaluate:
{target_text}

Assign an integer score from 1 to 5.

Scoring scale:
1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

Return only valid JSON in this format:

{{
  "score": 4,
  "reason": "Briefly explain why this score was assigned."
}}
"""

    raw_output = judge_callable(evaluation_prompt)
    parsed_output = extract_json(raw_output)

    score = int(parsed_output["score"])

    if score < 1 or score  >5:
        raise ValueError(f"Invalid score returned by judge: {score}")

    return GEvalResult(
        score=score,
        reason=parsed_output["reason"],
        evaluation_steps=evaluation_steps,
    )


def execute_geval_exercise():
    """
    Complete G-Eval exercise.

    Scenario:
    We evaluate whether an LLM-generated answer is clear, complete,
    and grounded in the provided enterprise policy.
    """

    target_text = """
External contractors can access internal company systems only through the
corporate VPN with multi-factor authentication. Their access must be approved
by the project owner and reviewed every 30 days. Direct access from public
networks is prohibited.
"""

    criteria_definition = """
Evaluate whether the answer is clear, complete, and policy-aligned.
The answer should be specific, avoid unsupported claims, and correctly explain
the access rules for external contractors.
"""

    judge_callable = build_ollama_judge(model_name="gemma4")

    evaluation_steps = generate_evaluation_steps(
        criteria_definition=criteria_definition,
        judge_callable=judge_callable,
    )

    result = apply_geval_rubric(
        target_text=target_text,
        criteria_definition=criteria_definition,
        evaluation_steps=evaluation_steps,
        judge_callable=judge_callable,
    )

    print("G-Eval Exercise Result")
    print("-" * 50)

    print("\nGenerated Evaluation Steps:")
    for index, step in enumerate(result.evaluation_steps, start=1):
        print(f"{index}. {step}")

    print("\nFinal Score:")
    print(result.score)

    print("\nReason:")
    print(result.reason)


if __name__ == "__main__":
    execute_geval_exercise()