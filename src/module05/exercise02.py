from deepeval.metrics import FaithfulnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase


def run_deepeval_verification():
    """
    Evaluates whether a generated RAG answer is faithful to the retrieved
    enterprise policy context.

    Scenario:
    A user asks whether external contractors can access internal systems.
    The RAG system retrieves internal policy fragments and generates an answer.
    DeepEval then checks whether the generated answer is grounded in the
    retrieved context.
    """
    retrieval_context = [
        (
            "Policy 401: External contractors may access internal systems only "
            "through the corporate VPN using multi-factor authentication."
        ),
        (
            "Policy 402: Contractor access must be approved by the project owner "
            "and reviewed every 30 days."
        ),
        (
            "Policy 403: Direct access from public networks to internal systems "
            "is prohibited."
        ),
    ]

    # in a real application you do not manually write this
    # you call the application llm
    generated_answer = (
        "External contractors can access internal systems only through the "
        "corporate VPN with multi-factor authentication. Their access must be "
        "approved by the project owner and reviewed every 30 days. Direct public "
        "network access is not allowed."
    )

    test_case_1 = LLMTestCase(
        input = "Can external contractorsaccess internal company systems?",
        actual_output= generated_answer,
        retrieval_context= retrieval_context
    )

    judge_model = OllamaModel(
       model = "gemma4",
        base_url="http://localhost:11434",
        temperature=0
    )

    metric = FaithfulnessMetric(
        threshold=0.7,
        model=judge_model,
        include_reason=True,
    )

    metric.measure(test_case_1)

    print(f"Input Query: {test_case_1.input}")
    print(f"Faithfulness Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Test State: {'PASSED' if metric.is_successful() else 'FAILED'}")

run_deepeval_verification()