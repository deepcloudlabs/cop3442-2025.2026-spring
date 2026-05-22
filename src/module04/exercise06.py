import statistics
from typing import List
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def compute_median(values: List[float]) -> float:
    """Compute the median of a numeric list."""
    return float(statistics.median(values))


@tool
def detect_outliers_iqr(values: List[float]) -> dict:
    """Detect outliers using the IQR rule and return bounds and outliers."""
    q1 = statistics.quantiles(values, n=4)[0]
    q3 = statistics.quantiles(values, n=4)[2]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = [x for x in values if x < lower or x > upper]
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "outliers": outliers
    }


model = ChatOllama(model="gemma4", temperature=0.0)

agent = create_agent(
    model=model,
    tools=[compute_median, detect_outliers_iqr],
    system_prompt=(
        "You are a data science teaching assistant. "
        "Use tools for numeric calculations and explain the statistical meaning."
    )
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "For [10, 12, 11, 13, 100], compute the median and detect outliers."}
    ]
})

print(result["messages"][-1].content)
