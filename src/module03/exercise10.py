# Agent prompt with a tool
# pip install -U langchain langchain_ollama

import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama


@tool
def compute_average_score(quiz_score: float, midterm_score: float) -> float:
    """Compute the arithmetic mean of quiz score and midterm score."""
    return (quiz_score + midterm_score) / 2


model = ChatOllama(model=os.getenv("OLLAMA_MODEL", "gemma4"), temperature=0.0)

agent = create_agent(
    model=model,
    tools=[compute_average_score],
    system_prompt=(
        "You are a data science teaching assistant. "
        "Use the tool when exact score calculation is required. "
        "After using the tool, explain the result in one clear sentence."
    )
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "A student has QuizScore 82 and MidtermScore 90. What is the average score?"
        }
    ]
})

print(result["messages"][-1].content)
