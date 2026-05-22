from typing import Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

LONG_TERM_MEMORY: Dict[str, List[str]] = {
    "user_001": [
        "Prefers Python code examples."
        "Teaches Introduction to Data Science"
        "Prefers academic but clear explanations"
    ]
}


def retrieve_user_memory(user_id: str) -> str:
    memories = LONG_TERM_MEMORY.get(user_id, [])
    if not memories:
        return "No stored preferences"
    return "\n".join(f"- {memory}" for memory in memories)


model = ChatOllama(model="gemma4", temperature=0.0)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an AI teaching assistant.

Relevant long-term memory:
{memory}

Use the memory only when it is relevant to the current task.
"""
    ),
    (
        "human",
        "{question}"
    )
])

chain = prompt | model | StrOutputParser()

answer = chain.invoke({
    "memory": retrieve_user_memory("user_001"),
    "question": "Explain conversational agency for undergraduate students."
})

print(answer)

