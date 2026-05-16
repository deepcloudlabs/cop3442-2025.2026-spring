# Simple prompt test harness
# pip install -U langchain langchain-ollama

import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model=os.getenv("OLLAMA_MODEL", "gemma4"), temperature=0.0)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a data science instructor.
Always return exactly three numbered points.
"""
    ),
    (
        "human",
        "Explain {topic} for undergraduate students."
    )
])

chain = prompt | model | StrOutputParser()

test_cases = [
    {"topic": "median imputation"},
    {"topic": "classification"},
    {"topic": "PCA"},
    {"topic": "ANOVA"}
]

for case in test_cases:
    print("=" * 80)
    print("TEST CASE:", case)
    assembled_prompt = prompt.format_messages(**case)

    print("\nASSEMBLED PROMPT:")
    for message in assembled_prompt:
        print(f"{message.type.upper()}: {message.content}")

    print("\nMODEL OUTPUT:")
    output = chain.invoke(case)
    print(output)
