# Dynamic prompt assembly without middleware
# pip install -U langchain langchain_ollama

import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model=os.getenv("OLLAMA_MODEL", "gemma4"), temperature=0.0)


def build_system_prompt(user_level: str) -> str:
    base = "You are a data science instructor."

    if user_level == "beginner":
        return base + " Explain with simple language and intuitive examples."
    elif user_level == "intermediate":
        return base + " Include formulas when useful and provide Python-oriented intuition."
    elif user_level == "advanced":
        return base + " Use technical terminology, assumptions, limitations, and edge cases."
    else:
        return base + " Explain clearly and accurately."


def answer_topic(topic: str, user_level: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", build_system_prompt(user_level)),
        ("human", "Explain this topic: {topic}")
    ])

    chain = prompt | model | StrOutputParser()
    return chain.invoke({"topic": topic})


print(answer_topic("DBSCAN clustering", "beginner"))
print(answer_topic("DBSCAN clustering", "advanced"))
