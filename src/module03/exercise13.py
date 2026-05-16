# Multi-step prompt decomposition with LangChain
# pip install -U langchain langchain-ollama

import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model=os.getenv("OLLAMA_MODEL", "gemma4"), temperature=0.0)

topic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a curriculum designer."
    ),
    (
        "human",
        "List five subtopics for a lecture on {main_topic}. Return only the list."
    )
])

slide_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an academic presentation writer."
    ),
    (
        "human",
        """
Main topic: {main_topic}

Subtopics:
{subtopics}

Create presentation-ready content.
For each subtopic, provide a slide title and three bullet points.
"""
    )
])

topic_chain = topic_prompt | model | StrOutputParser()
slide_chain = slide_prompt | model | StrOutputParser()

main_topic = "Prompt Content and Assembling the Prompt"

subtopics = topic_chain.invoke({"main_topic": main_topic})
slides = slide_chain.invoke({
    "main_topic": main_topic,
    "subtopics": subtopics
})

print("SUBTOPICS")
print(subtopics)

print("\nSLIDE CONTENT")
print(slides)
