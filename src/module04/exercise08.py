"""
Deterministic two-step workflow:
1. Extract requirements.
2. Generate an outline.
"""

import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="gemma4", temperature=0.0)

extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the main requirements from the user request as concise bullet points."),
    ("human", "{request}")
])

outline_prompt = ChatPromptTemplate.from_messages([
    ("system", "Create a structured lecture outline from the extracted requirements."),
    ("human", "Requirements:\n{requirements}\n\nCreate the outline.")
])

extract_chain = extract_prompt | model | StrOutputParser()
outline_chain = outline_prompt | model | StrOutputParser()

request = "Prepare lecture material on conversational agency with memory, tools, agents, workflows, and Python code."
requirements = extract_chain.invoke({"request": request})
outline = outline_chain.invoke({"requirements": requirements})

print("REQUIREMENTS")
print(requirements)
print("\nOUTLINE")
print(outline)
