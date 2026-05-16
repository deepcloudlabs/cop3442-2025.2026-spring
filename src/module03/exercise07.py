# Structured output using PydanticOutputParser
# pip install -U langchain langchain_ollama pydantic

import os
from typing import Literal

from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class ConceptExplanation(BaseModel):
    concept: str = Field(description="Name of the data science concept")
    definition: str = Field(description="Clear academic definition")
    importance: str = Field(description="Why the concept matters")
    example: str = Field(description="A realistic example")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Estimated difficulty for undergraduate students"
    )


parser = PydanticOutputParser(pydantic_object=ConceptExplanation)

model = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "gemma4"), temperature=0.0)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a data science instructor.
Return only the requested structured object.
{format_instructions}
"""
    ),
    (
        "human",
        "Explain the concept: {concept}"
    )
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

result = chain.invoke({
    "concept": "K-Means inertia"
})

print(result)
print(result.model_dump())
