# PromptTemplate with validation of input variables
# pip install -U langchain langchain_ollama

import os

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "gemma4"), temperature=0.0)

prompt = PromptTemplate(
    input_variables=["course", "topic", "student_level", "number_of_slides"],
    template="""
Course: {course}
Topic: {topic}
Student level: {student_level}

Create content for {number_of_slides} presentation slides.

For each slide:
- Provide a slide title.
- Provide 3 to 5 bullet points.
- Provide one instructor note.
- Use academically appropriate language.
"""
)

print("Expected variables:", prompt.input_variables)

chain = prompt | model | StrOutputParser()

result = chain.invoke({
    "course": "AIN-2002 Introduction to Data Science",
    "topic": "Prompt templates in LangChain",
    "student_level": "undergraduate",
    "number_of_slides": 5
})

print(result)
