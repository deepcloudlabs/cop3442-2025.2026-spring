from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field


class LessonPlan(BaseModel):
    title: str
    objectives: list[str]
    difficulty: str = Field(description="beginner, intermediate, or advanced")

parser = PydanticOutputParser(pydantic_object=LessonPlan)

prompt = PromptTemplate.from_template("""
Create a lesson plan about {topic}.

{format_instructions}
""").partial(format_instructions = parser.get_format_instructions())

llm = OllamaLLM(model="gemma4", temperature=0.0)

chain = prompt | llm | parser

chain.invoke({"topic": "retrieval-augmented generation"})