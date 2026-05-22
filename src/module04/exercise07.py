from typing import List, Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain.agents import create_agent


class LessonPlan(BaseModel):
    title: str = Field(description="Title of the lesson")
    target_audience: str = Field(description="Intended student audience")
    learning_objectives: List[str] = Field(description="Measurable learning objectives")
    key_concepts: List[str] = Field(description="Core concepts covered in the lesson")
    difficulty: Literal["beginner", "intermediate", "advanced"] = Field(description="Estimated difficulty")
    recommended_slide_count: int = Field(description="Recommended number of slides")


model = ChatOllama(model="gemma4", temperature=0.0)

agent = create_agent(
    model=model,
    tools=[],
    response_format=LessonPlan,
    system_prompt=(
        "You are a curriculum designer for an AI engineering program. "
        "Return a valid structured lesson plan."
    )
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Create a lesson plan for Conversational Agency and LLM Workflows."}
    ]
})

lesson_plan = result["structured_response"]
print(lesson_plan)
print(lesson_plan.model_dump())
