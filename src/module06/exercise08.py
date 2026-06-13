from pydantic import BaseModel, Field


class CourseFeedback(BaseModel):
    strengths: list[str] = Field(description="Positive aspects of the lecture")
    improvements: list[str] = Field(description="Concrete improvements")
    overall_score: int = Field(ge=1, le=10)


example = CourseFeedback(
    strengths=["Clear examples", "Good pacing"],
    improvements=["Add more exercises"],
    overall_score=8
)

print(example.model_dump_json(indent=2))
