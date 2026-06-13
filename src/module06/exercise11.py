import json
from pydantic import BaseModel, ValidationError


class GradeResult(BaseModel):
    grade: int
    feedback: str


raw = '{"grade": "excellent", "feedback": "Good conceptual explanation."}'

try:
    data = GradeResult.model_validate(json.loads(raw))
except ValidationError as e:
    print("Validation failed.")
    print(e.errors())
