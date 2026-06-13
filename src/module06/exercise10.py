from enum import Enum
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class GeneratedAnswerAudit(BaseModel):
    answer_relevant: bool
    grounded: bool
    risk_level: RiskLevel
    issues: list[str] = Field(default_factory=list)


audit = GeneratedAnswerAudit(
    answer_relevant=True,
    grounded=False,
    risk_level=RiskLevel.medium,
    issues=["No citation for one claim"]
)

print(audit.model_dump())
