from pydantic import BaseModel, Field
from typing import Optional


class SkillCategory(BaseModel):
    technical: list[str]
    soft: list[str]
    tools: list[str]
    certifications: list[str]


class FitScore(BaseModel):
    overall: int = Field(..., ge=0, le=100)
    rationale: str
    matched_keywords: list[str]
    missing_keywords: list[str]


class AtsScore(BaseModel):
    overall: int = Field(..., ge=0, le=100)
    rationale: str
    strengths: list[str]
    weaknesses: list[str]


class Suggestion(BaseModel):
    priority: str
    category: str
    suggestion: str
    example: Optional[str] = None


class AnalyzeResponse(BaseModel):
    candidate_name: Optional[str]
    contact_email: Optional[str]
    years_of_experience: Optional[float]
    current_title: Optional[str]
    summary: str
    skills: SkillCategory
    ats_score: AtsScore
    fit_score: Optional[FitScore]
    suggestions: list[Suggestion]
    raw_text_length: int
    model_used: str
