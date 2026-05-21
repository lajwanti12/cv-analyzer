import json
import re
from google import genai
from google.genai import types
from config import settings
from schemas import AnalyzeResponse, SkillCategory, FitScore, AtsScore, Suggestion

client = genai.Client(api_key=settings.gemini_api_key)

ANALYSIS_PROMPT = """You are an expert technical recruiter and career coach.
Analyze the resume text below and return ONLY a valid JSON object — no markdown fences, no explanation.

RESUME TEXT:
{resume_text}

{jd_section}

Return this exact JSON structure:
{{
  "candidate_name": "<string or null>",
  "contact_email": "<string or null>",
  "years_of_experience": <number or null>,
  "current_title": "<string or null>",
  "summary": "<2-3 sentence professional summary of this candidate>",
  "skills": {{
    "technical": ["<skill>"],
    "soft": ["<skill>"],
    "tools": ["<tool/framework>"],
    "certifications": ["<cert>"]
  }},
  "ats_score": {{
    "overall": <0-100>,
    "rationale": "<2-3 sentences explaining the ATS score>",
    "strengths": ["<what the resume does well for ATS>"],
    "weaknesses": ["<what hurts ATS compatibility>"]
  }},
  "fit_score": {fit_score_instruction},
  "suggestions": [
    {{
      "priority": "high|medium|low",
      "category": "content|format|keywords|experience",
      "suggestion": "<specific actionable suggestion>",
      "example": "<optional concrete example or null>"
    }}
  ]
}}

Rules:
- suggestions: provide exactly 5-7 items ordered by priority descending
- skills arrays: deduplicate, use canonical casing (e.g. "Python" not "python")
- ats_score.overall: integer 0-100 based on keyword density, formatting clarity, section completeness, quantified achievements, ATS-parseable structure
- ats_score.strengths: 2-4 specific things resume does well for ATS parsing
- ats_score.weaknesses: 2-4 specific things that hurt ATS compatibility
- fit_score.overall: integer 0-100 based on keyword overlap, experience match, seniority alignment
- If a field cannot be determined, use null
- Return ONLY the JSON. Any text outside the JSON will break parsing."""

FIT_SCORE_WITH_JD = """{
    "overall": <0-100>,
    "rationale": "<2-3 sentences explaining the score>",
    "matched_keywords": ["<keyword found in both resume and JD>"],
    "missing_keywords": ["<important JD keyword absent from resume>"]
  }"""

FIT_SCORE_WITHOUT_JD = "null"

JD_SECTION_TEMPLATE = """
JOB DESCRIPTION (score the resume against this):
{job_description}
"""


async def analyze_resume(resume_text: str, job_description: str | None) -> AnalyzeResponse:
    jd_section = (
        JD_SECTION_TEMPLATE.format(job_description=job_description)
        if job_description and job_description.strip()
        else ""
    )
    fit_instruction = FIT_SCORE_WITH_JD if jd_section else FIT_SCORE_WITHOUT_JD

    prompt = ANALYSIS_PROMPT.format(
        resume_text=resume_text[:12000],
        jd_section=jd_section,
        fit_score_instruction=fit_instruction,
    )

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    raw_json = _clean_json(response.text)
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned invalid JSON: {e}\nRaw response: {raw_json[:500]}")

    return _build_response(data, resume_text, settings.gemini_model)


def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _build_response(data: dict, resume_text: str, model: str) -> AnalyzeResponse:
    skills_raw = data.get("skills", {})
    skills = SkillCategory(
        technical=skills_raw.get("technical", []),
        soft=skills_raw.get("soft", []),
        tools=skills_raw.get("tools", []),
        certifications=skills_raw.get("certifications", []),
    )

    ats_raw = data.get("ats_score", {})
    ats_score = AtsScore(
        overall=ats_raw.get("overall", 0),
        rationale=ats_raw.get("rationale", ""),
        strengths=ats_raw.get("strengths", []),
        weaknesses=ats_raw.get("weaknesses", []),
    )

    fit_raw = data.get("fit_score")
    fit_score = None
    if fit_raw and isinstance(fit_raw, dict):
        fit_score = FitScore(
            overall=fit_raw.get("overall", 0),
            rationale=fit_raw.get("rationale", ""),
            matched_keywords=fit_raw.get("matched_keywords", []),
            missing_keywords=fit_raw.get("missing_keywords", []),
        )

    suggestions = [
        Suggestion(
            priority=s.get("priority", "medium"),
            category=s.get("category", "content"),
            suggestion=s.get("suggestion", ""),
            example=s.get("example"),
        )
        for s in data.get("suggestions", [])
    ]

    return AnalyzeResponse(
        candidate_name=data.get("candidate_name"),
        contact_email=data.get("contact_email"),
        years_of_experience=data.get("years_of_experience"),
        current_title=data.get("current_title"),
        summary=data.get("summary", ""),
        skills=skills,
        ats_score=ats_score,
        fit_score=fit_score,
        suggestions=suggestions,
        raw_text_length=len(resume_text),
        model_used=model,
    )
