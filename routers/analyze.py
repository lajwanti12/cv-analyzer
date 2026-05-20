from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from services.file_parser import extract_text
from services.gemini_service import analyze_resume
from schemas import AnalyzeResponse
from config import settings

router = APIRouter(prefix="/api", tags=["analyze"])

MAX_BYTES = settings.max_file_size_mb * 1024 * 1024


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(
    resume: UploadFile = File(..., description="Resume file: PDF, DOCX, or TXT"),
    job_description: Optional[str] = Form(None),
):
    if resume.size and resume.size > MAX_BYTES:
        raise HTTPException(413, f"File exceeds {settings.max_file_size_mb}MB limit.")

    text = await extract_text(resume)

    if not text.strip():
        raise HTTPException(
            422,
            "Could not extract any text from this file. "
            "If it's a scanned PDF, please convert it to text first.",
        )

    try:
        return await analyze_resume(text, job_description)
    except ValueError as e:
        raise HTTPException(500, str(e))
