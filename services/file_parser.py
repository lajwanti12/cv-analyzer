import io
from fastapi import UploadFile, HTTPException
import PyPDF2
from docx import Document


async def extract_text(file: UploadFile) -> str:
    content_type = file.content_type or ""
    raw = await file.read()

    if len(raw) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    if content_type == "application/pdf" or (file.filename or "").endswith(".pdf"):
        return _parse_pdf(raw)
    elif "wordprocessingml" in content_type or (file.filename or "").endswith(".docx"):
        return _parse_docx(raw)
    elif content_type.startswith("text/") or (file.filename or "").endswith(".txt"):
        return raw.decode("utf-8", errors="replace")
    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: '{content_type}'. Please upload a PDF, DOCX, or TXT file.",
        )


def _parse_pdf(data: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {e}")


def _parse_docx(data: bytes) -> str:
    try:
        doc = Document(io.BytesIO(data))
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse DOCX: {e}")
