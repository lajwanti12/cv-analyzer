from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.analyze import router
from config import settings

app = FastAPI(
    title="Resume Analyzer API",
    version="1.0.0",
    description="AI-powered resume analysis using Google Gemini",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.gemini_model}
