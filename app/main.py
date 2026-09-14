from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.fact_finder import router as fact_finder_router
from app.api.research import router as research_router
from app.core.config import settings


app = FastAPI(
    title="AI Business Analyzer API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(research_router)
app.include_router(fact_finder_router)


@app.get("/")
def home() -> dict[str, str]:
    return {
        "message": "FastAPI is running successfully",
    }


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
    }
