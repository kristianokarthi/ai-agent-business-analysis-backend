from fastapi import FastAPI
from app.api.research import router as research_router
from app.api.fact_finder import router as fact_finder_router
app = FastAPI(
    title="AI Business Analyzer API",
    version="1.0.0",
)

app.include_router(research_router)
app.include_router(fact_finder_router)
@app.get("/")
def home():
    return {
        "message": "FastAPI is running successfully"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy"
    }