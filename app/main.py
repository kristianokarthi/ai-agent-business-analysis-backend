from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.business_fundamentals import (
    router as business_fundamentals_router,
)
from app.api.customer_reputation import (
    router as customer_reputation_router,
)
from app.api.fact_finder import router as fact_finder_router
from app.api.market_competitor import (
    router as market_competitor_router,
)
from app.api.market_evidence import router as market_evidence_router
from app.api.public_signal_evidence import (
    router as public_signal_evidence_router,
)
from app.api.research import router as research_router
from app.api.report_context import router as report_context_router
from app.api.tavily_tools import router as tavily_tools_router
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
app.include_router(business_fundamentals_router)
app.include_router(market_competitor_router)
app.include_router(customer_reputation_router)
app.include_router(tavily_tools_router)
app.include_router(market_evidence_router)
app.include_router(public_signal_evidence_router)
app.include_router(report_context_router)


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
