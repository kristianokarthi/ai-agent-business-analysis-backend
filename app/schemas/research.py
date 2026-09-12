from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class ResearchPurpose(str, Enum):
    GENERAL_RESEARCH = "general_research"
    START_SIMILAR_BUSINESS = "start_similar_business"
    PARTNER_OR_SUPPLIER = "partner_or_supplier"
    STOCK_RESEARCH = "stock_research"


class ReportDepth(str, Enum):
    OVERVIEW = "overview"
    DEEP_ANALYSIS = "deep_analysis"


class FollowUpQuestion(BaseModel):
    id: str
    question: str
    options: list[str] = Field(default_factory=list)
    placeholder: str = ""


class ResearchRequest(BaseModel):
    company_request: str = Field(
        min_length=2,
        max_length=500,
        examples=["Analyse Coca-Cola"],
    )
    purpose: ResearchPurpose
    official_website: Optional[HttpUrl] = None
    report_depth: ReportDepth = ReportDepth.OVERVIEW
    follow_up_answers: dict[str, str] = Field(default_factory=dict)


class ResearchResponse(BaseModel):
    status: str
    company_request: str
    purpose: ResearchPurpose
    official_website: Optional[HttpUrl]
    report_depth: ReportDepth
    follow_up_answers: dict[str, str]
    message: str


class FollowUpQuestionsResponse(BaseModel):
    purpose: ResearchPurpose
    questions: list[FollowUpQuestion]