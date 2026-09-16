from enum import Enum

from pydantic import Field, HttpUrl, model_validator

from app.schemas.business_fundamentals import BusinessFundamentalsOutput
from app.schemas.fact_finder import FactFinderOutput, StrictSchema
from app.schemas.llm import LLMUsage
from app.schemas.market_competitor import (
    MarketCompetitorOutput,
    MarketSourceDocument,
)
from app.schemas.research import ResearchPurpose


class MarketCollectionStatus(str, Enum):
    COMPLETED = "completed"
    INSUFFICIENT_SOURCES = "insufficient_sources"


class MarketEvidenceCollectionInput(StrictSchema):
    purpose: ResearchPurpose
    follow_up_answers: dict[str, str] = Field(default_factory=dict)
    company_evidence: FactFinderOutput
    business_fundamentals: BusinessFundamentalsOutput

    @model_validator(mode="after")
    def validate_company_identity(
        self,
    ) -> "MarketEvidenceCollectionInput":
        company_name = (
            self.company_evidence.company_identity.name.strip().casefold()
        )
        analyzed_name = (
            self.business_fundamentals.company_name.strip().casefold()
        )
        if company_name != analyzed_name:
            raise ValueError(
                "Agent 1 and Agent 2 company names must match."
            )

        return self


class FailedMarketSource(StrictSchema):
    url: HttpUrl
    error: str


class MarketCollectorUsage(StrictSchema):
    provider: str = "tavily"
    search_requests: int = Field(ge=0, le=3)
    extraction_requests: int = Field(ge=0, le=1)
    searched_results: int = Field(ge=0)
    selected_urls: int = Field(ge=0, le=5)
    extracted_documents: int = Field(ge=0, le=5)
    credits_used: float = Field(ge=0)
    response_time_seconds: float = Field(ge=0)


class MarketEvidenceCollectionOutput(StrictSchema):
    status: MarketCollectionStatus
    company_name: str = Field(min_length=1)
    search_queries: list[str] = Field(min_length=1, max_length=3)
    market_documents: list[MarketSourceDocument] = Field(max_length=5)
    failed_sources: list[FailedMarketSource] = Field(max_length=5)
    usage: MarketCollectorUsage
    warnings: list[str]

    @model_validator(mode="after")
    def validate_status(self) -> "MarketEvidenceCollectionOutput":
        if (
            self.status == MarketCollectionStatus.COMPLETED
            and not self.market_documents
        ):
            raise ValueError(
                "Completed collection must contain market documents."
            )
        if (
            self.status == MarketCollectionStatus.INSUFFICIENT_SOURCES
            and self.market_documents
        ):
            raise ValueError(
                "Insufficient-source collection cannot contain documents."
            )

        return self


class MarketCompetitorResearchResponse(StrictSchema):
    result: MarketCompetitorOutput
    usage: LLMUsage
    collection: MarketEvidenceCollectionOutput
