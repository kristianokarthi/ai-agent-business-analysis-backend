from enum import Enum

from pydantic import Field, HttpUrl, model_validator

from app.schemas.customer_reputation import (
    CustomerReputationOutput,
    PublicSignalDocument,
)
from app.schemas.fact_finder import StrictSchema
from app.schemas.llm import LLMUsage
from app.schemas.research import ResearchPurpose


class PublicSignalCollectionStatus(str, Enum):
    COMPLETED = "completed"
    INSUFFICIENT_SOURCES = "insufficient_sources"


class PublicSignalCollectionInput(StrictSchema):
    company_name: str = Field(min_length=2, max_length=200)
    purpose: ResearchPurpose
    follow_up_answers: dict[str, str] = Field(default_factory=dict)


class FailedPublicSignalSource(StrictSchema):
    url: HttpUrl
    error: str


class PublicSignalCollectorUsage(StrictSchema):
    provider: str = "tavily"
    search_requests: int = Field(ge=0, le=3)
    extraction_requests: int = Field(ge=0, le=1)
    searched_results: int = Field(ge=0)
    selected_urls: int = Field(ge=0, le=8)
    extracted_documents: int = Field(ge=0, le=8)
    credits_used: float = Field(ge=0)
    response_time_seconds: float = Field(ge=0)


class PublicSignalCollectionOutput(StrictSchema):
    status: PublicSignalCollectionStatus
    company_name: str = Field(min_length=1)
    search_queries: list[str] = Field(min_length=1, max_length=3)
    signal_documents: list[PublicSignalDocument] = Field(max_length=8)
    failed_sources: list[FailedPublicSignalSource] = Field(max_length=8)
    usage: PublicSignalCollectorUsage
    warnings: list[str]

    @model_validator(mode="after")
    def validate_status(self) -> "PublicSignalCollectionOutput":
        if (
            self.status == PublicSignalCollectionStatus.COMPLETED
            and not self.signal_documents
        ):
            raise ValueError(
                "Completed collection must contain public signals."
            )
        if (
            self.status
            == PublicSignalCollectionStatus.INSUFFICIENT_SOURCES
            and self.signal_documents
        ):
            raise ValueError(
                "Insufficient-source collection cannot contain signals."
            )
        return self


class CustomerReputationResearchResponse(StrictSchema):
    result: CustomerReputationOutput
    usage: LLMUsage
    collection: PublicSignalCollectionOutput
