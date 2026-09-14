from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)

from app.schemas.llm import LLMUsage
from app.schemas.research import ResearchPurpose


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FactFinderStatus(str, Enum):
    COMPLETED = "completed"
    INSUFFICIENT_DATA = "insufficient_data"


class FactCategory(str, Enum):
    COMPANY_IDENTITY = "company_identity"
    HISTORY = "history"
    OWNERSHIP = "ownership"
    PRODUCTS_SERVICES = "products_services"
    CUSTOMERS_MARKETS = "customers_markets"
    OPERATIONS = "operations"
    PARTNERSHIPS = "partnerships"
    FINANCIAL = "financial"
    OTHER = "other"


class SourceDocument(StrictSchema):
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: HttpUrl
    content: str = Field(min_length=1)


class FactFinderRequest(StrictSchema):
    """Public request accepted from the Next.js application."""

    company_name: str = Field(min_length=2, max_length=200)
    official_website: HttpUrl
    purpose: ResearchPurpose
    follow_up_answers: dict[str, str] = Field(default_factory=dict)


class FactFinderInput(FactFinderRequest):
    """Internal agent input after the API collects source documents."""

    official_website: HttpUrl | None = None
    documents: list[SourceDocument] = Field(min_length=1)


class CompanyIdentity(StrictSchema):
    name: str
    legal_name: str | None
    industry: str | None
    headquarters: str | None
    official_website: str | None


class EvidenceFact(StrictSchema):
    fact_id: str
    statement: str
    category: FactCategory
    source_ids: list[str]


class CompanyClaim(StrictSchema):
    claim_id: str
    statement: str
    category: FactCategory
    source_ids: list[str]


class ConflictingClaim(StrictSchema):
    topic: str
    statements: list[str]
    source_ids: list[str]


class FactFinderOutput(StrictSchema):
    status: FactFinderStatus
    company_identity: CompanyIdentity
    verified_facts: list[EvidenceFact]
    company_claims: list[CompanyClaim]
    conflicting_claims: list[ConflictingClaim]
    missing_information: list[str]
    sources_used: list[str]

    @model_validator(mode="after")
    def prevent_duplicate_facts_and_claims(
        self,
    ) -> "FactFinderOutput":
        verified_statements = {
            fact.statement.strip().lower()
            for fact in self.verified_facts
        }
        claim_statements = {
            claim.statement.strip().lower()
            for claim in self.company_claims
        }

        if verified_statements.intersection(claim_statements):
            raise ValueError(
                "A statement cannot appear in both "
                "verified_facts and company_claims."
            )

        return self


class FactFinderAPIResponse(StrictSchema):
    result: FactFinderOutput
    usage: LLMUsage
