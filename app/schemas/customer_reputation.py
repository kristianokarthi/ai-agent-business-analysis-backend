from collections import Counter
from enum import Enum

from pydantic import Field, HttpUrl, model_validator

from app.schemas.business_fundamentals import ConfidenceLevel
from app.schemas.fact_finder import StrictSchema
from app.schemas.llm import LLMUsage
from app.schemas.research import ResearchPurpose


class PublicSignalStatus(str, Enum):
    COMPLETED = "completed"
    INSUFFICIENT_DATA = "insufficient_data"


class SignalSourceType(str, Enum):
    CUSTOMER_REVIEW = "customer_review"
    EMPLOYEE_REVIEW = "employee_review"
    NEWS = "news"
    FORUM = "forum"
    SOCIAL_MEDIA = "social_media"
    SURVEY = "survey"
    OTHER = "other"


class SignalSentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class PublicSignalDocument(StrictSchema):
    signal_id: str = Field(
        min_length=1,
        pattern=r"^signal_[A-Za-z0-9_-]+$",
    )
    title: str = Field(min_length=1)
    url: HttpUrl
    source_type: SignalSourceType
    content: str = Field(min_length=1, max_length=4000)
    rating: float | None = Field(
        default=None,
        ge=1,
        le=5,
    )


class CustomerReputationInput(StrictSchema):
    company_name: str = Field(min_length=1)
    purpose: ResearchPurpose
    follow_up_answers: dict[str, str] = Field(default_factory=dict)
    signal_documents: list[PublicSignalDocument] = Field(
        min_length=1,
        max_length=20,
    )

    @model_validator(mode="after")
    def require_unique_signal_ids(self) -> "CustomerReputationInput":
        signal_ids = [
            document.signal_id
            for document in self.signal_documents
        ]
        if len(signal_ids) != len(set(signal_ids)):
            raise ValueError("Signal IDs must be unique.")
        return self


class SignalAssessment(StrictSchema):
    signal_id: str = Field(min_length=1)
    sentiment: SignalSentiment
    summary: str = Field(min_length=1)
    themes: list[str] = Field(max_length=3)


class SentimentSummary(StrictSchema):
    classification: SignalSentiment
    analyzed_signal_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    neutral_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    mixed_count: int = Field(ge=0)
    unclear_count: int = Field(ge=0)
    confidence: ConfidenceLevel


class SignalTheme(StrictSchema):
    theme_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    signal_ids: list[str] = Field(min_length=1)
    signal_count: int = Field(ge=1)
    confidence: ConfidenceLevel

    @model_validator(mode="after")
    def validate_signal_count(self) -> "SignalTheme":
        unique_signal_ids = set(self.signal_ids)
        if len(unique_signal_ids) != len(self.signal_ids):
            raise ValueError(
                "A theme cannot cite the same signal more than once."
            )
        if self.signal_count != len(unique_signal_ids):
            raise ValueError(
                "signal_count must match the number of signal_ids."
            )
        return self


class SignalThemeDraft(StrictSchema):
    statement: str = Field(min_length=1)
    signal_ids: list[str] = Field(min_length=1)
    confidence: ConfidenceLevel


class CustomerReputationDraft(StrictSchema):
    signal_assessments: list[SignalAssessment] = Field(
        min_length=1,
        max_length=20,
    )
    praise_themes: list[SignalThemeDraft] = Field(max_length=3)
    complaint_themes: list[SignalThemeDraft] = Field(max_length=3)
    customer_pain_points: list[SignalThemeDraft] = Field(max_length=3)
    unmet_needs: list[SignalThemeDraft] = Field(max_length=3)
    reputation_risks: list[SignalThemeDraft] = Field(max_length=3)
    conflicting_signals: list[SignalThemeDraft] = Field(max_length=3)
    missing_information: list[str]
    overall_confidence: ConfidenceLevel


class CustomerReputationOutput(StrictSchema):
    status: PublicSignalStatus
    company_name: str = Field(min_length=1)
    sample_size: int = Field(ge=0)
    sentiment_summary: SentimentSummary
    signal_assessments: list[SignalAssessment] = Field(max_length=20)
    praise_themes: list[SignalTheme] = Field(max_length=5)
    complaint_themes: list[SignalTheme] = Field(max_length=5)
    customer_pain_points: list[SignalTheme] = Field(max_length=5)
    unmet_needs: list[SignalTheme] = Field(max_length=5)
    reputation_risks: list[SignalTheme] = Field(max_length=5)
    conflicting_signals: list[SignalTheme] = Field(max_length=5)
    missing_information: list[str]
    overall_confidence: ConfidenceLevel

    @model_validator(mode="after")
    def validate_counts_and_duplicates(
        self,
    ) -> "CustomerReputationOutput":
        assessment_ids = [
            assessment.signal_id
            for assessment in self.signal_assessments
        ]
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError(
                "Every signal must be assessed only once."
            )

        if self.sample_size != len(self.signal_assessments):
            raise ValueError(
                "sample_size must match signal_assessments."
            )

        counts = Counter(
            assessment.sentiment
            for assessment in self.signal_assessments
        )
        expected_counts = {
            SignalSentiment.POSITIVE: (
                self.sentiment_summary.positive_count
            ),
            SignalSentiment.NEUTRAL: (
                self.sentiment_summary.neutral_count
            ),
            SignalSentiment.NEGATIVE: (
                self.sentiment_summary.negative_count
            ),
            SignalSentiment.MIXED: (
                self.sentiment_summary.mixed_count
            ),
            SignalSentiment.UNCLEAR: (
                self.sentiment_summary.unclear_count
            ),
        }

        if self.sentiment_summary.analyzed_signal_count != self.sample_size:
            raise ValueError(
                "analyzed_signal_count must match sample_size."
            )

        for sentiment, expected_count in expected_counts.items():
            if counts[sentiment] != expected_count:
                raise ValueError(
                    f"{sentiment.value}_count does not match "
                    "signal assessments."
                )

        themes = [
            *self.praise_themes,
            *self.complaint_themes,
            *self.customer_pain_points,
            *self.unmet_needs,
            *self.reputation_risks,
            *self.conflicting_signals,
        ]
        theme_ids = [theme.theme_id for theme in themes]
        if len(theme_ids) != len(set(theme_ids)):
            raise ValueError("Theme IDs must be unique.")

        return self


class CustomerReputationAPIResponse(StrictSchema):
    result: CustomerReputationOutput
    usage: LLMUsage
