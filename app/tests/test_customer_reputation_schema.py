import pytest
from pydantic import ValidationError

from app.schemas.customer_reputation import (
    CustomerReputationInput,
    CustomerReputationOutput,
)


VALID_INPUT = {
    "company_name": "The Coca-Cola Company",
    "purpose": "start_similar_business",
    "follow_up_answers": {
        "target_location": "Chennai",
    },
    "signal_documents": [
        {
            "signal_id": "signal_review_1",
            "title": "Customer review one",
            "url": "https://example.com/reviews/1",
            "source_type": "customer_review",
            "content": "I like the taste and the product is easy to find.",
            "rating": 5,
        },
        {
            "signal_id": "signal_review_2",
            "title": "Customer review two",
            "url": "https://example.com/reviews/2",
            "source_type": "customer_review",
            "content": "The drink is too sweet. I want lower-sugar options.",
            "rating": 2,
        },
        {
            "signal_id": "signal_review_3",
            "title": "Customer review three",
            "url": "https://example.com/reviews/3",
            "source_type": "customer_review",
            "content": (
                "I enjoy the taste, but I am concerned about "
                "packaging waste."
            ),
            "rating": 3,
        },
        {
            "signal_id": "signal_news_1",
            "title": "Packaging initiative report",
            "url": "https://example.com/news/packaging",
            "source_type": "news",
            "content": (
                "The company announced a packaging-reduction initiative."
            ),
            "rating": None,
        },
    ],
}


VALID_OUTPUT = {
    "status": "completed",
    "company_name": "The Coca-Cola Company",
    "sample_size": 4,
    "sentiment_summary": {
        "classification": "mixed",
        "analyzed_signal_count": 4,
        "positive_count": 1,
        "neutral_count": 1,
        "negative_count": 1,
        "mixed_count": 1,
        "unclear_count": 0,
        "confidence": "low",
    },
    "signal_assessments": [
        {
            "signal_id": "signal_review_1",
            "sentiment": "positive",
            "summary": "Praises taste and availability.",
            "themes": ["taste", "availability"],
        },
        {
            "signal_id": "signal_review_2",
            "sentiment": "negative",
            "summary": "Criticizes sweetness and requests lower sugar.",
            "themes": ["sweetness", "lower sugar"],
        },
        {
            "signal_id": "signal_review_3",
            "sentiment": "mixed",
            "summary": "Praises taste but raises a packaging concern.",
            "themes": ["taste", "packaging"],
        },
        {
            "signal_id": "signal_news_1",
            "sentiment": "neutral",
            "summary": "Reports a packaging-reduction initiative.",
            "themes": ["packaging"],
        },
    ],
    "praise_themes": [
        {
            "theme_id": "praise_1",
            "statement": (
                "Two supplied customer reviews mention enjoying the taste."
            ),
            "signal_ids": [
                "signal_review_1",
                "signal_review_3",
            ],
            "signal_count": 2,
            "confidence": "medium",
        }
    ],
    "complaint_themes": [
        {
            "theme_id": "complaint_1",
            "statement": (
                "One supplied review describes the drink as too sweet."
            ),
            "signal_ids": ["signal_review_2"],
            "signal_count": 1,
            "confidence": "low",
        }
    ],
    "customer_pain_points": [],
    "unmet_needs": [
        {
            "theme_id": "need_1",
            "statement": (
                "One supplied review requests lower-sugar options."
            ),
            "signal_ids": ["signal_review_2"],
            "signal_count": 1,
            "confidence": "low",
        }
    ],
    "reputation_risks": [],
    "conflicting_signals": [
        {
            "theme_id": "conflict_1",
            "statement": (
                "The sample contains both a packaging concern and "
                "a report of a packaging-reduction initiative."
            ),
            "signal_ids": [
                "signal_review_3",
                "signal_news_1",
            ],
            "signal_count": 2,
            "confidence": "low",
        }
    ],
    "missing_information": [
        "The four-signal sample is too small for population conclusions."
    ],
    "overall_confidence": "low",
}


def test_customer_reputation_accepts_valid_input():
    result = CustomerReputationInput.model_validate(VALID_INPUT)

    assert result.company_name == "The Coca-Cola Company"
    assert len(result.signal_documents) == 4


def test_customer_reputation_rejects_duplicate_signal_ids():
    invalid_input = {
        **VALID_INPUT,
        "signal_documents": [
            VALID_INPUT["signal_documents"][0],
            {
                **VALID_INPUT["signal_documents"][1],
                "signal_id": "signal_review_1",
            },
        ],
    }

    with pytest.raises(
        ValidationError,
        match="Signal IDs must be unique",
    ):
        CustomerReputationInput.model_validate(invalid_input)


def test_customer_reputation_accepts_valid_output():
    result = CustomerReputationOutput.model_validate(VALID_OUTPUT)

    assert result.sample_size == 4
    assert result.sentiment_summary.mixed_count == 1


def test_customer_reputation_rejects_incorrect_sentiment_count():
    invalid_output = {
        **VALID_OUTPUT,
        "sentiment_summary": {
            **VALID_OUTPUT["sentiment_summary"],
            "positive_count": 2,
        },
    }

    with pytest.raises(
        ValidationError,
        match="positive_count does not match",
    ):
        CustomerReputationOutput.model_validate(invalid_output)


def test_customer_reputation_rejects_incorrect_theme_count():
    invalid_output = {
        **VALID_OUTPUT,
        "praise_themes": [
            {
                **VALID_OUTPUT["praise_themes"][0],
                "signal_count": 1,
            }
        ],
    }

    with pytest.raises(
        ValidationError,
        match="signal_count must match",
    ):
        CustomerReputationOutput.model_validate(invalid_output)
