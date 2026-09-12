import pytest
from pydantic import ValidationError

from app.schemas.fact_finder import (
    FactFinderInput,
    FactFinderOutput,
)


VALID_INPUT = {
    "company_name": "Coca-Cola",
    "official_website": "https://www.coca-colacompany.com",
    "purpose": "general_research",
    "follow_up_answers": {},
    "documents": [
        {
            "source_id": "source_1",
            "title": "About Coca-Cola",
            "url": "https://www.coca-colacompany.com/about-us",
            "content": (
                "The Coca-Cola Company is a beverage company."
            ),
        }
    ],
}


VALID_OUTPUT = {
    "status": "completed",
    "company_identity": {
        "name": "The Coca-Cola Company",
        "legal_name": "The Coca-Cola Company",
        "industry": "Beverages",
        "headquarters": "Atlanta, United States",
        "official_website": (
            "https://www.coca-colacompany.com"
        ),
    },
    "verified_facts": [
        {
            "fact_id": "fact_1",
            "statement": (
                "The company operates in the beverage industry."
            ),
            "category": "company_identity",
            "source_ids": ["source_1"],
        }
    ],
    "company_claims": [],
    "conflicting_claims": [],
    "missing_information": [
        "Detailed product-level revenue was not provided."
    ],
    "sources_used": ["source_1"],
}


def test_fact_finder_accepts_valid_input():
    result = FactFinderInput.model_validate(
        VALID_INPUT,
    )

    assert result.company_name == "Coca-Cola"
    assert result.official_website is not None
    assert result.purpose.value == "general_research"
    assert len(result.documents) == 1
    assert result.documents[0].source_id == "source_1"


def test_fact_finder_accepts_missing_website():
    input_without_website = {
        **VALID_INPUT,
        "official_website": None,
    }

    result = FactFinderInput.model_validate(
        input_without_website,
    )

    assert result.official_website is None


def test_fact_finder_rejects_unknown_fields():
    invalid_input = {
        **VALID_INPUT,
        "unexpected_field": "This should not be accepted",
    }

    with pytest.raises(ValidationError):
        FactFinderInput.model_validate(
            invalid_input,
        )


def test_fact_finder_accepts_valid_output():
    result = FactFinderOutput.model_validate(
        VALID_OUTPUT,
    )

    assert result.status.value == "completed"
    assert (
        result.company_identity.name
        == "The Coca-Cola Company"
    )
    assert len(result.verified_facts) == 1
    assert (
        result.verified_facts[0].source_ids
        == ["source_1"]
    )


def test_fact_finder_rejects_incomplete_output():
    invalid_output = {
        "status": "completed",
        "company_identity": {
            "name": "The Coca-Cola Company",
        },
    }

    with pytest.raises(ValidationError):
        FactFinderOutput.model_validate(
            invalid_output,
        )


def test_fact_finder_rejects_duplicate_fact_and_claim():
    invalid_output = {
        **VALID_OUTPUT,
        "company_claims": [
            {
                "claim_id": "claim_1",
                "statement": (
                    "The company operates in the "
                    "beverage industry."
                ),
                "category": "company_identity",
                "source_ids": ["source_1"],
            }
        ],
    }

    with pytest.raises(
        ValidationError,
        match="cannot appear in both",
    ):
        FactFinderOutput.model_validate(
            invalid_output,
        )