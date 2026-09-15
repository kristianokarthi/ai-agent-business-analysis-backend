import pytest

from app.agents.market_competitor import (
    InvalidMarketEvidenceReferenceError,
    validate_market_evidence_references,
)
from app.schemas.market_competitor import (
    MarketCompetitorInput,
    MarketCompetitorOutput,
)
from app.tests.test_market_competitor_schema import (
    VALID_INPUT,
    VALID_OUTPUT,
)


def test_market_evidence_validator_accepts_known_ids():
    agent_input = MarketCompetitorInput.model_validate(VALID_INPUT)
    output = MarketCompetitorOutput.model_validate(VALID_OUTPUT)

    validate_market_evidence_references(output, agent_input)


def test_market_evidence_validator_rejects_unknown_id():
    agent_input = MarketCompetitorInput.model_validate(VALID_INPUT)
    invalid_output = {
        **VALID_OUTPUT,
        "market_structure": {
            **VALID_OUTPUT["market_structure"],
            "evidence_ids": ["market_source_999"],
        },
    }
    output = MarketCompetitorOutput.model_validate(invalid_output)

    with pytest.raises(
        InvalidMarketEvidenceReferenceError,
        match="unknown evidence",
    ):
        validate_market_evidence_references(
            output,
            agent_input,
        )


def test_market_evidence_validator_rejects_wrong_company():
    agent_input = MarketCompetitorInput.model_validate(VALID_INPUT)
    output = MarketCompetitorOutput.model_validate(
        {
            **VALID_OUTPUT,
            "company_name": "Another Company",
        }
    )

    with pytest.raises(
        InvalidMarketEvidenceReferenceError,
        match="different company",
    ):
        validate_market_evidence_references(
            output,
            agent_input,
        )
