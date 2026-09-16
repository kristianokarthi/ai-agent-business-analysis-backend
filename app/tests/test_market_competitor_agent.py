import asyncio

import pytest

from app.agents.market_competitor import (
    InvalidMarketEvidenceReferenceError,
    MarketCompetitorAgent,
    validate_market_evidence_references,
)
from app.llm.groq_provider import StructuredLLMResult
from app.schemas.llm import LLMUsage
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


def test_empty_completed_analysis_is_marked_insufficient():
    agent_input = MarketCompetitorInput.model_validate(VALID_INPUT)
    empty_output = MarketCompetitorOutput.model_validate(
        {
            **VALID_OUTPUT,
            "competitors": [],
            "entry_barriers": [],
            "market_trends": [],
            "market_gaps": [],
            "competitive_risks": [],
        }
    )

    class FakeProvider:
        async def generate_structured(self, **_):
            return StructuredLLMResult(
                data=empty_output,
                usage=LLMUsage(
                    agent_name="market_competitor",
                    provider="groq",
                    model="test-model",
                    input_tokens=100,
                    output_tokens=50,
                    total_tokens=150,
                ),
            )

    response = asyncio.run(
        MarketCompetitorAgent(provider=FakeProvider()).run(agent_input)
    )

    assert response.data.status.value == "insufficient_data"
    assert any(
        "did not support competitor" in item
        for item in response.data.missing_information
    )
