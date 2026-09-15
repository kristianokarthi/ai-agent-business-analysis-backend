import pytest

from app.agents.customer_reputation import (
    InvalidPublicSignalReferenceError,
    validate_public_signal_references,
)
from app.schemas.customer_reputation import (
    CustomerReputationInput,
    CustomerReputationOutput,
)
from app.tests.test_customer_reputation_schema import (
    VALID_INPUT,
    VALID_OUTPUT,
)


def test_signal_validator_accepts_complete_known_sample():
    agent_input = CustomerReputationInput.model_validate(VALID_INPUT)
    output = CustomerReputationOutput.model_validate(VALID_OUTPUT)

    validate_public_signal_references(output, agent_input)


def test_signal_validator_rejects_omitted_signal():
    agent_input = CustomerReputationInput.model_validate(VALID_INPUT)
    assessments = VALID_OUTPUT["signal_assessments"][:-1]
    invalid_output = {
        **VALID_OUTPUT,
        "sample_size": 3,
        "sentiment_summary": {
            **VALID_OUTPUT["sentiment_summary"],
            "analyzed_signal_count": 3,
            "neutral_count": 0,
        },
        "signal_assessments": assessments,
        "conflicting_signals": [],
    }
    output = CustomerReputationOutput.model_validate(invalid_output)

    with pytest.raises(
        InvalidPublicSignalReferenceError,
        match="missing",
    ):
        validate_public_signal_references(
            output,
            agent_input,
        )


def test_signal_validator_rejects_unknown_theme_signal():
    agent_input = CustomerReputationInput.model_validate(VALID_INPUT)
    invalid_output = {
        **VALID_OUTPUT,
        "praise_themes": [
            {
                **VALID_OUTPUT["praise_themes"][0],
                "signal_ids": ["signal_unknown"],
                "signal_count": 1,
            }
        ],
    }
    output = CustomerReputationOutput.model_validate(invalid_output)

    with pytest.raises(
        InvalidPublicSignalReferenceError,
        match="unknown signals",
    ):
        validate_public_signal_references(
            output,
            agent_input,
        )


def test_signal_validator_rejects_wrong_company():
    agent_input = CustomerReputationInput.model_validate(VALID_INPUT)
    output = CustomerReputationOutput.model_validate(
        {
            **VALID_OUTPUT,
            "company_name": "Another Company",
        }
    )

    with pytest.raises(
        InvalidPublicSignalReferenceError,
        match="different company",
    ):
        validate_public_signal_references(
            output,
            agent_input,
        )
