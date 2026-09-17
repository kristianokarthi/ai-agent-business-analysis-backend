import pytest

from app.agents.customer_reputation import (
    InvalidPublicSignalReferenceError,
    finalize_customer_reputation_output,
    validate_public_signal_references,
)
from app.schemas.customer_reputation import (
    CustomerReputationDraft,
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


def test_finalizer_derives_counts_theme_ids_and_missing_assessments():
    agent_input = CustomerReputationInput.model_validate(VALID_INPUT)
    draft = CustomerReputationDraft.model_validate(
        {
            "signal_assessments": VALID_OUTPUT["signal_assessments"][:-1],
            "praise_themes": [
                {
                    "statement": VALID_OUTPUT["praise_themes"][0][
                        "statement"
                    ],
                    "signal_ids": [
                        "signal_review_1",
                        "signal_review_1",
                        "signal_review_3",
                    ],
                    "confidence": "medium",
                }
            ],
            "complaint_themes": [],
            "customer_pain_points": [],
            "unmet_needs": [],
            "reputation_risks": [],
            "conflicting_signals": [],
            "missing_information": ["The sample is small."],
            "overall_confidence": "low",
        }
    )

    output = finalize_customer_reputation_output(draft, agent_input)

    assert output.sample_size == 4
    assert output.sentiment_summary.analyzed_signal_count == 4
    assert output.sentiment_summary.positive_count == 1
    assert output.sentiment_summary.negative_count == 1
    assert output.sentiment_summary.mixed_count == 1
    assert output.sentiment_summary.unclear_count == 1
    assert output.praise_themes[0].theme_id == "praise_1"
    assert output.praise_themes[0].signal_count == 2
    assert output.signal_assessments[-1].sentiment.value == "unclear"
    assert "marked unclear" in output.missing_information[-1]

    validate_public_signal_references(output, agent_input)
