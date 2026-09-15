from app.llm.groq_provider import (
    GroqProvider,
    StructuredLLMResult,
)
from app.prompts.customer_reputation import (
    build_customer_reputation_prompt,
)
from app.schemas.customer_reputation import (
    CustomerReputationInput,
    CustomerReputationOutput,
)


class InvalidPublicSignalReferenceError(ValueError):
    """Raised when Agent 4 cites or omits supplied public signals."""


def validate_public_signal_references(
    output: CustomerReputationOutput,
    agent_input: CustomerReputationInput,
) -> None:
    available_ids = {
        document.signal_id
        for document in agent_input.signal_documents
    }
    assessed_ids = {
        assessment.signal_id
        for assessment in output.signal_assessments
    }

    if assessed_ids != available_ids:
        missing_ids = available_ids - assessed_ids
        unknown_ids = assessed_ids - available_ids
        details = []
        if missing_ids:
            details.append(
                "missing: " + ", ".join(sorted(missing_ids))
            )
        if unknown_ids:
            details.append(
                "unknown: " + ", ".join(sorted(unknown_ids))
            )
        raise InvalidPublicSignalReferenceError(
            "Signal assessments do not match the supplied sample ("
            + "; ".join(details)
            + ")."
        )

    themes = [
        *output.praise_themes,
        *output.complaint_themes,
        *output.customer_pain_points,
        *output.unmet_needs,
        *output.reputation_risks,
        *output.conflicting_signals,
    ]
    for theme in themes:
        unknown_ids = set(theme.signal_ids) - available_ids
        if unknown_ids:
            unknown = ", ".join(sorted(unknown_ids))
            raise InvalidPublicSignalReferenceError(
                f"Theme {theme.theme_id} cited unknown signals: "
                f"{unknown}."
            )

    expected_name = agent_input.company_name.strip().casefold()
    returned_name = output.company_name.strip().casefold()
    if returned_name != expected_name:
        raise InvalidPublicSignalReferenceError(
            "Agent 4 returned a different company name."
        )


class CustomerReputationAgent:
    def __init__(
        self,
        provider: GroqProvider | None = None,
    ) -> None:
        self.provider = provider or GroqProvider()

    async def run(
        self,
        agent_input: CustomerReputationInput,
    ) -> StructuredLLMResult[CustomerReputationOutput]:
        response = await self.provider.generate_structured(
            agent_name="customer_reputation",
            system_prompt=build_customer_reputation_prompt(
                agent_input.purpose,
            ),
            user_prompt=(
                "Analyze this public-signal sample for the selected "
                "company.\n\n"
                f"{agent_input.model_dump_json(indent=2)}"
            ),
            response_model=CustomerReputationOutput,
            max_tokens=3000,
            temperature=0,
        )

        validate_public_signal_references(
            response.data,
            agent_input,
        )

        return response
