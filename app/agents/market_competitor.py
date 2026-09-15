from collections.abc import Iterable

from app.llm.groq_provider import (
    GroqProvider,
    StructuredLLMResult,
)
from app.prompts.market_competitor import (
    build_market_competitor_prompt,
)
from app.schemas.market_competitor import (
    MarketCompetitorInput,
    MarketCompetitorOutput,
)


class InvalidMarketEvidenceReferenceError(ValueError):
    """Raised when Agent 3 cites evidence absent from its input."""


def _output_evidence_groups(
    output: MarketCompetitorOutput,
) -> Iterable[tuple[str, list[str]]]:
    yield (
        "market_definition",
        output.market_definition.evidence_ids,
    )
    yield (
        "market_structure",
        output.market_structure.evidence_ids,
    )

    for competitor in output.competitors:
        yield (
            f"competitor:{competitor.competitor_id}",
            competitor.evidence_ids,
        )

    sections = {
        "entry_barrier": output.entry_barriers,
        "market_trend": output.market_trends,
        "market_gap": output.market_gaps,
        "competitive_risk": output.competitive_risks,
    }
    for section_name, findings in sections.items():
        for finding in findings:
            yield (
                f"{section_name}:{finding.finding_id}",
                finding.evidence_ids,
            )


def validate_market_evidence_references(
    output: MarketCompetitorOutput,
    agent_input: MarketCompetitorInput,
) -> None:
    available_ids = {
        *(
            fact.fact_id
            for fact in agent_input.company_evidence.verified_facts
        ),
        *(
            claim.claim_id
            for claim in agent_input.company_evidence.company_claims
        ),
        *(
            finding.finding_id
            for finding in agent_input.business_fundamentals.findings
        ),
        *(
            document.source_id
            for document in agent_input.market_documents
        ),
    }

    for item_name, evidence_ids in _output_evidence_groups(output):
        unknown_ids = set(evidence_ids) - available_ids
        if unknown_ids:
            unknown = ", ".join(sorted(unknown_ids))
            raise InvalidMarketEvidenceReferenceError(
                f"{item_name} cited unknown evidence: {unknown}."
            )

    expected_name = (
        agent_input.company_evidence.company_identity.name
        .strip()
        .casefold()
    )
    returned_name = output.company_name.strip().casefold()
    if returned_name != expected_name:
        raise InvalidMarketEvidenceReferenceError(
            "Agent 3 returned a different company name."
        )


class MarketCompetitorAgent:
    def __init__(
        self,
        provider: GroqProvider | None = None,
    ) -> None:
        self.provider = provider or GroqProvider()

    async def run(
        self,
        agent_input: MarketCompetitorInput,
    ) -> StructuredLLMResult[MarketCompetitorOutput]:
        response = await self.provider.generate_structured(
            agent_name="market_competitor",
            system_prompt=build_market_competitor_prompt(
                agent_input.purpose,
            ),
            user_prompt=(
                "Analyse the following company, business, and market "
                "evidence.\n\n"
                f"{agent_input.model_dump_json(indent=2)}"
            ),
            response_model=MarketCompetitorOutput,
            max_tokens=3000,
            temperature=0,
        )

        validate_market_evidence_references(
            response.data,
            agent_input,
        )

        return response
