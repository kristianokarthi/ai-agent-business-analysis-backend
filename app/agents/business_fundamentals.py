from app.llm.groq_provider import (
    GroqProvider,
    StructuredLLMResult,
)
from app.prompts.business_fundamentals import (
    build_business_fundamentals_prompt,
)
from app.schemas.business_fundamentals import (
    BusinessFundamentalsInput,
    BusinessFundamentalsOutput,
    FindingBasis,
)
from app.schemas.fact_finder import FactFinderOutput


class InvalidEvidenceReferenceError(ValueError):
    """Raised when Agent 2 cites evidence that Agent 1 did not provide."""


def validate_evidence_references(
    output: BusinessFundamentalsOutput,
    evidence_pack: FactFinderOutput,
) -> None:
    fact_ids = {
        fact.fact_id
        for fact in evidence_pack.verified_facts
    }
    claim_ids = {
        claim.claim_id
        for claim in evidence_pack.company_claims
    }
    available_ids = fact_ids | claim_ids

    for finding in output.findings:
        referenced_ids = set(finding.evidence_ids)
        unknown_ids = referenced_ids - available_ids

        if unknown_ids:
            unknown = ", ".join(sorted(unknown_ids))
            raise InvalidEvidenceReferenceError(
                f"Finding {finding.finding_id} cited unknown evidence: "
                f"{unknown}."
            )

        if (
            finding.basis == FindingBasis.VERIFIED_FACT
            and not referenced_ids.issubset(fact_ids)
        ):
            raise InvalidEvidenceReferenceError(
                f"Finding {finding.finding_id} is marked verified_fact "
                "but cites a company claim."
            )

        if (
            finding.basis == FindingBasis.COMPANY_CLAIM
            and not referenced_ids.issubset(claim_ids)
        ):
            raise InvalidEvidenceReferenceError(
                f"Finding {finding.finding_id} is marked company_claim "
                "but cites a verified fact."
            )


class BusinessFundamentalsAgent:
    def __init__(
        self,
        provider: GroqProvider | None = None,
    ) -> None:
        self.provider = provider or GroqProvider()

    async def run(
        self,
        agent_input: BusinessFundamentalsInput,
    ) -> StructuredLLMResult[BusinessFundamentalsOutput]:
        system_prompt = build_business_fundamentals_prompt(
            agent_input.purpose,
        )
        user_prompt = (
            "Analyse this Agent 1 evidence pack and return business "
            "fundamentals.\n\n"
            f"{agent_input.model_dump_json(indent=2)}"
        )

        response = await self.provider.generate_structured(
            agent_name="business_fundamentals",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=BusinessFundamentalsOutput,
            max_tokens=2000,
            temperature=0.1,
        )

        validate_evidence_references(
            response.data,
            agent_input.evidence_pack,
        )

        return response
