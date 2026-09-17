from collections import Counter

from app.llm.groq_provider import (
    GroqProvider,
    StructuredLLMResult,
)
from app.prompts.customer_reputation import (
    build_customer_reputation_prompt,
)
from app.schemas.customer_reputation import (
    CustomerReputationDraft,
    CustomerReputationInput,
    CustomerReputationOutput,
    PublicSignalStatus,
    SentimentSummary,
    SignalAssessment,
    SignalSentiment,
    SignalTheme,
    SignalThemeDraft,
)


class InvalidPublicSignalReferenceError(ValueError):
    """Raised when Agent 4 cites or omits supplied public signals."""


def _overall_sentiment(
    counts: Counter[SignalSentiment],
) -> SignalSentiment:
    positive = counts[SignalSentiment.POSITIVE]
    negative = counts[SignalSentiment.NEGATIVE]
    neutral = counts[SignalSentiment.NEUTRAL]
    mixed = counts[SignalSentiment.MIXED]
    unclear = counts[SignalSentiment.UNCLEAR]

    if mixed or (positive and negative):
        return SignalSentiment.MIXED
    if positive > max(negative, neutral, unclear):
        return SignalSentiment.POSITIVE
    if negative > max(positive, neutral, unclear):
        return SignalSentiment.NEGATIVE
    if neutral >= max(positive, negative, unclear) and neutral:
        return SignalSentiment.NEUTRAL
    if unclear:
        return SignalSentiment.UNCLEAR
    return SignalSentiment.MIXED


def _normalize_assessments(
    draft: CustomerReputationDraft,
    agent_input: CustomerReputationInput,
) -> list[SignalAssessment]:
    available_ids = [
        document.signal_id
        for document in agent_input.signal_documents
    ]
    available_set = set(available_ids)
    assessments_by_id: dict[str, SignalAssessment] = {}
    for assessment in draft.signal_assessments:
        if (
            assessment.signal_id in available_set
            and assessment.signal_id not in assessments_by_id
        ):
            assessments_by_id[assessment.signal_id] = assessment

    return [
        assessments_by_id.get(signal_id)
        or SignalAssessment(
            signal_id=signal_id,
            sentiment=SignalSentiment.UNCLEAR,
            summary="No reliable assessment was generated for this signal.",
            themes=[],
        )
        for signal_id in available_ids
    ]


def _normalize_themes(
    drafts: list[SignalThemeDraft],
    prefix: str,
    available_ids: set[str],
) -> list[SignalTheme]:
    normalized: list[SignalTheme] = []
    for draft in drafts:
        signal_ids = list(
            dict.fromkeys(
                signal_id
                for signal_id in draft.signal_ids
                if signal_id in available_ids
            )
        )
        if not signal_ids:
            continue
        normalized.append(
            SignalTheme(
                theme_id=f"{prefix}_{len(normalized) + 1}",
                statement=draft.statement,
                signal_ids=signal_ids,
                signal_count=len(signal_ids),
                confidence=draft.confidence,
            )
        )
    return normalized


def finalize_customer_reputation_output(
    draft: CustomerReputationDraft,
    agent_input: CustomerReputationInput,
) -> CustomerReputationOutput:
    assessments = _normalize_assessments(draft, agent_input)
    counts = Counter(
        assessment.sentiment for assessment in assessments
    )
    available_ids = {
        document.signal_id
        for document in agent_input.signal_documents
    }

    missing_information = list(dict.fromkeys(draft.missing_information))
    draft_assessment_ids = [
        assessment.signal_id
        for assessment in draft.signal_assessments
    ]
    if (
        len(draft_assessment_ids) != len(available_ids)
        or len(set(draft_assessment_ids)) != len(draft_assessment_ids)
        or set(draft_assessment_ids) != available_ids
    ):
        missing_information.append(
            "Some model assessments were missing, duplicated, or unsupported; "
            "the affected supplied signals were marked unclear."
        )

    if any(
        set(theme.signal_ids) - available_ids
        for field in (
            "praise_themes",
            "complaint_themes",
            "customer_pain_points",
            "unmet_needs",
            "reputation_risks",
            "conflicting_signals",
        )
        for theme in getattr(draft, field)
    ):
        missing_information.append(
            "Unsupported theme references were excluded from the analysis."
        )

    theme_specs = (
        ("praise_themes", "praise"),
        ("complaint_themes", "complaint"),
        ("customer_pain_points", "pain_point"),
        ("unmet_needs", "unmet_need"),
        ("reputation_risks", "reputation_risk"),
        ("conflicting_signals", "conflict"),
    )
    themes = {
        field: _normalize_themes(
            getattr(draft, field),
            prefix,
            available_ids,
        )
        for field, prefix in theme_specs
    }

    return CustomerReputationOutput(
        status=PublicSignalStatus.COMPLETED,
        company_name=agent_input.company_name,
        sample_size=len(assessments),
        sentiment_summary=SentimentSummary(
            classification=_overall_sentiment(counts),
            analyzed_signal_count=len(assessments),
            positive_count=counts[SignalSentiment.POSITIVE],
            neutral_count=counts[SignalSentiment.NEUTRAL],
            negative_count=counts[SignalSentiment.NEGATIVE],
            mixed_count=counts[SignalSentiment.MIXED],
            unclear_count=counts[SignalSentiment.UNCLEAR],
            confidence=draft.overall_confidence,
        ),
        signal_assessments=assessments,
        missing_information=missing_information,
        overall_confidence=draft.overall_confidence,
        **themes,
    )


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
        self.provider = provider or GroqProvider(
            model="openai/gpt-oss-120b",
        )

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
            response_model=CustomerReputationDraft,
            max_tokens=4000,
            temperature=0,
        )

        final_output = finalize_customer_reputation_output(
            response.data,
            agent_input,
        )
        validate_public_signal_references(final_output, agent_input)

        return StructuredLLMResult(
            data=final_output,
            usage=response.usage,
        )
