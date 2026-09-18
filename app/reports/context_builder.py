import json
import math
from collections import defaultdict
from typing import Any, Iterable

from app.schemas.business_fundamentals import ConfidenceLevel
from app.schemas.report_context import (
    EvidenceCatalogEntry,
    ReportContext,
    ReportContextRequest,
    ReportSourceReference,
    ReportSourceType,
)


CONFIDENCE_SCORE = {
    ConfidenceLevel.HIGH: 3,
    ConfidenceLevel.MEDIUM: 2,
    ConfidenceLevel.LOW: 1,
}

PURPOSE_BUSINESS_PRIORITY = {
    "general_research": (
        "business_model",
        "products_services",
        "customer_segments",
        "value_proposition",
        "revenue_sources",
        "sales_distribution",
    ),
    "start_similar_business": (
        "business_model",
        "customer_segments",
        "value_proposition",
        "revenue_sources",
        "cost_drivers",
        "resources_capabilities",
        "sales_distribution",
    ),
    "partner_or_supplier": (
        "sales_distribution",
        "resources_capabilities",
        "operational_strengths",
        "operational_weaknesses",
        "business_model",
        "customer_segments",
    ),
    "stock_research": (
        "financial_highlights",
        "revenue_sources",
        "business_model",
        "operational_strengths",
        "operational_weaknesses",
        "cost_drivers",
        "resources_capabilities",
    ),
}

PURPOSE_FACT_PRIORITY = {
    "general_research": (
        "company_identity",
        "products_services",
        "customers_markets",
        "operations",
        "ownership",
    ),
    "start_similar_business": (
        "products_services",
        "customers_markets",
        "operations",
        "partnerships",
        "financial",
    ),
    "partner_or_supplier": (
        "operations",
        "partnerships",
        "products_services",
        "customers_markets",
        "ownership",
    ),
    "stock_research": (
        "financial",
        "ownership",
        "operations",
        "customers_markets",
        "products_services",
    ),
}


def _compact_text(value: str, max_characters: int = 700) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_characters:
        return compact
    return compact[: max_characters - 1].rstrip() + "…"


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = " ".join(value.split())
        key = compact.casefold()
        if compact and key not in seen:
            result.append(compact)
            seen.add(key)
    return result


def _ranked(items: list[Any], key, limit: int) -> list[Any]:
    indexed = list(enumerate(items))
    indexed.sort(key=lambda item: (-key(item[1]), item[0]))
    return [item for _, item in indexed[:limit]]


def _priority_score(value: str, priorities: tuple[str, ...]) -> int:
    try:
        return len(priorities) - priorities.index(value)
    except ValueError:
        return 0


def _estimate_tokens(payload: dict[str, Any]) -> int:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return math.ceil(len(serialized) / 4)


def _terminal_evidence_resolver(
    source_ids: set[str],
    parent_ids: dict[str, list[str]],
):
    def resolve(evidence_ids: Iterable[str]) -> list[str]:
        resolved: list[str] = []
        resolving: set[str] = set()

        def visit(evidence_id: str) -> None:
            if evidence_id in source_ids:
                if evidence_id not in resolved:
                    resolved.append(evidence_id)
                return
            if evidence_id in resolving:
                return
            parents = parent_ids.get(evidence_id)
            if not parents:
                if evidence_id not in resolved:
                    resolved.append(evidence_id)
                return
            resolving.add(evidence_id)
            for parent_id in parents:
                visit(parent_id)
            resolving.remove(evidence_id)

        for evidence_id in evidence_ids:
            visit(evidence_id)
        return resolved

    return resolve


def _collect_evidence_usage(
    value: Any,
    path: str = "context",
    usage: dict[str, set[str]] | None = None,
) -> dict[str, set[str]]:
    if usage is None:
        usage = defaultdict(set)
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in {"evidence_ids", "identity_evidence_ids"}:
                for evidence_id in child:
                    usage[evidence_id].add(path)
            else:
                _collect_evidence_usage(child, child_path, usage)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_evidence_usage(child, f"{path}[{index}]", usage)
    return usage


def _source_catalog(
    payload: dict[str, Any],
    sources: dict[str, ReportSourceReference],
) -> list[dict[str, Any]]:
    usage = _collect_evidence_usage(payload)
    catalog: list[dict[str, Any]] = []
    for evidence_id, used_by in usage.items():
        source = sources.get(evidence_id)
        if source is None:
            source = ReportSourceReference(
                evidence_id=evidence_id,
                title=f"Unresolved source {evidence_id}",
                url=None,
                source_type=ReportSourceType.UNKNOWN,
            )
        catalog.append(
            EvidenceCatalogEntry(
                **source.model_dump(),
                used_by=sorted(used_by),
            ).model_dump(mode="json")
        )
    return catalog


def _finding(
    *,
    finding_id: str,
    statement: str,
    category: str,
    evidence_ids: Iterable[str],
    confidence: ConfidenceLevel,
    resolve,
) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "statement": _compact_text(statement),
        "category": category,
        "evidence_ids": resolve(evidence_ids),
        "confidence": confidence.value,
    }


def _theme(theme, resolve) -> dict[str, Any]:
    return {
        "statement": _compact_text(theme.statement),
        "evidence_ids": resolve(theme.signal_ids),
        "confidence": theme.confidence.value,
    }


def _trim_order(purpose: str) -> list[tuple[str, ...]]:
    common = [
        ("reputation", "conflicting_signals"),
        ("reputation", "praise"),
        ("company_claims",),
        ("market", "market_trends"),
    ]
    if purpose == "stock_research":
        return common + [
            ("market", "market_gaps"),
            ("reputation", "pain_points"),
            ("reputation", "unmet_needs"),
            ("market", "entry_barriers"),
            ("business_findings",),
            ("verified_facts",),
        ]
    if purpose == "start_similar_business":
        return common + [
            ("reputation", "reputation_risks"),
            ("market", "competitive_risks"),
            ("business_findings",),
            ("verified_facts",),
        ]
    return common + [
        ("market", "market_gaps"),
        ("reputation", "pain_points"),
        ("reputation", "unmet_needs"),
        ("business_findings",),
        ("verified_facts",),
    ]


def _list_at(payload: dict[str, Any], path: tuple[str, ...]) -> list[Any]:
    value: Any = payload
    for key in path:
        value = value[key]
    return value


def build_report_context(request: ReportContextRequest) -> ReportContext:
    fact_output = request.company_evidence
    business_output = request.business_fundamentals
    market_output = request.market_analysis
    reputation_output = request.customer_reputation
    purpose = request.purpose.value

    source_list = [
        *request.company_sources,
        *request.market_sources,
        *request.public_signal_sources,
    ]
    sources = {source.evidence_id: source for source in source_list}

    official_url = fact_output.company_identity.official_website
    for source_id in fact_output.sources_used:
        sources.setdefault(
            source_id,
            ReportSourceReference(
                evidence_id=source_id,
                title="Official company source",
                url=official_url,
                source_type=ReportSourceType.COMPANY_OFFICIAL,
            ),
        )

    parent_ids: dict[str, list[str]] = {}
    for fact in fact_output.verified_facts:
        parent_ids[fact.fact_id] = fact.source_ids
    for claim in fact_output.company_claims:
        parent_ids[claim.claim_id] = claim.source_ids
    for finding in business_output.findings:
        parent_ids[finding.finding_id] = finding.evidence_ids
    for competitor in market_output.competitors:
        parent_ids[competitor.competitor_id] = competitor.evidence_ids
    for finding in (
        *market_output.entry_barriers,
        *market_output.market_trends,
        *market_output.market_gaps,
        *market_output.competitive_risks,
    ):
        parent_ids[finding.finding_id] = finding.evidence_ids

    resolve = _terminal_evidence_resolver(set(sources), parent_ids)
    fact_priorities = PURPOSE_FACT_PRIORITY[purpose]
    business_priorities = PURPOSE_BUSINESS_PRIORITY[purpose]

    facts = _ranked(
        fact_output.verified_facts,
        lambda item: _priority_score(item.category.value, fact_priorities),
        6,
    )
    business_findings = _ranked(
        business_output.findings,
        lambda item: (
            _priority_score(item.area.value, business_priorities) * 10
            + CONFIDENCE_SCORE[item.confidence]
        ),
        8,
    )
    competitors = _ranked(
        market_output.competitors,
        lambda item: CONFIDENCE_SCORE[item.confidence],
        3,
    )

    def market_findings(items) -> list[dict[str, Any]]:
        selected = _ranked(
            items,
            lambda item: CONFIDENCE_SCORE[item.confidence],
            3,
        )
        return [
            _finding(
                finding_id=item.finding_id,
                statement=item.statement,
                category="market",
                evidence_ids=item.evidence_ids,
                confidence=item.confidence,
                resolve=resolve,
            )
            for item in selected
        ]

    def themes(items) -> list[dict[str, Any]]:
        selected = _ranked(
            items,
            lambda item: (
                CONFIDENCE_SCORE[item.confidence] * 10
                + item.signal_count
            ),
            3,
        )
        return [_theme(item, resolve) for item in selected]

    summary = reputation_output.sentiment_summary
    payload: dict[str, Any] = {
        "company_name": fact_output.company_identity.name,
        "purpose": purpose,
        "follow_up_answers": request.follow_up_answers,
        "company_identity": fact_output.company_identity.model_dump(mode="json"),
        "identity_evidence_ids": resolve(fact_output.sources_used),
        "verified_facts": [
            _finding(
                finding_id=item.fact_id,
                statement=item.statement,
                category=item.category.value,
                evidence_ids=item.source_ids,
                confidence=ConfidenceLevel.HIGH,
                resolve=resolve,
            )
            for item in facts
        ],
        "company_claims": [
            _finding(
                finding_id=item.claim_id,
                statement=item.statement,
                category=item.category.value,
                evidence_ids=item.source_ids,
                confidence=ConfidenceLevel.LOW,
                resolve=resolve,
            )
            for item in fact_output.company_claims[:3]
        ],
        "conflicting_claims": [
            {
                "topic": item.topic,
                "statements": [_compact_text(value) for value in item.statements],
                "evidence_ids": resolve(item.source_ids),
            }
            for item in fact_output.conflicting_claims[:3]
        ],
        "business_findings": [
            _finding(
                finding_id=item.finding_id,
                statement=item.statement,
                category=item.area.value,
                evidence_ids=item.evidence_ids,
                confidence=item.confidence,
                resolve=resolve,
            )
            for item in business_findings
        ],
        "market": {
            "industry": market_output.market_definition.industry,
            "geographic_market": market_output.market_definition.geographic_market,
            "customer_groups": market_output.market_definition.customer_groups,
            "market_structure": market_output.market_structure.classification.value,
            "structure_explanation": _compact_text(
                market_output.market_structure.explanation
            ),
            "evidence_ids": resolve(
                [
                    *market_output.market_definition.evidence_ids,
                    *market_output.market_structure.evidence_ids,
                ]
            ),
            "competitors": [
                {
                    "name": item.name,
                    "competitor_type": item.competitor_type.value,
                    "positioning": _compact_text(item.positioning),
                    "strengths": item.strengths,
                    "weaknesses": item.weaknesses,
                    "evidence_ids": resolve(item.evidence_ids),
                    "confidence": item.confidence.value,
                }
                for item in competitors
            ],
            "entry_barriers": market_findings(market_output.entry_barriers),
            "market_trends": market_findings(market_output.market_trends),
            "market_gaps": market_findings(market_output.market_gaps),
            "competitive_risks": market_findings(
                market_output.competitive_risks
            ),
            "confidence": market_output.overall_confidence.value,
        },
        "reputation": {
            "sample_size": reputation_output.sample_size,
            "sentiment": summary.classification.value,
            "positive_count": summary.positive_count,
            "neutral_count": summary.neutral_count,
            "negative_count": summary.negative_count,
            "mixed_count": summary.mixed_count,
            "unclear_count": summary.unclear_count,
            "evidence_ids": resolve(
                assessment.signal_id
                for assessment in reputation_output.signal_assessments
            ),
            "praise": themes(reputation_output.praise_themes),
            "complaints": themes(reputation_output.complaint_themes),
            "pain_points": themes(reputation_output.customer_pain_points),
            "unmet_needs": themes(reputation_output.unmet_needs),
            "reputation_risks": themes(reputation_output.reputation_risks),
            "conflicting_signals": themes(
                reputation_output.conflicting_signals
            ),
            "confidence": reputation_output.overall_confidence.value,
        },
        "missing_information": _deduplicate(
            [
                *fact_output.missing_information,
                *business_output.missing_information,
                *market_output.missing_information,
                *reputation_output.missing_information,
                *request.collection_warnings,
                (
                    "Customer and reputation findings are based on a public "
                    f"sample of {reputation_output.sample_size} signals with "
                    f"{reputation_output.overall_confidence.value} confidence."
                ),
            ]
        ),
    }

    truncated = False
    while True:
        payload["evidence_catalog"] = _source_catalog(payload, sources)
        candidate = {
            **payload,
            "estimated_tokens": 0,
            "token_budget": request.max_context_tokens,
            "truncated_for_budget": truncated,
        }
        estimated = _estimate_tokens(candidate)
        if estimated <= request.max_context_tokens:
            candidate["estimated_tokens"] = estimated
            return ReportContext.model_validate(candidate)

        removed = False
        for path in _trim_order(purpose):
            values = _list_at(payload, path)
            minimum = 4 if path in {("business_findings",), ("verified_facts",)} else 1
            if len(values) > minimum:
                values.pop()
                truncated = True
                removed = True
                break
        if not removed:
            payload["missing_information"] = _deduplicate(
                [
                    *payload["missing_information"],
                    "Mandatory evidence and warning fields exceeded the requested "
                    "context budget and were preserved.",
                ]
            )
            payload["evidence_catalog"] = _source_catalog(payload, sources)
            candidate = {
                **payload,
                "estimated_tokens": 0,
                "token_budget": request.max_context_tokens,
                "truncated_for_budget": True,
            }
            candidate["estimated_tokens"] = _estimate_tokens(candidate)
            return ReportContext.model_validate(candidate)
