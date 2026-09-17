import asyncio
import re
from collections.abc import Iterable
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from app.schemas.customer_reputation import (
    PublicSignalDocument,
    SignalSourceType,
)
from app.schemas.public_signal_evidence import (
    FailedPublicSignalSource,
    PublicSignalCollectionInput,
    PublicSignalCollectionOutput,
    PublicSignalCollectionStatus,
    PublicSignalCollectorUsage,
)
from app.schemas.web_research import (
    SearchResult,
    TavilyExtractRequest,
    TavilyExtractResponse,
    TavilySearchRequest,
    TavilySearchResponse,
)
from app.search.tavily_provider import TavilyProvider


MAX_PUBLIC_SIGNAL_SEARCHES = 3
MAX_PUBLIC_SIGNALS = 8
MAX_SIGNAL_CHARACTERS = 4_000


class PublicSignalResearchProvider(Protocol):
    async def search(
        self,
        request: TavilySearchRequest,
    ) -> TavilySearchResponse: ...

    async def extract(
        self,
        request: TavilyExtractRequest,
    ) -> TavilyExtractResponse: ...


def _answer_for(
    answers: dict[str, str],
    keys: Iterable[str],
) -> str | None:
    normalized = {
        key.strip().casefold(): value.strip()
        for key, value in answers.items()
        if value.strip()
    }
    for key in keys:
        if normalized.get(key.casefold()):
            return normalized[key.casefold()]
    return None


def _compact_query(*parts: str) -> str:
    return " ".join(" ".join(parts).split())[:300].rstrip()


def build_public_signal_queries(
    request: PublicSignalCollectionInput,
) -> list[tuple[str, str]]:
    location = _answer_for(
        request.follow_up_answers,
        (
            "geographic_market",
            "target_location",
            "target_market",
            "location",
            "region",
            "country",
            "market",
        ),
    ) or "global"

    purpose_queries = {
        "general_research": (
            ("customer reviews complaints and praise", "general"),
            ("brand reputation and customer satisfaction", "general"),
            ("reputation controversy service quality", "news"),
        ),
        "start_similar_business": (
            ("customer reviews complaints and pain points", "general"),
            ("customer unmet needs product service feedback", "general"),
            ("brand reputation customer experience issues", "news"),
        ),
        "partner_or_supplier": (
            ("customer service reliability reviews", "general"),
            ("supplier partner reputation and complaints", "general"),
            ("relationship service quality reputation risk", "news"),
        ),
        "stock_research": (
            ("customer sentiment and brand perception", "general"),
            ("customer complaints product service issues", "general"),
            ("reputation risk controversy regulatory news", "news"),
        ),
    }[request.purpose.value]

    return [
        (
            _compact_query(request.company_name, focus, location),
            topic,
        )
        for focus, topic in purpose_queries
    ][:MAX_PUBLIC_SIGNAL_SEARCHES]


def _canonical_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            parsed.path.rstrip("/") or "/",
            parsed.query,
            "",
        )
    )


def _company_terms(company_name: str) -> set[str]:
    ignored = {"the", "company", "corporation", "inc", "limited"}
    return {
        term
        for term in re.findall(r"[a-z0-9]+", company_name.casefold())
        if len(term) > 2 and term not in ignored
    }


def select_public_signal_results(
    responses: list[TavilySearchResponse],
    company_name: str,
) -> list[SearchResult]:
    unique: dict[str, SearchResult] = {}
    for response in responses:
        for result in response.results:
            key = _canonical_url(str(result.url))
            existing = unique.get(key)
            if existing is None or (
                result.relevance_score > existing.relevance_score
            ):
                unique[key] = result

    terms = _company_terms(company_name)

    def score(result: SearchResult) -> float:
        searchable = f"{result.title} {result.content}".casefold()
        company_bonus = 0.2 if any(
            term in searchable for term in terms
        ) else 0
        return result.relevance_score + company_bonus

    selected: list[SearchResult] = []
    selected_urls: set[str] = set()

    # Keep at least one result from each intent before filling by score.
    for response in responses:
        for candidate in sorted(
            response.results,
            key=score,
            reverse=True,
        ):
            key = _canonical_url(str(candidate.url))
            if key not in selected_urls:
                selected.append(unique[key])
                selected_urls.add(key)
                break

    for candidate in sorted(unique.values(), key=score, reverse=True):
        if len(selected) >= MAX_PUBLIC_SIGNALS:
            break
        key = _canonical_url(str(candidate.url))
        if key not in selected_urls:
            selected.append(candidate)
            selected_urls.add(key)

    return selected[:MAX_PUBLIC_SIGNALS]


def classify_signal_source(
    url: str,
    title: str,
) -> SignalSourceType:
    hostname = (urlsplit(url).hostname or "").removeprefix("www.")
    searchable = f"{hostname} {title}".casefold()

    if any(marker in searchable for marker in ("glassdoor", "indeed")):
        return SignalSourceType.EMPLOYEE_REVIEW
    if any(
        marker in searchable
        for marker in (
            "trustpilot",
            "consumercomplaints",
            "mouthshut",
            "customer review",
            "product review",
        )
    ):
        return SignalSourceType.CUSTOMER_REVIEW
    if any(marker in searchable for marker in ("reddit", "quora", "forum")):
        return SignalSourceType.FORUM
    if any(
        marker in searchable
        for marker in ("x.com", "twitter", "facebook", "instagram")
    ):
        return SignalSourceType.SOCIAL_MEDIA
    if any(
        marker in searchable
        for marker in (
            "reuters",
            "bloomberg",
            "business-standard",
            "economictimes",
            "timesofindia",
            "news",
        )
    ):
        return SignalSourceType.NEWS
    if "survey" in searchable:
        return SignalSourceType.SURVEY
    return SignalSourceType.OTHER


def redact_contact_details(content: str) -> str:
    content = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[redacted email]",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(
        r"(?<!\w)@[A-Za-z0-9_]{2,}",
        "[redacted handle]",
        content,
    )
    def redact_phone(match: re.Match[str]) -> str:
        value = match.group(0)
        digit_count = sum(character.isdigit() for character in value)
        return (
            "[redacted phone]"
            if 10 <= digit_count <= 15
            else value
        )

    return re.sub(
        r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)",
        redact_phone,
        content,
    )


class PublicSignalCollector:
    def __init__(
        self,
        provider: PublicSignalResearchProvider | None = None,
    ) -> None:
        self.provider = provider or TavilyProvider()

    async def collect(
        self,
        request: PublicSignalCollectionInput,
    ) -> PublicSignalCollectionOutput:
        query_specs = build_public_signal_queries(request)
        search_responses = await asyncio.gather(
            *(
                self.provider.search(
                    TavilySearchRequest(
                        query=query,
                        topic=topic,
                        search_depth="basic",
                        max_results=5,
                        time_range="year" if topic == "news" else None,
                    )
                )
                for query, topic in query_specs
            )
        )
        selected_results = select_public_signal_results(
            search_responses,
            request.company_name,
        )
        credits = sum(
            response.usage.credits_used for response in search_responses
        )
        response_time = sum(
            response.usage.response_time_seconds
            for response in search_responses
        )
        searched_results = sum(
            len(response.results) for response in search_responses
        )

        if not selected_results:
            return self._build_output(
                request=request,
                query_specs=query_specs,
                search_requests=len(search_responses),
                searched_results=searched_results,
                selected_urls=0,
                extract_response=None,
                documents=[],
                credits_used=credits,
                response_time_seconds=response_time,
                warnings=[
                    "No relevant public signals were found. Agent 4 was not run."
                ],
            )

        extraction = await self.provider.extract(
            TavilyExtractRequest(
                urls=[result.url for result in selected_results],
                query=_compact_query(
                    request.company_name,
                    "customer feedback complaints praise reputation risk",
                ),
                extract_depth="basic",
                chunks_per_source=2,
            )
        )
        credits += extraction.usage.credits_used
        response_time += extraction.usage.response_time_seconds

        metadata = {
            _canonical_url(str(result.url)): result
            for result in selected_results
        }
        documents: list[PublicSignalDocument] = []
        for extracted in extraction.documents:
            content = redact_contact_details(extracted.content)
            content = content[:MAX_SIGNAL_CHARACTERS].strip()
            if not content:
                continue
            result = metadata.get(_canonical_url(str(extracted.url)))
            title = result.title if result else "Public signal source"
            source_type = classify_signal_source(
                str(extracted.url),
                title,
            )
            documents.append(
                PublicSignalDocument(
                    signal_id=(
                        f"signal_{source_type.value}_{len(documents) + 1}"
                    ),
                    title=title,
                    url=extracted.url,
                    source_type=source_type,
                    content=content,
                    rating=None,
                )
            )

        warnings: list[str] = []
        if extraction.failed_documents:
            warnings.append(
                f"{len(extraction.failed_documents)} selected source(s) "
                "could not be extracted."
            )
        if not documents:
            warnings.append(
                "No selected source returned usable public-signal content."
            )

        return self._build_output(
            request=request,
            query_specs=query_specs,
            search_requests=len(search_responses),
            searched_results=searched_results,
            selected_urls=len(selected_results),
            extract_response=extraction,
            documents=documents,
            credits_used=credits,
            response_time_seconds=response_time,
            warnings=warnings,
        )

    @staticmethod
    def _build_output(
        *,
        request: PublicSignalCollectionInput,
        query_specs: list[tuple[str, str]],
        search_requests: int,
        searched_results: int,
        selected_urls: int,
        extract_response: TavilyExtractResponse | None,
        documents: list[PublicSignalDocument],
        credits_used: float,
        response_time_seconds: float,
        warnings: list[str],
    ) -> PublicSignalCollectionOutput:
        failed_sources = (
            [
                FailedPublicSignalSource(url=item.url, error=item.error)
                for item in extract_response.failed_documents
            ]
            if extract_response
            else []
        )
        return PublicSignalCollectionOutput(
            status=(
                PublicSignalCollectionStatus.COMPLETED
                if documents
                else PublicSignalCollectionStatus.INSUFFICIENT_SOURCES
            ),
            company_name=request.company_name,
            search_queries=[query for query, _ in query_specs],
            signal_documents=documents,
            failed_sources=failed_sources,
            usage=PublicSignalCollectorUsage(
                search_requests=search_requests,
                extraction_requests=1 if extract_response else 0,
                searched_results=searched_results,
                selected_urls=selected_urls,
                extracted_documents=len(documents),
                credits_used=credits_used,
                response_time_seconds=response_time_seconds,
            ),
            warnings=warnings,
        )
