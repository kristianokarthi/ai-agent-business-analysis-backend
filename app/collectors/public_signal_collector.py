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
MIN_SIGNAL_CONTENT_CHARACTERS = 120
MAX_RESULTS_PER_DOMAIN = 2

SIGNAL_TERMS = {
    "boycott",
    "complaint",
    "complaints",
    "consumer",
    "controversy",
    "customer",
    "discussion",
    "experience",
    "feedback",
    "lawsuit",
    "perception",
    "product quality",
    "rating",
    "recall",
    "regulatory",
    "reputation",
    "review",
    "reviews",
    "satisfaction",
    "sentiment",
    "service quality",
    "survey",
}

LOW_QUALITY_HOSTS = {
    "bartleby.com",
    "brainly.in",
    "coursehero.com",
    "ivypanda.com",
    "scribd.com",
    "studocu.com",
    "ukessays.com",
}

TRUSTED_NEWS_HOSTS = {
    "bloomberg.com",
    "business-standard.com",
    "cnbc.com",
    "economictimes.indiatimes.com",
    "financialexpress.com",
    "hindustantimes.com",
    "indianexpress.com",
    "livemint.com",
    "ndtv.com",
    "reuters.com",
    "thehindu.com",
    "timesofindia.indiatimes.com",
}

REVIEW_OR_FORUM_HOSTS = {
    "consumercomplaints.in",
    "glassdoor.co.in",
    "glassdoor.com",
    "indeed.com",
    "mouthshut.com",
    "quora.com",
    "reddit.com",
    "trustpilot.com",
}


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
            ("customer reviews complaints product quality", "general"),
            ("consumer survey brand reputation satisfaction", "general"),
            ("controversy recall lawsuit regulatory action", "news"),
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


def _hostname(url: str) -> str:
    return (urlsplit(url).hostname or "").casefold().removeprefix("www.")


def _host_matches(hostname: str, domains: set[str]) -> bool:
    return any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in domains
    )


def _company_match_ratio(text: str, company_name: str) -> float:
    terms = _company_terms(company_name)
    if not terms:
        return 0
    normalized = text.casefold()
    matches = sum(term in normalized for term in terms)
    return matches / len(terms)


def _contains_signal_language(text: str) -> bool:
    normalized = text.casefold()
    return any(term in normalized for term in SIGNAL_TERMS)


def _is_weak_page(url: str, title: str) -> bool:
    parsed = urlsplit(url)
    hostname = _hostname(url)
    path_and_title = f"{parsed.path} {title}".casefold()
    if _host_matches(hostname, LOW_QUALITY_HOSTS):
        return True
    return any(
        marker in path_and_title
        for marker in (
            "contact-us",
            "contact us",
            "/contact/",
            "/faq",
            "frequently asked questions",
            "free essay",
            "report example",
        )
    )


def _is_relevant_result(
    result: SearchResult,
    company_name: str,
) -> bool:
    url = str(result.url)
    if _is_weak_page(url, result.title):
        return False
    searchable = f"{result.title} {url} {result.content}"
    return (
        _company_match_ratio(searchable, company_name) >= 0.6
        and _contains_signal_language(searchable)
    )


def _source_quality_bonus(result: SearchResult) -> float:
    hostname = _hostname(str(result.url))
    if _host_matches(hostname, TRUSTED_NEWS_HOSTS):
        return 0.3
    if _host_matches(hostname, REVIEW_OR_FORUM_HOSTS):
        return 0.25
    if hostname.endswith(".gov.in") or hostname.endswith(".gov"):
        return 0.3
    return 0


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

    def score(result: SearchResult) -> float:
        searchable = f"{result.title} {result.content}".casefold()
        company_bonus = 0.2 * _company_match_ratio(
            searchable,
            company_name,
        )
        return (
            result.relevance_score
            + company_bonus
            + _source_quality_bonus(result)
        )

    selected: list[SearchResult] = []
    selected_urls: set[str] = set()
    domain_counts: dict[str, int] = {}

    for candidate in sorted(unique.values(), key=score, reverse=True):
        if len(selected) >= MAX_PUBLIC_SIGNALS:
            break
        key = _canonical_url(str(candidate.url))
        hostname = _hostname(str(candidate.url))
        if (
            key in selected_urls
            or domain_counts.get(hostname, 0) >= MAX_RESULTS_PER_DOMAIN
            or not _is_relevant_result(candidate, company_name)
        ):
            continue
        selected.append(candidate)
        selected_urls.add(key)
        domain_counts[hostname] = domain_counts.get(hostname, 0) + 1

    return selected[:MAX_PUBLIC_SIGNALS]


def classify_signal_source(
    url: str,
    title: str,
    content: str = "",
) -> SignalSourceType:
    hostname = _hostname(url)
    searchable = f"{hostname} {title} {content[:500]}".casefold()

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
            "customer complaint",
        )
    ):
        return SignalSourceType.CUSTOMER_REVIEW
    if any(marker in searchable for marker in ("reddit", "quora", "forum")):
        return SignalSourceType.FORUM
    if any(
        marker in searchable
        for marker in (
            "x.com",
            "twitter",
            "facebook",
            "instagram",
            "linkedin.com",
        )
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
            "livemint",
            "financialexpress",
            "hindustantimes",
            "indianexpress",
            "thehindu",
            "ndtv",
            "cnbc",
            "news",
        )
    ):
        return SignalSourceType.NEWS
    if any(
        marker in searchable
        for marker in ("survey", "consumer research", "market research")
    ):
        return SignalSourceType.SURVEY
    return SignalSourceType.OTHER


def is_usable_signal_content(
    *,
    company_name: str,
    title: str,
    url: str,
    content: str,
) -> bool:
    compact_content = " ".join(content.split())
    if len(compact_content) < MIN_SIGNAL_CONTENT_CHARACTERS:
        return False
    if _is_weak_page(url, title):
        return False
    searchable = f"{title} {url} {compact_content}"
    return (
        _company_match_ratio(searchable, company_name) >= 0.6
        and _contains_signal_language(searchable)
    )


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
        discarded_documents = 0
        for extracted in extraction.documents:
            content = redact_contact_details(extracted.content)
            content = content[:MAX_SIGNAL_CHARACTERS].strip()
            result = metadata.get(_canonical_url(str(extracted.url)))
            title = result.title if result else "Public signal source"
            if not is_usable_signal_content(
                company_name=request.company_name,
                title=title,
                url=str(extracted.url),
                content=content,
            ):
                discarded_documents += 1
                continue
            source_type = classify_signal_source(
                str(extracted.url),
                title,
                content,
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
        if discarded_documents:
            warnings.append(
                f"{discarded_documents} extracted source(s) were discarded "
                "because they were weak, irrelevant, or too short."
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
