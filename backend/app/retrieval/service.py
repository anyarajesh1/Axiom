from dataclasses import dataclass

from app.config import settings
from app.retrieval.hybrid import retrieve_local
from app.retrieval.store import upsert_passages
from app.retrieval.tavily import ExternalSearchError, search_tavily
from app.schemas import Evidence


@dataclass(frozen=True)
class RetrievalOutcome:
    evidence: list[Evidence]
    external_search_used: bool


def retrieve_evidence(query: str, limit: int = 5) -> RetrievalOutcome:
    local = retrieve_local(query, limit)
    if local.evidence and local.relevance >= settings.RETRIEVAL_FALLBACK_THRESHOLD:
        return RetrievalOutcome(local.evidence, external_search_used=False)

    try:
        external_passages = search_tavily(query, max_results=max(limit, 5))
    except ExternalSearchError:
        if local.evidence:
            return RetrievalOutcome(local.evidence, external_search_used=False)
        raise

    if not external_passages:
        return RetrievalOutcome(local.evidence, external_search_used=False)

    upsert_passages(external_passages)
    expanded = retrieve_local(query, limit)
    return RetrievalOutcome(expanded.evidence, external_search_used=True)
