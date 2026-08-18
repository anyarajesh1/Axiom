from dataclasses import dataclass

from app.config import settings
from app.retrieval.corpus import CorpusPassage
from app.retrieval.hybrid import retrieve_local
from app.retrieval.store import passage_id, upsert_passages
from app.retrieval.tavily import ExternalSearchError, search_tavily
from app.schemas import Evidence


@dataclass(frozen=True)
class RetrievalOutcome:
    evidence: list[Evidence]
    external_search_used: bool


def external_evidence(
    query: str,
    passages: list[CorpusPassage],
    limit: int,
) -> list[Evidence]:
    query_tokens = set(query.lower().split())
    evidence: list[Evidence] = []
    for rank, passage in enumerate(passages[:limit], start=1):
        passage_tokens = set(passage.text.lower().split())
        overlap = (
            len(query_tokens & passage_tokens) / len(query_tokens)
            if query_tokens
            else 0
        )
        evidence.append(
            Evidence(
                id=passage_id(str(passage.source_url), passage.text),
                text=passage.text,
                source_name=passage.source_name,
                source_url=str(passage.source_url),
                category=passage.category,
                score=round(max(overlap, 1 / rank), 6),
            )
        )
    return evidence


def merge_evidence(
    local: list[Evidence],
    external: list[Evidence],
    limit: int,
) -> list[Evidence]:
    """Keep trusted corpus candidates while adding web-search coverage."""
    if limit <= 0:
        return []

    local_quota = min(len(local), max(1, (limit + 1) // 2))
    candidates = [*local[:local_quota], *external]
    candidates.extend(local[local_quota:])

    merged: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    for item in candidates:
        key = (str(item.source_url), " ".join(item.text.lower().split()))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) == limit:
            break
    return merged


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

    if settings.LOW_MEMORY_MODE:
        web_evidence = external_evidence(query, external_passages, limit)
        return RetrievalOutcome(
            merge_evidence(local.evidence, web_evidence, limit),
            external_search_used=True,
        )

    upsert_passages(external_passages)
    expanded = retrieve_local(query, limit)
    return RetrievalOutcome(expanded.evidence, external_search_used=True)
