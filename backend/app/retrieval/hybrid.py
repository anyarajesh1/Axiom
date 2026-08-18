import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.retrieval.store import dense_search, scroll_passages
from app.schemas import Evidence

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
RRF_K = 60
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


@dataclass(frozen=True)
class LocalRetrieval:
    evidence: list[Evidence]
    relevance: float


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def meaningful_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOP_WORDS}


def lexical_rank(
    query: str,
    records: list[tuple[UUID, dict[str, Any]]],
) -> list[tuple[UUID, float]]:
    query_tokens = tokenize(query)
    if not query_tokens or not records:
        return []

    documents = [tokenize(str(payload.get("text", ""))) for _, payload in records]
    document_frequency = Counter(
        token for document in documents for token in set(document)
    )
    average_length = sum(map(len, documents)) / len(documents) or 1
    query_counts = Counter(query_tokens)
    scores: list[tuple[UUID, float]] = []

    for (point_id, _), document in zip(records, documents, strict=True):
        frequencies = Counter(document)
        score = 0.0
        for token, query_count in query_counts.items():
            frequency = frequencies[token]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1 + (len(documents) - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + 1.5 * (
                0.25 + 0.75 * len(document) / average_length
            )
            score += (
                inverse_frequency
                * (frequency * 2.5 / denominator)
                * query_count
            )
        if score > 0:
            scores.append((point_id, score))

    return sorted(scores, key=lambda item: item[1], reverse=True)


def retrieve_local(query: str, limit: int = 5) -> LocalRetrieval:
    dense = dense_search(query, limit=max(limit * 2, 10))
    records = scroll_passages()
    payloads = {point_id: payload for point_id, payload in records}
    payloads.update({point_id: payload for point_id, payload, _ in dense})
    lexical = lexical_rank(query, records)

    fused: dict[UUID, float] = {}
    for rank, (point_id, _, _) in enumerate(dense, start=1):
        fused[point_id] = fused.get(point_id, 0) + 1 / (RRF_K + rank)
    for rank, (point_id, _) in enumerate(lexical, start=1):
        fused[point_id] = fused.get(point_id, 0) + 1 / (RRF_K + rank)

    ranked_ids = sorted(fused, key=fused.get, reverse=True)[:limit]
    max_fused = max(fused.values(), default=1)
    evidence = [
        Evidence(
            id=point_id,
            text=str(payloads[point_id]["text"]),
            source_name=str(payloads[point_id]["source_name"]),
            source_url=str(payloads[point_id]["source_url"]),
            category=str(payloads[point_id]["category"]),
            score=round(fused[point_id] / max_fused, 6),
        )
        for point_id in ranked_ids
    ]

    best_dense = max((score for _, _, score in dense), default=0.0)
    query_terms = meaningful_tokens(query)
    best_overlap = (
        max(
            (
                len(
                    query_terms
                    & meaningful_tokens(str(payload.get("text", "")))
                )
                / len(query_terms)
                for _, payload in records
            ),
            default=0.0,
        )
        if query_terms
        else 0.0
    )
    relevance = min(1.0, max(0.0, best_dense, best_overlap))
    return LocalRetrieval(evidence=evidence, relevance=relevance)
