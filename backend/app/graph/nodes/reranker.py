from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import settings
from app.retrieval.hybrid import meaningful_tokens
from app.schemas import Evidence

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

ScorePairs = Callable[[list[tuple[str, str]]], Sequence[float]]


class RerankerError(RuntimeError):
    """Raised when evidence reranking fails."""


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.RERANKER_MODEL)


def predict_relevance(pairs: list[tuple[str, str]]) -> Sequence[float]:
    return get_reranker().predict(
        pairs,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def sigmoid(value: float) -> float:
    bounded = max(-30.0, min(30.0, value))
    return 1 / (1 + math.exp(-bounded))


def lightweight_rerank(
    claim: str,
    evidence: list[Evidence],
    top_k: int,
) -> list[Evidence]:
    claim_tokens = meaningful_tokens(claim)
    reranked = []
    for item in evidence:
        evidence_tokens = meaningful_tokens(item.text)
        overlap = (
            len(claim_tokens & evidence_tokens) / len(claim_tokens)
            if claim_tokens
            else 0
        )
        relevance = min(1.0, max(item.score * 0.8, overlap))
        reranked.append(
            item.model_copy(update={"reranker_score": round(relevance, 6)})
        )
    return sorted(
        reranked,
        key=lambda item: item.reranker_score or 0,
        reverse=True,
    )[:top_k]


def rerank_evidence(
    claim: str,
    evidence: list[Evidence],
    top_k: int = 5,
    predict: ScorePairs | None = None,
) -> list[Evidence]:
    if not evidence:
        return []

    if predict is None and settings.LOW_MEMORY_MODE:
        return lightweight_rerank(claim, evidence, top_k)

    try:
        scores = list(
            (predict or predict_relevance)(
                [(claim, item.text) for item in evidence]
            )
        )
    except Exception as error:
        raise RerankerError("Axiom could not rerank the evidence.") from error

    if len(scores) != len(evidence):
        raise RerankerError("The reranker returned an unexpected result count.")

    reranked = [
        item.model_copy(
            update={"reranker_score": round(sigmoid(float(score)), 6)}
        )
        for item, score in zip(evidence, scores, strict=True)
    ]
    return sorted(
        reranked,
        key=lambda item: item.reranker_score or 0,
        reverse=True,
    )[:top_k]
