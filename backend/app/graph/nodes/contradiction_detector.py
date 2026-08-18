from uuid import UUID

from app.schemas import ContradictionSummary, EntailmentResult, Evidence

MIN_NLI_CONFIDENCE = 0.6
MIN_RERANKER_RELEVANCE = 0.05


def detect_contradictions(
    claim_id: UUID,
    evidence: list[Evidence],
    entailments: list[EntailmentResult],
) -> ContradictionSummary:
    evidence_by_id = {item.id: item for item in evidence}
    counts = {"entailment": 0, "contradiction": 0, "neutral": 0}

    for result in entailments:
        item = evidence_by_id.get(result.evidence_id)
        relevance = item.reranker_score if item else None
        if (
            item is None
            or relevance is None
            or relevance < MIN_RERANKER_RELEVANCE
            or result.confidence < MIN_NLI_CONFIDENCE
        ):
            counts["neutral"] += 1
            continue
        counts[result.label] += 1

    return ContradictionSummary(
        claim_id=claim_id,
        support_count=counts["entailment"],
        contradiction_count=counts["contradiction"],
        neutral_count=counts["neutral"],
        has_conflict=(
            counts["entailment"] > 0 and counts["contradiction"] > 0
        ),
    )
