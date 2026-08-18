from app.schemas import EntailmentResult, Evidence

MIN_COMBINED_SCORE = 0.05


def rank_evidence(
    evidence: list[Evidence],
    entailments: list[EntailmentResult],
    limit: int = 4,
) -> list[Evidence]:
    entailment_by_id = {result.evidence_id: result for result in entailments}
    ranked: list[Evidence] = []

    for item in evidence:
        result = entailment_by_id.get(item.id)
        if (
            result is None
            or result.label == "neutral"
            or item.reranker_score is None
        ):
            continue
        combined_score = item.reranker_score * result.confidence
        if combined_score < MIN_COMBINED_SCORE:
            continue
        ranked.append(
            item.model_copy(
                update={"combined_score": round(combined_score, 6)}
            )
        )

    return sorted(
        ranked,
        key=lambda item: item.combined_score or 0,
        reverse=True,
    )[:limit]
