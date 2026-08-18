from uuid import uuid4

from app.graph.nodes.contradiction_detector import detect_contradictions
from app.graph.nodes.evidence_ranker import rank_evidence
from app.schemas import EntailmentResult, Evidence


def make_evidence(reranker_score: float) -> Evidence:
    return Evidence(
        id=uuid4(),
        text="A passage long enough for Day 5 decision-node testing.",
        source_name="Test source",
        source_url="https://example.com/source",
        category="science",
        score=0.9,
        reranker_score=reranker_score,
    )


def make_result(
    evidence: Evidence,
    label: str,
    confidence: float,
) -> EntailmentResult:
    scores = {
        "contradiction": 0.05,
        "entailment": 0.05,
        "neutral": 0.05,
    }
    scores[label] = confidence
    return EntailmentResult(
        evidence_id=evidence.id,
        label=label,
        confidence=confidence,
        scores=scores,
    )


def test_contradiction_detector_flags_mixed_relevant_evidence() -> None:
    support = make_evidence(0.9)
    contradiction = make_evidence(0.8)
    irrelevant = make_evidence(0.001)

    summary = detect_contradictions(
        uuid4(),
        [support, contradiction, irrelevant],
        [
            make_result(support, "entailment", 0.9),
            make_result(contradiction, "contradiction", 0.8),
            make_result(irrelevant, "contradiction", 0.99),
        ],
    )

    assert summary.support_count == 1
    assert summary.contradiction_count == 1
    assert summary.neutral_count == 1
    assert summary.has_conflict is True


def test_ranker_combines_relevance_and_entailment_confidence() -> None:
    relevant = make_evidence(0.8)
    weak = make_evidence(0.01)

    ranked = rank_evidence(
        [weak, relevant],
        [
            make_result(weak, "contradiction", 0.99),
            make_result(relevant, "entailment", 0.75),
        ],
    )

    assert [item.id for item in ranked] == [relevant.id]
    assert ranked[0].combined_score == 0.6
