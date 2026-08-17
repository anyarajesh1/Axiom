from uuid import uuid4

from pytest import MonkeyPatch

from app.graph import verification
from app.retrieval.service import RetrievalOutcome
from app.schemas import EntailmentResult, Evidence


def make_evidence() -> Evidence:
    return Evidence(
        id=uuid4(),
        text="An EV battery can retain useful capacity beyond five years.",
        source_name="Test source",
        source_url="https://example.com/battery",
        category="technology",
        score=1.0,
    )


def test_verification_graph_runs_nodes_in_order(
    monkeypatch: MonkeyPatch,
) -> None:
    evidence = make_evidence()
    calls: list[str] = []

    def retrieve(*_args: object, **_kwargs: object) -> RetrievalOutcome:
        calls.append("retriever")
        return RetrievalOutcome([evidence], external_search_used=False)

    def rerank(*_args: object, **_kwargs: object) -> list[Evidence]:
        calls.append("reranker")
        return [evidence.model_copy(update={"reranker_score": 0.95})]

    def entail(*_args: object, **_kwargs: object) -> list[EntailmentResult]:
        calls.append("entailment")
        return [
            EntailmentResult(
                evidence_id=evidence.id,
                label="contradiction",
                confidence=0.9,
                scores={
                    "contradiction": 0.9,
                    "entailment": 0.05,
                    "neutral": 0.05,
                },
            )
        ]

    monkeypatch.setattr(verification, "retrieve_evidence", retrieve)
    monkeypatch.setattr(verification, "rerank_evidence", rerank)
    monkeypatch.setattr(verification, "score_entailment", entail)

    response = verification.verify_claim(
        "Every EV battery must be replaced after five years."
    )

    assert calls == ["retriever", "reranker", "entailment"]
    assert response.evidence[0].reranker_score == 0.95
    assert response.entailments[0].label == "contradiction"
