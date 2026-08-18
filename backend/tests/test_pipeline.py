from uuid import uuid4

from pytest import MonkeyPatch

from app.graph import pipeline
from app.retrieval.service import RetrievalOutcome
from app.schemas import Claim, EntailmentResult, Evidence, Verdict


def make_evidence() -> Evidence:
    return Evidence(
        id=uuid4(),
        text="EV batteries can retain useful capacity beyond five years.",
        source_name="Test source",
        source_url="https://example.com/battery",
        category="technology",
        score=1.0,
    )


def test_full_pipeline_runs_all_seven_nodes(
    monkeypatch: MonkeyPatch,
) -> None:
    claim = Claim(text="Every EV battery must be replaced after five years.")
    evidence = make_evidence()
    calls: list[str] = []

    def extract(_: str) -> list[Claim]:
        calls.append("extractor")
        return [claim]

    def retrieve(*_args: object, **_kwargs: object) -> RetrievalOutcome:
        calls.append("retriever")
        return RetrievalOutcome([evidence], external_search_used=False)

    def rerank(*_args: object, **_kwargs: object) -> list[Evidence]:
        calls.append("reranker")
        return [evidence.model_copy(update={"reranker_score": 0.8})]

    def entail(*_args: object, **_kwargs: object) -> list[EntailmentResult]:
        calls.append("entailment")
        return [
            EntailmentResult(
                evidence_id=evidence.id,
                label="contradiction",
                confidence=0.9,
                scores={
                    "contradiction": 0.9,
                    "entailment": 0.02,
                    "neutral": 0.08,
                },
            )
        ]

    def referee(*args: object, **_kwargs: object) -> Verdict:
        calls.append("referee")
        ranked = args[1]
        assert isinstance(ranked, list)
        return Verdict(
            claim_id=claim.id,
            label="contradicted",
            confidence=0.9,
            evidence_ids=[evidence.id],
            explanation="The evidence contradicts the five-year requirement.",
        )

    monkeypatch.setattr(pipeline, "extract_claims", extract)
    monkeypatch.setattr(pipeline, "retrieve_evidence", retrieve)
    monkeypatch.setattr(pipeline, "rerank_evidence", rerank)
    monkeypatch.setattr(pipeline, "score_entailment", entail)
    monkeypatch.setattr(pipeline, "decide_verdict", referee)

    result = pipeline.run_pipeline(
        "Every EV battery must be replaced after five years."
    )

    assert calls == [
        "extractor",
        "retriever",
        "reranker",
        "entailment",
        "referee",
    ]
    assert result.contradictions_by_claim[claim.id].contradiction_count == 1
    assert result.evidence_by_claim[claim.id][0].combined_score == 0.72
    assert result.verdicts[0].label == "contradicted"


def test_pipeline_degrades_to_insufficient_when_retrieval_fails(
    monkeypatch: MonkeyPatch,
) -> None:
    claim = Claim(text="A claim with unavailable evidence.")
    monkeypatch.setattr(pipeline, "extract_claims", lambda _: [claim])

    def fail(*_args: object, **_kwargs: object) -> RetrievalOutcome:
        from app.retrieval.store import VectorStoreError

        raise VectorStoreError("Unavailable")

    monkeypatch.setattr(pipeline, "retrieve_evidence", fail)

    result = pipeline.run_pipeline(claim.text)

    assert result.evidence_by_claim[claim.id] == []
    assert result.verdicts[0].label == "insufficient_evidence"
