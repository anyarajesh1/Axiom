import json
from uuid import uuid4

import pytest

from app.graph.nodes.referee import RefereeError, decide_verdict
from app.schemas import (
    Claim,
    ContradictionSummary,
    EntailmentResult,
    Evidence,
)


def make_inputs():
    claim = Claim(text="Every EV battery must be replaced after five years.")
    evidence = Evidence(
        id=uuid4(),
        text="EV batteries can retain useful capacity beyond five years.",
        source_name="Test source",
        source_url="https://example.com/battery",
        category="technology",
        score=1.0,
        reranker_score=0.8,
        combined_score=0.72,
    )
    entailment = EntailmentResult(
        evidence_id=evidence.id,
        label="contradiction",
        confidence=0.9,
        scores={
            "contradiction": 0.9,
            "entailment": 0.02,
            "neutral": 0.08,
        },
    )
    summary = ContradictionSummary(
        claim_id=claim.id,
        support_count=0,
        contradiction_count=1,
        neutral_count=0,
        has_conflict=False,
    )
    return claim, evidence, entailment, summary


def test_referee_returns_structured_verdict() -> None:
    claim, evidence, entailment, summary = make_inputs()
    response = json.dumps(
        {
            "label": "contradicted",
            "confidence": 0.9,
            "evidence_ids": [str(evidence.id)],
            "explanation": "The cited battery source contradicts the five-year requirement.",
        }
    )

    verdict = decide_verdict(
        claim,
        [evidence],
        [entailment],
        summary,
        request=lambda _: response,
    )

    assert verdict.claim_id == claim.id
    assert verdict.label == "contradicted"
    assert verdict.evidence_ids == [evidence.id]


def test_referee_rejects_invented_evidence_id() -> None:
    claim, evidence, entailment, summary = make_inputs()
    response = json.dumps(
        {
            "label": "contradicted",
            "confidence": 0.9,
            "evidence_ids": [str(uuid4())],
            "explanation": "The available source contradicts the claim.",
        }
    )

    with pytest.raises(RefereeError, match="unknown evidence ID"):
        decide_verdict(
            claim,
            [evidence],
            [entailment],
            summary,
            request=lambda _: response,
        )


def test_referee_skips_groq_when_no_evidence() -> None:
    claim, _, _, summary = make_inputs()

    verdict = decide_verdict(
        claim,
        [],
        [],
        summary,
        request=lambda _: pytest.fail("Groq should not be called"),
    )

    assert verdict.label == "insufficient_evidence"
    assert verdict.confidence == 0
