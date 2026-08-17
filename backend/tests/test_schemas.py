from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas import Claim, Evidence, PipelineState, Verdict


def test_claim_creates_an_id_and_preserves_source_span() -> None:
    claim = Claim(
        text="EV battery packs commonly last longer than five years.",
        source_span="battery packs commonly last longer than five years",
    )

    assert claim.id
    assert claim.source_span is not None


def test_verdict_rejects_confidence_outside_zero_to_one() -> None:
    with pytest.raises(ValidationError):
        Verdict(
            claim_id=uuid4(),
            label="supported",
            confidence=1.1,
            explanation="The cited source directly supports the claim.",
        )


def test_pipeline_state_groups_evidence_by_claim() -> None:
    claim = Claim(text="Lithium-ion batteries lose capacity over time.")
    evidence = Evidence(
        text="Lithium-ion cells gradually lose usable capacity as they age.",
        source_url="https://example.com/battery-aging",
        score=0.92,
    )
    state = PipelineState(
        input_text=claim.text,
        claims=[claim],
        evidence_by_claim={claim.id: [evidence]},
    )

    assert state.evidence_by_claim[claim.id][0].score == 0.92
