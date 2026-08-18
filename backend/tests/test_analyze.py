from collections.abc import Generator
from uuid import uuid4

from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

import app.routers.analyze as analyze_module
from app.db.models import Claim as ClaimRow
from app.db.models import Submission
from app.db.models import Verdict as VerdictRow
from app.db.session import create_db_and_tables, get_session
from app.main import app
from app.schemas import (
    Claim,
    ContradictionSummary,
    EntailmentResult,
    Evidence,
    PipelineState,
    Verdict,
)


def make_test_engine():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with test_engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    return test_engine


def test_analyze_persists_and_returns_complete_pipeline(
    monkeypatch: MonkeyPatch,
) -> None:
    test_engine = make_test_engine()
    create_db_and_tables(test_engine)

    def override_session() -> Generator[Session, None, None]:
        with Session(test_engine) as session:
            yield session

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
    verdict = Verdict(
        claim_id=claim.id,
        label="contradicted",
        confidence=0.9,
        evidence_ids=[evidence.id],
        explanation="The source contradicts the five-year requirement.",
    )
    pipeline_state = PipelineState(
        input_text=claim.text,
        claims=[claim],
        evidence_by_claim={claim.id: [evidence]},
        entailment_by_claim={claim.id: [entailment]},
        contradictions_by_claim={claim.id: summary},
        verdicts=[verdict],
    )
    monkeypatch.setattr(analyze_module, "run_pipeline", lambda _: pipeline_state)
    app.dependency_overrides[get_session] = override_session

    try:
        response = TestClient(app).post(
            "/analyze",
            json={"text": claim.text},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["verdicts"][0]["label"] == "contradicted"
    assert response.json()["evidence_by_claim"][str(claim.id)][0][
        "combined_score"
    ] == 0.72

    with Session(test_engine) as session:
        assert len(session.exec(select(Submission)).all()) == 1
        assert len(session.exec(select(ClaimRow)).all()) == 1
        stored_verdict = session.exec(select(VerdictRow)).one()
        assert stored_verdict.evidence_ids == [str(evidence.id)]


def test_analyze_returns_bad_gateway_for_extraction_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    from app.graph.nodes.extractor import ClaimExtractionError

    def fail(_: str) -> PipelineState:
        raise ClaimExtractionError("The claim extraction service is unavailable.")

    monkeypatch.setattr(analyze_module, "run_pipeline", fail)

    response = TestClient(app).post(
        "/analyze",
        json={"text": "A valid factual claim."},
    )

    assert response.status_code == 502
