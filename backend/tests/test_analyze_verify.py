from uuid import uuid4

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import app.routers.analyze as analyze_module
from app.main import app
from app.schemas import EntailmentResult, Evidence, VerifyClaimResponse

client = TestClient(app)


def test_analyze_verify_returns_ml_scores(monkeypatch: MonkeyPatch) -> None:
    evidence = Evidence(
        id=uuid4(),
        text="An EV battery can retain useful capacity beyond five years.",
        source_name="US Department of Energy",
        source_url="https://example.com/battery",
        category="technology",
        score=1.0,
        reranker_score=0.96,
    )
    result = EntailmentResult(
        evidence_id=evidence.id,
        label="contradiction",
        confidence=0.91,
        scores={
            "contradiction": 0.91,
            "entailment": 0.03,
            "neutral": 0.06,
        },
    )
    monkeypatch.setattr(
        analyze_module,
        "verify_claim",
        lambda *_: VerifyClaimResponse(
            claim="Every EV battery must be replaced after five years.",
            evidence=[evidence],
            entailments=[result],
            external_search_used=False,
        ),
    )

    response = client.post(
        "/analyze/verify",
        json={
            "claim": "Every EV battery must be replaced after five years."
        },
    )

    assert response.status_code == 200
    assert response.json()["evidence"][0]["reranker_score"] == 0.96
    assert response.json()["entailments"][0]["label"] == "contradiction"
