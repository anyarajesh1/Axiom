from uuid import uuid4

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import app.routers.analyze as analyze_module
from app.main import app
from app.retrieval.service import RetrievalOutcome
from app.retrieval.store import VectorStoreError
from app.schemas import Evidence

client = TestClient(app)


def test_analyze_evidence_returns_ranked_sources(
    monkeypatch: MonkeyPatch,
) -> None:
    evidence = Evidence(
        id=uuid4(),
        text="Earthquake magnitude is measured on a logarithmic scale.",
        source_name="USGS",
        source_url="https://www.usgs.gov/programs/earthquake-hazards",
        category="science",
        score=1.0,
    )
    monkeypatch.setattr(
        analyze_module,
        "retrieve_evidence",
        lambda *_: RetrievalOutcome([evidence], external_search_used=False),
    )

    response = client.post(
        "/analyze/evidence",
        json={"claim": "Earthquake magnitude is logarithmic."},
    )

    assert response.status_code == 200
    assert response.json()["evidence"][0]["source_name"] == "USGS"
    assert response.json()["external_search_used"] is False


def test_analyze_evidence_handles_vector_store_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    def fail(*_args: object) -> RetrievalOutcome:
        raise VectorStoreError("Axiom could not search its passage index.")

    monkeypatch.setattr(analyze_module, "retrieve_evidence", fail)

    response = client.post(
        "/analyze/evidence",
        json={"claim": "A valid claim for retrieval."},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Axiom could not search its passage index."
    }
