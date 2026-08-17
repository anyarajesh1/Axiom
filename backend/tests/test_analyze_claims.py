from collections.abc import Generator

from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

import app.routers.analyze as analyze_module
from app.db.models import Claim as ClaimRow
from app.db.models import Submission
from app.db.session import create_db_and_tables, get_session
from app.graph.nodes.extractor import ClaimExtractionError
from app.main import app
from app.schemas import Claim


def make_test_engine():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with test_engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    return test_engine


def test_analyze_claims_persists_submission_and_claims(
    monkeypatch: MonkeyPatch,
) -> None:
    test_engine = make_test_engine()
    create_db_and_tables(test_engine)

    def override_session() -> Generator[Session, None, None]:
        with Session(test_engine) as session:
            yield session

    monkeypatch.setattr(
        analyze_module,
        "extract_claims",
        lambda _: [
            Claim(
                text="EV battery packs often last longer than five years.",
                source_span="battery packs often last longer than five years",
            )
        ],
    )
    app.dependency_overrides[get_session] = override_session

    try:
        response = TestClient(app).post(
            "/analyze/claims",
            json={
                "text": "EV battery packs often last longer than five years."
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["claims"][0]["text"] == (
        "EV battery packs often last longer than five years."
    )

    with Session(test_engine) as session:
        assert len(session.exec(select(Submission)).all()) == 1
        assert len(session.exec(select(ClaimRow)).all()) == 1


def test_analyze_claims_returns_bad_gateway_for_extractor_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    def fail_extraction(_: str) -> list[Claim]:
        raise ClaimExtractionError("Groq returned an invalid claim extraction.")

    monkeypatch.setattr(analyze_module, "extract_claims", fail_extraction)

    response = TestClient(app).post(
        "/analyze/claims",
        json={"text": "The Moon orbits Earth."},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Groq returned an invalid claim extraction."
    }
