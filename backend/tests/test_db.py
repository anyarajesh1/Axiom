from sqlalchemy import inspect
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

from app.db.models import Claim, Submission
from app.db.session import create_db_and_tables, normalize_database_url


def make_test_engine():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with test_engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    return test_engine


def test_normalize_database_url_uses_psycopg_driver() -> None:
    original = "postgresql://user:password@example.com/axiom"

    assert normalize_database_url(original) == (
        "postgresql+psycopg://user:password@example.com/axiom"
    )


def test_create_db_and_tables_creates_day_two_tables() -> None:
    test_engine = make_test_engine()

    create_db_and_tables(test_engine)

    assert set(inspect(test_engine).get_table_names()) == {
        "claim",
        "submission",
        "verdict",
    }


def test_submission_and_claim_can_be_persisted() -> None:
    test_engine = make_test_engine()
    create_db_and_tables(test_engine)
    submission = Submission(input_text="Battery packs degrade over time.")
    submission_id = submission.id
    claim = Claim(
        submission_id=submission_id,
        text="Battery packs degrade over time.",
        source_span="Battery packs degrade over time",
    )

    with Session(test_engine) as session:
        session.add(submission)
        session.flush()
        session.add(claim)
        session.commit()
        stored_claim = session.exec(select(Claim)).one()

    assert stored_claim.submission_id == submission_id
