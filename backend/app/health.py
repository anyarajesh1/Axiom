import asyncio
from collections.abc import Callable

from groq import Groq
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

from app.config import settings


def check_groq() -> None:
    with Groq(
        api_key=settings.GROQ_API_KEY,
        timeout=10.0,
        max_retries=0,
    ) as client:
        client.models.list()


def check_qdrant() -> None:
    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=10,
        check_compatibility=False,
    )
    try:
        client.get_collections()
    finally:
        client.close()


def check_neon() -> None:
    database_url = settings.DATABASE_URL
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )

    engine = create_engine(
        database_url,
        connect_args={"connect_timeout": 10},
    )
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    finally:
        engine.dispose()


def run_check(check: Callable[[], None]) -> str:
    try:
        check()
    except Exception:   # noqa: BLE001
        return "red"
    return "green"


async def dependency_health() -> dict[str, str]:
    names = ("groq", "qdrant", "neon")
    checks = (check_groq, check_qdrant, check_neon)
    results = await asyncio.gather(
        *(asyncio.to_thread(run_check, check) for check in checks)
    )
    return dict(zip(names, results))
