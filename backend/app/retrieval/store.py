from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient, models

from app.config import settings
from app.retrieval.corpus import CorpusPassage
from app.retrieval.source_quality import is_acceptable_payload

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

VECTOR_SIZE = 384


class VectorStoreError(RuntimeError):
    """Raised when Axiom cannot use its vector store."""


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    vectors = get_embedder().encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=20,
    )


def passage_id(source_url: str, text: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"{source_url}\n{text}")


def passage_payload(passage: CorpusPassage) -> dict[str, str]:
    return {
        "text": passage.text,
        "source_name": passage.source_name,
        "source_url": str(passage.source_url),
        "category": passage.category,
    }


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(settings.QDRANT_COLLECTION):
        return

    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )


def upsert_passages(
    passages: Sequence[CorpusPassage],
    client: QdrantClient | None = None,
) -> int:
    if not passages:
        return 0

    qdrant = client or get_qdrant_client()
    try:
        ensure_collection(qdrant)
        vectors = embed_texts([passage.text for passage in passages])
        points = [
            models.PointStruct(
                id=passage_id(str(passage.source_url), passage.text),
                vector=vector,
                payload=passage_payload(passage),
            )
            for passage, vector in zip(passages, vectors, strict=True)
        ]
        qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
            wait=True,
        )
    except Exception as error:
        raise VectorStoreError("Axiom could not update its passage index.") from error

    return len(passages)


def dense_search(
    query: str,
    limit: int = 10,
    client: QdrantClient | None = None,
) -> list[tuple[UUID, dict[str, Any], float]]:
    qdrant = client or get_qdrant_client()
    try:
        response = qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=embed_texts([query])[0],
            limit=max(limit * 3, 30),
            with_payload=True,
        )
    except Exception as error:
        raise VectorStoreError("Axiom could not search its passage index.") from error

    results = [
        (UUID(str(point.id)), point.payload or {}, float(point.score))
        for point in response.points
        if is_acceptable_payload(point.payload or {})
    ]
    return results[:limit]


def scroll_passages(
    client: QdrantClient | None = None,
) -> list[tuple[UUID, dict[str, Any]]]:
    qdrant = client or get_qdrant_client()
    records: list[tuple[UUID, dict[str, Any]]] = []
    offset: Any = None

    try:
        while True:
            page, offset = qdrant.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(
                (UUID(str(record.id)), record.payload or {})
                for record in page
                if is_acceptable_payload(record.payload or {})
            )
            if offset is None:
                break
    except Exception as error:
        raise VectorStoreError("Axiom could not read its passage index.") from error

    return records
