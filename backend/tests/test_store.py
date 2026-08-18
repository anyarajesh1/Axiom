from unittest.mock import Mock

from pytest import MonkeyPatch
from qdrant_client import models

from app.retrieval import store
from app.retrieval.corpus import CorpusPassage


def make_passage() -> CorpusPassage:
    return CorpusPassage(
        text="Earthquake magnitude is measured on a logarithmic scale.",
        source_name="USGS",
        source_url="https://www.usgs.gov/programs/earthquake-hazards",
        category="science",
    )


def test_upsert_passages_creates_collection_and_uses_stable_id(
    monkeypatch: MonkeyPatch,
) -> None:
    client = Mock()
    client.collection_exists.return_value = False
    monkeypatch.setattr(store, "embed_texts", lambda _: [[0.0] * 384])
    passage = make_passage()

    count = store.upsert_passages([passage], client)

    assert count == 1
    client.create_collection.assert_called_once()
    points = client.upsert.call_args.kwargs["points"]
    assert points[0].id == store.passage_id(
        str(passage.source_url), passage.text
    )
    assert points[0].payload["source_name"] == "USGS"


def test_dense_search_returns_payload_and_score(
    monkeypatch: MonkeyPatch,
) -> None:
    client = Mock()
    point_id = store.passage_id("https://example.com", "passage")
    client.query_points.return_value = Mock(
        points=[
            models.ScoredPoint(
                id=point_id,
                version=1,
                score=0.82,
                payload={
                    "text": "A useful passage.",
                    "source_url": "https://example.com/source",
                },
            )
        ]
    )
    monkeypatch.setattr(store, "embed_texts", lambda _: [[0.0] * 384])

    results = store.dense_search("query", client=client)

    assert results == [
        (
            point_id,
            {
                "text": "A useful passage.",
                "source_url": "https://example.com/source",
            },
            0.82,
        )
    ]


def test_dense_search_filters_social_media_sources(
    monkeypatch: MonkeyPatch,
) -> None:
    client = Mock()
    client.query_points.return_value = Mock(
        points=[
            models.ScoredPoint(
                id=store.passage_id("https://facebook.com/post", "lyrics"),
                version=1,
                score=0.99,
                payload={
                    "text": "A song lyric that happens to match the claim.",
                    "source_url": "https://facebook.com/post",
                },
            )
        ]
    )
    monkeypatch.setattr(store, "embed_texts", lambda _: [[0.0] * 384])

    assert store.dense_search("Trees are green", client=client) == []
