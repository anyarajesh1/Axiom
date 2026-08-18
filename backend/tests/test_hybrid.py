from uuid import uuid4

from pytest import MonkeyPatch

from app.retrieval import hybrid
from app.retrieval.corpus import CorpusPassage


def payload(text: str, source: str) -> dict[str, str]:
    return {
        "text": text,
        "source_name": source,
        "source_url": f"https://example.com/{source.lower()}",
        "category": "science",
    }


def test_lexical_rank_prefers_matching_passage() -> None:
    earthquake_id = uuid4()
    moon_id = uuid4()
    records = [
        (
            earthquake_id,
            payload(
                "Earthquake magnitude uses a logarithmic scale.",
                "Earthquake",
            ),
        ),
        (moon_id, payload("The Moon orbits Earth.", "Moon")),
    ]

    ranked = hybrid.lexical_rank("earthquake magnitude scale", records)

    assert ranked[0][0] == earthquake_id


def test_hybrid_retrieval_merges_dense_and_lexical_results(
    monkeypatch: MonkeyPatch,
) -> None:
    dense_id = uuid4()
    lexical_id = uuid4()
    records = [
        (
            dense_id,
            payload("Carbon dioxide traps heat in the atmosphere.", "NASA"),
        ),
        (
            lexical_id,
            payload("Greenhouse gases contribute to warming.", "NOAA"),
        ),
    ]
    monkeypatch.setattr(
        hybrid,
        "dense_search",
        lambda *_args, **_kwargs: [(dense_id, records[0][1], 0.8)],
    )
    monkeypatch.setattr(hybrid, "scroll_passages", lambda: records)

    result = hybrid.retrieve_local("greenhouse gases warming", limit=2)

    assert {item.id for item in result.evidence} == {dense_id, lexical_id}
    assert result.evidence[0].score == 1.0
    assert result.relevance == 1.0


def test_common_words_do_not_make_an_unrelated_query_relevant(
    monkeypatch: MonkeyPatch,
) -> None:
    moon_id = uuid4()
    records = [
        (moon_id, payload("The Moon is a natural satellite.", "Moon")),
    ]
    monkeypatch.setattr(
        hybrid,
        "dense_search",
        lambda *_args, **_kwargs: [(moon_id, records[0][1], 0.2)],
    )
    monkeypatch.setattr(hybrid, "scroll_passages", lambda: records)

    result = hybrid.retrieve_local("The sky is blue.")

    assert result.relevance == 0.2


def test_low_memory_retrieval_skips_dense_embedding(
    monkeypatch: MonkeyPatch,
) -> None:
    earthquake_id = uuid4()
    records = [
        (
            earthquake_id,
            payload(
                "Earthquake magnitude uses a logarithmic scale.",
                "USGS",
            ),
        )
    ]

    def fail_dense(*_args: object, **_kwargs: object) -> list:
        raise AssertionError("Dense embedding must not run in low-memory mode")

    monkeypatch.setattr(hybrid.settings, "LOW_MEMORY_MODE", True)
    monkeypatch.setattr(hybrid, "dense_search", fail_dense)
    monkeypatch.setattr(hybrid, "scroll_passages", lambda: records)
    monkeypatch.setattr(hybrid, "load_corpus", list)

    result = hybrid.retrieve_local("earthquake magnitude logarithmic scale")

    assert result.evidence[0].id == earthquake_id
    assert result.relevance == 1.0


def test_low_memory_retrieval_includes_bundled_ev_evidence(
    monkeypatch: MonkeyPatch,
) -> None:
    passage = CorpusPassage(
        text=(
            "The expected lifetime of an electric vehicle battery is 12 to "
            "15 years in moderate climates and 8 to 12 years in extreme "
            "climates."
        ),
        source_name="US Department of Energy",
        source_url="https://afdc.energy.gov/fuels/electricity-benefits",
        category="climate_energy",
    )

    def fail_dense(*_args: object, **_kwargs: object) -> list:
        raise AssertionError("Dense embedding must not run in low-memory mode")

    monkeypatch.setattr(hybrid.settings, "LOW_MEMORY_MODE", True)
    monkeypatch.setattr(hybrid, "dense_search", fail_dense)
    monkeypatch.setattr(hybrid, "scroll_passages", list)
    monkeypatch.setattr(hybrid, "load_corpus", lambda: [passage])

    result = hybrid.retrieve_local(
        "Every electric vehicle battery must be replaced after five years."
    )

    assert result.evidence[0].source_name == "US Department of Energy"
    assert result.relevance >= 0.5
