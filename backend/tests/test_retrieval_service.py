from uuid import uuid4

from pytest import MonkeyPatch

from app.retrieval import service
from app.retrieval.corpus import CorpusPassage
from app.retrieval.hybrid import LocalRetrieval
from app.schemas import Evidence


def make_evidence() -> Evidence:
    return Evidence(
        id=uuid4(),
        text="A supporting passage long enough for this retrieval test.",
        source_name="Test source",
        source_url="https://example.com/source",
        category="science",
        score=1.0,
    )


def test_strong_local_result_skips_external_search(
    monkeypatch: MonkeyPatch,
) -> None:
    evidence = make_evidence()
    monkeypatch.setattr(
        service,
        "retrieve_local",
        lambda *_: LocalRetrieval([evidence], relevance=0.9),
    )
    external_search = lambda *_args, **_kwargs: []
    monkeypatch.setattr(service, "search_tavily", external_search)

    outcome = service.retrieve_evidence("a strong local claim")

    assert outcome.evidence == [evidence]
    assert outcome.external_search_used is False


def test_weak_result_expands_index_and_retries(
    monkeypatch: MonkeyPatch,
) -> None:
    evidence = make_evidence()
    calls = iter(
        [
            LocalRetrieval([], relevance=0.0),
            LocalRetrieval([evidence], relevance=0.9),
        ]
    )
    passage = CorpusPassage(
        text="An externally retrieved passage that can expand the index.",
        source_name="External source",
        source_url="https://example.com/external",
        category="web_search",
    )
    monkeypatch.setattr(service, "retrieve_local", lambda *_: next(calls))
    monkeypatch.setattr(service, "search_tavily", lambda *_args, **_kwargs: [passage])
    upsert = lambda passages: len(passages)
    monkeypatch.setattr(service, "upsert_passages", upsert)

    outcome = service.retrieve_evidence("a new claim")

    assert outcome.evidence == [evidence]
    assert outcome.external_search_used is True


def test_low_memory_fallback_returns_web_results_without_upsert(
    monkeypatch: MonkeyPatch,
) -> None:
    passage = CorpusPassage(
        text="Earthquake magnitude is measured on a logarithmic scale.",
        source_name="USGS",
        source_url="https://www.usgs.gov/programs/earthquake-hazards",
        category="science",
    )
    monkeypatch.setattr(service.settings, "LOW_MEMORY_MODE", True)
    monkeypatch.setattr(
        service,
        "retrieve_local",
        lambda *_: LocalRetrieval([], relevance=0),
    )
    monkeypatch.setattr(
        service,
        "search_tavily",
        lambda *_args, **_kwargs: [passage],
    )

    def fail_upsert(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("Web results must not be embedded in low-memory mode")

    monkeypatch.setattr(service, "upsert_passages", fail_upsert)

    outcome = service.retrieve_evidence("earthquake magnitude logarithmic")

    assert outcome.external_search_used is True
    assert outcome.evidence[0].source_name == "USGS"
