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
