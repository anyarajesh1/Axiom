import json
from pathlib import Path

import pytest

from app.retrieval.corpus import DEFAULT_CORPUS_PATH, load_corpus


def test_seed_corpus_is_valid_and_varied() -> None:
    passages = load_corpus()

    assert len(passages) >= 24
    assert len({passage.text for passage in passages}) == len(passages)
    assert {passage.category for passage in passages} >= {
        "ai",
        "climate_energy",
        "science",
        "technology",
    }
    assert DEFAULT_CORPUS_PATH.name == "seed_passages.jsonl"


def test_corpus_rejects_duplicate_passages(tmp_path: Path) -> None:
    record = {
        "text": "This passage is long enough to satisfy corpus validation.",
        "source_name": "Axiom Test",
        "source_url": "https://example.com/source",
        "category": "science",
    }
    corpus_path = tmp_path / "duplicate.jsonl"
    corpus_path.write_text(
        f"{json.dumps(record)}\n{json.dumps(record)}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate corpus passage on line 2"):
        load_corpus(corpus_path)
