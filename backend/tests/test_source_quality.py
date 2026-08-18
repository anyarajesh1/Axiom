from app.retrieval.source_quality import (
    is_acceptable_source,
    source_priority,
)


def test_social_and_lyric_platforms_are_rejected() -> None:
    assert is_acceptable_source("https://www.facebook.com/example") is False
    assert is_acceptable_source("https://genius.com/example-lyrics") is False
    assert is_acceptable_source("https://science.nasa.gov/example") is True


def test_government_and_educational_sources_are_prioritized() -> None:
    assert source_priority("https://www.usgs.gov/example") == 3
    assert source_priority("https://example.edu/research") == 3
    assert source_priority("https://example.com/blog") == 0
