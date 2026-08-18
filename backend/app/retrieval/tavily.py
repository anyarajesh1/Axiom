import httpx

from app.config import settings
from app.retrieval.corpus import CorpusPassage
from app.retrieval.source_quality import (
    is_acceptable_source,
    source_priority,
)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class ExternalSearchError(RuntimeError):
    """Raised when external evidence search fails."""


def search_tavily(query: str, max_results: int = 5) -> list[CorpusPassage]:
    try:
        response = httpx.post(
            TAVILY_SEARCH_URL,
            json={
                "api_key": settings.TAVILY_API_KEY,
                "query": f"{query} authoritative source explanation",
                "search_depth": "basic",
                "max_results": max(max_results * 2, 8),
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=20,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except (httpx.HTTPError, ValueError) as error:
        raise ExternalSearchError(
            "Axiom could not complete its external evidence search."
        ) from error

    passages: list[CorpusPassage] = []
    for result in results:
        content = " ".join(str(result.get("content", "")).split())
        source_url = result.get("url")
        if (
            len(content) < 20
            or not source_url
            or not is_acceptable_source(str(source_url))
        ):
            continue
        passages.append(
            CorpusPassage(
                text=content,
                source_name=str(result.get("title") or "Web source"),
                source_url=source_url,
                category="web_search",
            )
        )
    return sorted(
        passages,
        key=lambda passage: source_priority(str(passage.source_url)),
        reverse=True,
    )[:max_results]
