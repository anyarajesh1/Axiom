import httpx

from app.config import settings
from app.retrieval.corpus import CorpusPassage

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class ExternalSearchError(RuntimeError):
    """Raised when external evidence search fails."""


def search_tavily(query: str, max_results: int = 5) -> list[CorpusPassage]:
    try:
        response = httpx.post(
            TAVILY_SEARCH_URL,
            json={
                "api_key": settings.TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
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
        if len(content) < 20 or not source_url:
            continue
        passages.append(
            CorpusPassage(
                text=content,
                source_name=str(result.get("title") or "Web source"),
                source_url=source_url,
                category="web_search",
            )
        )
    return passages
