from typing import Any
from urllib.parse import urlparse

BLOCKED_DOMAINS = {
    "facebook.com",
    "genius.com",
    "instagram.com",
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "soundcloud.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "youtube.com",
}
TRUSTED_DOMAINS = {
    "apnews.com",
    "bbc.com",
    "britannica.com",
    "nature.com",
    "nationalgeographic.com",
    "reuters.com",
    "science.org",
}


def source_hostname(source_url: str) -> str:
    hostname = (urlparse(source_url).hostname or "").lower()
    return hostname.removeprefix("www.")


def domain_matches(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith(f".{domain}")


def is_acceptable_source(source_url: str) -> bool:
    hostname = source_hostname(source_url)
    return bool(hostname) and not any(
        domain_matches(hostname, domain) for domain in BLOCKED_DOMAINS
    )


def is_acceptable_payload(payload: dict[str, Any]) -> bool:
    return is_acceptable_source(str(payload.get("source_url", "")))


def source_priority(source_url: str) -> int:
    hostname = source_hostname(source_url)
    if hostname.endswith((".gov", ".edu")):
        return 3
    if any(domain_matches(hostname, domain) for domain in TRUSTED_DOMAINS):
        return 2
    if hostname.endswith(".org"):
        return 1
    return 0
