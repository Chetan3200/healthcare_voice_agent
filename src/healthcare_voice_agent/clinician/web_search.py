"""Small, read-only Exa search adapter for external clinical guidance."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


_EXA_SEARCH_URL = "https://api.exa.ai/search"
_EXTERNAL_GUIDANCE_WARNING = {
    "code": "EXTERNAL_GUIDANCE",
    "message": "External web guidance, not verified clinic-approved policy; web document versions are unavailable.",
    "evidence_ids": [],
}


class WebSearchError(Exception):
    """Safe caller-facing error for the external guidance provider."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ExaGuidanceSearch:
    """Retrieve Exa-extracted passages from a small approved domain allowlist."""

    def __init__(
        self,
        api_key: str | None,
        domains: tuple[str, ...] = ("nhs.uk", "nice.org.uk", "orthoinfo.aaos.org"),
        timeout: int = 15,
    ) -> None:
        self._api_key = api_key
        self._domains = tuple(domain.lower() for domain in domains)
        self._timeout = timeout

    def search(self, args: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        query = getattr(args, "query", None)
        if not isinstance(query, str) or not query.strip():
            raise WebSearchError("INVALID_ARGUMENT")
        if getattr(args, "version", None) is not None:
            raise WebSearchError("DOCUMENT_VERSION_NOT_FOUND")

        document_id = getattr(args, "document_id", None)
        include_domains = list(self._domains)
        if document_id is not None:
            parsed = self._approved_url(document_id)
            # Exa's documented filter accepts hostname/path prefixes; exact URL
            # matching is enforced again on every returned result below.
            include_domains = [parsed.hostname + (parsed.path or "/")]

        if not self._api_key or not self._api_key.strip():
            raise WebSearchError("BACKEND_UNAVAILABLE")

        payload = {
            "query": query,
            "numResults": args.top_k,
            "includeDomains": include_domains,
            "contents": {"highlights": True},
        }
        request = Request(
            _EXA_SEARCH_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"x-api-key": self._api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise WebSearchError("TIMEOUT") from exc
        except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise WebSearchError("BACKEND_UNAVAILABLE") from exc

        if not isinstance(response_data, dict) or not isinstance(response_data.get("results"), list):
            raise WebSearchError("BACKEND_UNAVAILABLE")

        matches = []
        for result in response_data["results"]:
            if not isinstance(result, dict):
                continue
            url = result.get("url")
            if not isinstance(url, str) or not self._is_approved_url(url):
                continue
            if document_id is not None and url != document_id:
                continue
            highlights = result.get("highlights") or []
            if not isinstance(highlights, list):
                raise WebSearchError("BACKEND_UNAVAILABLE")
            passages = [item for item in highlights if isinstance(item, str) and item.strip()]
            if not passages:
                continue
            matches.append({
                "document_id": url,
                "url": url,
                "title": result.get("title") if isinstance(result.get("title"), str) else "",
                "published_at": self._parse_published_at(result.get("publishedDate")),
                "document_version": None,
                "approval_status": "not_verified",
                "rank": len(matches) + 1,
                "passages": passages,
            })
            if len(matches) == args.top_k:
                break

        return {
            "query": query,
            "version_scope": "web_unversioned",
            "provider": "exa",
            "matches": matches,
            "as_of": datetime.now(timezone.utc),
        }, [dict(_EXTERNAL_GUIDANCE_WARNING)]

    def _approved_url(self, url: Any):
        if not isinstance(url, str):
            raise WebSearchError("INVALID_ARGUMENT")
        try:
            parsed = urlparse(url)
        except ValueError as exc:
            raise WebSearchError("INVALID_ARGUMENT") from exc
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password
                or not self._is_approved_host(parsed.hostname)):
            raise WebSearchError("INVALID_ARGUMENT")
        return parsed

    def _is_approved_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        return (parsed.scheme in {"http", "https"} and bool(parsed.hostname)
                and not parsed.username and not parsed.password and self._is_approved_host(parsed.hostname))

    def _is_approved_host(self, hostname: str) -> bool:
        hostname = hostname.lower()
        return any(hostname == domain or hostname.endswith("." + domain) for domain in self._domains)

    @staticmethod
    def _parse_published_at(value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else None
