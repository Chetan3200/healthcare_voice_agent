import json
from datetime import timezone
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from healthcare_voice_agent.clinician.web_search import ExaGuidanceSearch, WebSearchError


class Response:
    def __init__(self, payload): self.payload = payload
    def read(self): return json.dumps(self.payload).encode()
    def __enter__(self): return self
    def __exit__(self, *args): return False


def args(**changes):
    values = {"query": "wrist fracture follow-up", "top_k": 2, "document_id": None, "version": None}
    values.update(changes)
    return SimpleNamespace(**values)


def test_search_sends_extract_only_request_and_maps_exact_payload(monkeypatch):
    captured = {}
    payload = {"results": [
        {"url": "https://www.nhs.uk/conditions/wrist-fracture/", "title": "Wrist fracture", "publishedDate": "2026-01-02T03:04:05Z", "highlights": [" Keep the splint dry. ", ""]},
        {"url": "https://www.nice.org.uk/guidance/ng38", "title": "Guidance", "publishedDate": "not-a-date", "highlights": ["Relevant evidence."]},
    ]}
    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response(payload)
    monkeypatch.setattr("healthcare_voice_agent.clinician.web_search.urlopen", fake_urlopen)

    data, warnings = ExaGuidanceSearch("key", timeout=7).search(args())

    assert captured == {
        "url": "https://api.exa.ai/search", "headers": {"x-api-key": "key", "content-type": "application/json"},
        "payload": {"query": "wrist fracture follow-up", "numResults": 2, "includeDomains": ["nhs.uk", "nice.org.uk", "orthoinfo.aaos.org"], "contents": {"highlights": True}}, "timeout": 7,
    }
    assert data["query"] == "wrist fracture follow-up"
    assert "summary" not in captured["payload"]["contents"] and "text" not in captured["payload"]["contents"]
    assert data["version_scope"] == "web_unversioned" and data["provider"] == "exa"
    first, second = data["matches"]
    assert first == {"document_id": "https://www.nhs.uk/conditions/wrist-fracture/", "url": "https://www.nhs.uk/conditions/wrist-fracture/", "title": "Wrist fracture", "published_at": first["published_at"], "document_version": None, "approval_status": "not_verified", "rank": 1, "passages": [" Keep the splint dry. "]}
    assert first["published_at"].isoformat() == "2026-01-02T03:04:05+00:00"
    assert second == {"document_id": "https://www.nice.org.uk/guidance/ng38", "url": "https://www.nice.org.uk/guidance/ng38", "title": "Guidance", "published_at": None, "document_version": None, "approval_status": "not_verified", "rank": 2, "passages": ["Relevant evidence."]}
    assert data["as_of"].tzinfo is timezone.utc
    assert warnings == [{"code": "EXTERNAL_GUIDANCE", "message": "External web guidance, not verified clinic-approved policy; web document versions are unavailable.", "evidence_ids": []}]


def test_filters_wrong_domain_and_results_without_extracted_passages(monkeypatch):
    monkeypatch.setattr("healthcare_voice_agent.clinician.web_search.urlopen", lambda *_args, **_kwargs: Response({"results": [
        {"url": "https://example.com/bad", "highlights": ["Do not return."]},
        {"url": "https://nhs.uk/empty", "highlights": []},
        {"url": "https://nhs.uk/good", "highlights": ["Returned passage."]},
    ]}))
    data, _ = ExaGuidanceSearch("key").search(args())
    assert data["matches"] == [{"document_id": "https://nhs.uk/good", "url": "https://nhs.uk/good", "title": "", "published_at": None, "document_version": None, "approval_status": "not_verified", "rank": 1, "passages": ["Returned passage."]}]


def test_document_url_is_restricted_in_request_and_response(monkeypatch):
    captured = {}
    monkeypatch.setattr("healthcare_voice_agent.clinician.web_search.urlopen", lambda request, timeout: (captured.update(payload=json.loads(request.data)) or Response({"results": [
        {"url": "https://www.nhs.uk/conditions/wrist-fracture/other", "highlights": ["Prefix but not exact."]},
        {"url": "https://www.nhs.uk/conditions/wrist-fracture/", "highlights": ["Exact document."]},
    ]})))
    document_id = "https://www.nhs.uk/conditions/wrist-fracture/"
    data, _ = ExaGuidanceSearch("key").search(args(document_id=document_id))
    assert captured["payload"]["includeDomains"] == ["www.nhs.uk/conditions/wrist-fracture/"]
    assert [match["url"] for match in data["matches"]] == [document_id]


@pytest.mark.parametrize("search_args, code", [(args(query="   "), "INVALID_ARGUMENT"), (args(version="v1"), "DOCUMENT_VERSION_NOT_FOUND"), (args(document_id="https://example.com/x"), "INVALID_ARGUMENT")])
def test_rejects_invalid_external_search_scope(search_args, code):
    with pytest.raises(WebSearchError) as error:
        ExaGuidanceSearch("key").search(search_args)
    assert error.value.code == code


def test_missing_key_api_failure_and_empty_results_are_distinct(monkeypatch):
    with pytest.raises(WebSearchError) as error:
        ExaGuidanceSearch(None).search(args())
    assert error.value.code == "BACKEND_UNAVAILABLE"

    monkeypatch.setattr("healthcare_voice_agent.clinician.web_search.urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(HTTPError("url", 401, "no", {}, None)))
    with pytest.raises(WebSearchError) as error:
        ExaGuidanceSearch("key").search(args())
    assert error.value.code == "BACKEND_UNAVAILABLE"

    monkeypatch.setattr("healthcare_voice_agent.clinician.web_search.urlopen", lambda *_args, **_kwargs: Response({"results": []}))
    data, _ = ExaGuidanceSearch("key").search(args())
    assert data["matches"] == []
