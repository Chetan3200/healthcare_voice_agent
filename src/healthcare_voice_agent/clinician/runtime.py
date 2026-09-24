"""Server-managed case selection; read-only calls and a shared response envelope."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from time import monotonic
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from pydantic import ValidationError

from healthcare_voice_agent.config import ConfigurationError
from healthcare_voice_agent.tools.contracts import ResolvedCaseContext, TOOLS
from .case_selection import OPEN_CASE
from .records import ClinicalReadError, ClinicalRecords
from .web_search import ExaGuidanceSearch, WebSearchError

_MESSAGES = {
    "INVALID_ARGUMENT": "Supply the required fields with valid values; nullable fields must be explicit.",
    "IDENTITY_UNRESOLVED": "No case is selected. Open the caller-supplied exact case ID, or clarify which returned case ID to open. No additional DOB verification is required.",
    "CASE_CONTEXT_MISMATCH": "The requested case does not match the server-confirmed case.",
    "CASE_NOT_FOUND": "No matching accessible synthetic case was found. No case is selected.",
    "RECORD_NOT_FOUND": "No matching accessible record was found.",
    "RESULT_NOT_AVAILABLE": "No eligible completed model result is available.",
    "NOT_REVIEWED": "A clinician-reviewed report is not available; no draft has been substituted.",
    "AMBIGUOUS_RECORD": "Use the same-case candidates to select the requested study; clarify only if needed.",
    "RECORD_CONFLICT": "The stored current-record designation is inconsistent.",
    "DOCUMENT_VERSION_NOT_FOUND": "Exa does not provide verified document-version retrieval. No other version was substituted.",
    "TIMEOUT": "The lookup timed out. No result is available yet.",
    "BACKEND_UNAVAILABLE": "The data source is unavailable. Check local database or Exa configuration.",
    "INTERNAL_ERROR": "The source returned an invalid response. No clinical content was returned.",
}


def clinical_records(settings):
    if not all((settings.db_name, settings.db_user, settings.db_password)):
        raise ConfigurationError("POSTGRES_DB, POSTGRES_USER and POSTGRES_PASSWORD are required for clinical reads.")

    def connect():
        # A fresh, short-lived read-only connection per repository operation.
        connection = psycopg.connect(
            host=settings.db_host, port=settings.db_port, dbname=settings.db_name,
            user=settings.db_user.get_secret_value(), password=settings.db_password.get_secret_value(),
            connect_timeout=5, row_factory=dict_row,
            options="-c default_transaction_read_only=on -c statement_timeout=10000",
        )
        # Keep a selected model/report and its child rows on one database snapshot.
        connection.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        return connection
    return ClinicalRecords(connect, settings.user_id)


class ClinicianRuntime:
    def __init__(self, records, guidance, context, summary, trace=None, *,
                 demo_web_search_delay_seconds=0.0, sleep=asyncio.sleep):
        self.records, self.guidance = records, guidance
        self.context, self.summary, self.trace = context, summary, trace
        self.demo_web_search_delay_seconds = demo_web_search_delay_seconds
        self._sleep = sleep
        self.closed = False
        self.case_revision = context.case_context_version if context else 0

    async def invoke(self, name, arguments):
        started, request_id = monotonic(), uuid4()
        selecting = name == "open_case"
        if selecting:
            # Invalidate immediately, including invalid/cancelled/ambiguous switches.
            self.case_revision += 1
            self.context = self.summary = None
        revision = self.case_revision
        context = None if name == "search_clinic_instructions" else self.context
        contract = OPEN_CASE if selecting else TOOLS[name]
        data, warnings, error = None, [], None
        try:
            args = contract.validate_arguments_json(json.dumps(arguments))
        except (ValidationError, TypeError, ValueError):
            error = {"code": "INVALID_ARGUMENT", "message": _MESSAGES["INVALID_ARGUMENT"],
                     "retryable": False, "candidates": []}
        if error is None:
            try:
                if self.closed:
                    raise ClinicalReadError("BACKEND_UNAVAILABLE")
                if selecting:
                    matches = await asyncio.to_thread(self.records.find_cases, args)
                    if revision != self.case_revision or self.closed:
                        raise ClinicalReadError("CASE_CONTEXT_MISMATCH")
                    if not matches:
                        raise ClinicalReadError("CASE_NOT_FOUND")
                    # Validate identity data before installing it as trusted context.
                    from .case_selection import CaseIdentity
                    identities = [CaseIdentity.model_validate_json(json.dumps(
                        row, default=lambda value: value.isoformat())) for row in matches[:50]]
                    selected = identities[0] if args.case_id is not None and len(matches) == 1 else None
                    if selected:
                        self.summary = selected.model_dump(mode="json")
                        context = self.context = ResolvedCaseContext(
                            case_id=selected.case_id, case_context_version=revision)
                    data = {"state": "selected" if selected else "selection_required",
                            "selected_case": self.summary,
                            "candidates": [] if selected else [row.model_dump(mode="json") for row in identities],
                            "has_more": len(matches) > 50}
                elif name == "search_clinic_instructions":
                    if self.trace:
                        self.trace.emit("clinician_artificial_web_search_delay",
                                        tool_name=name,
                                        artificial_delay_seconds=self.demo_web_search_delay_seconds)
                    if self.demo_web_search_delay_seconds:
                        await self._sleep(self.demo_web_search_delay_seconds)
                        if self.closed:
                            raise ClinicalReadError("BACKEND_UNAVAILABLE")
                    data, warnings = await asyncio.to_thread(self.guidance.search, args)
                else:
                    data, warnings = await asyncio.to_thread(self.records.read, name, args, context)
            except (ClinicalReadError, WebSearchError) as exc:
                error = {"code": exc.code, "message": _MESSAGES[exc.code],
                         "retryable": exc.code in {"TIMEOUT", "BACKEND_UNAVAILABLE"},
                         "candidates": getattr(exc, "candidates", [])}
            except (TimeoutError, psycopg.errors.QueryCanceled):
                error = {"code": "TIMEOUT", "message": _MESSAGES["TIMEOUT"], "retryable": True, "candidates": []}
            except Exception:
                error = {"code": "BACKEND_UNAVAILABLE", "message": _MESSAGES["BACKEND_UNAVAILABLE"],
                         "retryable": True, "candidates": []}
        if name != "search_clinic_instructions" and (
                revision != self.case_revision or (not selecting and context != self.context)):
            data, warnings = None, []
            error = {"code": "CASE_CONTEXT_MISMATCH", "message": _MESSAGES["CASE_CONTEXT_MISMATCH"],
                     "retryable": False, "candidates": []}
        result = {"contract_version": "2.0.0" if name == "search_clinic_instructions" else "1.0.0",
                  "status": "error" if error else "ok", "data": None if error else data,
                  "error": error, "warnings": [] if error else warnings,
                  "meta": {"request_id": str(request_id), "tool_name": name,
                           "retrieved_at": datetime.now(timezone.utc).isoformat(),
                           "duration_ms": int((monotonic() - started) * 1000),
                           "case_context_version": revision if selecting else (
                               context.case_context_version if context else None)}}
        try:
            result = contract.validate_response_json(json.dumps(
                result, default=lambda value: value.isoformat(), allow_nan=False)).model_dump(mode="json")
        except (ValidationError, TypeError, ValueError, AttributeError):
            result.update(status="error", data=None, warnings=[], error={
                "code": "INTERNAL_ERROR", "message": _MESSAGES["INTERNAL_ERROR"],
                "retryable": False, "candidates": []})
        if self.trace:
            self.trace.emit("clinician_tool_result", tool_name=name, status=result["status"],
                            error_code=result["error"]["code"] if result["error"] else None,
                            request_id=str(request_id), duration_ms=result["meta"]["duration_ms"],
                            case_context_version=result["meta"]["case_context_version"]) 
        return result

    async def close(self):
        self.closed = True  # Connections close inside each read; there is no write to recover.


async def open_clinician_runtime(config, trace=None):
    records = clinical_records(config.clinician)
    try:
        await asyncio.to_thread(records.verify_access)
    except Exception:
        raise ConfigurationError("Synthetic clinical records are unavailable. Check local database access.") from None
    key = config.clinician.exa_api_key
    guidance = ExaGuidanceSearch(key.get_secret_value() if key else None,
                                 domains=config.clinician.search_domains)
    return ClinicianRuntime(
        records, guidance, None, None, trace,
        demo_web_search_delay_seconds=config.clinician.demo_web_search_delay_seconds,
    )
