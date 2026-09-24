"""Deterministic fixture loader for the Alembic 001 clinic schema.

Public test helpers:
    load_dataset(fixture_dir: Path) -> dict
        Verify the fixed fixture-file manifest hashes, parse JSON, and combine it.
    validate_dataset(dataset: dict) -> None
        Perform all validation without reading .env or opening a database.

The default CLI mode is an offline dry run. Database modes never update or delete
existing rows, and the invalid-fixture mode rolls back every attempted insert.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_DIR = ROOT / "fixtures"
DATASET_VERSION = "synthetic-clinic-v1"
REFERENCE_TIME = "2026-09-17T09:00:00Z"
SCHEMA_REVISION = "001"  # Fixture format remains the unchanged baseline schema.
SUPPORTED_DB_REVISIONS = frozenset({"001", "002", "003", "004", "005"})  # Additive booking revisions preserve fixture tables.
ADVISORY_LOCK_KEY = 7_240_010_017
FIXTURE_FILES = (
    "clinical_records.json",
    "clinic_documents.json",
    "record_scenarios.json",
    "document_scenarios.json",
    "invalid_records.json",
)
CLINICAL_TABLES = frozenset({
    "patients", "clinicians", "app_users", "cases", "studies", "model_results",
    "model_findings", "reviewed_reports", "appointments", "retrieval_indexes",
    "chunk_embeddings",
})
DOCUMENT_TABLES = frozenset({"clinic_documents", "document_versions", "document_chunks"})
EXPECTED_SQLSTATES = frozenset({"23502", "23503", "23505", "23514"})
CHECK_OPERATORS = frozenset({"in", "gte", "lt"})


class SeedError(Exception):
    """A safe-to-display fixture or seed failure."""


@dataclass(frozen=True)
class TableSpec:
    columns: tuple[str, ...]
    nullable: frozenset[str]
    primary_key: tuple[str, ...]
    id_columns: frozenset[str] = frozenset()
    timestamp_columns: frozenset[str] = frozenset()
    date_columns: frozenset[str] = frozenset()
    array_columns: frozenset[str] = frozenset()
    bool_columns: frozenset[str] = frozenset()
    int_columns: frozenset[str] = frozenset()
    float_columns: frozenset[str] = frozenset()


TABLE_SPECS: dict[str, TableSpec] = {
    "patients": TableSpec(
        ("patient_id", "display_name", "date_of_birth", "is_synthetic", "record_version", "created_at", "updated_at"),
        frozenset({"date_of_birth"}), ("patient_id",),
        id_columns=frozenset({"patient_id"}), date_columns=frozenset({"date_of_birth"}),
        timestamp_columns=frozenset({"created_at", "updated_at"}),
        bool_columns=frozenset({"is_synthetic"}), int_columns=frozenset({"record_version"}),
    ),
    "clinicians": TableSpec(
        ("clinician_id", "display_name", "is_synthetic", "created_at"), frozenset(), ("clinician_id",),
        id_columns=frozenset({"clinician_id"}), timestamp_columns=frozenset({"created_at"}),
        bool_columns=frozenset({"is_synthetic"}),
    ),
    "app_users": TableSpec(
        ("user_id", "display_name", "role", "is_active", "created_at"), frozenset(), ("user_id",),
        id_columns=frozenset({"user_id"}), timestamp_columns=frozenset({"created_at"}),
        bool_columns=frozenset({"is_active"}),
    ),
    "cases": TableSpec(
        ("case_id", "patient_id", "description", "status", "opened_at", "closed_at", "record_version", "created_at", "updated_at"),
        frozenset({"closed_at"}), ("case_id",),
        id_columns=frozenset({"case_id", "patient_id"}),
        timestamp_columns=frozenset({"opened_at", "closed_at", "created_at", "updated_at"}),
        int_columns=frozenset({"record_version"}),
    ),
    "studies": TableSpec(
        ("study_id", "case_id", "performed_at", "modality", "body_part", "laterality", "views", "acquisition_status", "image_ref", "record_version", "created_at", "updated_at"),
        frozenset({"performed_at", "image_ref"}), ("study_id",),
        id_columns=frozenset({"study_id", "case_id"}),
        timestamp_columns=frozenset({"performed_at", "created_at", "updated_at"}),
        array_columns=frozenset({"views"}), int_columns=frozenset({"record_version"}),
    ),
    "model_results": TableSpec(
        ("model_result_id", "study_id", "model_name", "model_version", "inference_status", "review_status", "generated_at", "record_status", "is_current", "summary", "score_description", "limitations", "record_version", "created_at", "updated_at"),
        frozenset({"generated_at", "summary", "score_description"}), ("model_result_id",),
        id_columns=frozenset({"model_result_id", "study_id"}),
        timestamp_columns=frozenset({"generated_at", "created_at", "updated_at"}),
        array_columns=frozenset({"limitations"}), bool_columns=frozenset({"is_current"}),
        int_columns=frozenset({"record_version"}),
    ),
    "model_findings": TableSpec(
        ("model_result_id", "finding_index", "label", "assessment", "confidence"),
        frozenset({"confidence"}), ("model_result_id", "finding_index"),
        id_columns=frozenset({"model_result_id"}), int_columns=frozenset({"finding_index"}),
        float_columns=frozenset({"confidence"}),
    ),
    "reviewed_reports": TableSpec(
        ("report_id", "study_id", "record_version", "review_status", "record_status", "is_current", "reviewed_by_clinician_id", "reviewed_at", "findings_text", "impression_text", "follow_up_recommendation", "supersedes_report_id", "created_at", "updated_at"),
        frozenset({"reviewed_by_clinician_id", "reviewed_at", "follow_up_recommendation", "supersedes_report_id"}),
        ("report_id",),
        id_columns=frozenset({"report_id", "study_id", "reviewed_by_clinician_id", "supersedes_report_id"}),
        timestamp_columns=frozenset({"reviewed_at", "created_at", "updated_at"}),
        bool_columns=frozenset({"is_current"}), int_columns=frozenset({"record_version"}),
    ),
    "appointments": TableSpec(
        ("appointment_id", "case_id", "appointment_type", "starts_at", "ends_at", "timezone", "status", "clinician_id", "location", "notes", "replaces_appointment_id", "record_version", "created_at", "updated_at"),
        frozenset({"ends_at", "clinician_id", "location", "notes", "replaces_appointment_id"}),
        ("appointment_id",),
        id_columns=frozenset({"appointment_id", "case_id", "clinician_id", "replaces_appointment_id"}),
        timestamp_columns=frozenset({"starts_at", "ends_at", "created_at", "updated_at"}),
        int_columns=frozenset({"record_version"}),
    ),
    "clinic_documents": TableSpec(
        ("document_id", "slug", "created_at"), frozenset(), ("document_id",),
        id_columns=frozenset({"document_id"}), timestamp_columns=frozenset({"created_at"}),
    ),
    "document_versions": TableSpec(
        ("document_id", "version", "title", "full_text", "tags", "approval_status", "publication_status", "published_at", "effective_from", "effective_to", "is_current", "created_at", "updated_at"),
        frozenset({"published_at", "effective_from", "effective_to"}), ("document_id", "version"),
        id_columns=frozenset({"document_id"}),
        timestamp_columns=frozenset({"published_at", "effective_from", "effective_to", "created_at", "updated_at"}),
        array_columns=frozenset({"tags"}), bool_columns=frozenset({"is_current"}),
    ),
    "document_chunks": TableSpec(
        ("chunk_id", "document_id", "document_version", "chunk_index", "section", "content", "updated_at"),
        frozenset(), ("chunk_id",), id_columns=frozenset({"chunk_id", "document_id"}),
        timestamp_columns=frozenset({"updated_at"}), int_columns=frozenset({"chunk_index"}),
    ),
    "retrieval_indexes": TableSpec(
        ("index_version", "embedding_model", "embedding_dimensions", "chunking_version", "distance_metric", "min_similarity", "created_at"),
        frozenset(), ("index_version",), timestamp_columns=frozenset({"created_at"}),
        int_columns=frozenset({"embedding_dimensions"}), float_columns=frozenset({"min_similarity"}),
    ),
    "chunk_embeddings": TableSpec(
        ("index_version", "chunk_id", "embedding", "embedding_dimensions", "created_at"),
        frozenset(), ("index_version", "chunk_id"), id_columns=frozenset({"chunk_id"}),
        timestamp_columns=frozenset({"created_at"}), int_columns=frozenset({"embedding_dimensions"}),
    ),
}

INSERT_ORDER = (
    "patients", "clinicians", "app_users", "cases", "studies", "model_results",
    "model_findings", "reviewed_reports", "appointments", "clinic_documents",
    "document_versions", "document_chunks", "retrieval_indexes", "chunk_embeddings",
)

ENUMS: dict[tuple[str, str], frozenset[str]] = {
    ("app_users", "role"): frozenset({"clinician", "staff"}),
    ("cases", "status"): frozenset({"open", "closed", "archived"}),
    ("studies", "modality"): frozenset({"XR", "CT", "MRI", "US", "OTHER"}),
    ("studies", "laterality"): frozenset({"left", "right", "bilateral", "not_applicable", "unknown"}),
    ("studies", "acquisition_status"): frozenset({"scheduled", "completed", "cancelled"}),
    ("model_results", "inference_status"): frozenset({"pending", "completed", "failed"}),
    ("model_results", "review_status"): frozenset({"unreviewed"}),
    ("model_results", "record_status"): frozenset({"active", "superseded", "withdrawn", "archived"}),
    ("model_findings", "assessment"): frozenset({"suspected", "not_detected", "indeterminate"}),
    ("reviewed_reports", "review_status"): frozenset({"draft", "reviewed"}),
    ("reviewed_reports", "record_status"): frozenset({"active", "superseded", "withdrawn", "archived"}),
    ("appointments", "appointment_type"): frozenset({"initial_consultation", "fracture_follow_up", "imaging", "physiotherapy", "other"}),
    ("appointments", "status"): frozenset({"scheduled", "confirmed", "cancelled", "completed", "no_show"}),
    ("document_versions", "approval_status"): frozenset({"draft", "approved"}),
    ("document_versions", "publication_status"): frozenset({"draft", "published", "withdrawn"}),
    ("retrieval_indexes", "distance_metric"): frozenset({"cosine"}),
}

NONBLANK = frozenset({
    ("patients", "display_name"), ("clinicians", "display_name"),
    ("app_users", "display_name"), ("studies", "body_part"),
    ("model_results", "model_name"), ("model_results", "model_version"),
    ("model_findings", "label"), ("clinic_documents", "slug"),
    ("document_versions", "title"), ("document_versions", "full_text"),
    ("document_chunks", "content"), ("retrieval_indexes", "index_version"),
    ("retrieval_indexes", "embedding_model"),
})

FOREIGN_KEYS = (
    ("cases", ("patient_id",), "patients", ("patient_id",)),
    ("studies", ("case_id",), "cases", ("case_id",)),
    ("model_results", ("study_id",), "studies", ("study_id",)),
    ("model_findings", ("model_result_id",), "model_results", ("model_result_id",)),
    ("reviewed_reports", ("study_id",), "studies", ("study_id",)),
    ("reviewed_reports", ("reviewed_by_clinician_id",), "clinicians", ("clinician_id",)),
    ("appointments", ("case_id",), "cases", ("case_id",)),
    ("appointments", ("clinician_id",), "clinicians", ("clinician_id",)),
    ("document_versions", ("document_id",), "clinic_documents", ("document_id",)),
    ("document_chunks", ("document_id", "document_version"), "document_versions", ("document_id", "version")),
    ("chunk_embeddings", ("chunk_id",), "document_chunks", ("chunk_id",)),
    ("chunk_embeddings", ("index_version", "embedding_dimensions"), "retrieval_indexes", ("index_version", "embedding_dimensions")),
)


def _fail(message: str) -> None:
    raise SeedError(message)


def _expect_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{context} must be a JSON object")
    return value


def _expect_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{context} must be a JSON array")
    return value


def _expect_keys(obj: Mapping[str, Any], required: set[str], allowed: set[str], context: str) -> None:
    keys = set(obj)
    missing = required - keys
    extra = keys - allowed
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(sorted(missing)))
        if extra:
            parts.append("unexpected " + ", ".join(sorted(extra)))
        _fail(f"{context} has " + "; ".join(parts))


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON contains duplicate object key {key!r}")
        result[key] = value
    return result


def _read_json_bytes(raw: bytes, basename: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail(f"{basename} is not UTF-8")
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_object)
    except SeedError:
        raise
    except (json.JSONDecodeError, ValueError):
        _fail(f"{basename} is not valid JSON")


def load_dataset(fixture_dir: Path) -> dict[str, Any]:
    """Load fixed fixtures after verifying every manifest SHA-256 digest.

    This function is entirely offline and never reads environment variables.
    """
    fixture_dir = Path(fixture_dir)
    manifest_path = fixture_dir / "manifest.json"
    try:
        manifest_raw = manifest_path.read_bytes()
    except OSError:
        _fail("Could not read fixtures/manifest.json")
    manifest = _expect_object(_read_json_bytes(manifest_raw, "manifest.json"), "manifest.json")
    _expect_keys(
        manifest,
        {"dataset_version", "reference_time", "schema_revision", "files", "table_counts", "scenario_counts"},
        {"dataset_version", "reference_time", "schema_revision", "files", "table_counts", "scenario_counts", "notice"},
        "manifest.json",
    )
    files = _expect_object(manifest["files"], "manifest.files")
    if set(files) != set(FIXTURE_FILES):
        _fail("manifest.files must contain exactly the fixed fixture-file whitelist")

    parsed: dict[str, Any] = {}
    for basename in FIXTURE_FILES:
        digest = files[basename]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            _fail(f"manifest digest for {basename} must be lowercase SHA-256 hex")
        try:
            raw = (fixture_dir / basename).read_bytes()
        except OSError:
            _fail(f"Could not read required fixture {basename}")
        if not hashlib.sha256(raw).hexdigest() == digest:
            _fail(f"SHA-256 verification failed for {basename}")
        parsed[basename] = _read_json_bytes(raw, basename)

    clinical = _expect_object(parsed["clinical_records.json"], "clinical_records.json")
    documents = _expect_object(parsed["clinic_documents.json"], "clinic_documents.json")
    for basename, obj in (("clinical_records.json", clinical), ("clinic_documents.json", documents)):
        _expect_keys(obj, {"dataset_version", "reference_time", "notice", "tables"},
                     {"dataset_version", "reference_time", "notice", "tables"}, basename)
    clinical_tables = _expect_object(clinical["tables"], "clinical_records.tables")
    document_tables = _expect_object(documents["tables"], "clinic_documents.tables")
    if set(clinical_tables) != CLINICAL_TABLES:
        _fail("clinical_records.tables must contain exactly the clinical fixture tables")
    if set(document_tables) != DOCUMENT_TABLES:
        _fail("clinic_documents.tables must contain exactly the document fixture tables")

    record_doc = _expect_object(parsed["record_scenarios.json"], "record_scenarios.json")
    document_doc = _expect_object(parsed["document_scenarios.json"], "document_scenarios.json")
    invalid_doc = _expect_object(parsed["invalid_records.json"], "invalid_records.json")
    scenario_metadata = {"dataset_version", "notice"}
    _expect_keys(record_doc, {"reference_time", "scenarios"}, {"reference_time", "scenarios"} | scenario_metadata, "record_scenarios.json")
    _expect_keys(document_doc, {"reference_time", "scenarios"}, {"reference_time", "scenarios"} | scenario_metadata, "document_scenarios.json")
    _expect_keys(invalid_doc, {"reference_time", "cases"}, {"reference_time", "cases"} | scenario_metadata, "invalid_records.json")

    return {
        "dataset_version": clinical["dataset_version"],
        "reference_time": clinical["reference_time"],
        "notice": clinical["notice"],
        "tables": {**clinical_tables, **document_tables},
        "record_scenarios": record_doc["scenarios"],
        "document_scenarios": document_doc["scenarios"],
        "invalid_records": invalid_doc["cases"],
        "manifest": manifest,
        "_source_metadata": {
            "clinical": {k: clinical[k] for k in ("dataset_version", "reference_time", "notice")},
            "documents": {k: documents[k] for k in ("dataset_version", "reference_time", "notice")},
            "record_scenarios": {k: record_doc[k] for k in ("dataset_version", "reference_time", "notice") if k in record_doc},
            "document_scenarios": {k: document_doc[k] for k in ("dataset_version", "reference_time", "notice") if k in document_doc},
            "invalid_records": {k: invalid_doc[k] for k in ("dataset_version", "reference_time", "notice") if k in invalid_doc},
        },
    }


def _parse_timestamp(value: Any, context: str) -> datetime:
    if not isinstance(value, str):
        _fail(f"{context} must be an ISO-8601 timestamp string")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        _fail(f"{context} is not a valid ISO-8601 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{context} must include a timezone offset")
    return parsed


def _parse_date(value: Any, context: str) -> date:
    if not isinstance(value, str):
        _fail(f"{context} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        _fail(f"{context} is not a valid ISO date")
    if "T" in value or len(value) != 10:
        _fail(f"{context} must use YYYY-MM-DD")
    return parsed


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _row_key(row: Mapping[str, Any], columns: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row[column] for column in columns)


def _validate_basic_row(table: str, row: Any, index: int) -> dict[str, Any]:
    context = f"tables.{table}[{index}]"
    row = _expect_object(row, context)
    spec = TABLE_SPECS[table]
    expected = set(spec.columns)
    if set(row) != expected:
        _expect_keys(row, expected, expected, context)
    for column in spec.columns:
        value = row[column]
        field = f"{context}.{column}"
        if value is None:
            if column not in spec.nullable:
                _fail(f"{field} may not be null")
            continue
        if column in spec.id_columns:
            if not isinstance(value, str) or not 1 <= len(value) <= 64:
                _fail(f"{field} must be a 1-64 character record ID")
        elif column in spec.timestamp_columns:
            _parse_timestamp(value, field)
        elif column in spec.date_columns:
            _parse_date(value, field)
        elif column in spec.array_columns:
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                _fail(f"{field} must be an array of strings")
        elif column in spec.bool_columns:
            if not isinstance(value, bool):
                _fail(f"{field} must be a boolean")
        elif column in spec.int_columns:
            if not _is_int(value):
                _fail(f"{field} must be an integer")
        elif column in spec.float_columns:
            if not _is_number(value):
                _fail(f"{field} must be a finite number")
        elif not isinstance(value, str):
            _fail(f"{field} must be a string")
        if (table, column) in NONBLANK and isinstance(value, str) and not value.strip():
            _fail(f"{field} may not be blank")
        allowed = ENUMS.get((table, column))
        if allowed is not None and value not in allowed:
            _fail(f"{field} has an unsupported enum value")
    return row


def _require_unique(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], context: str) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = _row_key(row, columns)
        if key in seen:
            _fail(f"{context} contains a duplicate key")
        seen.add(key)


def _build_index(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    return {_row_key(row, columns): row for row in rows}


def _validate_foreign_keys(tables: Mapping[str, list[dict[str, Any]]]) -> None:
    for child_table, child_cols, parent_table, parent_cols in FOREIGN_KEYS:
        parent_keys = {_row_key(row, parent_cols) for row in tables[parent_table]}
        for index, row in enumerate(tables[child_table]):
            key = _row_key(row, child_cols)
            if any(part is None for part in key):
                continue
            if key not in parent_keys:
                _fail(f"tables.{child_table}[{index}] has a missing {parent_table} parent")


def _validate_acyclic(rows: Sequence[Mapping[str, Any]], id_column: str, parent_column: str, context: str) -> None:
    parent = {row[id_column]: row[parent_column] for row in rows}
    for start in parent:
        seen: set[Any] = set()
        node: Any = start
        while node is not None:
            if node in seen:
                _fail(f"{context} contains a cycle")
            seen.add(node)
            node = parent.get(node)


def _authorized_cases(tables: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {"user_id": user["user_id"], "case_id": case["case_id"], "patient_id": case["patient_id"]}
        for user in tables["app_users"]
        if user["is_active"] and user["role"] in {"clinician", "staff"}
        for case in tables["cases"]
    ]


def _validate_check_value(table: str, column: str, value: Any, context: str) -> None:
    if table == "authorized_cases":
        if not isinstance(value, str) or not 1 <= len(value) <= 64:
            _fail(f"{context} must be a 1-64 character record ID")
        return
    spec = TABLE_SPECS[table]
    if value is None:
        if column not in spec.nullable:
            _fail(f"{context} may not be null")
    elif column in spec.timestamp_columns:
        _parse_timestamp(value, context)
    elif column in spec.date_columns:
        _parse_date(value, context)
    elif column in spec.bool_columns:
        if not isinstance(value, bool):
            _fail(f"{context} must be a boolean")
    elif column in spec.int_columns:
        if not _is_int(value):
            _fail(f"{context} must be an integer")
    elif column in spec.float_columns:
        if not _is_number(value):
            _fail(f"{context} must be a finite number")
    elif column in spec.array_columns:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            _fail(f"{context} must be an array of strings")
    elif not isinstance(value, str):
        _fail(f"{context} must be a string")


def _condition_matches(table: str, column: str, actual: Any, condition: Any) -> bool:
    if not isinstance(condition, dict):
        return actual == condition
    for operator, expected in condition.items():
        if operator == "in":
            if actual not in expected:
                return False
        elif operator == "gte":
            if actual is None or _comparable_value(table, column, actual) < _comparable_value(table, column, expected):
                return False
        elif operator == "lt":
            if actual is None or _comparable_value(table, column, actual) >= _comparable_value(table, column, expected):
                return False
    return True


def _comparable_value(table: str, column: str, value: Any) -> Any:
    if table != "authorized_cases":
        spec = TABLE_SPECS[table]
        if column in spec.timestamp_columns:
            return _parse_timestamp(value, f"check {table}.{column}")
        if column in spec.date_columns:
            return _parse_date(value, f"check {table}.{column}")
    return value


def _validate_scenarios(dataset: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    checks: list[tuple[str, dict[str, Any]]] = []
    scenario_ids: set[str] = set()
    tables = dataset["tables"]
    offline_tables: dict[str, list[dict[str, Any]]] = dict(tables)
    offline_tables["authorized_cases"] = _authorized_cases(tables)
    view_columns = {"user_id", "case_id", "patient_id"}

    for kind in ("record_scenarios", "document_scenarios"):
        scenarios = _expect_list(dataset[kind], kind)
        for index, item in enumerate(scenarios):
            context = f"{kind}[{index}]"
            item = _expect_object(item, context)
            required = {"id", "description", "record_refs", "expected_behavior"}
            allowed = required | {"checks"}
            _expect_keys(item, required, allowed, context)
            if not isinstance(item["id"], str) or not item["id"]:
                _fail(f"{context}.id must be a nonempty string")
            if item["id"] in scenario_ids:
                _fail("Scenario IDs must be unique across both scenario files")
            scenario_ids.add(item["id"])
            for key in ("description", "expected_behavior"):
                if not isinstance(item[key], str) or not item[key].strip():
                    _fail(f"{context}.{key} must be a nonblank string")
            refs = _expect_list(item["record_refs"], f"{context}.record_refs")
            if any(not isinstance(ref, str) or not ref for ref in refs):
                _fail(f"{context}.record_refs must contain nonempty strings")
            scenario_checks = item.get("checks", [])
            scenario_checks = _expect_list(scenario_checks, f"{context}.checks")
            for check_index, check in enumerate(scenario_checks):
                check_context = f"{context}.checks[{check_index}]"
                check = _expect_object(check, check_context)
                _expect_keys(check, {"table", "where", "count"}, {"table", "where", "count"}, check_context)
                table = check["table"]
                if table not in offline_tables:
                    _fail(f"{check_context}.table is not a trusted table or view")
                if not _is_int(check["count"]) or check["count"] < 0:
                    _fail(f"{check_context}.count must be a nonnegative integer")
                where = _expect_object(check["where"], f"{check_context}.where")
                allowed_columns = view_columns if table == "authorized_cases" else set(TABLE_SPECS[table].columns)
                for column, condition in where.items():
                    if column not in allowed_columns:
                        _fail(f"{check_context}.where uses an unknown column")
                    if isinstance(condition, dict):
                        if not condition or not set(condition) <= CHECK_OPERATORS:
                            _fail(f"{check_context}.where has an unsupported condition operator")
                        for operator, operand in condition.items():
                            values = operand if operator == "in" else [operand]
                            if operator == "in" and (not isinstance(operand, list) or not operand):
                                _fail(f"{check_context}.where in condition must be a nonempty array")
                            for value in values:
                                _validate_check_value(table, column, value, f"{check_context}.where.{column}")
                    else:
                        _validate_check_value(table, column, condition, f"{check_context}.where.{column}")
                actual_count = sum(
                    all(_condition_matches(table, column, row[column], condition) for column, condition in where.items())
                    for row in offline_tables[table]
                )
                if actual_count != check["count"]:
                    _fail(f"{check_context} expected count does not match the offline fixture rows")
                checks.append((item["id"], check))
    return checks


def _validate_invalid_cases(dataset: Mapping[str, Any]) -> None:
    cases = _expect_list(dataset["invalid_records"], "invalid_records")
    seen_ids: set[str] = set()
    for index, item in enumerate(cases):
        context = f"invalid_records[{index}]"
        item = _expect_object(item, context)
        required = {"id", "description", "table", "base_key", "patch", "expected_sqlstate"}
        _expect_keys(item, required, required, context)
        if not isinstance(item["id"], str) or not item["id"] or item["id"] in seen_ids:
            _fail(f"{context}.id must be unique and nonempty")
        seen_ids.add(item["id"])
        if not isinstance(item["description"], str) or not item["description"].strip():
            _fail(f"{context}.description must be nonblank")
        table = item["table"]
        if table not in TABLE_SPECS:
            _fail(f"{context}.table is not a trusted table")
        if table in {"retrieval_indexes", "chunk_embeddings"}:
            _fail(f"{context}.table may not generate retrieval indexes or embeddings")
        spec = TABLE_SPECS[table]
        base_key = _expect_object(item["base_key"], f"{context}.base_key")
        if set(base_key) != set(spec.primary_key):
            _fail(f"{context}.base_key must contain exactly the table primary key")
        patch = _expect_object(item["patch"], f"{context}.patch")
        if not patch or not set(patch) <= set(spec.columns):
            _fail(f"{context}.patch must use one or more known columns")
        if item["expected_sqlstate"] not in EXPECTED_SQLSTATES:
            _fail(f"{context}.expected_sqlstate is unsupported")
        matches = [row for row in dataset["tables"][table] if all(row[k] == v for k, v in base_key.items())]
        if len(matches) != 1:
            _fail(f"{context}.base_key must identify exactly one normal fixture row")
        # Confirm patching is side-effect free and creates a complete candidate. The
        # intentional candidate is not passed through normal-row validation.
        candidate = copy.deepcopy(matches[0])
        candidate.update(copy.deepcopy(patch))
        if set(candidate) != set(spec.columns):
            _fail(f"{context} does not produce a complete candidate row")


def validate_dataset(dataset: dict[str, Any]) -> None:
    """Validate a loaded dataset entirely offline; return None on success."""
    dataset = _expect_object(dataset, "dataset")
    required_dataset_keys = {
        "dataset_version", "reference_time", "notice", "tables", "record_scenarios",
        "document_scenarios", "invalid_records", "manifest", "_source_metadata",
    }
    _expect_keys(dataset, required_dataset_keys, required_dataset_keys, "dataset")
    if dataset["dataset_version"] != DATASET_VERSION:
        _fail("Dataset version is not synthetic-clinic-v1")
    if dataset["reference_time"] != REFERENCE_TIME:
        _fail("Dataset reference_time is not the fixed fixture reference time")
    _parse_timestamp(dataset["reference_time"], "dataset.reference_time")
    if not isinstance(dataset["notice"], str) or "synthetic" not in dataset["notice"].lower():
        _fail("Dataset notice must identify the content as synthetic")

    metadata = _expect_object(dataset["_source_metadata"], "dataset._source_metadata")
    clinical_meta = _expect_object(metadata.get("clinical"), "source clinical metadata")
    document_meta = _expect_object(metadata.get("documents"), "source document metadata")
    for name, meta in (("clinical", clinical_meta), ("documents", document_meta)):
        if meta.get("dataset_version") != DATASET_VERSION or meta.get("reference_time") != REFERENCE_TIME:
            _fail(f"{name} fixture metadata is inconsistent")
        if not isinstance(meta.get("notice"), str) or "synthetic" not in meta["notice"].lower():
            _fail(f"{name} fixture notice must identify synthetic content")
    for key in ("record_scenarios", "document_scenarios", "invalid_records"):
        source = _expect_object(metadata.get(key), f"source {key} metadata")
        if source.get("reference_time") != REFERENCE_TIME:
            _fail(f"{key} reference_time is inconsistent")
        if "dataset_version" in source and source["dataset_version"] != DATASET_VERSION:
            _fail(f"{key} dataset_version is inconsistent")
        if "notice" in source and (not isinstance(source["notice"], str) or not source["notice"].strip()):
            _fail(f"{key} notice must be a nonblank string")

    manifest = _expect_object(dataset["manifest"], "manifest")
    if manifest["dataset_version"] != DATASET_VERSION or manifest["reference_time"] != REFERENCE_TIME:
        _fail("Manifest dataset metadata is inconsistent")
    if manifest["schema_revision"] != SCHEMA_REVISION:
        _fail("Manifest schema_revision must be 001")

    tables = _expect_object(dataset["tables"], "tables")
    if set(tables) != set(TABLE_SPECS):
        _fail("Fixture tables must exactly match all Alembic 001 base tables")
    validated: dict[str, list[dict[str, Any]]] = {}
    for table, spec in TABLE_SPECS.items():
        rows = _expect_list(tables[table], f"tables.{table}")
        validated[table] = [_validate_basic_row(table, row, index) for index, row in enumerate(rows)]
        _require_unique(validated[table], spec.primary_key, f"tables.{table}")

    # Alternate/partial unique keys from revision 001.
    _require_unique(validated["clinic_documents"], ("slug",), "tables.clinic_documents.slug")
    _require_unique(validated["reviewed_reports"], ("study_id", "record_version"), "reviewed report revisions")
    _require_unique(validated["document_chunks"], ("document_id", "document_version", "chunk_index"), "document chunk positions")
    _require_unique(validated["retrieval_indexes"], ("index_version", "embedding_dimensions"), "retrieval index dimensions")
    for table, parent_col in (("model_results", "study_id"), ("reviewed_reports", "study_id"), ("document_versions", "document_id")):
        current = [row for row in validated[table] if row["is_current"]]
        _require_unique(current, (parent_col,), f"current rows in {table}")

    _validate_foreign_keys(validated)

    for table in ("patients", "clinicians"):
        if any(row["is_synthetic"] is not True for row in validated[table]):
            _fail(f"All {table} fixture rows must have is_synthetic=true")
    for table in ("patients", "cases", "studies", "model_results", "reviewed_reports", "appointments"):
        if any(row["record_version"] < 1 for row in validated[table]):
            _fail(f"All {table} record versions must be at least 1")
    for row in validated["cases"]:
        if row["closed_at"] is not None and _parse_timestamp(row["closed_at"], "cases.closed_at") < _parse_timestamp(row["opened_at"], "cases.opened_at"):
            _fail("A case closes before it opens")
    for row in validated["studies"]:
        if not row["body_part"].strip():
            _fail("Study body_part may not be blank")
    for row in validated["model_results"]:
        if row["inference_status"] == "completed" and row["generated_at"] is None:
            _fail("Completed model results require generated_at")
        if row["is_current"] and row["record_status"] != "active":
            _fail("Current model results must be active")
    for row in validated["model_findings"]:
        if row["finding_index"] < 0:
            _fail("Finding indexes must be nonnegative")
        if row["confidence"] is not None and not 0 <= float(row["confidence"]) <= 1:
            _fail("Finding confidence must be between 0 and 1")

    report_by_id = _build_index(validated["reviewed_reports"], ("report_id",))
    for row in validated["reviewed_reports"]:
        reviewed_fields_present = row["reviewed_by_clinician_id"] is not None and row["reviewed_at"] is not None
        if row["review_status"] == "reviewed" and not reviewed_fields_present:
            _fail("Reviewed reports require both reviewer and reviewed_at")
        if row["review_status"] == "draft" and (row["reviewed_by_clinician_id"] is not None or row["reviewed_at"] is not None):
            _fail("Draft reports may not have reviewer or reviewed_at")
        if row["is_current"] and row["record_status"] != "active":
            _fail("Current reports must be active")
        previous_id = row["supersedes_report_id"]
        if previous_id is not None:
            if previous_id == row["report_id"]:
                _fail("A report may not supersede itself")
            previous = report_by_id.get((previous_id,))
            if previous is None or previous["study_id"] != row["study_id"]:
                _fail("A superseded report must exist in the same study")
            if previous["record_version"] >= row["record_version"]:
                _fail("A report must supersede a lower revision")
    _validate_acyclic(validated["reviewed_reports"], "report_id", "supersedes_report_id", "reviewed report history")

    appointment_by_id = _build_index(validated["appointments"], ("appointment_id",))
    for row in validated["appointments"]:
        try:
            ZoneInfo(row["timezone"])
        except (ZoneInfoNotFoundError, ValueError):
            _fail("Appointment timezone must be a valid IANA timezone name")
        if row["ends_at"] is not None and _parse_timestamp(row["ends_at"], "appointments.ends_at") < _parse_timestamp(row["starts_at"], "appointments.starts_at"):
            _fail("An appointment ends before it starts")
        previous_id = row["replaces_appointment_id"]
        if previous_id is not None:
            if previous_id == row["appointment_id"]:
                _fail("An appointment may not replace itself")
            previous = appointment_by_id.get((previous_id,))
            if previous is None or previous["case_id"] != row["case_id"]:
                _fail("A replaced appointment must exist in the same case")
    _validate_acyclic(validated["appointments"], "appointment_id", "replaces_appointment_id", "appointment replacement history")

    for row in validated["document_versions"]:
        if not 1 <= len(row["version"]) <= 32:
            _fail("Document version must contain 1-32 characters")
        published = row["publication_status"] == "published"
        if published and (row["approval_status"] != "approved" or row["published_at"] is None or row["effective_from"] is None):
            _fail("Published documents require approval, published_at, and effective_from")
        if row["is_current"] and (not published or row["approval_status"] != "approved"):
            _fail("Current document versions must be approved and published")
        if row["effective_to"] is not None:
            if row["effective_from"] is None or _parse_timestamp(row["effective_to"], "document_versions.effective_to") <= _parse_timestamp(row["effective_from"], "document_versions.effective_from"):
                _fail("Document effective_to must be after effective_from")
        if row["published_at"] is not None and row["effective_from"] is not None:
            if _parse_timestamp(row["published_at"], "document_versions.published_at") > _parse_timestamp(row["effective_from"], "document_versions.effective_from"):
                _fail("Document published_at may not be after effective_from")
    version_by_key = _build_index(validated["document_versions"], ("document_id", "version"))
    for row in validated["document_chunks"]:
        if row["chunk_index"] < 0:
            _fail("Document chunk indexes must be nonnegative")
        parent = version_by_key[(row["document_id"], row["document_version"])]
        if row["content"] not in parent["full_text"]:
            _fail("Every document chunk must be a verbatim parent-text substring")

    if validated["retrieval_indexes"] or validated["chunk_embeddings"]:
        _fail("Normal fixtures must not contain retrieval indexes or embeddings")

    manifest_counts = _expect_object(manifest["table_counts"], "manifest.table_counts")
    if set(manifest_counts) != set(TABLE_SPECS):
        _fail("manifest.table_counts must list exactly all Alembic 001 tables")
    for table, rows in validated.items():
        count = manifest_counts[table]
        if not _is_int(count) or count < 0 or count != len(rows):
            _fail(f"Manifest row count is incorrect for {table}")

    _validate_scenarios(dataset)
    _validate_invalid_cases(dataset)
    scenario_counts = _expect_object(manifest["scenario_counts"], "manifest.scenario_counts")
    expected_scenario_keys = {"record_scenarios", "document_scenarios", "invalid_records"}
    if set(scenario_counts) != expected_scenario_keys:
        _fail("manifest.scenario_counts has incorrect keys")
    expected_counts = {
        "record_scenarios": len(dataset["record_scenarios"]),
        "document_scenarios": len(dataset["document_scenarios"]),
        "invalid_records": len(dataset["invalid_records"]),
    }
    for key, expected in expected_counts.items():
        if not _is_int(scenario_counts[key]) or scenario_counts[key] != expected:
            _fail(f"Manifest scenario count is incorrect for {key}")


def _topological_rows(rows: Sequence[dict[str, Any]], id_column: str, parent_column: str) -> list[dict[str, Any]]:
    pending = {row[id_column]: row for row in rows}
    emitted: set[Any] = set()
    ordered: list[dict[str, Any]] = []
    while pending:
        ready = sorted(
            (key for key, row in pending.items() if row[parent_column] is None or row[parent_column] in emitted),
            key=str,
        )
        if not ready:
            _fail(f"Could not order self-referencing rows for {parent_column}")
        for key in ready:
            ordered.append(pending.pop(key))
            emitted.add(key)
    return ordered


def _db_value(table: str, column: str, value: Any) -> Any:
    if value is None:
        return None
    spec = TABLE_SPECS[table]
    if column in spec.timestamp_columns:
        return _parse_timestamp(value, f"{table}.{column}")
    if column in spec.date_columns:
        return _parse_date(value, f"{table}.{column}")
    return value


def _normalize(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value
        return value.astimezone(timezone.utc)
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, float) and value == 0:
        return 0.0
    return value


def _assert_revision(cursor: Any) -> None:
    cursor.execute("SELECT to_regclass('public.alembic_version') IS NOT NULL")
    row = cursor.fetchone()
    if row is None or row[0] is not True:
        _fail("Database is missing public.alembic_version; apply Alembic revision 001 first")
    cursor.execute("SELECT version_num FROM public.alembic_version ORDER BY version_num")
    versions = [row[0] for row in cursor.fetchall()]
    if len(versions) != 1 or versions[0] not in SUPPORTED_DB_REVISIONS:
        _fail("Database Alembic revision must be one of the supported additive revisions 001 through 005")


def _select_fixture_row(cursor: Any, table: str, row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    from psycopg import sql

    spec = TABLE_SPECS[table]
    query = sql.SQL("SELECT {} FROM {}.{} WHERE {}").format(
        sql.SQL(", ").join(sql.Identifier(column) for column in spec.columns),
        sql.Identifier("clinic"), sql.Identifier(table),
        sql.SQL(" AND ").join(
            sql.SQL("{} = %s").format(sql.Identifier(column)) for column in spec.primary_key
        ),
    )
    values = [_db_value(table, column, row[column]) for column in spec.primary_key]
    cursor.execute(query, values)
    return cursor.fetchone()


def _row_matches(table: str, fixture: Mapping[str, Any], database_row: Sequence[Any]) -> bool:
    spec = TABLE_SPECS[table]
    expected = [_normalize(_db_value(table, column, fixture[column])) for column in spec.columns]
    actual = [_normalize(value) for value in database_row]
    return expected == actual


def _insert_or_compare(cursor: Any, table: str, row: Mapping[str, Any]) -> str:
    from psycopg import sql

    spec = TABLE_SPECS[table]
    columns = spec.columns
    query = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({}) ON CONFLICT DO NOTHING").format(
        sql.Identifier("clinic"), sql.Identifier(table),
        sql.SQL(", ").join(sql.Identifier(column) for column in columns),
        sql.SQL(", ").join(sql.Placeholder() for _ in columns),
    )
    values = [_db_value(table, column, row[column]) for column in columns]
    cursor.execute(query, values)
    if cursor.rowcount == 1:
        return "inserted"
    existing = _select_fixture_row(cursor, table, row)
    if existing is None:
        _fail(f"Insert conflict in {table} did not match the fixture primary key")
    if not _row_matches(table, row, existing):
        _fail(f"Existing {table} row differs from the deterministic fixture; no data was changed")
    return "skipped"


def _where_sql(table: str, where: Mapping[str, Any]) -> tuple[Any, list[Any]]:
    from psycopg import sql

    clauses: list[Any] = []
    values: list[Any] = []
    for column, condition in where.items():
        identifier = sql.Identifier(column)
        conditions = condition if isinstance(condition, dict) else {"eq": condition}
        for operator, operand in conditions.items():
            if operator == "eq":
                if operand is None:
                    clauses.append(sql.SQL("{} IS NULL").format(identifier))
                else:
                    clauses.append(sql.SQL("{} = %s").format(identifier))
                    values.append(_db_value(table, column, operand) if table != "authorized_cases" else operand)
            elif operator == "in":
                placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in operand)
                clauses.append(sql.SQL("{} IN ({})").format(identifier, placeholders))
                values.extend(_db_value(table, column, value) if table != "authorized_cases" else value for value in operand)
            elif operator == "gte":
                clauses.append(sql.SQL("{} >= %s").format(identifier))
                values.append(_db_value(table, column, operand) if table != "authorized_cases" else operand)
            elif operator == "lt":
                clauses.append(sql.SQL("{} < %s").format(identifier))
                values.append(_db_value(table, column, operand) if table != "authorized_cases" else operand)
    return (sql.SQL(" AND ").join(clauses) if clauses else sql.SQL("TRUE"), values)


def _run_db_checks(cursor: Any, dataset: Mapping[str, Any]) -> int:
    from psycopg import sql

    checked = 0
    for scenario_id, check in _validate_scenarios(dataset):
        table = check["table"]
        where_sql, values = _where_sql(table, check["where"])
        query = sql.SQL("SELECT count(*) FROM {}.{} WHERE {}").format(
            sql.Identifier("clinic"), sql.Identifier(table), where_sql
        )
        cursor.execute(query, values)
        actual = cursor.fetchone()[0]
        if actual != check["count"]:
            _fail(f"Database count check failed for scenario {scenario_id}")
        checked += 1
    return checked


def _verify_fixture_rows(cursor: Any, dataset: Mapping[str, Any]) -> int:
    matched = 0
    for table in INSERT_ORDER:
        for row in dataset["tables"][table]:
            existing = _select_fixture_row(cursor, table, row)
            if existing is None:
                _fail(f"Database is missing a seeded {table} fixture row")
            if not _row_matches(table, row, existing):
                _fail(f"A seeded {table} row differs from the deterministic fixture")
            matched += 1
    return matched


def _load_db_environment() -> dict[str, Any]:
    # Called only by explicit database modes. Offline validation never reads .env.
    from dotenv import load_dotenv

    env_path = ROOT / ".env"
    if not env_path.is_file():
        _fail("Missing .env; create private local database settings first")
    load_dotenv(env_path)
    required = ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")
    if any(not os.getenv(key) for key in required):
        _fail("Required local database settings are missing")
    try:
        port = int(os.getenv("DB_PORT", "5432"))
    except ValueError:
        _fail("DB_PORT must be an integer from 1 to 65535")
    if not 1 <= port <= 65535:
        _fail("DB_PORT must be an integer from 1 to 65535")
    return {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "port": port,
        "dbname": os.environ["POSTGRES_DB"],
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "connect_timeout": 5,
    }


def _connect() -> Any:
    import psycopg

    try:
        return psycopg.connect(**_load_db_environment(), autocommit=True)
    except SeedError:
        raise
    except (psycopg.Error, OSError, ValueError):
        _fail("Database connection failed; check local settings and database status privately")


def apply_dataset(dataset: Mapping[str, Any]) -> tuple[int, int, int]:
    """Insert missing fixture rows atomically, without updating existing data."""
    import psycopg

    connection = _connect()
    inserted = skipped = checks = 0
    try:
        with connection.cursor() as cursor:
            cursor.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
            try:
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", (ADVISORY_LOCK_KEY,))
                _assert_revision(cursor)
                for table in INSERT_ORDER:
                    rows = dataset["tables"][table]
                    if table == "reviewed_reports":
                        rows = _topological_rows(rows, "report_id", "supersedes_report_id")
                    elif table == "appointments":
                        rows = _topological_rows(rows, "appointment_id", "replaces_appointment_id")
                    for row in rows:
                        result = _insert_or_compare(cursor, table, row)
                        inserted += result == "inserted"
                        skipped += result == "skipped"
                checks = _run_db_checks(cursor, dataset)
                cursor.execute("COMMIT")
            except Exception:
                cursor.execute("ROLLBACK")
                raise
    except SeedError:
        raise
    except psycopg.Error:
        _fail("Database apply failed; run --verify-db to establish the database state before retrying. No existing rows were overwritten and credentials were not printed")
    finally:
        connection.close()
    return inserted, skipped, checks


def verify_database(dataset: Mapping[str, Any]) -> tuple[int, int]:
    """Read-only verification of all fixture rows and scenario count checks."""
    import psycopg

    connection = _connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            try:
                _assert_revision(cursor)
                matched = _verify_fixture_rows(cursor, dataset)
                checks = _run_db_checks(cursor, dataset)
            finally:
                cursor.execute("ROLLBACK")
    except SeedError:
        raise
    except psycopg.Error:
        _fail("Database verification failed; credentials and existing row values were not printed")
    finally:
        connection.close()
    return matched, checks


def _invalid_candidate(dataset: Mapping[str, Any], invalid: Mapping[str, Any]) -> dict[str, Any]:
    matches = [
        row for row in dataset["tables"][invalid["table"]]
        if all(row[column] == value for column, value in invalid["base_key"].items())
    ]
    candidate = copy.deepcopy(matches[0])
    candidate.update(copy.deepcopy(invalid["patch"]))
    return candidate


def check_invalid_database(dataset: Mapping[str, Any]) -> int:
    """Confirm invalid candidates fail at exact SQLSTATEs; persist nothing."""
    import psycopg
    from psycopg import sql

    connection = _connect()
    checked = 0
    try:
        with connection.cursor() as cursor:
            cursor.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
            try:
                _assert_revision(cursor)
                _verify_fixture_rows(cursor, dataset)
                _run_db_checks(cursor, dataset)
                for index, invalid in enumerate(dataset["invalid_records"]):
                    table = invalid["table"]
                    spec = TABLE_SPECS[table]
                    candidate = _invalid_candidate(dataset, invalid)
                    savepoint = sql.Identifier(f"invalid_fixture_{index}")
                    cursor.execute(sql.SQL("SAVEPOINT {}").format(savepoint))
                    observed: str | None = None
                    succeeded = False
                    try:
                        query = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({})").format(
                            sql.Identifier("clinic"), sql.Identifier(table),
                            sql.SQL(", ").join(sql.Identifier(column) for column in spec.columns),
                            sql.SQL(", ").join(sql.Placeholder() for _ in spec.columns),
                        )
                        cursor.execute(query, [_db_value(table, column, candidate[column]) for column in spec.columns])
                        succeeded = True
                    except psycopg.Error as exc:
                        observed = exc.sqlstate
                    finally:
                        cursor.execute(sql.SQL("ROLLBACK TO SAVEPOINT {}").format(savepoint))
                        cursor.execute(sql.SQL("RELEASE SAVEPOINT {}").format(savepoint))
                    if succeeded:
                        _fail(f"Invalid fixture {invalid['id']} was unexpectedly accepted; it was rolled back")
                    if observed != invalid["expected_sqlstate"]:
                        _fail(f"Invalid fixture {invalid['id']} returned SQLSTATE {observed or 'unknown'}, expected {invalid['expected_sqlstate']}")
                    checked += 1
            finally:
                cursor.execute("ROLLBACK")
    except SeedError:
        raise
    except psycopg.Error:
        _fail("Invalid-fixture database check failed; all attempts were rolled back")
    finally:
        connection.close()
    return checked


def _total_rows(dataset: Mapping[str, Any]) -> int:
    return sum(len(rows) for rows in dataset["tables"].values())


def _print_counts(dataset: Mapping[str, Any]) -> None:
    for table in INSERT_ORDER:
        print(f"  {table}: {len(dataset['tables'][table])}")
    print(f"  record_scenarios: {len(dataset['record_scenarios'])}")
    print(f"  document_scenarios: {len(dataset['document_scenarios'])}")
    print(f"  invalid_records: {len(dataset['invalid_records'])}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate or load deterministic Alembic-001 fixtures")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="offline validation only (default)")
    modes.add_argument("--apply", action="store_true", help="atomically insert missing normal rows")
    modes.add_argument("--verify-db", action="store_true", help="read-only database verification")
    modes.add_argument("--check-invalid-db", action="store_true", help="rollback-only invalid-row SQLSTATE checks")
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR, help="fixture directory (default: project fixtures)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        dataset = load_dataset(args.fixture_dir)
        validate_dataset(dataset)
        if args.apply:
            inserted, skipped, checks = apply_dataset(dataset)
            print(f"Apply OK: inserted {inserted}, identical existing {skipped}, database checks {checks}.")
        elif args.verify_db:
            matched, checks = verify_database(dataset)
            print(f"Database verification OK: matched {matched} fixture rows; checks {checks}.")
        elif args.check_invalid_db:
            checked = check_invalid_database(dataset)
            print(f"Invalid database checks OK: {checked} attempts rejected at exact expected SQLSTATEs; all rolled back.")
        else:
            print(f"Dry run OK: deterministic offline validation passed for {_total_rows(dataset)} normal rows.")
            _print_counts(dataset)
            print("Embeddings/indexes were not generated; retrieval_indexes and chunk_embeddings fixtures are empty.")
        return 0
    except SeedError as exc:
        print(f"Seed failed: {exc}", file=sys.stderr)
        return 1
    except Exception:
        # Deliberately avoid exception reprs: driver errors can include connection
        # details or existing values. The detailed cause remains available to tests
        # by calling the public helpers directly for expected SeedError failures.
        print("Seed failed unexpectedly; no credentials or existing row values were printed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
