"""Offline schema/export examples; not backend authorization or booking tests."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from healthcare_voice_agent.tools.contracts import TOOLS
from healthcare_voice_agent.tools.frontdesk_contracts import (
    FRONTDESK_TOOLS, AvailableSlot, AvailableSlotsData, FindAvailableSlotsArgs,
    PreparedBookingData, export_frontdesk_contracts, validate_example_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "docs/contracts/frontdesk-tool-examples.json"
BUNDLE = json.loads(EXAMPLES.read_text())
BY_ID = {item["example_id"]: item for item in BUNDLE["examples"]}


def validate(model, data):
    return model.model_validate_json(json.dumps(data))


def find_args(**overrides):
    return {**BY_ID["available_slots"]["arguments"], **overrides}


def slot(**overrides):
    original = BY_ID["available_slots"]["expected_response"]["data"]["slots"][0]
    return {**copy.deepcopy(original), **overrides}


def test_four_new_contracts_do_not_change_existing_five():
    assert set(FRONTDESK_TOOLS) == {"find_available_slots", "prepare_booking", "book_appointment", "get_booking_status"}
    assert len(TOOLS) == 5
    assert not set(TOOLS).intersection(FRONTDESK_TOOLS)
    assert {name for name, tool in FRONTDESK_TOOLS.items() if tool.read_only} == {"get_booking_status"}


def test_all_companion_examples_validate():
    assert validate_example_bundle(EXAMPLES) == 13


def test_checked_in_schema_matches_python_export():
    exported = json.loads((ROOT / "docs/contracts/frontdesk-tool-contracts.json").read_text())
    assert exported == export_frontdesk_contracts()


def test_every_input_field_is_required_even_when_nullable():
    for tool in FRONTDESK_TOOLS.values():
        schema = tool.export()["input_schema"]
        assert set(schema["required"]) == set(schema["properties"])
        assert schema["additionalProperties"] is False
    for field in ("earliest_start_time", "latest_start_time", "clinician_id"):
        payload = find_args()
        payload.pop(field)
        with pytest.raises(ValidationError):
            validate(FindAvailableSlotsArgs, payload)
    validate(FindAvailableSlotsArgs, find_args(earliest_start_time=None, latest_start_time=None, clinician_id=None))


@pytest.mark.parametrize("overrides", [
    {"start_date": "2026-2-01"}, {"start_date": "2026-02-29"},
    {"start_date": "2026-04-31"}, {"start_date": "2026-10-01T00:00:00Z"},
    {"end_date": "2026-09-30"}, {"end_date": "2026-11-01"},
    {"earliest_start_time": "9:00"}, {"earliest_start_time": "24:00"},
    {"earliest_start_time": "09:00:00"}, {"latest_start_time": "12:60"},
    {"earliest_start_time": "17:00", "latest_start_time": "09:00"},
    {"max_results": 0}, {"max_results": 101}, {"max_results": True},
    {"max_results": "3"}, {"clinician_id": 123}, {"appointment_type": "unknown"},
    {"extra_setting": True},
])
def test_invalid_search_fields_are_not_coerced(overrides):
    with pytest.raises(ValidationError):
        validate(FindAvailableSlotsArgs, find_args(**overrides))


def test_date_and_daily_time_boundaries_are_inclusive():
    result = validate(FindAvailableSlotsArgs, find_args(end_date="2026-10-31", earliest_start_time="09:00", latest_start_time="09:00"))
    assert result.earliest_start_time == result.latest_start_time


@pytest.mark.parametrize("tool,example", [("prepare_booking", "prepared_not_booked"), ("book_appointment", "confirmed_booking"), ("get_booking_status", "pending_lookup_not_booking_success")])
def test_extra_controller_fields_are_not_llm_arguments(tool, example):
    args = BY_ID[example]["arguments"]
    for field in ("confirmed", "session_id", "confirmation_event_id", "case_context_version"):
        with pytest.raises(ValidationError):
            FRONTDESK_TOOLS[tool].validate_arguments_json(json.dumps({**args, field: True}))


def test_new_opaque_identifiers_are_uuid_not_arbitrary_strings():
    with pytest.raises(ValidationError):
        FRONTDESK_TOOLS["prepare_booking"].validate_arguments_json(json.dumps({"case_id": "1042", "slot_id": "invented-slot"}))


@pytest.mark.parametrize("overrides", [
    {"timezone": "Moon/Imaginary"}, {"location": "   "},
    {"starts_at": "2026-10-01T09:00:00"},
    {"starts_at": "2026-10-01T09:00:00Z"},
    {"ends_at": "2026-10-01T09:00:00+05:30"},
    {"ends_at": "2026-10-01T08:30:00+05:30"},
])
def test_slot_requires_valid_zone_aware_forward_interval(overrides):
    with pytest.raises(ValidationError):
        validate(AvailableSlot, slot(**overrides))


def test_dst_fold_compares_instants_not_just_wall_clock():
    result = validate(AvailableSlot, slot(timezone="America/New_York", starts_at="2026-11-01T01:30:00-04:00", ends_at="2026-11-01T01:15:00-05:00"))
    assert (result.ends_at - result.starts_at).total_seconds() == 45 * 60
    with pytest.raises(ValidationError):
        validate(AvailableSlot, slot(timezone="America/New_York", starts_at="2026-03-08T02:30:00-05:00", ends_at="2026-03-08T04:00:00-04:00"))


def test_slot_lists_are_unique_ordered_and_bounded():
    data = copy.deepcopy(BY_ID["available_slots"]["expected_response"]["data"])
    for slots in ([slot(), slot()], [slot(starts_at="2026-10-02T09:00:00+05:30", ends_at="2026-10-02T09:30:00+05:30", slot_id="00000000-0000-4000-8000-000000000999"), slot()], [slot()] * 6):
        with pytest.raises(ValidationError):
            validate(AvailableSlotsData, {**data, "slots": slots})
    with pytest.raises(ValidationError):
        validate(AvailableSlotsData, {**data, "slots": [], "has_more": True})
    with pytest.raises(ValidationError):
        validate(AvailableSlotsData, {**data, "as_of": "2026-10-02T10:00:00Z"})


def test_prepared_hold_cannot_outlast_appointment():
    data = copy.deepcopy(BY_ID["prepared_not_booked"]["expected_response"]["data"])
    with pytest.raises(ValidationError):
        validate(PreparedBookingData, {**data, "hold_expires_at": "2026-10-02T10:00:00Z"})
    with pytest.raises(ValidationError):
        validate(PreparedBookingData, {**data, "draft_id": data["booking_request_id"]})


def test_status_branches_cannot_claim_booked_without_an_appointment():
    tool = FRONTDESK_TOOLS["get_booking_status"]
    response = copy.deepcopy(BY_ID["pending_lookup_not_booking_success"]["expected_response"])
    response["data"]["booking_state"] = "booked"
    with pytest.raises(ValidationError):
        tool.validate_response_json(json.dumps(response))
    response["data"]["booking_state"] = "failed"
    with pytest.raises(ValidationError):
        tool.validate_response_json(json.dumps(response))


def test_pending_expired_boundary_and_booked_persistence():
    tool = FRONTDESK_TOOLS["get_booking_status"]
    response = copy.deepcopy(BY_ID["expired_at_exact_boundary"]["expected_response"])
    response["data"]["booking_state"] = "pending"
    with pytest.raises(ValidationError):
        tool.validate_response_json(json.dumps(response))
    response["data"]["booking_state"] = "expired"
    response["data"]["as_of"] = "2026-09-30T10:01:59Z"
    with pytest.raises(ValidationError):
        tool.validate_response_json(json.dumps(response))
    committed = tool.validate_response_json(json.dumps(BY_ID["booked_after_old_hold_expiry"]["expected_response"]))
    assert committed.status == "ok" and committed.data.booking_state == "booked"


def test_response_metadata_and_error_envelope_are_not_interchangeable():
    response = copy.deepcopy(BY_ID["missing_confirmation"]["expected_response"])
    response["meta"]["tool_name"] = "prepare_booking"
    with pytest.raises(ValidationError):
        FRONTDESK_TOOLS["book_appointment"].validate_response_json(json.dumps(response))
    response = copy.deepcopy(BY_ID["missing_confirmation"]["expected_response"])
    response["data"] = BY_ID["confirmed_booking"]["expected_response"]["data"]
    with pytest.raises(ValidationError):
        FRONTDESK_TOOLS["book_appointment"].validate_response_json(json.dumps(response))
