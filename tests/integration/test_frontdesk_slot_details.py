"""Offline persisted-slot read checks; no providers or live database."""
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import update

from healthcare_voice_agent.booking import tables as t
from healthcare_voice_agent.booking.service import BookingError
from tests.integration.test_frontdesk_booking import clinic, good


def test_offered_slot_details_are_local_and_do_not_prepare(clinic):
    offered = good(clinic.search())['slots'][0]
    before = {table.name: clinic.count(table) for table in (
        t.appointments, t.drafts, t.holds, t.allocations, t.offers)}
    actual = clinic.service.get_slot_details(clinic.context, slot_id=offered['slot_id'])
    assert actual == offered
    assert actual['starts_at'].endswith('+05:30')
    assert {table.name: clinic.count(table) for table in (
        t.appointments, t.drafts, t.holds, t.allocations, t.offers)} == before


def test_unoffered_slot_is_not_exposed(clinic):
    with pytest.raises(BookingError) as error:
        clinic.service.get_slot_details(clinic.context, slot_id=clinic.slot_ids[0])
    assert error.value.code == 'SLOT_NOT_OFFERED'
    assert clinic.count(t.drafts) == 0


def test_other_session_cannot_reuse_offer(clinic):
    offered = good(clinic.search())['slots'][0]
    other = clinic.new_context('CASE-A')
    with pytest.raises(BookingError) as error:
        clinic.service.get_slot_details(other, slot_id=offered['slot_id'])
    assert error.value.code == 'SLOT_NOT_OFFERED'


def test_stale_context_rejected(clinic):
    offered = good(clinic.search())['slots'][0]
    clinic.service.advance_conversation(clinic.context)
    with pytest.raises(BookingError) as error:
        clinic.service.get_slot_details(clinic.context, slot_id=offered['slot_id'])
    assert error.value.code == 'SESSION_CONTEXT_MISMATCH'


def test_no_selected_case_rejected(clinic):
    context = clinic.new_context(None)
    with pytest.raises(BookingError) as error:
        clinic.service.get_slot_details(context, slot_id=clinic.slot_ids[0])
    assert error.value.code == 'IDENTITY_UNRESOLVED'


@pytest.mark.parametrize('slot_id', [None, '', 'not-a-uuid', True])
def test_invalid_slot_identifier_is_safe(clinic, slot_id):
    with pytest.raises(BookingError) as error:
        clinic.service.get_slot_details(clinic.context, slot_id=slot_id)
    assert error.value.code == 'INVALID_ARGUMENT'


def test_unknown_slot_is_not_found(clinic):
    with pytest.raises(BookingError) as error:
        clinic.service.get_slot_details(clinic.context, slot_id=uuid4())
    assert error.value.code == 'RECORD_NOT_FOUND'


def test_different_clinic_slot_is_not_exposed(clinic):
    offered = good(clinic.search())['slots'][0]
    clinic.service.register_clinic('OTHER', 'Asia/Kolkata')
    with clinic.engine.begin() as connection:
        connection.execute(update(t.slots).where(t.slots.c.slot_id == clinic.slot_ids[0])
                           .values(clinic_id='OTHER'))
    # Select the moved identity directly; no cross-clinic calendar data leaks.
    with pytest.raises(BookingError) as error:
        clinic.service.get_slot_details(clinic.context, slot_id=clinic.slot_ids[0])
    assert error.value.code == 'RECORD_NOT_FOUND'
