"""Real SQLite-backed native handlers for the two observed wrong-time failures."""
import asyncio
from datetime import time
from types import SimpleNamespace

import pytest

from healthcare_voice_agent.booking import tables as t
from healthcare_voice_agent.demo import tools
from healthcare_voice_agent.demo.runtime import DemoRuntime
from tests.integration.test_frontdesk_booking import clinic, good, rule, DAY


def offered(clinic, kind, hour, minute):
    result = good(clinic.search(appointment_type=kind,
        earliest_start_time=f'{hour:02}:{minute:02}', latest_start_time=f'{hour:02}:{minute:02}'))
    assert len(result['slots']) == 1
    return result['slots'][0]


def native_call(clinic, handler, args):
    session = DemoRuntime(clinic.engine, clinic.service, clinic.context,
        {'timezone': 'Asia/Kolkata'}, final_turn_epoch=0)
    result, node = asyncio.run(handler(args, SimpleNamespace(state={'session': session})))
    assert node is None
    return result, session


@pytest.mark.parametrize('selected_minute,expected_status', [(0, 'error'), (30, 'ok')])
def test_physio_ten_thirty_requires_ten_thirty_start(clinic, selected_minute, expected_status):
    clinic.service.publish_schedule('CLINIC', (rule(kind='physiotherapy'),), DAY, DAY)
    selected = offered(clinic, 'physiotherapy', 10, selected_minute)
    result, session = native_call(clinic, tools.book_slot, {
        'slot_id': selected['slot_id'], 'requested_starts_at': f'{DAY}T10:30:00+05:30'})
    assert result['status'] == expected_status
    if expected_status == 'error':
        assert result['error']['code'] == 'SLOT_TIME_MISMATCH'
        assert clinic.count(t.appointments) == clinic.count(t.drafts) == clinic.count(t.holds) == 0
        assert session.pending is None
    else:
        assert result['data']['appointment']['starts_at'] == f'{DAY}T10:30:00+05:30'
        assert clinic.count(t.appointments) == 1


@pytest.mark.parametrize('selected_minute,expected_status', [(0, 'error'), (30, 'ok')])
def test_imaging_three_thirty_reschedule_keeps_source_on_mismatch(clinic, selected_minute, expected_status):
    clinic.service.publish_schedule('CLINIC', (rule(kind='imaging', closes=time(17)),), DAY, DAY)
    source_slot = offered(clinic, 'imaging', 10, 30)
    source_draft = good(clinic.prepare(source_slot['slot_id']))
    clinic.confirm(source_draft)
    good(clinic.book(source_draft))
    source = clinic.service.list_bookings(clinic.context)[0]
    selected = offered(clinic, 'imaging', 15, selected_minute)
    result, session = native_call(clinic, tools.reschedule_appointment, {
        'appointment_id': source['appointment_id'], 'record_version': source['record_version'],
        'slot_id': selected['slot_id'], 'requested_starts_at': f'{DAY}T15:30:00+05:30'})
    assert result['status'] == expected_status
    rows = clinic.rows(t.appointments)
    if expected_status == 'error':
        assert result['error']['code'] == 'SLOT_TIME_MISMATCH'
        assert len(rows) == 1 and rows[0]['status'] == 'confirmed'
        assert rows[0]['record_version'] == source['record_version']
        assert clinic.count(t.drafts) == 1
        assert clinic.count(t.reschedules) == clinic.count(t.holds) == 0
        assert session.pending is None
    else:
        assert result['data']['appointment']['starts_at'] == f'{DAY}T15:30:00+05:30'
        assert sorted(row['status'] for row in rows) == ['cancelled', 'confirmed']
        assert clinic.count(t.reschedules) == 1
