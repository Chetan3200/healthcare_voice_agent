"""Backend schedule rules and compatibility checks; no database access."""
from datetime import date, time, timedelta, timezone
from types import SimpleNamespace

import pytest

from healthcare_voice_agent.booking.scheduling import ScheduleRule, generate_slots
from scripts import seed


def rule(**changes):
    return ScheduleRule(**{"appointment_type": "fracture_follow_up", "clinician_id": "SYN-DR",
        "weekdays": (0, 1, 2, 3, 4, 5, 6), "opens_at": time(9), "closes_at": time(10),
        "duration_minutes": 30, "location": "Synthetic room", **changes})


def test_backend_hours_duration_eligibility_and_inclusive_days():
    rows = list(generate_slots((rule(),), date(2026, 10, 5), date(2026, 10, 6), "Asia/Kolkata"))
    assert len(rows) == 4
    assert all(row["clinician_id"] == "SYN-DR" for row in rows)
    assert all(row["ends_at"] - row["starts_at"] == timedelta(minutes=30) for row in rows)
    assert rows[0]["starts_at"].hour == 3 and rows[0]["starts_at"].minute == 30
    assert list(generate_slots((rule(weekdays=(6,)),), date(2026, 10, 5), date(2026, 10, 5), "Asia/Kolkata")) == []


@pytest.mark.parametrize("day", [date(2026, 3, 8), date(2026, 11, 1)])
def test_dst_gaps_and_ambiguous_slots_are_not_guessed(day):
    rows = list(generate_slots((rule(opens_at=time(0), closes_at=time(4)),), day, day, "America/New_York"))
    assert rows
    from zoneinfo import ZoneInfo
    zone = ZoneInfo("America/New_York")
    for row in rows:
        assert row["ends_at"] - row["starts_at"] == timedelta(minutes=30)
        for key in ("starts_at", "ends_at"):
            local = row[key].astimezone(zone)
            assert local.replace(fold=0).utcoffset() == local.replace(fold=1).utcoffset()


@pytest.mark.parametrize("changes", [
    {"duration_minutes": True}, {"duration_minutes": 0}, {"duration_minutes": 241},
    {"opens_at": time(19)}, {"closes_at": time(8)}, {"weekdays": (7,)},
    {"weekdays": (False,)}, {"weekdays": ()}, {"location": " "},
    {"appointment_type": "invented"}, {"opens_at": time(9, 0, 1)},
    {"opens_at": time(9, tzinfo=timezone.utc)},
])
def test_invalid_backend_schedule_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        rule(**changes)


@pytest.mark.parametrize("versions", [["001"], ["002"], ["003"], ["004"], ["005"]])
def test_existing_fixture_format_is_compatible_with_additive_booking_revision(versions):
    cursor = SimpleNamespace(execute=lambda *args: None, fetchone=lambda: (True,),
        fetchall=lambda: [(value,) for value in versions])
    seed._assert_revision(cursor)
    assert seed.SCHEMA_REVISION == "001"


@pytest.mark.parametrize("versions", [[], ["006"], ["001", "002"], ["unrecognized"]])
def test_seed_does_not_assume_unknown_migrations_are_compatible(versions):
    cursor = SimpleNamespace(execute=lambda *args: None, fetchone=lambda: (True,),
        fetchall=lambda: [(value,) for value in versions])
    with pytest.raises(seed.SeedError):
        seed._assert_revision(cursor)
