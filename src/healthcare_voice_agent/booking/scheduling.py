"""Trusted backend schedule configuration. Never expose these fields as LLM args."""
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

APPOINTMENT_TYPES = frozenset({"initial_consultation", "fracture_follow_up", "imaging", "physiotherapy", "other"})


@dataclass(frozen=True)
class ScheduleRule:
    appointment_type: str
    clinician_id: str
    weekdays: tuple[int, ...]  # Monday=0, Sunday=6
    opens_at: time
    closes_at: time
    duration_minutes: int
    location: str

    def __post_init__(self):
        if self.appointment_type not in APPOINTMENT_TYPES:
            raise ValueError("Unsupported appointment type")
        if not self.clinician_id.strip() or not self.location.strip():
            raise ValueError("Clinician and location are required")
        if not self.weekdays or any(type(day) is not int or not 0 <= day <= 6 for day in self.weekdays):
            raise ValueError("Weekdays must be integers 0 through 6")
        if type(self.duration_minutes) is not int or not 5 <= self.duration_minutes <= 240:
            raise ValueError("Duration must be 5 through 240 minutes")
        if (self.opens_at.tzinfo or self.closes_at.tzinfo or self.opens_at >= self.closes_at
                or self.opens_at.second or self.opens_at.microsecond
                or self.closes_at.second or self.closes_at.microsecond):
            raise ValueError("Opening hours must be same-day, minute-precision local times")


def _unique_local(value: datetime, zone: ZoneInfo) -> datetime | None:
    first = value.replace(tzinfo=zone, fold=0)
    second = value.replace(tzinfo=zone, fold=1)
    if first.utcoffset() != second.utcoffset():
        return None  # Ambiguous or nonexistent DST times are never guessed.
    if first.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) != value:
        return None
    return first


def generate_slots(rules: tuple[ScheduleRule, ...], start_date: date, end_date: date, timezone_name: str):
    """Inclusive date range, fixed durations; skip DST-ambiguous/gap intervals."""
    zone = ZoneInfo(timezone_name)
    if start_date > end_date or (end_date - start_date).days >= 366:
        raise ValueError("Schedule publication must cover 1 through 366 days")
    current = start_date
    while current <= end_date:
        for rule in rules:
            if current.weekday() not in rule.weekdays:
                continue
            duration = timedelta(minutes=rule.duration_minutes)
            start = datetime.combine(current, rule.opens_at)
            close = datetime.combine(current, rule.closes_at)
            while start + duration <= close:
                aware_start = _unique_local(start, zone)
                aware_end = _unique_local(start + duration, zone)
                if (aware_start and aware_end and
                    aware_end.astimezone(timezone.utc) - aware_start.astimezone(timezone.utc) == duration):
                    yield {"appointment_type": rule.appointment_type,
                           "clinician_id": rule.clinician_id, "location": rule.location,
                           "starts_at": aware_start.astimezone(timezone.utc),
                           "ends_at": aware_end.astimezone(timezone.utc)}
                start += duration
        if current == end_date:
            break
        current += timedelta(days=1)
