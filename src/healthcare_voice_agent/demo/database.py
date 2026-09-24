"""Fixed-target, opt-in PostgreSQL demo setup. Never reads the clinic .env.

The compose project, port, database, user and volume are separate from the clinic.
Alembic renders SQL offline; that SQL is executed ONLY on the verified demo
connection. No migration-env connection override or ambient DB settings are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
import io
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import URL, create_engine, text

ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = ROOT / ".demo" / "frontdesk"
COMPOSE_PROJECT = "fracture-frontdesk-demo"
DATABASE_NAME = "frontdesk_synthetic_demo"
DATABASE_USER = "frontdesk_demo_owner"
DATABASE_HOST = "127.0.0.1"
DATABASE_PORT = 55432
DEMO_CLINIC_ID = "DEMO-CLINIC"
DEMO_CASE_ID = "1042"
DEMO_STAFF_ID = "SYN-USER-STAFF"
DEMO_TIMEZONE = "Asia/Kolkata"
MARKER_PREFIX = "healthcare-voice-agent:frontdesk-synthetic-demo:v1:"


class DemoDatabaseError(RuntimeError):
    """A safe message without connection details or database credentials."""


@dataclass(frozen=True)
class DemoSettings:
    instance_id: UUID
    state_dir: Path = field(repr=False)
    password: str = field(repr=False)

    @property
    def marker(self):
        return MARKER_PREFIX + str(self.instance_id)

    @property
    def password_file(self):
        return self.state_dir / "db-password.txt"

    def connection_kwargs(self):
        return {"host": DATABASE_HOST, "port": DATABASE_PORT, "dbname": DATABASE_NAME,
                "user": DATABASE_USER, "password": self.password, "connect_timeout": 5}


def _read_private(path: Path) -> str:
    if path.is_symlink():
        raise DemoDatabaseError("Demo settings must not be symlinks.")
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise DemoDatabaseError("Demo settings must be owner-only files (mode 600).")
        return path.read_text(encoding="utf-8")
    except OSError:
        raise DemoDatabaseError("Demo settings are missing. Run setup_frontdesk_demo.py --prepare.") from None


def _write_private(path: Path, value: str):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def load_demo_settings(state_dir: Path | None = None) -> DemoSettings:
    """Read only the private demo files. Never fall back to main DB credentials."""
    directory = Path(state_dir) if state_dir is not None else STATE_DIR
    try:
        values = json.loads(_read_private(directory / "settings.json"))
        expected = {"schema_version", "instance_id", "project", "host", "port", "database", "user"}
        if set(values) != expected or values["schema_version"] != 1:
            raise ValueError
        fixed = (values["project"], values["host"], values["port"], values["database"], values["user"])
        if fixed != (COMPOSE_PROJECT, DATABASE_HOST, DATABASE_PORT, DATABASE_NAME, DATABASE_USER):
            raise ValueError
        instance_id = UUID(values["instance_id"])
        password = _read_private(directory / "db-password.txt").strip()
        if len(password) < 32 or not all(c.isalnum() or c in "-_" for c in password):
            raise ValueError
        return DemoSettings(instance_id, directory.resolve(), password)
    except (ValueError, TypeError, KeyError):
        raise DemoDatabaseError("Demo settings are invalid or point outside the isolated target.") from None


def create_demo_settings(state_dir: Path | None = None) -> DemoSettings:
    directory = Path(state_dir) if state_dir is not None else STATE_DIR
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = directory / "settings.json"
    password_file = directory / "db-password.txt"
    if manifest.exists() or password_file.exists():
        # A partial setup is not permission to replace a password and orphan a volume.
        return load_demo_settings(directory)
    values = {"schema_version": 1, "instance_id": str(uuid4()), "project": COMPOSE_PROJECT,
              "host": DATABASE_HOST, "port": DATABASE_PORT, "database": DATABASE_NAME, "user": DATABASE_USER}
    _write_private(password_file, secrets.token_urlsafe(48) + "\n")
    _write_private(manifest, json.dumps(values, indent=2) + "\n")
    return load_demo_settings(directory)


def build_demo_engine(settings: DemoSettings | None = None):
    """Construct a lazy fixed-target engine; does not connect or read root .env."""
    settings = settings or load_demo_settings()
    url = URL.create("postgresql+psycopg", host=DATABASE_HOST, port=DATABASE_PORT,
        database=DATABASE_NAME, username=DATABASE_USER, password=settings.password)
    engine = create_engine(url, hide_parameters=True, echo=False, pool_pre_ping=True,
                           connect_args={"connect_timeout": 5,
                                         "options": "-c statement_timeout=5000 -c lock_timeout=2000"},
                           pool_size=3, max_overflow=2, pool_timeout=5)
    engine._frontdesk_demo_settings = settings
    return engine


def _settings_for_engine(engine):
    url = engine.url
    if (url.drivername, url.host, url.port, url.database, url.username) != (
            "postgresql+psycopg", DATABASE_HOST, DATABASE_PORT, DATABASE_NAME, DATABASE_USER):
        raise DemoDatabaseError("Refusing an engine outside the fixed isolated demo database.")
    settings = getattr(engine, "_frontdesk_demo_settings", None)
    if not isinstance(settings, DemoSettings):
        raise DemoDatabaseError("Use build_demo_engine to obtain the verified demo connection.")
    return settings


def _check_identity(connection, settings, *, allow_unmarked_empty=False):
    row = connection.execute(text("SELECT current_database(), current_user, "
        "shobj_description(oid, 'pg_database') FROM pg_database WHERE datname=current_database()" )).one()
    if row[0] != DATABASE_NAME or row[1] != DATABASE_USER:
        raise DemoDatabaseError("Connected database identity does not match the isolated demo.")
    if row[2] == settings.marker:
        return
    if not allow_unmarked_empty or row[2] is not None:
        raise DemoDatabaseError("Database does not have this demo instance's identity marker.")
    occupied = connection.execute(text("""
        SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname NOT IN ('public','information_schema')
                      AND nspname NOT LIKE 'pg_%')
          OR EXISTS(SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                    WHERE n.nspname='public')
          OR EXISTS(SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
                    WHERE n.nspname='public')
    """)).scalar_one()
    if occupied:
        raise DemoDatabaseError("Refusing to initialize an unknown nonempty database.")
    from psycopg import sql
    statement = sql.SQL("COMMENT ON DATABASE {} IS {}").format(sql.Identifier(DATABASE_NAME), sql.Literal(settings.marker))
    raw = connection.connection.driver_connection
    connection.exec_driver_sql(statement.as_string(raw))


def _alembic_head():
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    config = Config(str(ROOT / "alembic.ini"))
    heads = ScriptDirectory.from_config(config).get_heads()
    if len(heads) != 1:
        raise DemoDatabaseError("Demo setup requires one unambiguous Alembic head.")
    return heads[0]


def migrate_demo_database(engine):
    """Apply Alembic-generated SQL only to the verified demo engine.

    Rendering uses offline mode, so migrations/env.py never loads root .env or
    creates its normal clinic engine. The generated BEGIN/COMMIT owns atomic DDL.
    """
    from alembic import command
    from alembic.config import Config
    settings = _settings_for_engine(engine)
    with engine.begin() as connection:
        _check_identity(connection, settings, allow_unmarked_empty=True)
        tracked = connection.execute(text("SELECT to_regclass('public.alembic_version') IS NOT NULL")).scalar_one()
        versions = connection.execute(text("SELECT version_num FROM public.alembic_version")).scalars().all() if tracked else []
    if len(versions) > 1:
        raise DemoDatabaseError("Demo database has ambiguous migration state.")
    current = versions[0] if versions else "base"
    if current == _alembic_head():
        return current
    output = io.StringIO()
    config = Config(str(ROOT / "alembic.ini"), output_buffer=output)
    command.upgrade(config, f"{current}:head", sql=True)
    # Verify the instance again immediately before executing the generated SQL.
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        _check_identity(connection, settings)
        cursor = connection.connection.driver_connection.cursor()
        try:
            cursor.execute(output.getvalue(), prepare=False)
            while cursor.nextset():
                pass
        except BaseException:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()
    return _alembic_head()


def seed_demo_database(engine):
    """Preserve original fixture bytes/rows; additional demo bookings are retained."""
    from scripts import seed
    settings = _settings_for_engine(engine)
    dataset = seed.load_dataset(seed.DEFAULT_FIXTURE_DIR)
    seed.validate_dataset(dataset)
    inserted = skipped = 0
    with engine.begin() as connection:
        _check_identity(connection, settings)
        # Also serialize with the application's booking writes.
        connection.execute(text("SELECT id FROM clinic.booking_guard WHERE id=1 FOR UPDATE")).one()
        cursor = connection.connection.driver_connection.cursor()
        try:
            seed._assert_revision(cursor)
            for table in seed.INSERT_ORDER:
                rows = dataset["tables"][table]
                if table == "reviewed_reports":
                    rows = seed._topological_rows(rows, "report_id", "supersedes_report_id")
                elif table == "appointments":
                    rows = seed._topological_rows(rows, "appointment_id", "replaces_appointment_id")
                for row in rows:
                    outcome = seed._insert_or_compare(cursor, table, row)
                    inserted += outcome == "inserted"
                    skipped += outcome == "skipped"
            matched = seed._verify_fixture_rows(cursor, dataset)
            # Do not run fixed scenario total counts: retained test bookings can
            # legitimately change them. Check every original row instead.
        finally:
            cursor.close()
    return {"inserted": inserted, "unchanged": skipped, "verified_fixture_rows": matched}


def link_existing_demo_booking(engine):
    """Link only AP-1042-01 to bounded booking provenance in the isolated demo.

    This data-only operation never creates or changes an appointment, publishes
    a schedule, runs a migration, starts a container, or reads clinic .env data.
    """
    from scripts import seed
    from healthcare_voice_agent.demo.seeded_bookings import (
        SeededBookingLinkError, canonical_seeded_booking, link_existing_seeded_booking,
    )

    settings = _settings_for_engine(engine)
    try:
        dataset = seed.load_dataset(seed.DEFAULT_FIXTURE_DIR)
        seed.validate_dataset(dataset)
        canonical = canonical_seeded_booking(dataset)

        def validate_target(connection, expected):
            _check_identity(connection, settings)
            versions = connection.execute(text(
                "SELECT version_num FROM public.alembic_version ORDER BY version_num")).scalars().all()
            if versions != [_alembic_head()]:
                raise SeededBookingLinkError("Demo schema is not at the required revision.")
            identity = connection.execute(text("""
                SELECT p.is_synthetic, c.patient_id, c.status,
                       clinician.is_synthetic, staff.is_active, staff.role, clinic.timezone
                FROM clinic.cases c
                JOIN clinic.patients p ON p.patient_id=c.patient_id
                JOIN clinic.clinicians clinician ON clinician.clinician_id=:clinician
                JOIN clinic.app_users staff ON staff.user_id=:staff
                JOIN clinic.booking_clinics clinic ON clinic.clinic_id=:clinic
                WHERE c.case_id=:case_id
            """), {"case_id": expected.case_id, "clinician": expected.clinician_id,
                    "staff": expected.staff_id, "clinic": expected.clinic_id}).one_or_none()
            if (identity is None or identity[0] is not True or identity[1] != expected.patient_id
                    or identity[2] != "open" or identity[3] is not True
                    or identity[4] is not True or identity[5] != "staff"
                    or identity[6] != expected.timezone):
                raise SeededBookingLinkError(
                    "The isolated synthetic patient, case, clinician, staff, or clinic identity changed.")

        def validate_fixture_rows(connection, expected):
            cursor = connection.connection.driver_connection.cursor()
            try:
                for table, row in expected.fixture_rows.items():
                    existing = seed._select_fixture_row(cursor, table, row)
                    if existing is None or not seed._row_matches(table, row, existing):
                        raise SeededBookingLinkError(
                            "Canonical fixture rows differ in the demo database; no provenance was added.")
            finally:
                cursor.close()

        result = link_existing_seeded_booking(engine, canonical,
            validate_target=validate_target, validate_canonical_rows=validate_fixture_rows)
        return {"database": DATABASE_NAME, "host": DATABASE_HOST, "port": DATABASE_PORT,
                "revision": _alembic_head(), **result}
    except DemoDatabaseError:
        raise
    except (SeededBookingLinkError, seed.SeedError) as exc:
        raise DemoDatabaseError(str(exc)) from None
    except Exception:
        raise DemoDatabaseError(
            "Existing synthetic booking provenance could not be verified; no appointment was created or changed.") from None


def publish_demo_schedule(engine):
    from healthcare_voice_agent.booking.service import BookingService
    from healthcare_voice_agent.booking.scheduling import ScheduleRule
    settings = _settings_for_engine(engine)
    with engine.connect() as connection:
        _check_identity(connection, settings)
        today = connection.execute(text("SELECT (clock_timestamp() AT TIME ZONE :zone)::date"), {"zone": DEMO_TIMEZONE}).scalar_one()
    service = BookingService(engine)
    service.register_clinic(DEMO_CLINIC_ID, DEMO_TIMEZONE)
    # Alternative appointment types share clinician capacity; the backend checks
    # overlap across distinct slot IDs. These are synthetic approved demo rules.
    rules = tuple(ScheduleRule(kind, clinician, (0, 1, 2, 3, 4), time(9), time(17), duration,
        "Synthetic Front Desk, Room " + room) for kind, clinician, duration, room in (
        ("initial_consultation", "SYN-CLIN-01", 30, "1"),
        ("fracture_follow_up", "SYN-CLIN-01", 30, "1"),
        ("imaging", "SYN-CLIN-02", 30, "2"),
        ("physiotherapy", "SYN-CLIN-03", 30, "3"),
        ("other", "SYN-CLIN-03", 30, "3"),
    ))
    ids = service.publish_schedule(DEMO_CLINIC_ID, rules, today, today + timedelta(days=14))
    return {"published_or_reused_slots": len(ids), "start_date": today.isoformat(),
            "end_date": (today + timedelta(days=14)).isoformat()}


def demo_case_summary(engine):
    """Fixed, pre-resolved fictional case. Not an identity/authentication service."""
    settings = _settings_for_engine(engine)
    with engine.connect() as connection:
        _check_identity(connection, settings)
        row = connection.execute(text("""SELECT c.case_id, p.display_name, p.is_synthetic, c.status
            FROM clinic.cases c JOIN clinic.patients p USING(patient_id) WHERE c.case_id=:case_id"""),
            {"case_id": DEMO_CASE_ID}).mappings().one_or_none()
        staff = connection.execute(text("SELECT is_active, role FROM clinic.app_users WHERE user_id=:user_id"),
            {"user_id": DEMO_STAFF_ID}).one_or_none()
        if not row or not row["is_synthetic"] or row["status"] != "open" or not staff or not staff[0] or staff[1] != "staff":
            raise DemoDatabaseError("The fixed synthetic case or demo staff principal is unavailable.")
        return {"case_id": DEMO_CASE_ID, "patient_display_name": row["display_name"],
                "staff_id": DEMO_STAFF_ID, "clinic_id": DEMO_CLINIC_ID, "timezone": DEMO_TIMEZONE}


def verify_demo_database(engine):
    """Read-only readiness check. Does not migrate, seed or publish schedules."""
    settings = _settings_for_engine(engine)
    with engine.connect() as connection:
        _check_identity(connection, settings)
        versions = connection.execute(text("SELECT version_num FROM public.alembic_version")).scalars().all()
        if versions != [_alembic_head()]:
            raise DemoDatabaseError("Demo schema is not at the current migration head.")
        zone = connection.execute(text("SELECT timezone FROM clinic.booking_clinics WHERE clinic_id=:clinic"),
            {"clinic": DEMO_CLINIC_ID}).scalar_one_or_none()
        if zone != DEMO_TIMEZONE:
            raise DemoDatabaseError("Synthetic clinic configuration is missing.")
        count = connection.execute(text("SELECT count(*) FROM clinic.booking_slots WHERE clinic_id=:clinic "
            "AND enabled AND starts_at>clock_timestamp()"), {"clinic": DEMO_CLINIC_ID}).scalar_one()
        if not count:
            raise DemoDatabaseError("No future demo slots. Rerun setup_frontdesk_demo.py --prepare.")
    summary = demo_case_summary(engine)
    return {"database": DATABASE_NAME, "host": DATABASE_HOST, "port": DATABASE_PORT,
            "revision": versions[0], "future_slots": count, **summary}


def start_demo_container(settings: DemoSettings):
    """Start only the named isolated compose project, without image pulls."""
    docker = next((p for p in ("/usr/local/bin/docker", "/Applications/Docker.app/Contents/Resources/bin/docker",
                              "/opt/homebrew/bin/docker", "/usr/bin/docker") if Path(p).is_file()), None)
    if docker is None:
        raise DemoDatabaseError("Docker CLI is unavailable. Start Docker Desktop and install its CLI.")
    environment = {key: os.environ[key] for key in ("PATH", "HOME", "USER", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG") if key in os.environ}
    environment["FRONTDESK_DEMO_PASSWORD_FILE"] = str(settings.password_file)
    command = [docker, "compose", "--env-file", os.devnull, "--project-name", COMPOSE_PROJECT,
        "--file", str(ROOT / "compose.frontdesk-demo.yaml"), "up", "--detach", "--wait",
        "--wait-timeout", "100", "--pull", "never", "db"]
    try:
        subprocess.run(command, cwd=ROOT, env=environment, check=True, timeout=120,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (subprocess.SubprocessError, OSError):
        raise DemoDatabaseError("Isolated Docker startup failed. Check Docker Desktop, cached image and port 55432; no main DB settings were used.") from None


def prepare_demo_database(state_dir: Path | None = None):
    settings = create_demo_settings(state_dir)
    start_demo_container(settings)
    engine = build_demo_engine(settings)
    try:
        migrate_demo_database(engine)
        fixture_result = seed_demo_database(engine)
        schedule_result = publish_demo_schedule(engine)
        return {**verify_demo_database(engine), **fixture_result, **schedule_result}
    except DemoDatabaseError:
        raise
    except Exception:
        raise DemoDatabaseError("Demo preparation could not be verified. The main clinic database was not selected; retain the demo files and retry after checking the isolated container.") from None
    finally:
        engine.dispose()
