"""Isolation/configuration tests; no Docker execution, .env reads or database calls."""
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import URL

from healthcare_voice_agent.demo import database as demo


@pytest.fixture
def settings(tmp_path):
    return demo.create_demo_settings(tmp_path / "state")


def test_settings_are_private_fixed_target_and_reused(settings):
    assert settings.password_file.stat().st_mode & 0o777 == 0o600
    assert (settings.state_dir / "settings.json").stat().st_mode & 0o777 == 0o600
    assert settings.password not in repr(settings)
    assert settings.password not in (settings.state_dir / "settings.json").read_text()
    assert demo.create_demo_settings(settings.state_dir) == settings
    assert settings.connection_kwargs()["port"] == 55432
    assert settings.connection_kwargs()["dbname"] == "frontdesk_synthetic_demo"


@pytest.mark.parametrize("field,value", [("port",5433),("database","fracture_clinic"),("host","remote"),
    ("user","clinic_owner"),("project","fracture-clinic"),("schema_version",2)])
def test_tampered_targets_are_rejected(settings, field, value):
    path = settings.state_dir / "settings.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(demo.DemoDatabaseError):
        demo.load_demo_settings(settings.state_dir)


def test_partial_settings_never_rotate_or_overwrite_password(settings):
    before = settings.password_file.read_bytes()
    (settings.state_dir / "settings.json").unlink()
    with pytest.raises(demo.DemoDatabaseError):
        demo.create_demo_settings(settings.state_dir)
    assert settings.password_file.read_bytes() == before


def test_world_readable_or_symlink_settings_fail_closed(settings, tmp_path):
    settings.password_file.chmod(0o644)
    with pytest.raises(demo.DemoDatabaseError):
        demo.load_demo_settings(settings.state_dir)
    settings.password_file.chmod(0o600)
    target = tmp_path / "password-copy"
    target.write_bytes(settings.password_file.read_bytes())
    settings.password_file.unlink()
    settings.password_file.symlink_to(target)
    with pytest.raises(demo.DemoDatabaseError):
        demo.load_demo_settings(settings.state_dir)


def test_engine_construction_is_lazy_ignores_main_db_environment(settings, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://untrusted/main")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "fracture_clinic")
    calls = []
    def factory(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(url=url)
    monkeypatch.setattr(demo, "create_engine", factory)
    engine = demo.build_demo_engine(settings)
    assert engine.url.host == "127.0.0.1" and engine.url.port == 55432
    assert engine.url.database == "frontdesk_synthetic_demo"
    assert calls[0][1]["hide_parameters"] is True
    assert demo._settings_for_engine(engine) == settings
    engine.url = engine.url.set(port=5433)
    with pytest.raises(demo.DemoDatabaseError):
        demo._settings_for_engine(engine)


def test_compose_command_never_uses_main_env_or_pulls_image(settings, monkeypatch):
    calls = []
    monkeypatch.setattr(demo.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(Path, "is_file", lambda path: True)
    monkeypatch.setenv("POSTGRES_PASSWORD", "main-secret-must-not-enter")
    monkeypatch.setenv("POSTGRES_DB", "fracture_clinic")
    demo.start_demo_container(settings)
    command, options = calls[0]
    assert command[command.index("--project-name")+1] == "fracture-frontdesk-demo"
    assert command[command.index("--env-file")+1] == os.devnull
    assert command[command.index("--pull")+1] == "never"
    assert "POSTGRES_PASSWORD" not in options["env"]
    assert "POSTGRES_DB" not in options["env"]
    assert options["env"]["FRONTDESK_DEMO_PASSWORD_FILE"] == str(settings.password_file)
    assert "down" not in command


class Result:
    def __init__(self, value): self.value = value
    def one(self): return self.value
    def scalar_one(self): return self.value


class FakeConnection:
    def __init__(self, identity, occupied=False):
        self.identity = identity
        self.occupied = occupied
        self.calls = []
        self.connection = SimpleNamespace(driver_connection=None)
    def execute(self, query):
        self.calls.append(str(query))
        return Result(self.identity if len(self.calls) == 1 else self.occupied)
    def exec_driver_sql(self, value): self.calls.append(value)


def test_existing_wrong_identity_is_never_modified(settings):
    connection = FakeConnection(("fracture_clinic", demo.DATABASE_USER, settings.marker))
    with pytest.raises(demo.DemoDatabaseError):
        demo._check_identity(connection, settings, allow_unmarked_empty=True)
    assert len(connection.calls) == 1


def test_unknown_nonempty_database_is_never_claimed(settings):
    connection = FakeConnection((demo.DATABASE_NAME, demo.DATABASE_USER, None), occupied=True)
    with pytest.raises(demo.DemoDatabaseError, match="nonempty"):
        demo._check_identity(connection, settings, allow_unmarked_empty=True)
    assert len(connection.calls) == 2


def test_unknown_marker_fails_even_on_empty_database(settings):
    connection = FakeConnection((demo.DATABASE_NAME, demo.DATABASE_USER, "unrelated"))
    with pytest.raises(demo.DemoDatabaseError):
        demo._check_identity(connection, settings, allow_unmarked_empty=True)
    assert len(connection.calls) == 1


def test_verified_marker_does_not_write(settings):
    connection = FakeConnection((demo.DATABASE_NAME, demo.DATABASE_USER, settings.marker))
    demo._check_identity(connection, settings)
    assert len(connection.calls) == 1


def test_compose_uses_separate_volume_and_loopback_only():
    source = (demo.ROOT / "compose.frontdesk-demo.yaml").read_text()
    assert "127.0.0.1:55432:5432" in source
    assert "frontdesk_demo_data" in source and "clinic_db_data" not in source
    assert "POSTGRES_PASSWORD_FILE" in source and "pull_policy: never" in source
    assert "${POSTGRES_DB" not in source and "${DB_PORT" not in source


def test_script_default_is_check_only(monkeypatch, capsys):
    from scripts import setup_frontdesk_demo as script
    engine = SimpleNamespace(dispose=lambda: None)
    monkeypatch.setattr(script, "build_demo_engine", lambda: engine)
    monkeypatch.setattr(script, "verify_demo_database", lambda value: {
        "database": demo.DATABASE_NAME, "revision": script.EXPECTED_SCHEMA_REVISION})
    monkeypatch.setattr(script, "prepare_demo_database", lambda: pytest.fail("Default must not prepare"))
    assert script.main([]) == 0
    assert demo.DATABASE_NAME in capsys.readouterr().out


def test_setup_and_seed_accept_current_direct_request_revision():
    from scripts import seed
    from scripts import setup_frontdesk_demo as script
    cursor = SimpleNamespace(execute=lambda *args: None, fetchone=lambda: (True,),
        fetchall=lambda: [("005",)])
    assert script.EXPECTED_SCHEMA_REVISION == demo._alembic_head() == "005"
    assert "005" in seed.SUPPORTED_DB_REVISIONS
    seed._assert_revision(cursor)


def test_link_existing_mode_is_explicit_data_only(monkeypatch, capsys):
    from scripts import setup_frontdesk_demo as script
    calls = []
    engine = SimpleNamespace(dispose=lambda: calls.append("dispose"))
    monkeypatch.setattr(script, "build_demo_engine", lambda: engine)
    monkeypatch.setattr(script, "link_existing_demo_booking", lambda value: {
        "database": demo.DATABASE_NAME, "revision": script.EXPECTED_SCHEMA_REVISION,
        "appointment_id": "AP-1042-01", "status": "linked"})
    monkeypatch.setattr(script, "verify_demo_database", lambda value: pytest.fail("Link mode is not readiness mode"))
    monkeypatch.setattr(script, "prepare_demo_database", lambda: pytest.fail("Link mode must not prepare"))
    assert script.main(["--link-existing"]) == 0
    assert calls == ["dispose"]
    assert "AP-1042-01" in capsys.readouterr().out


def test_link_existing_rejects_any_non_demo_engine_before_connecting():
    engine = SimpleNamespace(
        url=URL.create("postgresql+psycopg", host="127.0.0.1", port=5432,
            database="fracture_clinic", username="clinic_owner"),
        connect=lambda: pytest.fail("A non-demo engine must never connect"),
    )
    with pytest.raises(demo.DemoDatabaseError, match="outside the fixed isolated demo"):
        demo.link_existing_demo_booking(engine)


def test_setup_rejects_stale_demo_revision(monkeypatch, capsys):
    from scripts import setup_frontdesk_demo as script
    engine = SimpleNamespace(dispose=lambda: None)
    monkeypatch.setattr(script, "build_demo_engine", lambda: engine)
    monkeypatch.setattr(script, "verify_demo_database", lambda value: {
        "database": demo.DATABASE_NAME, "revision": "003"})
    with pytest.raises(SystemExit):
        script.main([])
    assert "packaged front-desk backend" in capsys.readouterr().err


def test_readiness_errors_do_not_expose_credentials(monkeypatch, capsys):
    from scripts import setup_frontdesk_demo as script
    def fail(): raise RuntimeError("private-password-marker")
    monkeypatch.setattr(script, "build_demo_engine", fail)
    with pytest.raises(SystemExit):
        script.main([])
    assert "private-password-marker" not in capsys.readouterr().err
