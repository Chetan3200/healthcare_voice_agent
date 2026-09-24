"""Offline repair-helper guards. No packages are downloaded or installed."""
import builtins
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import repair_charset_normalizer as repair

VERSION = "3.5.1"
URL = "https://files.pythonhosted.org/packages/aa/bb/fixture/charset_normalizer-3.5.1-py3-none-any.whl"
HASH = "sha256:" + "a" * 64


def write_lock(root, *, url=URL, digest=HASH, duplicate_package=False, duplicate_wheel=False, version=VERSION):
    wheel = f'{{url = "{url}", hash = "{digest}"}}'
    entry = (f'[[package]]\nname = "charset-normalizer"\nversion = "{version}"\n'
             'source = {registry = "https://pypi.org/simple"}\n'
             f'wheels = [{wheel}{", " + wheel if duplicate_wheel else ""}]\n')
    (root / "uv.lock").write_text(entry + (entry if duplicate_package else ""))


@pytest.fixture
def project(tmp_path):
    interpreter = tmp_path / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\nexit 99\n")
    interpreter.chmod(0o700)
    site = tmp_path / ".venv/lib/python3.11/site-packages"
    dist = site / "charset_normalizer-3.5.1.dist-info"
    dist.mkdir(parents=True)
    metadata = dist / "METADATA"
    metadata.write_text("Metadata-Version: 2.1\nName: charset-normalizer\nVersion: 3.5.1\n")
    module = site / "charset_normalizer"
    module.mkdir()
    (module / "__init__.py").write_text("raise AssertionError('Never import this package')\n")
    native = module / "cd.cpython-311-darwin.so"
    native.write_bytes(b"fake native file: never loaded")
    write_lock(tmp_path)
    return SimpleNamespace(root=tmp_path, site=site, metadata=metadata, module=module,
                           native=native, interpreter=interpreter)


@pytest.fixture(autouse=True)
def forbid_installation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No real installer may run in these tests")
    monkeypatch.setattr(repair.subprocess, "run", forbidden)


@pytest.mark.parametrize("argv", [[], ["--check"]])
def test_default_and_check_are_read_only_and_do_not_import_package(project, monkeypatch, capsys, argv):
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.startswith("charset_normalizer"):
            raise AssertionError("Native package import is forbidden")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    before = project.native.read_bytes()
    assert repair.main(argv, root=project.root) == 0
    output = capsys.readouterr().out
    assert '"action": "not_applied"' in output
    assert '"native_extensions_present": true' in output
    assert '"version": "3.5.1"' in output
    assert project.native.read_bytes() == before
    assert "not a malware assessment" in output


def test_official_wheel_uses_locked_version_and_digest(project):
    version, wheel = repair.locked_wheel(project.root)
    assert version == VERSION
    assert wheel == URL + "#sha256=" + "a" * 64


@pytest.mark.parametrize("url", [
    URL.replace("https:", "http:"),
    URL.replace("files.pythonhosted.org", "untrusted.invalid"),
    URL.replace("files.pythonhosted.org", "user:private-secret@files.pythonhosted.org"),
    URL.replace("files.pythonhosted.org", "files.pythonhosted.org:443"),
    URL + "?key=private-secret", URL + "#private-secret",
    URL.replace("/packages/", "/not-packages/"),
    URL.replace("py3-none-any", "cp311-cp311-macosx_10_9_universal2"),
    URL.replace("3.5.1", "3.5.2"),
])
def test_invalid_wheel_identity_is_rejected_without_exposing_input(project, url):
    write_lock(project.root, url=url)
    with pytest.raises(repair.RepairError) as error:
        repair.locked_wheel(project.root)
    assert "private-secret" not in str(error.value)


@pytest.mark.parametrize("digest", ["md5:" + "a" * 32, "sha256:" + "a" * 63,
                                      "sha256:" + "g" * 64, "sha256:"])
def test_invalid_hash_rejected(project, digest):
    write_lock(project.root, digest=digest)
    with pytest.raises(repair.RepairError):
        repair.locked_wheel(project.root)


@pytest.mark.parametrize("option", ["duplicate_package", "duplicate_wheel"])
def test_ambiguous_lock_rejected(project, option):
    write_lock(project.root, **{option: True})
    with pytest.raises(repair.RepairError):
        repair.locked_wheel(project.root)


def test_wrong_installed_version_rejected_before_installation(project):
    project.metadata.write_text("Name: charset-normalizer\nVersion: 3.4.0\n")
    with pytest.raises(repair.RepairError, match="uniquely match"):
        repair.install_locked_pure_python(project.root)


def test_duplicate_installed_metadata_rejected(project):
    duplicate = project.site / "charset_normalizer-3.4.0.dist-info"
    duplicate.mkdir()
    (duplicate / "METADATA").write_text("Name: charset-normalizer\nVersion: 3.4.0\n")
    with pytest.raises(repair.RepairError, match="uniquely match"):
        repair.inspect_project(project.root)


def test_missing_project_interpreter_rejected(project):
    project.interpreter.unlink()
    with pytest.raises(repair.RepairError, match="interpreter|python"):
        repair.inspect_project(project.root)


def test_explicit_install_uses_only_same_locked_package_and_verifies_result(project, monkeypatch):
    calls = []
    monkeypatch.setattr(repair.shutil, "which", lambda name: "/trusted/uv")
    before_lock = (project.root / "uv.lock").read_bytes()
    def fake_install(command, **kwargs):
        calls.append((command, kwargs))
        project.native.unlink()  # Simulate wheel replacement only in this test fixture.
    monkeypatch.setattr(repair.subprocess, "run", fake_install)
    result = repair.install_locked_pure_python(project.root)
    command, kwargs = calls[0]
    assert command == ["/trusted/uv", "pip", "install", "--python", str(project.interpreter),
                       "--no-deps", "--reinstall", "--no-cache", "--link-mode", "copy",
                       URL + "#sha256=" + "a" * 64]
    assert kwargs == {"cwd": project.root, "check": True, "timeout": 120,
                      "capture_output": True, "text": True}
    assert result["action"] == "pure_python_package_installed"
    assert not result["native_extensions_present"]
    assert not result["security_settings_changed"] and not result["voice_demo_started"]
    assert (project.root / "uv.lock").read_bytes() == before_lock


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("private-secret", 120),
    subprocess.CalledProcessError(1, "private-secret", stderr="credential-private-secret")])
def test_installer_errors_are_sanitized(project, monkeypatch, capsys, failure):
    monkeypatch.setattr(repair.shutil, "which", lambda name: "/trusted/uv")
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(repair.subprocess, "run", fail)
    assert repair.main(["--install"], root=project.root) == 1
    captured = capsys.readouterr()
    assert "private-secret" not in captured.err + captured.out
    assert "outcome is unverified" in captured.err


def test_compiled_remnant_after_install_is_not_reported_as_success(project, monkeypatch):
    monkeypatch.setattr(repair.shutil, "which", lambda name: "/trusted/uv")
    monkeypatch.setattr(repair.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(repair.RepairError, match="could not be verified"):
        repair.install_locked_pure_python(project.root)


def test_no_uv_never_attempts_installation(project, monkeypatch):
    access = repair.os.access
    monkeypatch.setattr(repair.shutil, "which", lambda name: None)
    monkeypatch.setattr(repair.os, "access", lambda path, mode: False if str(path) == "/opt/homebrew/bin/uv" else access(path, mode))
    with pytest.raises(repair.RepairError, match="uv is unavailable"):
        repair.install_locked_pure_python(project.root)


def test_mutually_exclusive_cli_modes(project):
    with pytest.raises(SystemExit):
        repair.main(["--check", "--install"], root=project.root)
