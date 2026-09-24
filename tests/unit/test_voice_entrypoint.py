"""Offline CLI and local-runner policy checks; no server or providers start."""

import argparse
import json

import pytest

from healthcare_voice_agent.config import ConfigurationError, load_config
from healthcare_voice_agent.voice.server import LocalRunnerParser


def test_voice_defaults_and_overrides():
    config = load_config(None, environ={"VOICE_PORT": "8008", "VOICE_MAX_SESSION_SECONDS": "90"})
    assert config.voice.host == "127.0.0.1"
    assert config.voice.port == 8008
    assert config.voice.max_session_seconds == 90
    assert config.voice.input_sample_rate == 16000
    assert config.voice.output_sample_rate == 24000


@pytest.mark.parametrize("name,value", [
    ("VOICE_PORT", "80"), ("VOICE_PORT", "65536"),
    ("VOICE_IDLE_TIMEOUT_SECONDS", "0"), ("VOICE_MAX_SESSION_SECONDS", "nan"),
    ("VOICE_START_TIMEOUT_SECONDS", "-1"), ("VOICE_MODLE", "anything"),
    ("VOICE_HOST", "0.0.0.0"),
])
def test_invalid_voice_settings_are_rejected(name, value):
    with pytest.raises(ConfigurationError):
        load_config(None, environ={name: value})


def test_runner_defaults_are_applied_after_its_arguments_exist(tmp_path):
    config = load_config(None, environ={"VOICE_PORT": "8008"})
    parser = LocalRunnerParser(config=config, run_dir=tmp_path, command_args=[])
    # Mimic the public runner adding its defaults after the app creates a parser.
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--transport", default=None)
    parser.add_argument("--ice-servers", default="unwanted-environment-value")
    args = parser.parse_args()
    assert (args.host, args.port, args.transport) == ("127.0.0.1", 8008, "webrtc")
    assert args.ice_servers == []
    assert args.app_config is config
    assert args.run_root == tmp_path
    with pytest.raises(SystemExit):
        parser.parse_args(["--host", "0.0.0.0"])


def test_serve_requires_live_opt_in_before_importing_runtime(tmp_path, monkeypatch):
    from healthcare_voice_agent import __main__ as entry
    monkeypatch.setattr(entry, "load_config", lambda _: load_config(None, environ={}))
    with pytest.raises(SystemExit) as error:
        entry.main(["--serve", "--run-dir", str(tmp_path / "run")])
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_inspection_and_serving_are_mutually_exclusive():
    from healthcare_voice_agent.__main__ import main
    with pytest.raises(SystemExit):
        main(["--serve", "--check-config"])


def test_serve_passes_frozen_config_and_records_it(tmp_path, monkeypatch):
    from healthcare_voice_agent import __main__ as entry
    from healthcare_voice_agent.voice import server
    config = load_config(None, environ={
        "LIVE_API_ENABLED": "true", "OPENAI_API_KEY": "dummy-offline-key",
    })
    calls = []
    monkeypatch.setattr(entry, "load_config", lambda _: config)
    monkeypatch.setattr(server, "start_server", lambda *args: calls.append(args))
    directory = tmp_path / "run"
    assert entry.main(["--serve", "--run-dir", str(directory)]) == 0
    assert calls[0][0] is config
    assert calls[0][1] == directory
    assert isinstance(calls[0][2], argparse.ArgumentParser)
    record = (directory / "config.json").read_text()
    assert "dummy-offline-key" not in record
    assert json.loads(record)["voice_policy"]["clinic_tools_enabled"] is False


def test_console_script_executes_the_discoverable_main_module(monkeypatch):
    from healthcare_voice_agent import cli
    calls = []
    monkeypatch.setattr(cli.runpy, "run_module", lambda *a, **kw: calls.append((a, kw)))
    cli.main()
    assert calls == [(("healthcare_voice_agent",), {"run_name": "__main__", "alter_sys": True})]
