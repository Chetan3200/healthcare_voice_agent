"""Bundled UI/route smoke test. No microphone, providers, or network sockets.

Skipped until the runner/WebRTC extras are installed. A local ASGI test client
checks the real Pipecat routes; Uvicorn's socket server is not started.
"""

import inspect
import socket
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("aiortc")
pytest.importorskip("pipecat_ai_prebuilt")


def test_console_discovery_and_bundled_browser_routes(tmp_path, monkeypatch):
    import uvicorn
    from fastapi.testclient import TestClient
    from pipecat.runner import run as pipecat_runner
    from healthcare_voice_agent import cli, config as configuration
    from healthcare_voice_agent.voice import providers

    config = configuration.load_config(None, environ={
        "LIVE_API_ENABLED": "true",
        "OPENAI_API_KEY": "dummy-offline-key", "VOICE_PORT": "8008",
    })
    monkeypatch.setattr(configuration, "load_config", lambda *a, **kw: config)

    def unexpected(*a, **kw):
        raise AssertionError("UI/start-route tests must not construct providers or access network sockets")

    monkeypatch.setattr(socket.socket, "connect", unexpected)
    monkeypatch.setattr(socket.socket, "connect_ex", unexpected)
    monkeypatch.setattr(socket, "create_connection", unexpected)
    for name in ("build_stt", "build_llm", "build_tts"):
        monkeypatch.setattr(providers, name, unexpected)

    served = []

    def serve_in_process(app, *, host, port):
        assert (host, port) == ("127.0.0.1", 8008)
        # This runs while the console adapter's real __main__ module is active.
        module = pipecat_runner._get_bot_module()
        assert module.__name__ == "__main__"
        assert inspect.iscoroutinefunction(module.bot)
        with TestClient(app) as client:
            response = client.get("/client/")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
            assert "<html" in response.text.lower()
            assert client.get("/", follow_redirects=False).status_code in (302, 307)
            response = client.post("/start", json={"transport": "webrtc"})
            assert response.status_code == 200
            assert response.json()["sessionId"]
            # /start allocates negotiation metadata; it does not run a bot.
            assert client.post("/start", json={"transport": "daily"}).status_code == 400
        served.append(True)

    monkeypatch.setattr(uvicorn, "run", serve_in_process)
    monkeypatch.setattr(sys, "argv", [
        "healthcare-voice-agent", "--serve", "--env-file", str(tmp_path / "absent.env"),
        "--run-dir", str(tmp_path / "run"),
    ])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0
    assert served == [True]
