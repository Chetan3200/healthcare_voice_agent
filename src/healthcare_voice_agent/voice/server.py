"""Thin adapter around Pipecat's bundled local development server.

There are no custom signaling routes or frontend assets. Importing this module
is offline; only start_server() imports Pipecat's optional runner dependencies.
"""

from __future__ import annotations

import argparse
from importlib.util import find_spec
from pathlib import Path

from healthcare_voice_agent.config import AppConfig


class VoiceDependencyError(RuntimeError):
    """Missing optional browser-runtime dependencies."""


class LocalRunnerParser(argparse.ArgumentParser):
    """Apply local-only defaults after Pipecat adds its runner arguments.

    Pipecat 1.11 exposes main(parser=...), but no main(bot=..., argv=...).
    This public argparse adapter preserves normal CLI parsing without rewriting
    sys.argv, and passes the frozen configuration to each bot through cli_args.
    """

    def __init__(self, *, config: AppConfig, run_dir: Path,
                 command_args: list[str] | None, **kwargs):
        super().__init__(**kwargs)
        self._config = config
        self._run_dir = run_dir
        self._command_args = command_args

    def parse_args(self, args=None, namespace=None):
        self.set_defaults(
            host=self._config.voice.host, port=self._config.voice.port,
            transport="webrtc", ice_servers=[],
        )
        result = super().parse_args(
            self._command_args if args is None else args, namespace,
        )
        if result.host != "127.0.0.1" or result.transport != "webrtc":
            self.error("This demo supports loopback WebRTC only.")
        result.app_config = self._config
        result.run_root = self._run_dir
        return result


def start_server(config: AppConfig, run_dir: Path,
                 app_parser: argparse.ArgumentParser, argv: list[str] | None) -> None:
    config.require_live_ready()
    if config.agent_mode == "frontdesk_demo":
        from healthcare_voice_agent.demo.database import build_demo_engine, verify_demo_database
        demo_engine = None
        try:
            demo_engine = build_demo_engine()
            verify_demo_database(demo_engine)
        except Exception:
            raise VoiceDependencyError("The isolated front-desk database is not ready. Run scripts/setup_frontdesk_demo.py --check (or explicitly --prepare).") from None
        finally:
            if demo_engine is not None:
                demo_engine.dispose()
    if config.agent_mode == "clinician":
        from healthcare_voice_agent.clinician.runtime import clinical_records
        try:
            clinical_records(config.clinician).verify_access()
        except Exception:
            raise VoiceDependencyError("The synthetic clinical records are not accessible. Run scripts/voice_clinician.sh --check. No database changes were made.") from None
    missing = [name for name in (
        "pipecat", "fastapi", "uvicorn", "aiortc", "pipecat_ai_prebuilt",
    ) if find_spec(name) is None]
    if missing:
        raise VoiceDependencyError(
            "Browser voice dependencies are missing. In your normal Terminal, run: "
            "uv sync --locked --extra voice --no-cache --link-mode copy"
        )

    from healthcare_voice_agent.voice.tracing import install_log_redaction

    # The runner resets Loguru sinks. A global patcher survives that reset.
    install_log_redaction(config.redaction_secrets())
    if config.stt.provider == "whisper" or config.tts.provider == "kokoro":
        from healthcare_voice_agent.voice.providers import warm_local_models, ProviderDependencyError
        print("Loading and warming selected local speech models before accepting connections...")
        try:
            warm_local_models(config)
        except ProviderDependencyError as exc:
            raise VoiceDependencyError(str(exc)) from None
        except Exception:
            raise VoiceDependencyError("Local model warmup failed. Check local runtime dependencies and model files.") from None
    from pipecat.runner.run import main as run_pipecat

    parser = LocalRunnerParser(
        config=config, run_dir=run_dir, command_args=argv,
        parents=[app_parser], add_help=False,
    )
    if config.agent_mode == "frontdesk_demo":
        print("Isolated synthetic front desk: fixed fictional patient, separate PostgreSQL database; no clinical tools.")
        print("The agent uses clear requests directly, without an extra confirmation turn.")
    if config.agent_mode == "clinician":
        print("Synthetic clinician assistant: open_case plus five read-only tools; no preselected patient. Exa is external guidance, not clinic-approved policy.")
        if not config.clinician.exa_api_key:
            print("EXA_API_KEY is missing: clinical reads work, but web guidance search is unavailable.")
    print(f"Browser UI: http://{config.voice.host}:{config.voice.port}/client/")
    if any(stage.provider == "openai" for stage in (config.stt, config.llm, config.tts)):
        print("Click Connect to start a session (selected OpenAI services are paid). Use synthetic speech only.")
    else:
        print("Click Connect to start a session using the configured model servers. Use synthetic speech only.")
    print(f"Session logs: {run_dir / 'sessions'}")
    run_pipecat(parser=parser)
