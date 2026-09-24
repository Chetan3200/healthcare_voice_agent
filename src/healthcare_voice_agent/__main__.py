"""Inspect configuration offline, or explicitly serve the local browser voice demo."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from healthcare_voice_agent.config import ConfigurationError, load_config, write_run_config


async def bot(runner_args) -> None:
    """Pipecat discovers this symbol on the executed __main__ module."""
    from healthcare_voice_agent.voice.session import run_session
    await run_session(runner_args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file", type=Path, default=Path(".env"),
        help="Dotenv path; default is .env in the current working directory",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-config", action="store_true", help="Print resolved non-secret settings offline")
    mode.add_argument("--serve", action="store_true", help="Serve the local voice demo; requires live opt-in, plus a key for any selected OpenAI stage")
    mode.add_argument("--prepare-local-models", action="store_true", help="Download pinned selected local model assets; no OpenAI calls")
    parser.add_argument("--run-dir", type=Path, help="New run directory; defaults to runs/<unique-id>")
    args = parser.parse_args(argv)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    try:
        config = load_config(args.env_file)
        if args.serve:
            config.require_live_ready()
        run_dir = args.run_dir or Path("runs") / run_id
        record = write_run_config(config, run_dir)
    except (ConfigurationError, OSError) as exc:
        parser.error(str(exc))
    if args.prepare_local_models:
        from healthcare_voice_agent.voice.providers import prepare_local_models, ProviderDependencyError
        try:
            assets = prepare_local_models(config)
        except ProviderDependencyError as exc:
            parser.error(str(exc))
        print(json.dumps(assets, indent=2))
        print("Local model files prepared. No OpenAI API calls were made.")
        return 0
    if args.check_config:
        print(json.dumps(config.public_dict(), indent=2, allow_nan=False))
    print(f"Non-secret run configuration: {record}")
    if args.serve:
        from healthcare_voice_agent.voice.server import VoiceDependencyError, start_server
        try:
            start_server(config, run_dir, parser, argv)
        except (ConfigurationError, VoiceDependencyError) as exc:
            parser.error(str(exc))
    else:
        print("Offline check only. No provider services or API calls were started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
