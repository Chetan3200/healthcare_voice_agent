"""Console-script adapter for Pipecat's __main__.bot discovery convention."""

import runpy


def main() -> None:
    # Match `python -m healthcare_voice_agent`. The installed console-script
    # wrapper itself does not expose bot(), so execute the real package module.
    runpy.run_module("healthcare_voice_agent", run_name="__main__", alter_sys=True)
