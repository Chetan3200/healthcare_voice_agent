"""Inspect, or explicitly install, the locked pure-Python charset-normalizer wheel.

Default/--check is read-only. --install replaces only this package in the project
.venv, using the same locked version and SHA-256. It never imports its native
extension, changes macOS security settings, or starts the voice demo.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "charset-normalizer"


class RepairError(RuntimeError):
    """A fixed diagnostic that never includes subprocess output or credentials."""


def locked_wheel(root: Path) -> tuple[str, str]:
    """Return the pinned version and official pure-Python wheel with its hash."""
    try:
        lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
        packages = [p for p in lock["package"] if p.get("name") == PACKAGE]
        if len(packages) != 1:
            raise ValueError
        package = packages[0]
        version = package["version"]
        if (not isinstance(version, str) or not re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", version)
                or package.get("source") != {"registry": "https://pypi.org/simple"}):
            raise ValueError
        filename = f"charset_normalizer-{version}-py3-none-any.whl"
        wheels = [w for w in package["wheels"] if urlsplit(w["url"]).path.rsplit("/", 1)[-1] == filename]
        if len(wheels) != 1:
            raise ValueError
        wheel = wheels[0]
        url = urlsplit(wheel["url"])
        digest = wheel["hash"]
        if (url.scheme != "https" or url.netloc != "files.pythonhosted.org"
                or url.username is not None or url.password is not None
                or url.query or url.fragment or not url.path.startswith("/packages/")
                or not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)):
            raise ValueError
        return version, wheel["url"] + "#" + digest.replace(":", "=", 1)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise RepairError("The lock must contain one official, hash-pinned pure-Python charset-normalizer wheel.") from None


def inspect_project(root: Path) -> tuple[dict, str]:
    version, wheel = locked_wheel(root)
    interpreter = root / ".venv" / "bin" / "python"
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise RepairError("The project .venv/bin/python is missing or not executable.")
    site = root / ".venv" / "lib" / "python3.11" / "site-packages"
    installed = [d for d in metadata.distributions(path=[str(site)])
                 if re.sub(r"[-_.]+", "-", d.metadata.get("Name", "")).lower() == PACKAGE]
    if len(installed) != 1 or installed[0].version != version:
        raise RepairError("The installed project package must uniquely match the version in uv.lock; no change was made.")
    module = site / "charset_normalizer"
    if not module.is_dir() or not module.resolve().is_relative_to(site.resolve()):
        raise RepairError("The project package directory is missing or outside its site-packages directory.")
    native = sum(p.is_file() and p.suffix in {".so", ".dylib", ".pyd"} for p in module.rglob("*"))
    return {"package": PACKAGE, "version": version, "pure_python_wheel_available": True,
            "native_extensions_present": bool(native), "native_extension_count": native}, wheel


def install_locked_pure_python(root: Path) -> dict:
    before, wheel = inspect_project(root)
    uv = shutil.which("uv")
    if not uv and os.access("/opt/homebrew/bin/uv", os.X_OK):
        uv = "/opt/homebrew/bin/uv"
    if not uv:
        raise RepairError("uv is unavailable. No package change was attempted.")
    command = [uv, "pip", "install", "--python", str(root / ".venv" / "bin" / "python"),
               "--no-deps", "--reinstall", "--no-cache", "--link-mode", "copy", wheel]
    try:
        subprocess.run(command, cwd=root, check=True, timeout=120,
                       capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        raise RepairError("Package installation failed or timed out. Its outcome is unverified; rerun --check. Raw installer output was withheld.") from None
    after, _ = inspect_project(root)
    if after["version"] != before["version"] or after["native_extensions_present"]:
        raise RepairError("Installation completed but the same-version pure-Python package could not be verified. Rerun --check.")
    return {**after, "action": "pure_python_package_installed",
            "security_settings_changed": False, "voice_demo_started": False}


def main(argv=None, *, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only inspection (default)")
    mode.add_argument("--install", action="store_true", help="Explicitly download and replace only the locked package with its pure-Python build")
    args = parser.parse_args(argv)
    try:
        if args.install:
            result = install_locked_pure_python(root)
        else:
            result, _ = inspect_project(root)
            result["action"] = "not_applied"
            result["next_step"] = ".venv/bin/python scripts/repair_charset_normalizer.py --install"
        print(json.dumps(result, indent=2))
        print("This is not a malware assessment or a live voice test. No macOS security settings are changed.")
        return 0
    except RepairError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("Package inspection failed. No native extension was loaded; details were withheld.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
