"""Safe per-dependency diagnostics; never imports native runtimes or models."""

import importlib
from types import SimpleNamespace

import pytest

from healthcare_voice_agent.voice import providers


@pytest.mark.parametrize("error,reason", [
    (ImportError("code signature invalid /private-marker/lib.so"), "macos_library_policy"),
    (OSError("library load disallowed by system policy private-marker"), "macos_library_policy"),
    (ImportError("[metal::load_device] No Metal device available private-marker"), "metal_unavailable"),
    (ModuleNotFoundError("No module named private-marker"), "missing_dependency"),
    (ImportError("unknown failure private-marker"), "native_import_failed"),
])
def test_import_diagnostic_is_specific_without_raw_error(monkeypatch, error, reason):
    def fail(name):
        raise error
    monkeypatch.setattr(importlib, "import_module", fail)
    with pytest.raises(providers.ProviderDependencyError) as caught:
        providers._import_local_dependency("torch", "torch")
    message = str(caught.value)
    assert "torch" in message
    assert f"[{reason}]" in message
    assert "private-marker" not in message
    assert caught.value.__suppress_context__ is True
    if reason == "metal_unavailable":
        assert "Reinstalling packages does not fix GPU access" in message


def test_scipy_wrapper_preserves_nested_system_policy_category():
    original = ImportError("library load disallowed by system policy private-marker")
    wrapper = ImportError("scipy install seems broken")
    wrapper.__cause__ = original
    assert providers._native_import_reason(wrapper) == "macos_library_policy"


def test_exception_cycle_is_bounded():
    error = ImportError("unknown")
    error.__cause__ = error
    assert providers._native_import_reason(error) == "native_import_failed"


def test_whisper_transitive_imports_identified_before_mlx(monkeypatch):
    monkeypatch.setattr(providers.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(providers.platform, "machine", lambda: "arm64")
    calls = []
    modules = {
        "torch": object(), "tiktoken": object(), "scipy": object(),
        "mlx.core": object(), "mlx_whisper": object(),
        "huggingface_hub": SimpleNamespace(snapshot_download=object()),
        "mlx_whisper.transcribe": SimpleNamespace(ModelHolder=object()),
    }
    def fake_import(name, package):
        calls.append((name, package))
        return modules[name]
    monkeypatch.setattr(providers, "_import_local_dependency", fake_import)
    result = providers._load_whisper_components()
    assert [package for _, package in calls[:3]] == ["torch", "tiktoken", "scipy"]
    assert result == (modules["mlx.core"], modules["mlx_whisper"],
                      modules["huggingface_hub"].snapshot_download,
                      modules["mlx_whisper.transcribe"].ModelHolder)
