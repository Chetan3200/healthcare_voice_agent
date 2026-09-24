"""Exercise the real HF cache completeness check without downloads or MLX."""

import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("huggingface_hub")
from huggingface_hub import snapshot_download
from huggingface_hub.errors import IncompleteSnapshotError

from healthcare_voice_agent.voice import local_assets, providers


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Whisper cache regression must not use the network")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "local-models"
    hub_cache = root / "huggingface"
    repo = hub_cache / "models--mlx-community--whisper-large-v3-turbo"
    snapshot = repo / "snapshots" / local_assets.WHISPER_REVISION
    snapshot.mkdir(parents=True)
    for name in local_assets.WHISPER_FILES:
        (snapshot / name).write_bytes(b"fixture")
    trees = repo / "trees"
    trees.mkdir()
    # Preparation intentionally downloaded two files from a four-file repo.
    (trees / f"{local_assets.WHISPER_REVISION}.json").write_text(json.dumps({
        "format_version": 1,
        "files": {name: {"size": 7, "blob_id": "1" * 40} for name in (
            ".gitattributes", "README.md", *local_assets.WHISPER_FILES,
        )},
    }))
    loaded = []
    holder = SimpleNamespace(get_model=lambda path, dtype: loaded.append((path, dtype)))
    module = object()
    monkeypatch.setattr(providers, "model_cache", lambda: root)
    monkeypatch.setattr(providers, "_whisper_runtime", None)
    monkeypatch.setattr(providers, "_load_whisper_components", lambda: (
        SimpleNamespace(float16="fp16"), module, snapshot_download, holder,
    ))
    return SimpleNamespace(snapshot=snapshot, hub_cache=hub_cache, loaded=loaded,
                           module=module, holder=holder)


def test_partial_repo_is_complete_for_the_actual_model_files(cache):
    # Reproduce the previous failure with the real pinned HF SDK.
    with pytest.raises(IncompleteSnapshotError, match="README.md"):
        snapshot_download(repo_id=local_assets.WHISPER_MODEL,
                          revision=local_assets.WHISPER_REVISION,
                          cache_dir=str(cache.hub_cache), token=False,
                          local_files_only=True)
    # Application load must request only the same files as preparation.
    assert providers._get_whisper_runtime() == (cache.module, str(cache.snapshot))
    assert cache.loaded == [(str(cache.snapshot), "fp16")]
    assert not (cache.snapshot / "README.md").exists()
    assert not (cache.snapshot / ".gitattributes").exists()


@pytest.mark.parametrize("missing", local_assets.WHISPER_FILES)
def test_missing_required_file_still_fails_closed(cache, missing):
    (cache.snapshot / missing).unlink()
    with pytest.raises(providers.ProviderDependencyError, match="prepare-local-models"):
        providers._get_whisper_runtime()
    assert cache.loaded == []


@pytest.mark.parametrize("error", [ValueError("private-marker"),
                                     ImportError("No Metal device available private-marker")])
def test_initialization_error_does_not_prescribe_redownload(cache, error):
    def fail(*args):
        raise error
    cache.holder.get_model = fail
    with pytest.raises(providers.ProviderDependencyError) as caught:
        providers._get_whisper_runtime()
    message = str(caught.value)
    assert "model files were found" in message
    assert "prepare-local-models" not in message
    assert "private-marker" not in message
    if isinstance(error, ImportError):
        assert "Metal GPU access" in message


def test_launcher_never_synchronizes_dependencies():
    launcher = Path(__file__).resolve().parents[2] / "scripts/voice_local.sh"
    invocation = launcher.read_text().split('exec "$UV_BIN"', 1)[1]
    assert "--no-sync" in invocation
    assert "--no-python-downloads" in invocation
