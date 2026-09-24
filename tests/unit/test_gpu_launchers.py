"""Offline shell contract tests. Stub executables, no GPU, models or network."""

import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/serve_gpu_model.sh"
MANIFEST = json.loads((ROOT / "deploy/gpu-models.lock.json").read_text())


def executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nset -eu\n" + content)
    path.chmod(0o700)
    return path


@pytest.fixture
def env(tmp_path):
    # Use a minimal environment so local .env, keys and server variables cannot
    # enter subprocesses. Fake git only resolves the configured source revision.
    bin_dir = tmp_path / "bin"
    executable(bin_dir / "git", 'printf "%s\\n" "$FAKE_COMMIT"\n')
    return {"PATH": str(bin_dir) + os.pathsep + os.environ["PATH"], "HOME": str(tmp_path)}


def run(args, env):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env,
                          capture_output=True, text=True, timeout=10)


def test_manifest_has_immutable_source_and_model_revisions():
    for model in MANIFEST["models"].values():
        for name in ("runtime_commit", "model_revision"):
            assert len(model[name]) == 40
            assert all(c in "0123456789abcdef" for c in model[name])
    nemo = MANIFEST["models"]["nemotron"]
    assert nemo["model_revision"] in nemo["artifact_url"]
    assert len(nemo["artifact_sha256"]) == 64
    assert nemo["artifact_size_bytes"] == 741548352


@pytest.mark.parametrize("args", [[], ["missing"], ["breeze", "--host", "0.0.0.0"]])
def test_bad_usage_does_not_start_anything(args, env):
    assert run(args, env).returncode == 2


def test_help_does_not_need_runtime(env):
    result = run(["--help"], env)
    assert result.returncode == 0
    assert "docs/gpu-models.md" in result.stdout


@pytest.mark.parametrize("provider", ["nemotron", "breeze", "hybrid_diffusion"])
def test_missing_runtime_fails_before_inference(provider, env):
    assert run([provider], env).returncode != 0


def prepare_breeze(tmp_path, env):
    source = tmp_path / "breeze source"
    source.mkdir()
    model = tmp_path / "breeze model"
    for file in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors.index.json", "model-00001-of-00002.safetensors",
                 "model-00002-of-00002.safetensors", "audio_tokenizer/config.json",
                 "audio_tokenizer/model.safetensors"):
        path = model / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture")
    python = executable(tmp_path / "env/bin/python", 'printf "%s\\n" "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" "$PWD" "$@"\n')
    env.update(BREEZE_SOURCE_DIR=str(source), BREEZE_MODEL_PATH=str(model),
               BREEZE_PYTHON=str(python), FAKE_COMMIT=MANIFEST["models"]["breeze"]["runtime_commit"])
    return source, model


@pytest.mark.parametrize("fast", ["false", "true"])
def test_breeze_loopback_offline_correct_module_and_fast_opt_in(tmp_path, env, fast):
    source, model = prepare_breeze(tmp_path, env)
    env["GPU_BREEZE_FAST"] = fast
    result = run(["breeze"], env)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:3] == ["1", "1", str(source)]
    assert lines[3:10] == ["-m", "breeze_infer.api", str(model), "--host", "127.0.0.1", "--port", "7861"]
    assert ("--fast-all" in lines) == (fast == "true")


def test_breeze_missing_checkpoint_file_rejected(tmp_path, env):
    _, model = prepare_breeze(tmp_path, env)
    (model / "audio_tokenizer/model.safetensors").unlink()
    result = run(["breeze"], env)
    assert result.returncode != 0
    assert "missing or empty" in result.stderr


def test_wrong_runtime_revision_rejected(tmp_path, env):
    prepare_breeze(tmp_path, env)
    env["FAKE_COMMIT"] = "0" * 40
    result = run(["breeze"], env)
    assert result.returncode != 0
    assert "revision" in result.stderr


def prepare_hybrid(tmp_path, env):
    source = tmp_path / "hybrid source"
    executable(source / "eval/scripts/serve.sh", 'printf "%s\\n" "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" "$HOST" "$PORT" "$MAX_RUNNING_REQUESTS" "$MEM_FRACTION_STATIC" "$CUDA_GRAPH_BS" "$@"\n')
    cache = tmp_path / "cache"
    executable(cache / "venvs/hybrid-diffusion-eval/bin/python", "exit 99\n")
    model = tmp_path / "hybrid model"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (model / name).write_text("synthetic fixture")
    env.update(HYBRID_SOURCE_DIR=str(source), HYBRID_MODEL_PATH=str(model),
               HYBRID_DIFFUSION_CACHE_ROOT=str(cache),
               FAKE_COMMIT=MANIFEST["models"]["hybrid_diffusion"]["runtime_commit"])
    return model


@pytest.mark.parametrize("mode", ["self-spec", "diffusion", "causal"])
def test_hybrid_loopback_limits_parser_and_offline_flags(tmp_path, env, mode):
    model = prepare_hybrid(tmp_path, env)
    env.update(GPU_HYBRID_MODE=mode, HOST="0.0.0.0", PORT="1234", MAX_RUNNING_REQUESTS="100")
    result = run(["hybrid_diffusion"], env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "1", "1", "127.0.0.1", "30000", "1", "0.40", "1", mode, str(model), "--",
        "--served-model-name", "yuchen-zhu-zyc/HybridDiffusion-2B", "--tool-call-parser", "qwen3_coder",
    ]


@pytest.mark.parametrize("value", ["0.000", "1", "-1", "nan", "0.5;echo unsafe"])
def test_hybrid_rejects_invalid_memory_fraction(tmp_path, env, value):
    prepare_hybrid(tmp_path, env)
    env["GPU_HYBRID_MEM_FRACTION"] = value
    assert run(["hybrid_diffusion"], env).returncode != 0


def test_nemotron_rejects_wrong_artifact_digest_and_inherited_engines(tmp_path, env):
    source = tmp_path / "nemo"
    executable(source / "build/cuda-server/bin/nemo-speech", "echo should-not-run; exit 99\n")
    model = tmp_path / "synthetic.gguf"
    model.write_bytes(b"not a real model")
    env.update(NEMO_SOURCE_DIR=str(source), NEMO_MODEL_PATH=str(model),
               FAKE_COMMIT=MANIFEST["models"]["nemotron"]["runtime_commit"])
    result = run(["nemotron"], env)
    assert result.returncode != 0
    assert "checksum" in result.stderr
    env["NEMO_SPEECH_TTS_MODEL_PATH"] = "magpie"
    result = run(["nemotron"], env)
    assert result.returncode != 0
    assert "Unset NEMO_SPEECH_" in result.stderr
    assert "should-not-run" not in result.stdout


def test_application_profile_selects_servers_without_installing_or_enabling_live(tmp_path, env):
    app = tmp_path / "app"
    app.mkdir()
    (app / "scripts").mkdir()
    launcher = app / "scripts/voice_gpu.sh"
    launcher.write_bytes((ROOT / "scripts/voice_gpu.sh").read_bytes())
    executable(app / ".venv/bin/python", "exit 99\n")
    uv = executable(tmp_path / "bin/uv", 'printf "%s\\n" "$STT_PROVIDER" "$STT_BASE_URL" "$LLM_PROVIDER" "$TTS_PROVIDER" "${LIVE_API_ENABLED-unset}" "$HF_HUB_OFFLINE" "$@"\n')
    env["UV_BIN"] = str(uv)
    result = subprocess.run(["sh", str(launcher), "--check-config"], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "nemotron", "ws://127.0.0.1:8080/v1/audio/transcriptions/realtime",
        "hybrid_diffusion", "breeze", "unset", "1", "run", "--no-sync",
        "--no-python-downloads", "--locked", "--extra", "voice", "python",
        "-m", "healthcare_voice_agent", "--check-config",
    ]
    assert not (app / ".env").exists()
