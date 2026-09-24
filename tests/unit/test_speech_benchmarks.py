"""Exercise benchmark bookkeeping with fake speech engines, never model calls."""

import asyncio
import importlib.util
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pipecat")
ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Benchmark tests must be offline")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


@pytest.fixture
def tts_benchmark(tmp_path, monkeypatch):
    benchmark = module("benchmark_local_speech")
    created = []
    class Engine:
        def create(self, text, **kwargs):
            return np.full(1200, .1, dtype=np.float32), 24000
    def create(threads):
        created.append(threads)
        return Engine()
    monkeypatch.setattr(benchmark, "_create_kokoro_runtime", create)
    output = tmp_path / "tts"
    args = SimpleNamespace(output=output, repeats=2, threads=[2, 4], chunk_chars=[0, 60])
    assert asyncio.run(benchmark.run(args)) == 0
    return benchmark, output, created


def test_fixed_cpu_comparison_has_all_rows_and_frozen_wavs(tts_benchmark):
    _, output, created = tts_benchmark
    assert created == [2, 4]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["rows"] == 32 and summary["failures"] == 0
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert all("".join(row["piece_texts"]) == row["synthetic_text"] for row in rows)
    assert all(row["first_pcm_seconds"] <= row["total_seconds"] for row in rows)
    assert (rows[0]["cpu_threads"], rows[0]["first_chunk_chars"]) == (2, 0)
    assert (rows[16]["cpu_threads"], rows[16]["first_chunk_chars"]) == (4, 60)
    manifest = json.loads((output / "audio/manifest.json").read_text())
    assert len(manifest["items"]) == 4
    assert all((output / "audio" / item["file"]).is_file() for item in manifest["items"])


def test_whisper_comparison_reuses_identical_frozen_inputs(tts_benchmark, tmp_path, monkeypatch):
    _, source, _ = tts_benchmark
    benchmark = module("benchmark_whisper_reuse")
    calls = []
    def transcribe(audio, language, reuse_encoder):
        calls.append((audio, language, reuse_encoder))
        return {"text": "कृपया reports लाएँ", "language": "hi", "local_decode_path": "single_encode" if reuse_encoder else "stock"}
    monkeypatch.setattr(benchmark, "_whisper_transcribe", transcribe)
    output = tmp_path / "stt"
    assert benchmark.run(SimpleNamespace(audio_dir=source / "audio", output=output, repeats=2)) == 0
    assert len(calls) == 18  # Two untimed warmups plus 16 measured calls.
    assert all(language is None for _, language, _ in calls)
    for index in range(2, len(calls), 2):
        assert calls[index][0] is calls[index + 1][0]
        assert calls[index][2] is not calls[index + 1][2]
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["comparisons"]) == 4
    assert all(row["all_normalized_texts_match"] and row["all_languages_match"] for row in summary["comparisons"])
    assert benchmark.word_error_rate("कृपया reports लाएँ।", "कृपया reports लाएँ") == 0
    audio_file = next((source / "audio").glob("*.wav"))
    audio_file.write_bytes(audio_file.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        benchmark.load_inputs(source / "audio")


def test_failed_native_warmup_is_reported_not_counted_as_measurement(tmp_path, monkeypatch):
    benchmark = module("benchmark_local_speech")
    def fail(*args, **kwargs):
        raise OSError("private native path marker")
    monkeypatch.setattr(benchmark, "_create_kokoro_runtime", fail)
    output = tmp_path / "blocked"
    assert asyncio.run(benchmark.run(SimpleNamespace(output=output, repeats=2, threads=[2, 4], chunk_chars=[0, 60]))) == 2
    report = (output / "blocked.json").read_text()
    assert "private native path marker" not in report
    assert not (output / "summary.json").exists()
