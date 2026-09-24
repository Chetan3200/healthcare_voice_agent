#!/usr/bin/env python3
"""Bounded, cache-only CPU benchmark for fixed synthetic Kokoro speech inputs.

It measures local synthesis timing only.  Piece-ready playback-gap estimates are
scheduling estimates, not real playback or browser/end-to-end latency.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import statistics
import sys
import time
import wave
from importlib import metadata
from pathlib import Path
from typing import Any

from healthcare_voice_agent.voice import local_tts
from healthcare_voice_agent.voice.providers import _create_kokoro_runtime, _native_import_reason
from healthcare_voice_agent.voice.speech_chunks import split_for_first_audio

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "evals/scenarios/local-speech-optimization-v1.json"
SAMPLE_RATE = 24000
EXPECTED_CASE_IDS = ("short_english", "long_english", "hindi", "mixed_hindi_english")


def sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("kokoro-onnx", "onnxruntime", "numpy", "pipecat-ai"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def load_cases() -> list[dict[str, str]]:
    fixture = json.loads(SCENARIO.read_text(encoding="utf-8"))
    cases = fixture.get("cases")
    if not isinstance(cases, list) or tuple(item.get("id") for item in cases) != EXPECTED_CASE_IDS:
        raise ValueError("fixed synthetic scenario fixture is invalid")
    if any(not isinstance(item.get("text"), str) or not item["text"].strip() for item in cases):
        raise ValueError("fixed synthetic scenario has invalid text")
    return [{"id": item["id"], "text": item["text"]} for item in cases]


def route(text: str) -> tuple[str, str]:
    if local_tts._has_devanagari_letter(text):
        return "hf_alpha", "hi"
    return "af_heart", "en-us"


def percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    return sorted(values)[max(0, math.ceil(len(values) * 0.95) - 1)]


async def heartbeat(stop: asyncio.Event, lags: list[float]) -> None:
    """Record event-loop scheduling lag at an approximately 10 ms cadence."""
    deadline = time.perf_counter() + 0.01
    while not stop.is_set():
        await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        now = time.perf_counter()
        lags.append(max(0.0, now - deadline))
        deadline += 0.01


async def synthesize_case(engine: Any, case: dict[str, str], chunk_chars: int) -> tuple[dict[str, Any], list[bytes] | None]:
    text = case["text"]
    voice, language = route(text)
    pieces = split_for_first_audio(text, chunk_chars)
    if not pieces or "".join(pieces) != text:
        raise RuntimeError("fixed splitter text-preservation check failed")
    started = time.perf_counter()
    stop, lags = asyncio.Event(), []
    monitor = asyncio.create_task(heartbeat(stop, lags))
    pcm_parts: list[bytes] = []
    ready, durations = [], []
    failure = None
    try:
        for part in pieces:
            audio, sample_rate = await asyncio.to_thread(
                engine.create, part, voice=voice, lang=language, speed=1.0
            )
            pcm = local_tts._pcm16(audio, sample_rate)
            pcm_parts.append(pcm)
            ready.append(time.perf_counter() - started)
            durations.append(len(pcm) / (2 * SAMPLE_RATE))
    except Exception:
        failure = "synthesis_or_audio_validation_failed"
    finally:
        finished = time.perf_counter()
        stop.set()
        await monitor
    total = finished - started
    gaps = []
    playback_end = ready[0] + durations[0] if ready else 0.0
    for piece_ready, duration in zip(ready[1:], durations[1:]):
        gaps.append(max(0.0, piece_ready - playback_end))
        playback_end = max(playback_end, piece_ready) + duration
    row: dict[str, Any] = {
        "case_id": case["id"],
        "synthetic_text": text,
        "text_sha256": sha256(text),
        "voice": voice,
        "language": language,
        "pieces": len(pieces),
        "piece_texts": list(pieces),
        "piece_ready_seconds": ready,
        "piece_audio_seconds": durations,
        "estimated_playback_gap_seconds": gaps,
        "estimated_playback_gap_note": "Readiness-versus-prior-duration estimate only; no playback device was measured.",
        "first_pcm_seconds": ready[0] if ready else None,
        "total_seconds": total,
        "total_audio_seconds": sum(durations),
        "heartbeat_lag_p95_ms": percentile95(lags) * 1000,
        "heartbeat_lag_max_ms": (max(lags) if lags else 0.0) * 1000,
        "heartbeat_samples": len(lags),
        "status": "failed" if failure else "ok",
    }
    if failure:
        row["failure"] = failure
        return row, None
    return row, pcm_parts


async def warm_engine(engine: Any) -> None:
    """Warm both fixed routes without recording them as benchmark samples."""
    for text in ("Synthetic English warmup.", "सिंथेटिक हिंदी वार्मअप।"):
        voice, language = route(text)
        audio, sample_rate = await asyncio.to_thread(engine.create, text, voice=voice, lang=language, speed=1.0)
        local_tts._pcm16(audio, sample_rate)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(pcm)


def median_range(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [row[field] for row in rows if row["status"] == "ok" and row[field] is not None]
    return {"median": statistics.median(values), "min": min(values), "max": max(values)} if values else {}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["cpu_threads"], row["first_chunk_chars"], row["case_id"]), []).append(row)
    return [
        {
            "cpu_threads": threads, "first_chunk_chars": chars, "case_id": case_id,
            "samples": len(group), "successful_samples": sum(row["status"] == "ok" for row in group),
            "first_pcm_seconds": median_range(group, "first_pcm_seconds"),
            "total_seconds": median_range(group, "total_seconds"),
            "total_audio_seconds": median_range(group, "total_audio_seconds"),
        }
        for (threads, chars, case_id), group in sorted(groups.items())
    ]


def blocked(output: Path, message: str, settings: dict[str, Any]) -> int:
    write_json(output / "blocked.json", {"status": "blocked", "reason": message, "settings": settings,
                                           "detail_policy": "Native exception details intentionally omitted."})
    print("BLOCKED: local Kokoro initialization or warmup failed; wrote blocked.json", file=sys.stderr)
    return 2


async def run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise ValueError("--output must name a new directory")
    if set(args.threads) != {2, 4} or len(args.threads) != 2:
        raise ValueError("--threads must contain exactly 2 and 4")
    if set(args.chunk_chars) != {0, 60} or len(args.chunk_chars) != 2:
        raise ValueError("--chunk-chars must contain exactly 0 and 60")
    args.output.mkdir(parents=True)
    cases = load_cases()
    settings = {"repeats": args.repeats, "threads": args.threads, "chunk_chars": args.chunk_chars}
    environment = {
        "platform": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine()},
        "package_versions": package_versions(),
        "provider_model_policy": "Kokoro v1.0 cache-only CPU engine via providers._create_kokoro_runtime; no downloads, hosted providers, .env reads, or browser measurement.",
        "scenario_sha256": sha256(SCENARIO.read_bytes()),
    }
    write_json(args.output / "run_metadata.json", {
        "suite_id": "local-speech-optimization-v1", "settings": settings, **environment,
    })
    engines: dict[int, Any] = {}
    for threads in sorted(args.threads):
        try:
            engines[threads] = _create_kokoro_runtime(threads)
            await warm_engine(engines[threads])
        except Exception as exc:
            return blocked(args.output, "cache_only_engine_initialization_or_warmup_failed:" + _native_import_reason(exc), settings)

    rows: list[dict[str, Any]] = []
    baseline: list[dict[str, Any]] = []
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as log:
        for repeat in range(1, args.repeats + 1):
            ordered_cases = cases if repeat % 2 else list(reversed(cases))
            settings_order = [(threads, chars) for threads in args.threads for chars in args.chunk_chars]
            if not repeat % 2:
                settings_order.reverse()
            for threads, chunk_chars in settings_order:
                for case in ordered_cases:
                    print(f"progress repeat={repeat} threads={threads} first_chunk_chars={chunk_chars} case={case['id']}", flush=True)
                    row, pcm_parts = await synthesize_case(engines[threads], case, chunk_chars)
                    row.update({"repeat": repeat, "cpu_threads": threads, "first_chunk_chars": chunk_chars})
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    log.flush()
                    rows.append(row)
                    if threads == 2 and chunk_chars == 0 and repeat == 1 and pcm_parts is not None:
                        baseline.append({"case": case, "pcm": b"".join(pcm_parts)})
    audio_dir = args.output / "audio"
    audio_dir.mkdir()
    manifest = []
    for item in baseline:
        filename = f"{item['case']['id']}.wav"
        destination = audio_dir / filename
        write_wav(destination, item["pcm"])
        manifest.append({"case_id": item["case"]["id"], "synthetic_text": item["case"]["text"],
                         "text_sha256": sha256(item["case"]["text"]), "file": filename,
                         "wav_sha256": sha256(destination.read_bytes()), "sample_rate": SAMPLE_RATE,
                         "format": "mono PCM16 little-endian WAV", "source": "threads=2, first_chunk_chars=0, repeat=1"})
    write_json(audio_dir / "manifest.json", {"purpose": "Frozen synthetic local-TTS outputs for later Whisper inputs.", "items": manifest})
    failures = sum(row["status"] != "ok" for row in rows)
    write_json(args.output / "summary.json", {
        "suite_id": "local-speech-optimization-v1", "status": "failed" if failures else "completed", "settings": settings, **environment,
        "rows": len(rows), "failures": failures,
        "summary_by_setting_case": summarize(rows),
        "measurement_scope": "Offline local synthesis only. Estimated playback gaps are not playback measurements; no browser or end-to-end latency was measured.",
    })
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", nargs="+", type=int, default=[2, 4], choices=[2, 4])
    parser.add_argument("--chunk-chars", nargs="+", type=int, default=[0, 60], choices=[0, 60])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run(parse_args())))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
