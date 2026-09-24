#!/usr/bin/env python3
"""Frozen local Whisper comparison using synthetic WAVs from the Kokoro benchmark.

No hosted API, .env, downloads or installs. Run from the project root in a normal
Terminal with Metal access. Both modes use exactly the same checksum-verified
16-kHz inputs. Reference WER is a synthetic-TTS proxy, not human ASR accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from healthcare_voice_agent.voice.providers import _whisper_transcribe, ProviderDependencyError


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def words(text):
    # Whitespace tokens, with Unicode word characters retained. Hindi combining
    # marks remain present; only punctuation-category characters are removed.
    import unicodedata
    return "".join(c if not unicodedata.category(c).startswith("P") else " " for c in text.casefold()).split()


def word_error_rate(reference, hypothesis):
    ref, hyp = words(reference), words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for index, left in enumerate(ref, 1):
        current = [index]
        for j, right in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1] / max(1, len(ref))


def load_inputs(audio_dir):
    manifest = json.loads((audio_dir / "manifest.json").read_text())
    if len(manifest["items"]) != 4:
        raise ValueError("Expected all four frozen synthetic inputs")
    inputs = []
    for item in manifest["items"]:
        file = (audio_dir / item["file"]).resolve()
        if file.parent != audio_dir.resolve():
            raise ValueError("Invalid audio manifest path")
        if hashlib.sha256(file.read_bytes()).hexdigest() != item["wav_sha256"]:
            raise ValueError("Frozen WAV checksum mismatch")
        with wave.open(str(file), "rb") as handle:
            if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 24000):
                raise ValueError("Expected mono PCM16 at 24 kHz")
            waveform = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768
        audio = resample_poly(waveform, 2, 3).astype(np.float32)
        inputs.append((item, audio))
    return inputs


def run(args):
    if args.output.exists():
        raise ValueError("--output must be a new directory")
    inputs = load_inputs(args.audio_dir)
    args.output.mkdir(parents=True)
    write_json(args.output / "manifest.json", {
        "source_audio_manifest_sha256": hashlib.sha256((args.audio_dir / "manifest.json").read_bytes()).hexdigest(),
        "inputs": [dict(item, inference_pcm_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
                        inference_sample_rate=16000) for item, audio in inputs],
        "python": sys.version, "platform": platform.platform(), "repeats": args.repeats,
        "policy": "Cache-only, automatic language detection, temperature=0, unchanged model, no APIs.",
    })
    try:
        # Warm each code path on identical synthetic input outside timed samples.
        for reuse in (False, True):
            _whisper_transcribe(inputs[0][1], language=None, reuse_encoder=reuse)
    except ProviderDependencyError as exc:
        write_json(args.output / "blocked.json", {"status": "blocked", "reason": str(exc)})
        print("BLOCKED: native Whisper runtime unavailable; see blocked.json")
        return 2
    rows = []
    with (args.output / "results.jsonl").open("w") as out:
        for repeat in range(1, args.repeats + 1):
            cases = inputs if repeat % 2 else list(reversed(inputs))
            modes = (False, True) if repeat % 2 else (True, False)
            for item, audio in cases:
                for reuse in modes:
                    print(f"repeat={repeat} case={item['case_id']} reuse_encoder={reuse}", flush=True)
                    started = time.perf_counter()
                    try:
                        result = _whisper_transcribe(audio, language=None, reuse_encoder=reuse)
                    except Exception:
                        write_json(args.output / "blocked.json", {
                            "status": "blocked", "reason": "native_inference_failed", "completed_rows": len(rows),
                        })
                        return 2
                    row = {"repeat": repeat, "case_id": item["case_id"], "reuse_encoder": reuse,
                           "seconds": time.perf_counter() - started,
                           "audio_seconds": len(audio) / 16000,
                           "decode_path": result.get("local_decode_path", "stock"),
                           "text": result["text"], "language": result["language"],
                           "synthetic_reference_wer": word_error_rate(item["synthetic_text"], result["text"])}
                    rows.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
    comparisons = []
    for item, _ in inputs:
        old = [r for r in rows if r["case_id"] == item["case_id"] and not r["reuse_encoder"]]
        new = [r for r in rows if r["case_id"] == item["case_id"] and r["reuse_encoder"]]
        comparisons.append({
            "case_id": item["case_id"],
            "stock_median_seconds": statistics.median(r["seconds"] for r in old),
            "optimized_median_seconds": statistics.median(r["seconds"] for r in new),
            "all_normalized_texts_match": all(words(a["text"]) == words(b["text"]) for a, b in zip(old, new)),
            "all_languages_match": all(a["language"] == b["language"] for a, b in zip(old, new)),
            "optimized_paths": [r["decode_path"] for r in new],
            "stock_reference_wer": [r["synthetic_reference_wer"] for r in old],
            "optimized_reference_wer": [r["synthetic_reference_wer"] for r in new],
        })
    write_json(args.output / "summary.json", {
        "status": "completed", "comparisons": comparisons,
        "limits": "Fixed synthetic local inference only; no browser timing, held-out test accuracy or automatic tuning.",
    })
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()
    try:
        raise SystemExit(run(args))
    except ValueError as exc:
        parser.error(str(exc))
