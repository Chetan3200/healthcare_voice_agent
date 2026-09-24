"""Pinned local model identities. No provider imports, downloads or inference."""

from pathlib import Path

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
WHISPER_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
WHISPER_FILES = ("config.json", "weights.safetensors")
KOKORO_RELEASE = "model-files-v1.0"
KOKORO_FILES = {
    "kokoro-v1.0.onnx": {"size": 325532387, "sha256": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"},
    "voices-v1.0.bin": {"size": 28214398, "sha256": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"},
}
KOKORO_RELEASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/" + KOKORO_RELEASE
)


def model_cache() -> Path:
    """Run from the project root, like the existing .env/run-directory policy."""
    return Path("models/local-voice").resolve()


def local_asset_record(stt_provider: str, tts_provider: str) -> dict:
    result = {}
    if stt_provider == "whisper":
        result["stt"] = {"model": WHISPER_MODEL, "revision": WHISPER_REVISION,
                         "runtime": "mlx-whisper", "device": "Metal GPU",
                         "mode": "VAD-segmented, not native streaming"}
    if tts_provider == "kokoro":
        result["tts"] = {"model": "kokoro-v1.0", "release": KOKORO_RELEASE,
                         "runtime": "kokoro-onnx", "device": "CPUExecutionProvider",
                         "files": KOKORO_FILES,
                         "mode": "text-chunk synthesis then PCM chunk delivery"}
    return result
