"""Validated provider settings and secret-free run records.

Read .env once at the application boundary. Environment variables take priority.
This module neither imports speech providers nor contacts external services.
"""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Mapping
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import (
    BaseModel, ConfigDict, Field, SecretStr, StringConstraints,
    ValidationError, field_validator, model_validator,
)

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
DemoWebSearchDelaySeconds = Annotated[float, Field(ge=0, le=15, allow_inf_nan=False)]


class ConfigurationError(ValueError):
    """A configuration problem that is safe to display without input values."""


class SettingsModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class ProviderSettings(SettingsModel):
    supported_providers: ClassVar[tuple[str, ...]] = ("openai",)
    provider: NonEmpty = "openai"
    model: NonEmpty
    base_url: NonEmpty | None
    # Only self-hosted stages use these keys. Never borrow the OpenAI credential.
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)

    @field_validator("provider")
    @classmethod
    def supported_provider(cls, value: str) -> str:
        if value not in cls.supported_providers:
            raise ValueError("Unsupported provider for this stage; no fallback is used")
        return value

    @field_validator("base_url")
    @classmethod
    def endpoint_without_credentials(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            url = urlsplit(value)
            _ = url.port  # Reject malformed port numbers too.
        except ValueError:
            raise ValueError("Endpoint must be a valid URL") from None
        if not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("Endpoint must have a host and no credentials, query, or fragment")
        if url.scheme not in {"https", "wss", "http", "ws"}:
            raise ValueError("Endpoint must use HTTPS/WSS, or HTTP/WS on loopback")
        if url.scheme in {"http", "ws"} and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Unencrypted endpoints are allowed only on loopback")
        return value.rstrip("/")


class STTSettings(ProviderSettings):
    supported_providers: ClassVar[tuple[str, ...]] = ("openai", "whisper", "nemotron")
    model: NonEmpty = "gpt-live-transcribe"
    base_url: NonEmpty | None = "wss://api.openai.com/v1/realtime"
    language: NonEmpty | None = "en"  # English by default; None is an explicit auto-detect opt-in.
    ws_close_timeout_seconds: PositiveSeconds = 2.0
    finalize_transcripts: bool = True
    max_segment_seconds: PositiveSeconds = 60.0
    max_pending_segments: Annotated[int, Field(ge=1, le=32)] = 8
    inference_timeout_seconds: PositiveSeconds = 30.0
    reuse_encoder: bool = True

    @model_validator(mode="before")
    @classmethod
    def local_defaults(cls, value):
        if isinstance(value, dict) and value.get("provider") == "nemotron":
            value = dict(value)
            value.setdefault("model", "nemotron-3.5-asr-streaming-0.6b")
            # Full WebSocket URL, not the OpenAI realtime endpoint.
            value.setdefault("base_url", "ws://127.0.0.1:8080/v1/audio/transcriptions/realtime")
        if isinstance(value, dict) and value.get("provider") == "whisper":
            value = dict(value)
            value.setdefault("model", "mlx-community/whisper-large-v3-turbo")
            if value.get("model") == "large-v3-turbo":
                value["model"] = "mlx-community/whisper-large-v3-turbo"
            value.setdefault("base_url", None)
            if value["base_url"] == "":
                value["base_url"] = None
        return value

    @model_validator(mode="after")
    def compatible_settings(self) -> STTSettings:
        if self.provider == "whisper":
            if self.model != "mlx-community/whisper-large-v3-turbo":
                raise ValueError("Whisper requires STT_MODEL=mlx-community/whisper-large-v3-turbo")
            if self.base_url is not None:
                raise ValueError("Local Whisper has no endpoint; unset STT_BASE_URL")
            if not self.finalize_transcripts:
                raise ValueError("Local Whisper requires STT_FINALIZE_TRANSCRIPTS=true")
            if self.language not in {None, "en", "hi"}:
                raise ValueError("Local Whisper supports STT_LANGUAGE=auto, en, or hi in this experiment")
            return self
        if self.base_url is None or urlsplit(self.base_url).scheme not in {"ws", "wss"}:
            raise ValueError("Streaming STT requires a WebSocket endpoint")
        if self.provider == "nemotron":
            if self.model != "nemotron-3.5-asr-streaming-0.6b":
                raise ValueError("Nemotron requires STT_MODEL=nemotron-3.5-asr-streaming-0.6b")
            if urlsplit(self.base_url).hostname == "api.openai.com":
                raise ValueError("Nemotron requires its own STT_BASE_URL")
            if not self.finalize_transcripts:
                raise ValueError("Nemotron requires STT_FINALIZE_TRANSCRIPTS=true")
            return self
        if self.model == "gpt-live-transcribe" and self.language != "en":
            raise ValueError("gpt-live-transcribe requires STT_LANGUAGE=en for this English-only profile")
        return self


class HTTPSettings(ProviderSettings):
    base_url: NonEmpty | None = "https://api.openai.com/v1"
    timeout_seconds: PositiveSeconds = 30.0
    connect_timeout_seconds: PositiveSeconds = 5.0
    keepalive_expiry_seconds: PositiveSeconds = 60.0

    @field_validator("base_url")
    @classmethod
    def http_endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if urlsplit(value).scheme not in {"http", "https"}:
            raise ValueError("LLM and TTS require HTTP endpoints")
        return value


class LLMSettings(HTTPSettings):
    supported_providers: ClassVar[tuple[str, ...]] = ("openai", "hybrid_diffusion")
    model: NonEmpty = "gpt-4.1-mini-2025-04-14"
    max_output_tokens: Annotated[int, Field(gt=0)] = 512

    @model_validator(mode="before")
    @classmethod
    def self_hosted_defaults(cls, value):
        if isinstance(value, dict) and value.get("provider") == "hybrid_diffusion":
            value = dict(value)
            value.setdefault("model", "yuchen-zhu-zyc/HybridDiffusion-2B")
            value.setdefault("base_url", "http://127.0.0.1:30000/v1")
        return value

    @model_validator(mode="after")
    def require_endpoint(self):
        if self.base_url is None:
            raise ValueError("LLM requires an HTTP endpoint")
        if self.provider == "hybrid_diffusion":
            if self.model != "yuchen-zhu-zyc/HybridDiffusion-2B":
                raise ValueError("HybridDiffusion requires LLM_MODEL=yuchen-zhu-zyc/HybridDiffusion-2B")
            if urlsplit(self.base_url).hostname == "api.openai.com":
                raise ValueError("HybridDiffusion requires its own LLM_BASE_URL")
        return self


DEFAULT_TTS_INSTRUCTIONS = (
    "Speak in clear, natural English with a warm, calm, conversational tone, as if "
    "talking with one person. Use natural intonation and short, meaningful pauses. "
    "Read appointment dates and times carefully. Avoid flat, robotic, exaggerated, "
    "or sing-song speech. Read only the supplied English text faithfully, preserving "
    "its wording. Do not translate, transliterate, add commentary, or read these "
    "instructions aloud."
)


class TTSSettings(HTTPSettings):
    supported_providers: ClassVar[tuple[str, ...]] = ("openai", "kokoro", "breeze")
    model: NonEmpty = "gpt-4o-mini-tts"
    voice: NonEmpty = "coral"
    instructions: NonEmpty | None = DEFAULT_TTS_INSTRUCTIONS
    # PCM audio downloaded per streaming chunk, not text aggregation or speed.
    audio_chunk_ms: Annotated[int, Field(ge=20, le=1000)] = 100
    language: Literal["auto", "en", "hi", "zh"] = "en"
    breeze_cfg_scale: Annotated[float, Field(gt=0, le=10, allow_inf_nan=False)] = 4.0
    hindi_voice: NonEmpty = "hf_alpha"
    speed: Annotated[float, Field(ge=0.5, le=2.0, allow_inf_nan=False)] = 1.0
    inference_timeout_seconds: PositiveSeconds = 30.0
    cpu_threads: Annotated[int, Field(ge=1, le=8)] = 2
    first_chunk_chars: Annotated[int, Field(ge=0, le=256)] = 60

    @field_validator("first_chunk_chars")
    @classmethod
    def valid_first_chunk_size(cls, value):
        if value != 0 and value < 32:
            raise ValueError("TTS_FIRST_CHUNK_CHARS must be 0 (disabled) or 32 to 256")
        return value

    @model_validator(mode="before")
    @classmethod
    def local_defaults(cls, value):
        if isinstance(value, dict) and value.get("provider") == "breeze":
            value = dict(value)
            for key, default in {
                "model": "BreezeBlue/Breeze-TTS-2", "voice": "S0",
                "instructions": "A warm, calm conversational voice.",
                "base_url": "http://127.0.0.1:7861",
            }.items():
                value.setdefault(key, default)
        if isinstance(value, dict) and value.get("provider") == "kokoro":
            value = dict(value)
            for key, default in {"model": "kokoro-v1.0", "voice": "af_heart",
                                 "instructions": None, "base_url": None}.items():
                value.setdefault(key, default)
            if value["base_url"] == "":
                value["base_url"] = None
        return value

    @field_validator("instructions", mode="before")
    @classmethod
    def optional_instructions(cls, value):
        if isinstance(value, str) and value.strip().lower() in {"", "none", "off", "disabled"}:
            return None
        return value

    @model_validator(mode="after")
    def compatible_instructions(self) -> TTSSettings:
        if self.provider == "breeze":
            if self.model != "BreezeBlue/Breeze-TTS-2":
                raise ValueError("Breeze requires TTS_MODEL=BreezeBlue/Breeze-TTS-2")
            if self.base_url is None or urlsplit(self.base_url).hostname == "api.openai.com":
                raise ValueError("Breeze requires its own TTS_BASE_URL")
            if self.instructions is None:
                raise ValueError("Breeze voice design requires TTS_INSTRUCTIONS")
            if self.voice != "S0" or self.speed != 1.0:
                raise ValueError("Breeze voice design requires TTS_VOICE=S0 and TTS_SPEED=1")
            if self.language not in {"auto", "en", "zh"}:
                raise ValueError("Breeze supports English/Chinese; use TTS_LANGUAGE=auto, en, or zh")
            return self
        if self.provider == "kokoro":
            if self.language not in {"auto", "en", "hi"}:
                raise ValueError("Kokoro supports TTS_LANGUAGE=auto, en, or hi")
            if self.model != "kokoro-v1.0":
                raise ValueError("Kokoro requires TTS_MODEL=kokoro-v1.0")
            if self.base_url is not None:
                raise ValueError("Local Kokoro has no endpoint; unset TTS_BASE_URL")
            if self.instructions is not None:
                raise ValueError("Kokoro does not support delivery instructions; set TTS_INSTRUCTIONS=off")
            if not self.voice.startswith(("af_", "am_", "bf_", "bm_")):
                raise ValueError("Set TTS_VOICE to a Kokoro English voice, e.g. af_heart")
            if not self.hindi_voice.startswith(("hf_", "hm_")):
                raise ValueError("Set TTS_HINDI_VOICE to a Kokoro Hindi voice, e.g. hf_alpha")
            return self
        if self.base_url is None:
            raise ValueError("OpenAI TTS requires an HTTP endpoint")
        if self.instructions is not None and self.model in {"tts-1", "tts-1-hd"}:
            raise ValueError("This TTS model does not support instructions; set TTS_INSTRUCTIONS=off")
        return self


class VoiceSettings(SettingsModel):
    """Local runtime settings; session and idle cutoffs are disabled by default."""

    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: Annotated[int, Field(ge=1024, le=65535)] = 7860
    input_sample_rate: Literal[16000] = 16000
    output_sample_rate: Literal[24000] = 24000
    smart_turn_stop_seconds: PositiveSeconds = 1.5
    idle_timeout_seconds: PositiveSeconds | None = None
    max_session_seconds: PositiveSeconds | None = None
    setup_timeout_seconds: PositiveSeconds = 20.0
    start_timeout_seconds: PositiveSeconds = 20.0
    cancel_timeout_seconds: PositiveSeconds = 10.0

    @field_validator("idle_timeout_seconds", "max_session_seconds", mode="before")
    @classmethod
    def optional_cutoff(cls, value):
        if isinstance(value, str) and value.strip().lower() in {"", "none", "off", "disabled"}:
            return None
        return value


class ClinicianSettings(SettingsModel):
    """Local synthetic clinical reads and a separately keyed public web search."""
    user_id: NonEmpty = "SYN-USER-CLIN"
    db_host: Literal["127.0.0.1", "localhost", "::1"] = "127.0.0.1"
    db_port: Annotated[int, Field(ge=1, le=65535)] = 5433
    db_name: NonEmpty | None = None
    db_user: SecretStr | None = Field(default=None, exclude=True, repr=False)
    db_password: SecretStr | None = Field(default=None, exclude=True, repr=False)
    exa_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    search_domains: tuple[NonEmpty, ...] = ("nhs.uk", "nice.org.uk", "orthoinfo.aaos.org")
    demo_web_search_delay_seconds: DemoWebSearchDelaySeconds = 0.0


class AppConfig(SettingsModel):
    agent_mode: Literal["conversation", "frontdesk_demo", "clinician"] = "conversation"
    stt: STTSettings = Field(default_factory=STTSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    openai_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    live_api_enabled: bool = False
    clinician: ClinicianSettings = Field(default_factory=ClinicianSettings)

    @model_validator(mode="after")
    def demo_pipeline_requirements(self):
        if self.agent_mode in {"frontdesk_demo", "clinician"}:
            if any(stage.provider != "openai" for stage in (self.stt, self.llm, self.tts)):
                raise ValueError("The front-desk and clinician profiles require OpenAI STT, LLM and TTS.")
            if not self.stt.finalize_transcripts:
                raise ValueError("The tool-enabled voice profiles require finalized transcripts.")
        return self

    @property
    def llm_http_max_retries(self) -> int:
        """One native request retry for clinician LLM calls; no turn/tool replay."""
        return 1 if self.agent_mode == "clinician" else 0

    def require_live_ready(self) -> None:
        """Require live opt-in; an OpenAI key only when an OpenAI stage is selected."""
        if not self.live_api_enabled:
            raise ConfigurationError("Live APIs are disabled. Set LIVE_API_ENABLED=true to use live services.")
        uses_openai = any(stage.provider == "openai" for stage in (self.stt, self.llm, self.tts))
        if uses_openai and (not self.openai_api_key or not self.openai_api_key.get_secret_value().strip()):
            raise ConfigurationError("OPENAI_API_KEY is required for live OpenAI services.")

    def redaction_secrets(self) -> tuple[str, ...]:
        """Private values for log masking only. Never print or serialize this tuple."""
        keys = (self.openai_api_key, self.stt.api_key, self.llm.api_key, self.tts.api_key,
                self.clinician.exa_api_key, self.clinician.db_password)
        return tuple(key.get_secret_value() for key in keys if key)

    def public_dict(self) -> dict:
        """Allow only non-secret configuration into logs and run records."""
        result = self.model_dump(mode="json")  # API key is excluded by its field definition.
        result["openai_key_configured"] = bool(
            self.openai_api_key and self.openai_api_key.get_secret_value().strip()
        )
        result["clinician"]["exa_key_configured"] = bool(self.clinician.exa_api_key)
        return result


# Explicit mappings prevent unrelated database variables/secrets entering voice logs.
_STAGE_ENV = {
    "stt": {
        "provider": "STT_PROVIDER", "model": "STT_MODEL", "base_url": "STT_BASE_URL",
        "api_key": "STT_API_KEY",
        "language": "STT_LANGUAGE", "ws_close_timeout_seconds": "STT_WS_CLOSE_TIMEOUT_SECONDS",
        "finalize_transcripts": "STT_FINALIZE_TRANSCRIPTS",
        "max_segment_seconds": "STT_MAX_SEGMENT_SECONDS",
        "max_pending_segments": "STT_MAX_PENDING_SEGMENTS",
        "inference_timeout_seconds": "STT_INFERENCE_TIMEOUT_SECONDS",
        "reuse_encoder": "STT_REUSE_ENCODER",
    },
    "llm": {
        "provider": "LLM_PROVIDER", "model": "LLM_MODEL", "base_url": "LLM_BASE_URL",
        "api_key": "LLM_API_KEY",
        "timeout_seconds": "LLM_TIMEOUT_SECONDS", "connect_timeout_seconds": "LLM_CONNECT_TIMEOUT_SECONDS",
        "max_output_tokens": "LLM_MAX_OUTPUT_TOKENS",
        "keepalive_expiry_seconds": "LLM_KEEPALIVE_EXPIRY_SECONDS",
    },
    "tts": {
        "provider": "TTS_PROVIDER", "model": "TTS_MODEL", "base_url": "TTS_BASE_URL",
        "api_key": "TTS_API_KEY", "breeze_cfg_scale": "TTS_BREEZE_CFG_SCALE",
        "timeout_seconds": "TTS_TIMEOUT_SECONDS", "connect_timeout_seconds": "TTS_CONNECT_TIMEOUT_SECONDS",
        "voice": "TTS_VOICE", "instructions": "TTS_INSTRUCTIONS",
        "audio_chunk_ms": "TTS_AUDIO_CHUNK_MS",
        "keepalive_expiry_seconds": "TTS_KEEPALIVE_EXPIRY_SECONDS",
        "language": "TTS_LANGUAGE", "hindi_voice": "TTS_HINDI_VOICE",
        "speed": "TTS_SPEED", "inference_timeout_seconds": "TTS_INFERENCE_TIMEOUT_SECONDS",
        "cpu_threads": "TTS_CPU_THREADS", "first_chunk_chars": "TTS_FIRST_CHUNK_CHARS",
    },
    "voice": {
        "port": "VOICE_PORT",
        "smart_turn_stop_seconds": "VOICE_SMART_TURN_STOP_SECONDS",
        "idle_timeout_seconds": "VOICE_IDLE_TIMEOUT_SECONDS",
        "max_session_seconds": "VOICE_MAX_SESSION_SECONDS",
        "setup_timeout_seconds": "VOICE_SETUP_TIMEOUT_SECONDS",
        "start_timeout_seconds": "VOICE_START_TIMEOUT_SECONDS",
        "cancel_timeout_seconds": "VOICE_CANCEL_TIMEOUT_SECONDS",
    },
}


def load_config(
    env_file: str | Path | None = ".env", *, environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load a CWD-relative .env, then environment overrides, without mutating os.environ.

    Run from the repository root or pass an explicit env_file path. Missing .env
    files use defaults/environment. Use env_file=None and environ={} for fully
    isolated offline tests. Variable interpolation is disabled so settings never
    silently depend on other secrets.
    """
    values = dict(dotenv_values(env_file, interpolate=False)) if env_file is not None else {}
    values.update(os.environ if environ is None else environ)
    known_keys = {key for mapping in _STAGE_ENV.values() for key in mapping.values()} | {"AGENT_MODE"}
    unknown_keys = sorted(
        key for key in values
        if key.startswith(("STT_", "LLM_", "TTS_", "VOICE_", "AGENT_")) and key not in known_keys
    )
    if unknown_keys:
        raise ConfigurationError("Unknown provider setting or voice setting names: " + ", ".join(unknown_keys))
    data: dict = {
        stage: {field: values[key] for field, key in mapping.items() if key in values}
        for stage, mapping in _STAGE_ENV.items()
    }
    if (
        "language" in data["stt"]
        and str(data["stt"]["language"]).strip().lower() == "auto"
    ):
        data["stt"]["language"] = None
    data["agent_mode"] = values.get("AGENT_MODE", "conversation")
    if data["agent_mode"] == "clinician":
        mapping = {"user_id": "CLINICIAN_USER_ID",
                   "db_host": "DB_HOST", "db_port": "DB_PORT", "db_name": "POSTGRES_DB",
                   "db_user": "POSTGRES_USER", "db_password": "POSTGRES_PASSWORD",
                   "exa_api_key": "EXA_API_KEY",
                   "demo_web_search_delay_seconds": "CLINICIAN_DEMO_WEB_SEARCH_DELAY_SECONDS"}
        data["clinician"] = {field: values[key] for field, key in mapping.items() if values.get(key)}
        if values.get("EXA_SEARCH_DOMAINS"):
            data["clinician"]["search_domains"] = tuple(
                value.strip() for value in values["EXA_SEARCH_DOMAINS"].split(",") if value.strip())
    data["openai_api_key"] = values.get("OPENAI_API_KEY") or None
    data["live_api_enabled"] = values.get("LIVE_API_ENABLED", "false")
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        messages = [
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors(include_input=False, include_context=False, include_url=False)
        ]
        raise ConfigurationError("; ".join(messages)) from None


def write_run_config(config: AppConfig, run_dir: str | Path) -> Path:
    """Write one exclusive, non-secret config.json for a run; never overwrite it."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in (
        "healthcare-voice-agent", "pipecat-ai", "openai", "httpx2",
        "pipecat-ai-prebuilt", "aiortc", "onnxruntime",
        "mlx-whisper", "mlx", "mlx-metal", "kokoro-onnx", "espeakng-loader",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    from healthcare_voice_agent.voice.local_assets import local_asset_record
    record = {
        "schema_version": 1,
        "local_models": local_asset_record(config.stt.provider, config.tts.provider),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": versions,
        "configuration": config.public_dict(),
        "provider_policy": {
            "llm_http_max_retries": config.llm_http_max_retries,
            "tts_http_max_retries": 0, "stt_auto_reconnect": False,
            "self_hosted_native_validation": "not established by offline tests",
            "hybrid_diffusion_thinking_enabled": False if config.llm.provider == "hybrid_diffusion" else None,
        },
        "voice_policy": {
            "transport": "SmallWebRTCTransport",
            "vad": "SileroVADAnalyzer",
            "turn_end": "LocalSmartTurnAnalyzerV3 (bundled v3.2)",
            "incomplete_turn_filtering": (
                "native conversational LLM gate after Smart Turn (default 5s/10s waits)"
                if config.agent_mode == "clinician" else "disabled"
            ),
            "user_turn_stop_timeout_seconds": 15.0 if config.agent_mode == "clinician" else 5.0,
            "provider_turn_detection": False,
            "stt_finalization": (
                "FIFO local segments with a speech-generation guard"
                if config.stt.provider == "whisper" else
                "Nemotron WebSocket results guarded by local speech generation"
                if config.stt.provider == "nemotron" else
                "ordered committed items with a local speech-generation guard"
                if config.stt.finalize_transcripts else "stock Pipecat transcription metadata"
            ),
            "microphone_gate": "RTVI client-ready; discard pre-ready audio",
            "clinic_tools_enabled": config.agent_mode == "clinician",
            "frontdesk_tools_enabled": config.agent_mode == "frontdesk_demo",
            "booking_database": "isolated synthetic demo only" if config.agent_mode == "frontdesk_demo" else None,
            "booking_confirmation": "direct request, no extra confirmation turn" if config.agent_mode == "frontdesk_demo" else None,
            "raw_audio_recording": False,
            "timing_scope": "server-observed, not measured browser playback",
        },
    }
    target = directory / "config.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return target
