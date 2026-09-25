"""Construct provider-specific Pipecat services in one place.

Factories are synchronous constructors, not network/session starters. Pipecat
opens the STT socket during setup; inference happens only when a pipeline runs.
Imports are lazy so configuration checks and offline tests need no voice extras.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from healthcare_voice_agent.config import AppConfig, HTTPSettings
from healthcare_voice_agent.voice.http_timing import HTTPConnectionTiming

if TYPE_CHECKING:
    from pipecat.processors.frame_processor import FrameProcessor
    from healthcare_voice_agent.voice.tracing import SessionTrace


class ProviderDependencyError(RuntimeError):
    """Optional speech dependencies have not been installed."""


@dataclass(frozen=True)
class _OpenAIComponents:
    stt: type
    llm: type
    tts: type
    async_client: type
    http_client: type
    timeout: type
    connection_limits: Any


def _load_openai_components() -> _OpenAIComponents:
    # Do not import these in tools, prompts, or the pipeline assembly module.
    try:
        # OpenAI 3.x uses HTTPX2. Get both the client and Timeout from the SDK
        # rather than mixing a legacy httpx.Timeout with an HTTPX2 client.
        from openai import AsyncOpenAI, DefaultAsyncHttpx2Client, Timeout, DEFAULT_CONNECTION_LIMITS
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.stt import OpenAIRealtimeSTTService
        from pipecat.services.openai.tts import OpenAITTSService
        from healthcare_voice_agent.voice.openai_stt import CommittedTranscriptSTTMixin
    except ModuleNotFoundError:
        raise ProviderDependencyError(
            "Voice dependencies are missing. In Terminal, run: uv sync --locked --extra voice"
        ) from None
    class CommittedOpenAIRealtimeSTTService(CommittedTranscriptSTTMixin, OpenAIRealtimeSTTService):
        pass

    return _OpenAIComponents(
        stt=CommittedOpenAIRealtimeSTTService, llm=OpenAILLMService, tts=OpenAITTSService,
        async_client=AsyncOpenAI, http_client=DefaultAsyncHttpx2Client, timeout=Timeout,
        connection_limits=DEFAULT_CONNECTION_LIMITS,
    )


def _http_timeout(settings: HTTPSettings, components: _OpenAIComponents) -> Any:
    # A timeout for each HTTP phase, NOT an end-to-end turn deadline.
    return components.timeout(
        settings.timeout_seconds, connect=settings.connect_timeout_seconds,
    )


def _http_client(settings: HTTPSettings, components: _OpenAIComponents,
                 *, stage: str, trace: SessionTrace | None = None) -> Any:
    # The pinned SDK exports an HTTPX2 Limits dataclass. Copy it so only idle
    # expiry changes, never the global defaults, pool capacities, or retries.
    kwargs = {
        "timeout": _http_timeout(settings, components),
        "limits": replace(components.connection_limits,
                          keepalive_expiry=settings.keepalive_expiry_seconds),
    }
    if settings.provider != "openai":
        # Self-hosted traffic must not inherit an ambient outbound proxy or
        # follow a redirect to a different model server.
        kwargs.update(trust_env=False, follow_redirects=False)
    if trace is not None:
        timing = HTTPConnectionTiming(trace, stage=stage)
        kwargs["event_hooks"] = {"request": [timing.on_request], "response": [timing.on_response]}
    return components.http_client(**kwargs)


class _CloseOwnedClient:
    """Pipecat 1.11.0 does not close these owned SDK clients in cleanup."""

    async def cleanup(self) -> None:
        try:
            await super().cleanup()
        finally:
            if not getattr(self, "_owned_client_closed", False):
                await self._client.close()
                self._owned_client_closed = True


def build_stt(config: AppConfig) -> FrameProcessor:
    """Build streaming, transcription-only STT with local turn control.

    The configured timeout is ONLY the WebSocket close-handshake timeout.
    Connection/turn deadlines will belong to the session runtime, not a fake
    'request timeout' argument that Pipecat silently ignores.
    """
    config.require_live_ready()
    if config.stt.provider == "whisper":
        return _build_local_whisper(config)
    if config.stt.provider == "nemotron":
        from healthcare_voice_agent.voice.nemotron_stt import NemotronSTTService
        stage = config.stt
        return NemotronSTTService(
            model=stage.model, base_url=stage.base_url, language=stage.language,
            api_key=stage.api_key.get_secret_value() if stage.api_key else None,
            max_segment_seconds=stage.max_segment_seconds,
            max_pending_segments=stage.max_pending_segments,
            inference_timeout_seconds=stage.inference_timeout_seconds,
            ws_close_timeout_seconds=stage.ws_close_timeout_seconds,
            finalize_transcripts=stage.finalize_transcripts,
        )
    components = _load_openai_components()
    stage = config.stt
    return components.stt(
        api_key=config.openai_api_key.get_secret_value(),
        base_url=stage.base_url,
        settings=components.stt.Settings(
            model=stage.model,
            language=stage.language,  # Settings(None) explicitly omits a forced language.
        ),
        turn_detection=False,
        finalize_transcripts=stage.finalize_transcripts,
        ws_close_timeout=stage.ws_close_timeout_seconds,
        reconnect_on_error=False,
    )


def build_llm(config: AppConfig, *, system_instruction: str | None = None,
              trace: SessionTrace | None = None) -> FrameProcessor:
    """Build a text/tool LLM; the caller supplies its agent instructions.

    Pipecat 1.11.0 ignores a stock http_client/timeout kwarg for this service.
    Its public create_client hook is the supported place to configure the SDK.
    """
    config.require_live_ready()
    components = _load_openai_components()
    stage = config.llm

    class ConfiguredOpenAILLM(_CloseOwnedClient, components.llm):
        def _compose_system_instruction(self):
            super()._compose_system_instruction()
            if config.agent_mode == "clinician":
                from pipecat.processors.aggregators.async_tool_messages import ASYNC_TOOL_INSTRUCTIONS
                # Native default forbids standalone async replies; the clinician
                # uses one idle result-handoff path instead.
                # Keep native async messages/cancellation; replace no SDK code.
                instruction = self._settings.system_instruction
                if isinstance(instruction, str):
                    instruction = instruction.replace(ASYNC_TOOL_INSTRUCTIONS, "").rstrip()
                    self._settings.system_instruction = instruction or None

        def create_client(
            self, api_key=None, base_url=None, organization=None, project=None,
            default_headers=None, **kwargs,
        ):
            timeout = _http_timeout(stage, components)
            return components.async_client(
                api_key=api_key, base_url=base_url, organization=organization,
                project=project, default_headers=default_headers,
                http_client=_http_client(stage, components, stage="llm", trace=trace),
                timeout=timeout, max_retries=config.llm_http_max_retries,
            )

        async def get_chat_completions(self, context):
            delivery = getattr(self, "_clinician_delivery", None)
            if delivery is not None:
                # Limit only the model request, not the native tool callback context.
                context = delivery.inference_context(context)
            return await super().get_chat_completions(context)

        async def _process_context(self, context):
            delivery = getattr(self, "_clinician_delivery", None)
            if delivery is not None:
                delivery.begin_inference()
            try:
                await super()._process_context(context)
            except asyncio.CancelledError:
                if config.agent_mode == "clinician":
                    from healthcare_voice_agent.voice.reply_tts import DiscardLLMReplyFrame

                    # Native cancellation emits an end frame in finally, before
                    # forwarding the interruption. Do not let it flush partial text.
                    await self.push_frame(DiscardLLMReplyFrame())
                raise  # No change to task/tool cancellation or interruption policy.

        async def push_error_frame(self, error, force_treat_as_permanent=False):
            if config.agent_mode == "clinician":
                from healthcare_voice_agent.voice.reply_tts import DiscardLLMReplyFrame

                # Discard before error callbacks can cancel the request, including
                # permanent failures that never reach the spoken-fallback branch.
                await self.push_frame(DiscardLLMReplyFrame())
            # Native classification must precede any fallback: permanent/unusable
            # failures retain their stop behavior. Any eligible SDK retry is done.
            await super().push_error_frame(error, force_treat_as_permanent=force_treat_as_permanent)
            if config.agent_mode == "clinician" and not error.fatal and self.is_usable:
                from healthcare_voice_agent.voice.reply_tts import LLMErrorFallbackFrame

                # A failed incomplete response must not leave a re-engagement
                # timer that would trigger another hidden LLM request afterward.
                await self._cancel_incomplete_timeout()
                # An ordered data frame clears any partial buffered reply before
                # going straight to TTS. Native interruption/cancellation still applies.
                await self.push_frame(LLMErrorFallbackFrame())
                if trace is not None:
                    trace.emit("llm_error_fallback", action="queued_for_tts")

    settings = {
        "model": stage.model,
        "max_completion_tokens": stage.max_output_tokens,
    }
    if stage.provider in {"hybrid_diffusion", "qwen"}:
        from healthcare_voice_agent.voice.hybrid_llm import SelfHostedLLMMixin

        class ConfiguredSelfHostedLLM(SelfHostedLLMMixin, ConfiguredOpenAILLM):
            pass

        service = ConfiguredSelfHostedLLM(
            # The SDK requires a nonempty value even for an unauthenticated
            # loopback server. Never forward the private OpenAI key here.
            api_key=stage.api_key.get_secret_value() if stage.api_key else "not-required",
            base_url=stage.base_url,
            settings=components.llm.Settings(**settings),
            retry_on_timeout=False,
        )
        # Avoid the base constructor's system-instruction DEBUG log.
        if system_instruction is not None:
            service._settings.system_instruction = system_instruction
        service._model_server_deadline_seconds = stage.timeout_seconds
        service._model_server_label = "Qwen" if stage.provider == "qwen" else "HybridDiffusion"
        return service
    if config.agent_mode == "frontdesk_demo":
        settings["extra"] = {"parallel_tool_calls": False}
    if system_instruction is not None:
        settings["system_instruction"] = system_instruction
    return ConfiguredOpenAILLM(
        api_key=config.openai_api_key.get_secret_value(),
        base_url=stage.base_url,
        settings=components.llm.Settings(**settings),
        retry_on_timeout=False,
        run_in_parallel=config.agent_mode != "frontdesk_demo",
    )


def build_tts(config: AppConfig, *, trace: SessionTrace | None = None) -> FrameProcessor:
    """Build streaming HTTP TTS, with configured timeouts and owned cleanup."""
    config.require_live_ready()
    if config.tts.provider == "kokoro":
        return _build_local_kokoro(config, trace=trace)
    if config.tts.provider == "breeze":
        from healthcare_voice_agent.voice.breeze_tts import BreezeTTSService
        stage = config.tts
        return BreezeTTSService(
            base_url=stage.base_url, model=stage.model, instructions=stage.instructions,
            audio_chunk_ms=stage.audio_chunk_ms,
            inference_timeout_seconds=stage.inference_timeout_seconds,
            connect_timeout_seconds=stage.connect_timeout_seconds,
            api_key=stage.api_key.get_secret_value() if stage.api_key else None,
            cfg_scale=stage.breeze_cfg_scale,
        )
    components = _load_openai_components()
    stage = config.tts

    class ConfiguredOpenAITTS(_CloseOwnedClient, components.tts):
        @property
        def chunk_size(self) -> int:
            """Read smaller PCM chunks without changing synthesis or text."""
            # OpenAI PCM is mono signed 16-bit. Keep whole samples.
            return (self.sample_rate * stage.audio_chunk_ms // 1000) * 2

    service = ConfiguredOpenAITTS(
        api_key=config.openai_api_key.get_secret_value(),
        base_url=stage.base_url,
        http_client=_http_client(stage, components, stage="tts", trace=trace),
        settings=components.tts.Settings(
            model=stage.model, voice=stage.voice, instructions=stage.instructions,
        ),
        sample_rate=24000,  # OpenAI's fixed PCM rate, not a provider-neutral setting.
        # Do not expire the audio context while its HTTP request is still waiting.
        # Normal HTTP completion closes the context immediately, without this wait.
        stop_frame_timeout_s=stage.connect_timeout_seconds + stage.timeout_seconds,
    )
    # Pipecat's TTS constructor does not expose this SDK setting. It is a public
    # SDK attribute; disable automatic retries for bounded initial experiments.
    service._client.max_retries = 0
    return service


# Native model objects are immutable/shared; audio, transcripts and TTS contexts
# belong to each service instance. Locks live INSIDE the synchronous callbacks:
# cancelling an asyncio waiter cannot release a lock around still-running native
# inference or allow a reconnected session to race the old call.
import hashlib
import platform
import threading
from pathlib import Path

from healthcare_voice_agent.voice.local_assets import (
    WHISPER_MODEL, WHISPER_REVISION, WHISPER_FILES, KOKORO_FILES, KOKORO_RELEASE_URL,
    model_cache,
)

_whisper_lock = threading.Lock()
_kokoro_lock = threading.Lock()
_whisper_runtime = None
_kokoro_runtime = None
_kokoro_runtime_threads = None


def _native_import_reason(error: BaseException) -> str:
    """Classify import failures without exposing raw paths, messages or secrets."""
    seen = set()
    current = error
    messages = []
    missing = False
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        missing = missing or isinstance(current, ModuleNotFoundError)
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__
    message = " ".join(messages)
    if "no metal device" in message or "metal device unavailable" in message:
        return "metal_unavailable"
    if "code signature" in message or "library load disallowed by system policy" in message:
        return "macos_library_policy"
    if missing:
        return "missing_dependency"
    return "native_import_failed"


def _import_local_dependency(module_name: str, package: str):
    """Only call with constant, allowlisted module/package names below."""
    import importlib
    try:
        return importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        reason = _native_import_reason(exc)
        if reason == "metal_unavailable":
            detail = (
                "Metal GPU access is unavailable to this process. Use a regular local Terminal, "
                "not a sandboxed/headless process. Reinstalling packages does not fix GPU access."
            )
        elif reason == "macos_library_policy":
            detail = (
                "macOS blocked a native library. The targeted reinstall may have omitted this "
                "dependency; follow docs/local-speech.md from a normal Terminal. "
                "Do not disable macOS security checks."
            )
        elif reason == "missing_dependency":
            detail = "A required dependency is missing. Install the voice and local-voice extras."
        else:
            detail = "Native import failed. See docs/local-speech.md for a direct import diagnostic."
        raise ProviderDependencyError(f"Local dependency {package} failed [{reason}]. {detail}") from None


def _load_whisper_components():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ProviderDependencyError("Local Whisper requires Apple Silicon macOS; no CPU fallback is used.")
    # mlx-whisper imports these transitively. Check them separately so a broken
    # torch/tiktoken/scipy extension is not mislabeled as an MLX installation issue.
    _import_local_dependency("torch", "torch")
    _import_local_dependency("tiktoken", "tiktoken")
    _import_local_dependency("scipy", "scipy")
    mx = _import_local_dependency("mlx.core", "mlx")
    module = _import_local_dependency("mlx_whisper", "mlx-whisper")
    hub = _import_local_dependency("huggingface_hub", "huggingface-hub")
    transcription = _import_local_dependency("mlx_whisper.transcribe", "mlx-whisper")
    return mx, module, hub.snapshot_download, transcription.ModelHolder


def _load_kokoro_components():
    try:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        return ort, Kokoro
    except (ImportError, OSError):
        raise ProviderDependencyError(
            "Local Kokoro dependencies are unavailable. Run: uv sync --locked --extra voice --extra local-voice"
        ) from None


def _sha256_file(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_kokoro_file(file: Path, spec: dict) -> bool:
    return (file.is_file() and file.stat().st_size == spec["size"]
            and (spec["sha256"] is None or _sha256_file(file) == spec["sha256"]))


def prepare_local_models(config: AppConfig) -> dict:
    """Explicit public-weight downloads only. Never uses the OpenAI key or APIs.

    Serving is cache-only: a model cannot begin downloading inside a speech turn.
    Atomic temporary files and pinned hashes prevent reusing partial downloads.
    """
    result = {}
    cache = model_cache()
    if config.stt.provider == "whisper":
        try:
            from huggingface_hub import snapshot_download as download
        except ImportError:
            raise ProviderDependencyError("Install the local-voice extra before preparing models.") from None
        try:
            download(repo_id=WHISPER_MODEL, revision=WHISPER_REVISION,
                     cache_dir=str(cache / "huggingface"), token=False,
                     allow_patterns=list(WHISPER_FILES))
        except Exception:
            raise ProviderDependencyError("Whisper model download failed. Retry --prepare-local-models.") from None
        result["whisper"] = {"model": WHISPER_MODEL, "revision": WHISPER_REVISION}
    if config.tts.provider == "kokoro":
        from urllib.request import urlopen
        folder = cache / "kokoro"
        folder.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name, spec in KOKORO_FILES.items():
            dest = folder / name
            if not _valid_kokoro_file(dest, spec):
                from tempfile import NamedTemporaryFile
                temporary = None
                try:
                    with NamedTemporaryFile(dir=folder, suffix=".partial", delete=False) as handle:
                        temporary = Path(handle.name)
                        with urlopen(f"{KOKORO_RELEASE_URL}/{name}", timeout=60) as response:
                            for block in iter(lambda: response.read(1024 * 1024), b""):
                                handle.write(block)
                    if not _valid_kokoro_file(temporary, spec):
                        raise ValueError("asset integrity check failed")
                    temporary.replace(dest)
                except Exception:
                    raise ProviderDependencyError("Kokoro model download/integrity check failed. Retry --prepare-local-models.") from None
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            hashes[name] = _sha256_file(dest)
        result["kokoro"] = hashes
    return result


def _get_whisper_runtime():
    global _whisper_runtime
    # Caller holds _whisper_lock, including during initial model loading.
    if _whisper_runtime is None:
        mx, module, download, holder = _load_whisper_components()
        try:
            local = download(repo_id=WHISPER_MODEL, revision=WHISPER_REVISION,
                             cache_dir=str(model_cache() / "huggingface"), token=False,
                             local_files_only=True, allow_patterns=list(WHISPER_FILES))
            for name in WHISPER_FILES:
                if not (Path(local) / name).is_file():
                    raise FileNotFoundError
        except Exception:
            raise ProviderDependencyError(
                "Required cached Whisper config/weights are unavailable. Run --prepare-local-models to prepare the selected files."
            ) from None
        try:
            holder.get_model(local, mx.float16)
        except Exception as exc:
            reason = _native_import_reason(exc)
            if reason == "metal_unavailable":
                detail = "Metal GPU access is unavailable to this process. Use a regular local graphical Terminal."
            elif reason == "macos_library_policy":
                detail = "macOS blocked a native library during model initialization. Do not disable security checks."
            else:
                detail = "Model initialization failed; inspect the local MLX runtime, not the download step."
            raise ProviderDependencyError(
                f"Whisper model files were found, but loading failed ({type(exc).__name__}). {detail}"
            ) from None
        _whisper_runtime = (module, local)
    return _whisper_runtime


def _create_kokoro_runtime(cpu_threads: int = 2):
    """Cache-only engine factory, also used by the fixed local CPU benchmark."""
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= 8:
        raise ProviderDependencyError("Kokoro CPU threads must be between 1 and 8.")
    ort, kokoro = _load_kokoro_components()
    folder = model_cache() / "kokoro"
    if not all(_valid_kokoro_file(folder / name, spec) for name, spec in KOKORO_FILES.items()):
        raise ProviderDependencyError("Kokoro assets are missing or invalid. Run --prepare-local-models first.")
    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = cpu_threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(folder / "kokoro-v1.0.onnx"), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        return kokoro.from_session(session, str(folder / "voices-v1.0.bin"))
    except Exception:
        raise ProviderDependencyError("Kokoro could not load its ONNX model or phonemizer. See the normal-Terminal reinstall instructions in docs/local-speech.md.") from None


def _get_kokoro_runtime(*, cpu_threads: int | None = None):
    global _kokoro_runtime, _kokoro_runtime_threads
    # Caller holds _kokoro_lock. Never swap the engine under a live native call.
    if _kokoro_runtime is None:
        selected = cpu_threads if cpu_threads is not None else 2
        _kokoro_runtime = _create_kokoro_runtime(selected)
        _kokoro_runtime_threads = selected
    elif cpu_threads is not None and cpu_threads != _kokoro_runtime_threads:
        raise ProviderDependencyError("Kokoro CPU thread changes require a server restart.")
    return _kokoro_runtime


def _whisper_reuse_dependencies(local):
    # Provider imports remain confined to this factory boundary. The holder
    # returns the already loaded FP16 model; this does not download or reload it.
    from importlib import import_module
    from healthcare_voice_agent.voice.whisper_reuse import WhisperReuseDependencies
    audio_ops = import_module("mlx_whisper.audio")
    decoding = import_module("mlx_whisper.decoding")
    tokenizer = import_module("mlx_whisper.tokenizer")
    transcription = import_module("mlx_whisper.transcribe")
    mx = import_module("mlx.core")
    return transcription.ModelHolder.get_model(local, mx.float16), WhisperReuseDependencies(
        log_mel_spectrogram=audio_ops.log_mel_spectrogram,
        pad_or_trim=audio_ops.pad_or_trim,
        decoding_options_factory=decoding.DecodingOptions,
        get_tokenizer=tokenizer.get_tokenizer,
        n_samples=audio_ops.N_SAMPLES, n_frames=audio_ops.N_FRAMES,
    )


def _whisper_transcribe(audio, *, language=None, reuse_encoder=False):
    with _whisper_lock:
        module, local = _get_whisper_runtime()
        used_stock = False
        def stock(value):
            nonlocal used_stock
            used_stock = True
            return module.transcribe(
                value, path_or_hf_repo=local, language=language, task="transcribe",
                temperature=0.0, condition_on_previous_text=False, verbose=None,
            )
        if not reuse_encoder:
            return stock(audio)
        from healthcare_voice_agent.voice.whisper_reuse import transcribe_with_encoder_reuse
        model, dependencies = _whisper_reuse_dependencies(local)
        result = transcribe_with_encoder_reuse(
            audio, model=model, language=language, dependencies=dependencies,
            stock_transcribe=stock,
        )
        result["local_decode_path"] = "stock_fallback" if used_stock else "single_encode"
        return result


def _kokoro_synthesize(text, *, voice, lang, speed):
    with _kokoro_lock:
        return _get_kokoro_runtime().create(text, voice=voice, lang=lang, speed=speed)


def warm_local_models(config: AppConfig) -> None:
    """Cache-only model load and synthetic warmup before accepting connections."""
    if config.stt.provider == "whisper":
        import numpy as np
        _whisper_transcribe(np.zeros(16000, dtype=np.float32), language=config.stt.language,
                            reuse_encoder=config.stt.reuse_encoder)
    if config.tts.provider == "kokoro":
        with _kokoro_lock:
            engine = _get_kokoro_runtime(cpu_threads=config.tts.cpu_threads)
            if config.tts.voice not in engine.voices or config.tts.hindi_voice not in engine.voices:
                raise ProviderDependencyError("Configured Kokoro voice is not present in the pinned voices file.")
        _kokoro_synthesize("Local speech check.", voice=config.tts.voice, lang="en-us", speed=config.tts.speed)
        _kokoro_synthesize("नमस्ते।", voice=config.tts.hindi_voice, lang="hi", speed=config.tts.speed)


def _build_local_whisper(config: AppConfig):
    from functools import partial
    from healthcare_voice_agent.voice.local_stt import LocalWhisperSTTService
    # Fail before browser setup rather than downloading on the first utterance.
    with _whisper_lock:
        _get_whisper_runtime()
    return LocalWhisperSTTService(
        transcribe=partial(_whisper_transcribe, language=config.stt.language,
                           reuse_encoder=config.stt.reuse_encoder),
        model=config.stt.model, language=config.stt.language,
        max_segment_seconds=config.stt.max_segment_seconds,
        max_pending_segments=config.stt.max_pending_segments,
        inference_timeout_seconds=config.stt.inference_timeout_seconds,
    )


def _build_local_kokoro(config: AppConfig, *, trace: SessionTrace | None = None):
    from healthcare_voice_agent.voice.local_tts import LocalKokoroTTSService
    with _kokoro_lock:
        engine = _get_kokoro_runtime(cpu_threads=config.tts.cpu_threads)
        if config.tts.voice not in engine.voices or config.tts.hindi_voice not in engine.voices:
            raise ProviderDependencyError("Configured Kokoro voice is not present in the pinned voices file.")
    return LocalKokoroTTSService(
        synthesize=_kokoro_synthesize, model=config.tts.model,
        voice=config.tts.voice, hindi_voice=config.tts.hindi_voice,
        language=config.tts.language, speed=config.tts.speed,
        audio_chunk_ms=config.tts.audio_chunk_ms,
        inference_timeout_seconds=config.tts.inference_timeout_seconds,
        first_chunk_chars=config.tts.first_chunk_chars,
        trace=trace,
    )
