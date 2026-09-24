"""One bounded MLX Whisper encoder-reuse path for short local segments.

This module deliberately imports no MLX, NumPy, or ``mlx_whisper`` symbols.
``providers.py`` must inject the already imported model and the few native
operations named by :class:`WhisperReuseDependencies`.

For a non-empty segment of at most one Whisper window, the helper builds the
same transcription mel window as mlx-whisper 0.4.2 and calls
``model.decode(window, options(language=None))``. ``DecodingTask`` encodes that
window once and performs automatic language detection from the encoded audio
features before decoding.

There is one intentional semantic difference from stock ``transcribe`` for
short multilingual clips with automatic language selection. Stock language
identification uses the first 30 seconds of the mel made after appending 30
seconds of waveform silence. Its transcription window then slices only the
content frames and pads those normalized mel frames with zeros. This helper
uses that latter transcription window for both language identification and
decoding so the encoder result can be reused. Accuracy and speed therefore
require live MLX/Metal validation.

The optimization is intentionally narrow: task="transcribe", temperature=0,
no prompt, no word timestamps, and one complete decoding window. Longer clips,
empty inputs, malformed decode results, and timestamp sequences which make the
stock seek loop revisit the window use the injected unchanged stock fallback.
This is not a general replacement for transcribe: arbitrary decode options are
not accepted. Both paths use the pinned native DecodingOptions defaults. Any
future provider prompt, suppression, precision or decoding-policy change must
update both paths together. condition_on_previous_text=False matters only when
there is another decoding pass, which is delegated to the stock path here.
No model, package, or process-global state is patched or mutated here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WhisperReuseDependencies:
    """Native operations and constants supplied by the provider boundary."""

    log_mel_spectrogram: Callable[..., Any]
    pad_or_trim: Callable[..., Any]
    decoding_options_factory: Callable[..., Any]
    get_tokenizer: Callable[..., Any]
    n_samples: int
    n_frames: int


def transcribe_with_encoder_reuse(
    audio: Any,
    *,
    model: Any,
    language: str | None,
    dependencies: WhisperReuseDependencies,
    stock_transcribe: Callable[[Any], Mapping[str, Any]],
    no_speech_threshold: float | None = 0.6,
    logprob_threshold: float | None = -1.0,
) -> dict[str, str]:
    """Transcribe one eligible segment with a single encoder pass.

    ``stock_transcribe`` must be a caller-owned closure or partial containing
    the existing baseline arguments, including task="transcribe",
    temperature=0.0, condition_on_previous_text=False, and the selected model
    path. It is called with the original ``audio`` object, never an altered mel.

    The returned contract is intentionally limited to ``text`` and
    ``language``, which are the fields consumed by the local STT adapter.
    Decoder/runtime exceptions propagate. Only known ineligibility or an
    unsupported timestamp/result structure selects the stock fallback.
    """

    mel = dependencies.log_mel_spectrogram(
        audio,
        n_mels=model.dims.n_mels,
        padding=dependencies.n_samples,
    )
    content_frames = mel.shape[-2] - dependencies.n_frames
    if content_frames <= 0 or content_frames > dependencies.n_frames:
        return _stock_result(audio, stock_transcribe)

    segment = mel[:content_frames]
    segment = dependencies.pad_or_trim(
        segment,
        dependencies.n_frames,
        axis=-2,
    )

    decode_language = language
    if decode_language is None and not model.is_multilingual:
        decode_language = "en"
    options = dependencies.decoding_options_factory(
        task="transcribe",
        language=decode_language,
        temperature=0.0,
    )
    result = model.decode(segment, options)

    result_language = getattr(result, "language", None)
    raw_tokens = getattr(result, "tokens", None)
    if not isinstance(result_language, str) or not _plain_integer_tokens(raw_tokens):
        return _stock_result(audio, stock_transcribe)
    tokens = list(raw_tokens)

    no_speech_prob = getattr(result, "no_speech_prob", None)
    avg_logprob = getattr(result, "avg_logprob", None)
    if not isinstance(no_speech_prob, (int, float)) or not isinstance(
        avg_logprob, (int, float)
    ):
        return _stock_result(audio, stock_transcribe)

    if no_speech_threshold is not None:
        should_skip = no_speech_prob > no_speech_threshold
        if logprob_threshold is not None and avg_logprob > logprob_threshold:
            should_skip = False
        if should_skip:
            # Stock skips the whole window before interpreting its timestamps.
            return {"text": "", "language": result_language}

    tokenizer = dependencies.get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=result_language,
        task="transcribe",
    )
    text = _complete_window_text(tokens, tokenizer)
    if text is None:
        # Stock would seek to a timestamp inside this window and decode again.
        # Returning the first decode would silently omit its unfinished suffix.
        return _stock_result(audio, stock_transcribe)

    return {"text": text, "language": result_language}


def _plain_integer_tokens(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(
        isinstance(token, int) and not isinstance(token, bool) for token in value
    )


def _complete_window_text(tokens: list[int], tokenizer: Any) -> str | None:
    """Mirror the non-word-timestamp parts of mlx-whisper's seek decision.

    ``None`` means the token structure tells stock ``transcribe`` to revisit a
    suffix of the current window. Complete multi-segment windows are retained,
    including stock's removal of empty or zero-duration timestamp segments.
    """

    timestamp_begin = tokenizer.timestamp_begin
    timestamp_mask = [token >= timestamp_begin for token in tokens]
    consecutive = [
        index + 1
        for index in range(len(timestamp_mask) - 1)
        if timestamp_mask[index] and timestamp_mask[index + 1]
    ]
    if not consecutive:
        segment_text = tokenizer.decode([token for token in tokens if token < tokenizer.eot])
        if not segment_text.strip():
            return ""
        return tokenizer.decode(tokens)

    single_timestamp_ending = len(timestamp_mask) >= 2 and timestamp_mask[-2:] == [
        False,
        True,
    ]
    if not single_timestamp_ending:
        return None

    kept_tokens: list[int] = []
    last_slice = 0
    for current_slice in [*consecutive, len(tokens)]:
        segment_tokens = tokens[last_slice:current_slice]
        if (
            not segment_tokens
            or segment_tokens[0] < timestamp_begin
            or segment_tokens[-1] < timestamp_begin
        ):
            return None
        start_timestamp = segment_tokens[0] - timestamp_begin
        end_timestamp = segment_tokens[-1] - timestamp_begin
        text_tokens = [token for token in segment_tokens if token < tokenizer.eot]
        segment_text = tokenizer.decode(text_tokens)
        if start_timestamp != end_timestamp and segment_text.strip():
            kept_tokens.extend(segment_tokens)
        last_slice = current_slice

    return tokenizer.decode(kept_tokens)


def _stock_result(
    audio: Any,
    stock_transcribe: Callable[[Any], Mapping[str, Any]],
) -> dict[str, str]:
    result = stock_transcribe(audio)
    text = result.get("text")
    language = result.get("language")
    if not isinstance(text, str) or not isinstance(language, str):
        raise TypeError("stock Whisper transcription must return text and language strings")
    return {"text": text, "language": language}


__all__ = ["WhisperReuseDependencies", "transcribe_with_encoder_reuse"]
