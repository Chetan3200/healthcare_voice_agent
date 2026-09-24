"""Offline contract tests for the bounded Whisper encoder-reuse helper.

The fakes model the relevant mlx-whisper 0.4.2 contracts without importing MLX,
NumPy, downloading weights, or executing native code.
"""

from types import SimpleNamespace

import pytest

from healthcare_voice_agent.voice.whisper_reuse import (
    WhisperReuseDependencies,
    transcribe_with_encoder_reuse,
)


N_FRAMES = 3000
N_SAMPLES = 480000
TIMESTAMP_BEGIN = 100
EOT = 90


class FakeMel:
    def __init__(self, frames, n_mels=80, label="mel"):
        self.shape = (frames, n_mels)
        self.label = label

    def __getitem__(self, item):
        assert isinstance(item, slice)
        assert item.start is None
        assert item.step is None
        return FakeMel(item.stop, self.shape[-1], label="content")


class FakeTokenizer:
    timestamp_begin = TIMESTAMP_BEGIN
    eot = EOT

    _pieces = {1: " first", 2: " later", 3: " speech"}

    def decode(self, tokens):
        return "".join(self._pieces.get(token, "") for token in tokens if token < TIMESTAMP_BEGIN)


class FakeModel:
    def __init__(self, results, *, multilingual=True):
        self.dims = SimpleNamespace(n_mels=80)
        self.is_multilingual = multilingual
        self.num_languages = 99 if multilingual else 0
        self._results = list(results)
        self.decode_calls = 0
        self.encoder_calls = 0
        self.language_detection_calls = 0
        self.segments = []
        self.options = []

    def decode(self, segment, options):
        self.decode_calls += 1
        self.segments.append(segment)
        self.options.append(options)
        self.encoder_calls += 1
        if options.language is None:
            self.language_detection_calls += 1
        value = self._results.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def decode_result(
    *,
    language="hi",
    tokens=(TIMESTAMP_BEGIN, 1, TIMESTAMP_BEGIN + 20),
    no_speech_prob=0.1,
    avg_logprob=-0.2,
):
    return SimpleNamespace(
        language=language,
        tokens=tokens,
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
    )


def harness(content_frames, *, stock=None):
    calls = SimpleNamespace(mel=[], pad=[], options=[], tokenizer=[], stock=[])

    def log_mel(audio, **kwargs):
        calls.mel.append((audio, kwargs))
        return FakeMel(content_frames + N_FRAMES)

    def pad_or_trim(mel, length, *, axis):
        calls.pad.append((mel.shape, length, axis))
        return FakeMel(length, mel.shape[-1], label="padded-transcription-window")

    def options_factory(**kwargs):
        calls.options.append(dict(kwargs))
        return SimpleNamespace(**kwargs)

    def get_tokenizer(multilingual, **kwargs):
        calls.tokenizer.append((multilingual, kwargs))
        return FakeTokenizer()

    def stock_transcribe(audio):
        calls.stock.append(audio)
        return stock or {"text": " stock complete", "language": "en", "segments": []}

    dependencies = WhisperReuseDependencies(
        log_mel_spectrogram=log_mel,
        pad_or_trim=pad_or_trim,
        decoding_options_factory=options_factory,
        get_tokenizer=get_tokenizer,
        n_samples=N_SAMPLES,
        n_frames=N_FRAMES,
    )
    return dependencies, stock_transcribe, calls


def run(audio, model, dependencies, stock_transcribe, *, language=None, **kwargs):
    return transcribe_with_encoder_reuse(
        audio,
        model=model,
        language=language,
        dependencies=dependencies,
        stock_transcribe=stock_transcribe,
        **kwargs,
    )


def test_eligible_automatic_language_uses_one_encode_for_detection_and_decode():
    audio = object()
    dependencies, stock, calls = harness(1200)
    model = FakeModel([decode_result(language="hi")])

    result = run(audio, model, dependencies, stock)

    assert result == {"text": " first", "language": "hi"}
    assert model.decode_calls == model.encoder_calls == 1
    assert model.language_detection_calls == 1
    assert calls.stock == []
    assert calls.mel == [(audio, {"n_mels": 80, "padding": N_SAMPLES})]
    assert calls.pad == [((1200, 80), N_FRAMES, -2)]
    assert calls.options == [{"task": "transcribe", "language": None, "temperature": 0.0}]
    assert calls.tokenizer == [(True, {
        "num_languages": 99, "language": "hi", "task": "transcribe",
    })]


@pytest.mark.parametrize(
    "multilingual,requested,detected,expected_option,expected_language_detections",
    [
        (True, "hi", "hi", "hi", 0),
        (False, None, "en", "en", 0),
    ],
)
def test_requested_and_monolingual_language_behavior(
    multilingual, requested, detected, expected_option, expected_language_detections
):
    dependencies, stock, calls = harness(800)
    model = FakeModel([decode_result(language=detected)], multilingual=multilingual)

    result = run("audio", model, dependencies, stock, language=requested)

    assert result["language"] == detected
    assert model.options[0].language == expected_option
    assert model.language_detection_calls == expected_language_detections
    assert calls.stock == []


@pytest.mark.parametrize(
    "no_speech_prob,avg_logprob,expected_text",
    [
        (0.7, -1.2, ""),
        (0.7, -0.5, " first"),
        (0.5, -2.0, " first"),
    ],
)
def test_stock_no_speech_and_logprob_threshold_interaction(
    no_speech_prob, avg_logprob, expected_text
):
    dependencies, stock, _ = harness(600)
    model = FakeModel([decode_result(
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
    )])

    result = run("audio", model, dependencies, stock)

    assert result == {"text": expected_text, "language": "hi"}


def test_complete_multi_segment_timestamp_window_stays_on_fast_path():
    dependencies, stock, calls = harness(2200)
    tokens = (
        TIMESTAMP_BEGIN,
        1,
        TIMESTAMP_BEGIN + 20,
        TIMESTAMP_BEGIN + 20,
        2,
        TIMESTAMP_BEGIN + 50,
    )
    model = FakeModel([decode_result(tokens=tokens)])

    result = run("audio", model, dependencies, stock)

    assert result == {"text": " first later", "language": "hi"}
    assert model.encoder_calls == 1
    assert calls.stock == []


def test_partial_timestamp_seek_falls_back_so_later_speech_is_not_dropped():
    dependencies, stock, calls = harness(
        2500,
        stock={"text": " first later speech", "language": "hi", "segments": [1, 2]},
    )
    # Consecutive timestamps followed by unfinished text make stock transcribe
    # seek to the last completed timestamp and decode the suffix again.
    partial_tokens = (
        TIMESTAMP_BEGIN,
        1,
        TIMESTAMP_BEGIN + 20,
        TIMESTAMP_BEGIN + 20,
        2,
    )
    model = FakeModel([decode_result(tokens=partial_tokens)])

    result = run("original-audio", model, dependencies, stock)

    assert result == {"text": " first later speech", "language": "hi"}
    assert model.encoder_calls == 1
    assert calls.stock == ["original-audio"]


def test_longer_than_one_window_uses_stock_without_decoding():
    dependencies, stock, calls = harness(3001)
    model = FakeModel([decode_result()])

    result = run("long-audio", model, dependencies, stock)

    assert result == {"text": " stock complete", "language": "en"}
    assert model.decode_calls == model.encoder_calls == 0
    assert calls.pad == []
    assert calls.stock == ["long-audio"]


@pytest.mark.parametrize("tokens", [None, "not-token-ids", (TIMESTAMP_BEGIN, True)])
def test_unsupported_decode_result_structure_uses_stock(tokens):
    dependencies, stock, calls = harness(1000)
    model = FakeModel([decode_result(tokens=tokens)])

    result = run("audio", model, dependencies, stock)

    assert result == {"text": " stock complete", "language": "en"}
    assert calls.stock == ["audio"]


def test_decode_failure_propagates_and_does_not_poison_a_later_call():
    dependencies, stock, calls = harness(900)
    model = FakeModel([
        RuntimeError("synthetic decoder failure"),
        decode_result(language="en", tokens=(TIMESTAMP_BEGIN, 3, TIMESTAMP_BEGIN + 10)),
    ])

    with pytest.raises(RuntimeError, match="synthetic decoder failure"):
        run("first", model, dependencies, stock)

    assert run("second", model, dependencies, stock) == {
        "text": " speech",
        "language": "en",
    }
    assert model.decode_calls == model.encoder_calls == 2
    assert calls.stock == []
    assert calls.options == [
        {"task": "transcribe", "language": None, "temperature": 0.0},
        {"task": "transcribe", "language": None, "temperature": 0.0},
    ]
    assert calls.options[0] is not calls.options[1]
