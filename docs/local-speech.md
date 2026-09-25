# Archived local speech experiment: Kokoro + Whisper large-v3-turbo

The experiment's launcher and benchmark helpers have been retired. These notes
preserve historical implementation and verification details, not current startup
instructions. See [the README](../README.md) and [GPU setup](gpu-models.md) for
current launch commands.

This profile replaces **both speech stages together**, as requested. The text LLM
remains the configured OpenAI model. It is not an all-local agent, and it does not
use another hosted provider. OpenAI speech remains the default outside this profile.

## macOS loading errors

If macOS rejects a native library, do not disable Gatekeeper, re-sign third-party
binaries, or clear quarantine flags. Try installation from your normal Terminal:

```bash
uv sync --locked --extra voice --extra local-voice --no-cache --link-mode copy \
  --reinstall-package mlx --reinstall-package mlx-metal --reinstall-package espeakng-loader \
  --reinstall-package torch --reinstall-package tiktoken --reinstall-package scipy
```

The first three packages are not the entire Whisper import chain. `mlx-whisper`
also imports Torch, tiktoken and SciPy. Reinstalling only MLX/eSpeak can succeed
while these transitive native libraries remain blocked. Startup now names the
failing dependency and reports a sanitized category:

- `macos_library_policy`: macOS rejected a native library; reinstall that package
  from a normal Terminal and resolve any security prompt normally.
- `metal_unavailable`: the binary imported but this process cannot access the
  Metal GPU. Reinstalling packages does **not** fix this. Use a normal local
  graphical Terminal rather than a sandboxed/headless process. An error in the
  assistant's background process does not prove your Terminal has the same issue.
- `missing_dependency` or `native_import_failed`: diagnose the named package.

For example, from the project root, a direct dependency diagnostic is:

```bash
.venv/bin/python -c 'import torch, tiktoken, scipy; import mlx.core; import mlx_whisper; print("Whisper imports OK")'
```

Factory errors are sanitized and never silently switch to CPU Whisper, a smaller
model, or an OpenAI speech endpoint. Do not raise speech timeouts to hide a broken
import. Installation success does not establish successful runtime loading.

## Selected implementations

| Stage | Implementation | Important distinction |
| --- | --- | --- |
| ASR | `mlx-community/whisper-large-v3-turbo`, MLX GPU, pinned HF revision | Complete VAD-segment transcription, **not native streaming ASR** |
| LLM | Existing OpenAI configuration | Still sends finalized text to OpenAI and incurs API charges |
| TTS | Kokoro v1.0 ONNX, CPU (default 2 intra-op / 1 inter-op threads) | Delivers PCM between smaller first-request synthesis pieces, **not native sample-level synthesis streaming** |

Whisper uses `task=transcribe`, temperature 0, automatic language detection by
default, and no previous-utterance text conditioning. It receives 16 kHz mono
PCM in memory. The model is not the English-only distilled checkpoint. MLX is used
instead of whisper.cpp to stay Python-first on Apple Silicon. Whisper source
revision, Kokoro release/checksums, runtimes and package versions are recorded in
`config.json`. No native speed or audible-quality claim follows from configuration.

Kokoro preserves the supplied text and delivers 24 kHz mono PCM16 in at most
`TTS_AUDIO_CHUNK_MS` chunks (100 ms default). It does not accept OpenAI delivery
instructions; the experiment used `TTS_INSTRUCTIONS=off`. Text aggregation remains
Pipecat's normal aggregation. For the first request in each audio context, the
adapter chooses at most one conservative punctuation boundary near a soft
60-character target, looking up to twice that target when necessary. Both sides
must contain substantial speech; it never falls back to an ordinary space or
creates a one-/two-word tail. The whole remainder stays intact. If no suitable
boundary exists, the original text stays in one request, even above the target.
It emits the first piece's PCM before starting the remainder. Later original
requests in the context stay intact. No word is split or dropped, and both pieces
use the voice/language chosen for the original text.
This may change prosody at joins; it requires a listening check. Changing PCM
frame size alone still cannot shorten native synthesis.

### English / Hindi routing

- `TTS_LANGUAGE=auto`: a Devanagari **letter** selects Hindi for that entire TTS
  chunk; otherwise English. Digits or punctuation alone do not trigger Hindi.
- English voice: `TTS_VOICE=af_heart`.
- Hindi voice: `TTS_HINDI_VOICE=hf_alpha`.
- `TTS_LANGUAGE=en` or `hi` forces one route for listening comparisons.
- `TTS_SPEED=1.0` by default.

This is a transparent script heuristic, not language identification. Romanized
Hindi is treated as English in auto mode. Mixed-script Hinglish uses Hindi for the
whole chunk, so borrowed English words may be pronounced poorly. Different chunks
can use different voices. Do not claim seamless Hinglish from these settings.
The response-language prompt remains unchanged; no translation/rewrite is added.

## Turn correctness, interruption and resource bounds

- Existing Silero VAD and Smart Turn remain the turn owners. Their thresholds,
  1.5 s continuous-silence fallback and transcript waiting are unchanged.
- Whisper segments are queued FIFO. Speech-generation metadata is reserved before
  awaits, checked again when emitting, and checked by the existing stop strategy.
  An earlier segment retains its text but cannot finalize newer speech.
- `STT_FINALIZE_TRANSCRIPTS=true` is mandatory for local Whisper.
- Defaults: 60 s maximum segment (including pre-roll), 8 outstanding segments,
  30 s inference deadline. Overflow/missing or invalid inference fails closed,
  rather than replying using only the surviving words. Empty/no-speech recognition
  does not synthesize a transcript or fabricate a final flag.
- Disconnect/cancel drops pending local ASR output. TTS interruption invalidates
  any in-flight result before it can emit old audio into a newer context. The
  session's disconnect/error paths call `runner.cancel`, not graceful `EndFrame`.
  Native TTS `EndFrame` draining remains intentional; each reconnect gets fresh
  services and contexts, never a reused pipeline.
- Native inference already running in a thread cannot be forcibly cancelled.
  Process-wide thread locks prevent overlapping model calls after reconnect;
  cancelled waiters discard their eventual output. A stuck inference can still
  occupy the model until it returns, causing the next request to time out.
- Models are shared, but session audio buffers, transcripts, TTS contexts and LLM
  conversations are not. Raw microphone audio is not written to disk.
- Local inference failures produce sanitized permanent service errors with no
  provider/runtime fallback or automatic retry. Known unsupported Whisper
  fast-path cases use the same-model stock transcription algorithm, not a
  different provider. Pipecat's TTS audio-context watchdog remains unchanged (3 s).
  While Kokoro synthesis is pending, the adapter refreshes only that existing
  context every second using Pipecat 1.11's non-audio keepalive sentinel. The
  existing 30 s **total** inference deadline covers every piece of the original
  text request, including thread scheduling and waiting for the shared engine
  lock; neither pieces nor keepalives restart it. They stop
  on completion, cancellation, interruption, or timeout. An idle context with no
  synthesis still expires normally. No silence is injected, and the native
  consumer records TTFB/TTFA only from real PCM, never a keepalive.
- `local_tts_synthesis_started`, `finished`, `cancelled`, `discarded`, and `failed`
  events record context ID and elapsed synthesis-wait time (including lock wait),
  not the raw text/audio or backend exception. Successful results include audio
  duration. Failure distinguishes an inference timeout from other synthesis or
  audio-validation errors. These help separate slow synthesis from context
  cancellation in the next live run; they are not GPU/CPU kernel timings.

## Paired experiment and rollback

Keep the OpenAI LLM, prompts, Smart Turn settings, playback device and input script
fixed. First confirm short English, Hindi and Hinglish replies, numbers/IDs,
corrections, a longer pause, barge-in, disconnect during speech and reconnect.
Report the effective providers, cold/warm status, language counts, first transcript,
first LLM token and first server audio separately. Local Whisper starts inference
after a VAD segment ends, unlike the existing streaming OpenAI ASR, so a smaller
model/network-free stage is not automatically a lower-latency voice loop.

Changing both stages tests the requested pair, but does not attribute a speed or
quality difference to just one model. Stop after the fixed comparison for review.
Do not wire clinical tools into this experiment yet.

Provider changes require a server restart, not just a browser reconnect. Do not
run `uv sync` or change extras underneath a running server.

## Offline verification

```bash
.venv/bin/python -m pytest
uv lock --check
```

Offline tests inject local model callbacks and validate adapter/factory behavior,
including actual Pipecat frame/context lifecycle, ordered segments, stale results,
PCM conversion, chunk boundaries, exact text, timeouts, failure and cache policy.
They do not establish real model speed, audible TTS quality, or browser microphone
behavior. Those require successful native loading and a listening session.

### Implementation-pass verification (2026-09-21)

- 364 offline tests passed; Python compilation and `uv lock --check` passed.
- The paired profile resolved Whisper + Kokoro + the existing OpenAI LLM.
- Both pinned model bundles were downloaded; Kokoro checksums were verified.
- Native warmup was **blocked** in the background execution environment: macOS
  rejected MLX and the bundled eSpeak native libraries with a system-policy load
  error. No quarantine/security settings were changed. Follow the normal-Terminal
  reinstall step above; it is a next step, not proof that loading will succeed.
- No live browser conversation, local ASR accuracy, audible Kokoro quality or
  local end-to-end latency result was established in this pass.

### Kokoro context-watchdog correction (2026-09-21)

- Session `8b1e9eed9a8c491786d557f67c9f8361` received its TTS request at
  11:36:50.720 local time, then reported a zero-audio context 3.003 seconds later.
  This matches the native idle watchdog expiring while the adapter awaited its
  whole-chunk synthesis result. It does not establish that Kokoro itself returned
  empty audio or tell us when native synthesis would have completed.
- A failing regression reproduced that same error using Pipecat's real context
  consumer and an injected, blocked synthesis callback. With the bounded context
  keepalive fix, a synthetic 3.15-second wait survives the unchanged 3-second
  watchdog and delivers real test PCM. This is a lifecycle test, not a measured
  Kokoro speed or quality result.
- **387 offline tests passed**, including native context consumption, bounded
  inference deadline, interruption, cancellation, cleanup, late-result discard,
  first-audio metrics, post-interruption speech and sanitized trace events.
- Restart the local server to load this correction. A new browser listening run
  is still required to establish actual Kokoro completion time and audio quality.
  No dependencies, model files or macOS security settings were changed for this fix.

## Bounded speech optimizations (2026-09-21)

The local profile now defaults to:

| Setting | Default | Meaning |
| --- | --- | --- |
| `STT_REUSE_ENCODER` | `true` | Single native decode for eligible complete windows of at most 30 s. Automatic language detection reuses the same encoded features. |
| `TTS_FIRST_CHUNK_CHARS` | `60` | Soft punctuation-boundary target for the first request only; at most two pieces, never forced space cuts. `0` disables. |
| `TTS_CPU_THREADS` | `2` | ONNX intra-op threads, configurable 1–8; inter-op remains 1. Changes require a server restart. |

**Whisper:** `whisper_reuse.py` uses the installed native decoder, not a different
model or a forced language. It preserves the fixed greedy transcription and
silence rules. Empty/long inputs and timestamp structures that require another
window use unchanged stock transcription, rather than dropping unfinished speech.
Errors still fail closed. `stt_final.decode_path` records `single_encode` or
`stock_fallback` when the optimization is enabled.

This is not guaranteed numerically equivalent to the old path: for short inputs,
stock language detection uses waveform-silence padding, while the single-encode
path detects language from the zero-padded transcription mel window. Language
selection/text must therefore be compared on real native inference. A fallback
after a tentative decode can also be slower than stock alone. The 30-second
encoder window itself, FIFO segmentation, and all generation guards are unchanged.

**Kokoro:** text is preserved codepoint-for-codepoint across pieces. The original
request retains one usage record, one deadline and one text/context entry. First
PCM is delivered before the remainder is synthesized. Interruption stops starting
later pieces and discards stale in-flight output. Sentence aggregation, voice
routing, model, speed and sample format remain unchanged. Smaller requests can
alter intonation, pauses and total generated duration; no quality equivalence is
claimed without listening. The native context consumer still owns first-audio
metrics, and first-request tracking is removed on context completion/interruption.

**CPU tuning is not claimed complete:** the default stays at two threads until a
measured 2-versus-4 comparison justifies a change. The background attempt could
initialize ONNX but macOS blocked eSpeak at first phonemization; MLX imports also
reported Metal unavailable. Automated normal-Terminal launch was unavailable in
that environment. No dependencies, model files, quarantine attributes or security
settings were changed. Native timing and listening verification are **blocked**
there, not evidence of slow or fast optimized hardware execution.

### Optimization-pass offline verification

- **449 tests passed** after integration, including single-decode eligibility,
  completeness fallback, lossless first-request splitting, cross-piece deadlines,
  cancellation and interruption, native context metrics, and fake-engine benchmark
  accounting. This test count does not include successful native model timing.
- `uv lock --check --offline --no-python-downloads`, Python compilation and shell
  syntax checks passed. All 15 fingerprinted protected project files (including
  private `.env`, prompts, dependency files, Compose, fixtures and migrations)
  were unchanged.
- Native timing, actual transcript comparison and audible join quality were not
  verified for this historical optimization pass. CPU thread default was
  deliberately not changed without data.

### Conservative phrase-boundary correction

The previous whitespace fallback could split `marine life` and generate `life.`
as an independent request. The corrected helper makes **at most one early cut**
at punctuation, with at least three word-like tokens and 16–24 characters on each
side (depending on the configured target). It ignores cuts inside quotations or
parentheses, common abbreviations/initials and numeric notation. If uncertain,
it leaves the input whole. This remains a punctuation heuristic, not a grammar
parser or a guarantee of identical prosody.

The Lakshadweep regression now yields exactly:

1. `Yes, Lakshadweep is a group of beautiful islands and a union territory of India, `
2. `known for its beaches and marine life.`

Exact whitespace/text reconstruction and the actual adapter call order are tested
offline. ASR, voices, sample format, deadlines and interruption behavior are
unchanged. Native audio/listening was not rerun for this correction. Some first
audio may arrive later because phrases are allowed to exceed the soft character
target rather than being broken at arbitrary spaces. Restart the server to load
the updated helper.

Verification for this correction: **483 offline tests passed**, lockfile validation,
Python compilation and shell syntax checks passed. The nine protected source and
configuration baselines were unchanged, including ASR, providers, TTS lifecycle,
private `.env`, prompts and dependency files.
