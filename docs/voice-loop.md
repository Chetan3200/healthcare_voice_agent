# Smallest browser voice loop

## Scope and verification boundary

This document describes the default `conversation` connectivity profile, not
the separate [clinician assistant](clinician-demo.md) or [front desk](frontdesk-demo.md).
The default profile has no tools,
database connection, retrieval, embeddings, or patient context. Use synthetic
speech only. Do not speak real patient information or credentials.

The code and offline tests cover pipeline assembly, local turn-model loading,
real Pipecat worker startup/readiness/cancellation with fake services, per-session state,
disconnect/deadline cleanup, redacted traces, and safe browser errors. An optional
ASGI test checks the real bundled UI routes after the browser extras are installed.
These checks do **not** verify a microphone, live model access, speech quality,
actual playback, or audible disconnect/reconnect reliability.

## Install

Run in your normal Terminal, from the repository root:

```bash
uv sync --locked --extra voice --no-cache --link-mode copy
uv run --locked --extra voice pytest
uv run --locked --extra voice python -m healthcare_voice_agent --check-config --env-file .env
```

The updated extra includes Pipecat's `runner` and `webrtc` dependencies as well
as OpenAI. All versions are in `uv.lock`. Do not install them from the browser
agent on this Mac, disable macOS security checks, or change quarantine flags.

Tests use dummy credentials and fake services/network guards, not paid inference.
The browser-route test is skipped if FastAPI, aiortc, or the prebuilt UI is absent.

## Live prerequisites and startup

Add your key privately to the existing `.env` and preserve all database values:

- `OPENAI_API_KEY`: your private key. Never paste it into chat or commit it.
- `LIVE_API_ENABLED=true`: explicit opt-in to live services.

There is no budget requirement. Automatic idle and session cutoffs default to
disabled. If an older `.env` explicitly sets them, remove those values or set
`VOICE_IDLE_TIMEOUT_SECONDS=none` and `VOICE_MAX_SESSION_SECONDS=none`.
The default models are documented in [providers.md](providers.md).

```bash
uv run --locked --extra voice python -m healthcare_voice_agent --serve --env-file .env
```

Open **http://127.0.0.1:7860/client/**. Choose the local WebRTC transport if the
bundled UI offers a transport choice. Allow microphone access, select the right
microphone/speaker, click Connect, wait for readiness, and speak first. Headphones reduce echo.
There is no unsolicited greeting.

Starting the server alone does not construct provider services. Negotiating a
browser connection starts them and can incur usage, including streaming silence.
Disconnect when finished. Ctrl+C stops the development server.

The host is fixed to `127.0.0.1`; configure `VOICE_PORT` if necessary. This is an
unauthenticated local development server, not a public deployment. Do not put it
behind a public tunnel. No external ICE servers are configured by this app.
Pipecat's own client may explicitly request a public STUN fallback.

## Pipeline and turn ownership

```text
bundled browser UI / SmallWebRTCTransport
  -> microphone input (16 kHz mono PCM inside Python)
  -> client-ready audio gate
  -> streaming STT
  -> user context aggregator: Silero VAD + Local Smart Turn v3.2
  -> text LLM with the simple agent/prompts.md instruction
  -> streaming TTS
  -> transport output (24 kHz mono PCM inside Python)
  -> assistant context aggregator
```

The wire transport is WebRTC, not raw PCM HTTP. Pipecat handles transport codec
conversion. The OpenAI STT adapter resamples the 16 kHz input to its required
24 kHz stream internally. Its server-side turn detection is explicitly disabled.
There is only one local VAD/turn owner. The user aggregator owns analyzer cleanup.
A separate startup gate drops microphone frames until RTVI client-ready; it is
not a second turn controller. This is explicit because SmallWebRTC in 1.11 does
not honor the generic `audio_in_stream_on_start` flag. Pre-ready audio is not
buffered or replayed.
The local ONNX assets are bundled in the pinned Pipecat package, not downloaded
from a model hub at session startup.

Provider construction stays in `voice/providers.py`; `voice/pipeline.py` has no
OpenAI service/SDK imports or hardcoded model names. Every connection constructs
new providers, context aggregators, and a worker runner. A reconnect does not
reuse the previous conversation. Browser request bodies cannot change the server
configuration, prompt, or trace directory.

## Response-language policy

The active project is English-only. The packaged prompt always requests English
and no longer selects among languages or follows mixed-script transcription into
a different response language. STT defaults to English; the gpt-live adapter sends
`languages: ["en"]` using the model-specific wire field. TTS instructions also
request English without translation or transliteration.

Restart the server to load the provider settings and code, then reconnect. The
old `response-language-v1.json` cases remain historical multilingual fixtures,
not the current response policy or a measured evaluation. Offline tests check
configuration, real-adapter outgoing request shape, prompt forwarding and
reconnect behavior. They do not prove perfect English recognition.

## Transcript-wait latency patch

The first latency patch is enabled by `STT_FINALIZE_TRANSCRIPTS=true`. It marks
only ordered, relevant committed transcripts as finalized, while keeping local
VAD and Smart Turn. Additional generation/item checks prevent older results from
satisfying a newer speech segment's stop gate. The native p99 fallback and global
stuck-turn watchdog remain in place.

See [the latency patch guide](latency-finalization.md) for the exact scope,
protocol assumptions, trace fields, frozen manual cases, and baseline switch.
Restart the server to load this code/configuration change; reconnect alone is
not sufficient. Offline tests verify the control flow, not live latency savings.

## Smaller audio chunks and shorter silence fallback

The current experiment uses `TTS_AUDIO_CHUNK_MS=100` and
`VOICE_SMART_TURN_STOP_SECONDS=1.5`, replacing the earlier 500 ms audio read chunks
and 3-second Smart Turn fallback. Models, TTS style/speed, sentence aggregation,
VAD thresholds, and transcript-finality guards are unchanged. The fallback is
continuous silence, not a mandatory delay on each turn.

See [configuration, rollback, and listening checks](providers.md#audio-buffering-and-silence-fallback-experiment).
Both settings require a server restart. Compare latency together with playback
gaps and premature replies during thinking pauses; neither real-world speed nor
quality is established by offline tests.

## Session limits and recovery

| Setting | Default | Meaning |
| --- | ---: | --- |
| `VOICE_IDLE_TIMEOUT_SECONDS` | `none` | No automatic idle cutoff; optional positive seconds |
| `VOICE_MAX_SESSION_SECONDS` | `none` | No automatic session cutoff; optional positive seconds |
| `VOICE_SETUP_TIMEOUT_SECONDS` | 20 | Bound processor setup, including STT setup |
| `VOICE_START_TIMEOUT_SECONDS` | 20 | Bound propagation of the pipeline start frame |
| `VOICE_CANCEL_TIMEOUT_SECONDS` | 10 | Maximum framework wait for the cancellation frame to drain |

With both cutoffs disabled, the call stays active until you disconnect or a
service fails. Startup, network, and cleanup timeouts are still used for stalled
operations. These are not total conversation-duration limits.

Disconnect requests immediate cancellation, not graceful playback of queued
speech. Errors stop the demo session instead of automatically retrying a broken
provider. Check the local trace, correct the problem, then reconnect. Errors
shown in the browser are deliberately generic; provider details stay in the
redacted server trace. Pipecat handles its normal interruption flow. There are
no additional custom clinical correction/generation guards in this step.

## Logs

Each server invocation writes a non-secret configuration record. Each connection
writes its own folder:

```text
runs/<run-id>/
  config.json
  sessions/<random-session-id>/
    config.json
    prompt.txt
    events.jsonl
```

Each JSONL event includes UTC time, monotonic elapsed time, a sequence number,
session ID, event name, and event data. Selected frame records also carry their
frame ID/source. TTS request events carry the TTS context ID.

Important event groups:

- `stt_interim` / `stt_final`: provider transcript frames, kept separate.
- `user_turn_committed`: the text actually added by the user aggregator.
- `assistant_generated_text`: generated LLM text, not speech already heard.
- `tts_input`: the prepared text sent to TTS.
- `tts_output_text` / `assistant_context_turn`: output-linked text/context,
  including interruption status on completed assistant turns.
- `metrics`: tagged Pipecat metric types for stage timing and usage where emitted.
- `user_to_first_bot_speech_latency`: server-observed latency, not browser-measured
  time to audible playback.
- `http_request_start`, `http_connection_phase`, `http_response_headers`, and
  `http_request_failed`: selected connection setup/reuse evidence for the LLM and
  TTS clients. These contain no URLs, headers, or request/response bodies. See
  [the connection-reuse experiment](providers.md#http-connection-reuse-experiment)
  for the 60-second idle expiry, timing meanings, limitations, and 5-second rollback.
- `processor_setup`, lifecycle, timeout, interruption, and redacted error events.

Traces use private files, remain under ignored `runs/`, and mask configured keys,
recognizable credentials, and credential fields. Console messages and exception
summaries are masked through Loguru and stdlib logging hooks. Uvicorn access
records retain their five sanitized arguments (including the integer status
code), because its formatter unpacks these fields. Other stdlib messages retain
the existing rendered-message redaction behavior. Regression tests exercise the
real access formatter, including colored output and repeated redactor installs.
These are credential filters, not a general PHI scrubber. The requested transcripts are retained, so
synthetic-only usage is essential. Structured traces do not serialize raw audio,
full contexts, or raw exception objects. Selected JSONL writes are synchronous
and small; instrumentation overhead has not been benchmarked.

This step does not record WAV files. It rejects the hidden
`PIPECAT_SMART_TURN_LOG_DATA=true` recording switch. Deliberate raw-audio capture
and richer generation/case correlation belong to later audited evaluation work.
Do not claim exact heard-so-far text after an interruption: placing the assistant
aggregator after output helps, but provider timestamps and actual browser playback
must still be tested with the selected TTS service.

## Manual checkpoint and observed status

The user has reported a working microphone/speaker loop and barge-in. The
pre-TTS-style session `c4609ed0024e4a9b8beac5e8502329bf` in run
`20260920T174452Z-41a5f054` was inspected during the earlier multilingual experiment: its saved prompt matched the then-current
language policy, the English and Hindi questions received matching response
text, and an explicit English-only preference held for two subsequent Hindi
questions. The session ended with `session_finished: client_disconnected`.

This was not a complete language-suite pass: the first place name was transcribed
as Latin-script `Bangalore`, so the original mixed-script English-name edge case
was not reproduced. The mixed-language population retry received Hindi rather
than a clear Hinglish response. Two other response attempts were interrupted
before generating text; one followed an interim `Yeah` transcript. No raw audio
was recorded, so the trace cannot establish whether that detection was genuine
speech. These observations are separate from TTS delivery quality.

The first TTS delivery-instruction change is described in
[provider configuration](providers.md#tts-delivery-instructions). Its outgoing
request wiring is tested offline; audible improvement still needs a listening
check. Preserve the remaining lifecycle checks below rather than assuming they
all passed from a normal disconnect or from the user-reported basic loop.

Test the real microphone and speaker path:

1. Connect and say a simple fictional greeting. Confirm a final transcript and an
   audible reply, not just generated text in the UI.
2. Ask a short English question. Verify that the transcript and reply remain
   English; note recognition errors separately from response-generation errors.
3. Disconnect while idle. Verify a terminal `session_finished` event.
4. Reconnect and repeat. Verify a new session folder and no old conversation state.
5. Disconnect during a reply, then reconnect once. Confirm old audio does not
   continue playing and the new session responds.

Record success or the exact failure for each check. The checkpoint is complete
only after the actual microphone/speaker/provider path passes. A constructor,
mock-lifecycle, or HTTP-route test is not a substitute.

## Pinned framework boundary

Pipecat 1.11 discovers `bot()` on the executed `__main__` module. The small console
adapter uses Python's `runpy` to match `python -m healthcare_voice_agent`; it does
not depend on scanning unrelated files in the working directory. A custom public
argparse parser sets local WebRTC defaults and carries the frozen config through
`runner_args.cli_args`. Signaling routes and `/client` remain Pipecat-owned.

Auto-RTVI stays enabled for the bundled client's ready handshake. The narrow
`SafeRTVIProcessor` override replaces two **private**, version-pinned error-send
methods because 1.11 has no public error-redaction option. Its regression test
must be revisited before upgrading Pipecat.
