# Provider configuration

## What this stage implements

`config.py` validates settings and writes secret-free run records.
`voice/providers.py` centralizes construction of provider-specific services.
Individual adapters implement their protocol and lifecycle contracts. The voice
pipeline uses its three factories, not provider clients directly. Clinical tool backends are still absent.

The [minimal browser pipeline](voice-loop.md) is now implemented. Offline tests
check construction, wiring, and lifecycle behavior. The browser loop has been
manually exercised and session traces inspected; clinical accuracy and TTS
naturalness have not been benchmarked.

## Initial, configurable baseline

| Stage | Provider/model | Pipecat adapter |
| --- | --- | --- |
| ASR | OpenAI `gpt-live-transcribe` | `OpenAIRealtimeSTTService`, with a commit-aware finalization subclass |
| Text LLM/tools | OpenAI `gpt-4.1-mini-2025-04-14` | `OpenAILLMService`, with a small client-configuration subclass |
| TTS | OpenAI `gpt-4o-mini-tts`, voice `coral` | `OpenAITTSService`, with configurable PCM chunk size and owned-client cleanup |

These defaults are initial candidates, not evidence of model access or clinic
accuracy. Pipecat 1.11.0 and OpenAI SDK 3.16.2 are pinned in the optional `voice`
extra. All resolved dependency versions are in `uv.lock`.

ASR uses a transcription-only WebSocket, not native speech-to-speech. The adapter
sets `turn_detection=False`, so the local VAD/Smart Turn logic owns the turn
boundary. The current project is English-only: `STT_LANGUAGE=en` is the default
and is required for `gpt-live-transcribe` in this profile. The pinned library only
builds the legacy singular language field, so the project adapter changes the
final websocket `session.update` transcription payload to **`languages: ["en"]`**
and removes `language`. This also applies after reconnects and settings updates.
Legacy models such as `gpt-4o-transcribe` keep their singular field.

This follows the model-specific [OpenAI realtime transcription contract](https://developers.openai.com/api/docs/guides/realtime-transcription).
Outgoing JSON is tested through the real adapter with mocked websocket I/O;
English recognition accuracy and provider acceptance still require a live run.
Auto-language behavior remains opt-in for compatible local experiments, not the
active OpenAI front-desk profile. No dependency upgrade is required.

## Configuration and secrets

Settings come from `.env`, overridden by process environment variables. Missing
settings use the defaults documented in `.env.example`. Loading config does not
mutate the process environment. The default `.env` is relative to the current
working directory, not the installed package location. Run commands from the
repository root, or pass `--env-file` explicitly. If the file is missing, defaults
and process environment are used. Dotenv interpolation is disabled.

- Never replace an existing `.env` with `.env.example`; retain database settings.
- API keys are SecretStr fields, excluded from representation and serialization.
- The public record reports only whether an OpenAI key was supplied.
- Endpoint URLs cannot contain credentials, query strings, or fragments.
- Plain HTTP/WS endpoints are accepted only on loopback for future local models.
- Unknown STT_/LLM_/TTS_/VOICE_ setting names are errors, not silently ignored knobs.
- ASR supports `openai`, local `whisper`, or self-hosted `nemotron`.
  TTS supports `openai`, local `kokoro`, or self-hosted `breeze`.
  LLM supports `openai` or self-hosted `hybrid_diffusion`.
  Unsupported stage/provider combinations fail explicitly.
- Self-hosted stages use optional `STT_API_KEY`, `LLM_API_KEY`, `TTS_API_KEY`
  Bearer credentials, excluded from records and included in log masking.
  They never inherit `OPENAI_API_KEY`. That key is required only if at least
  one OpenAI stage remains. `LIVE_API_ENABLED=true` is always required.
- The [paired local speech profile](local-speech.md) selects Whisper large-v3-turbo
  on MLX and Kokoro ONNX on CPU without modifying `.env` or the OpenAI defaults.

Inspect settings without installing voice dependencies or supplying a key:

```bash
uv run --locked python -m healthcare_voice_agent --check-config
```

Every entrypoint invocation writes `runs/<unique-id>/config.json`, including
model names, endpoints, timeout settings, and Python/package versions.
The key and database settings are excluded. `--run-dir` can select a new
output directory; an existing config.json will never be overwritten. `runs/` is
ignored by Git.

## STT transcript finalization

`STT_FINALIZE_TRANSCRIPTS=true` enables ordered, segment-aware finalization and
consumer-side generation checks. Completed text can then release a Smart
Turn-complete turn without waiting out the STT p99 deadline. It does not shorten
VAD thresholds, disable transcript waiting, or change models. The exact protocol,
offline verification boundaries, remaining watchdog behavior, and rollback
command are documented in [the latency patch guide](latency-finalization.md).

Restart the server after changing this setting. For OpenAI, use `false` to restore
stock adapter and turn-strategy behavior for a comparison. Local Whisper requires
`true`: FIFO segmented inference and consumer-side generation checks are mandatory.
Nemotron also requires `true`, using ordered explicit commit acknowledgements.
Protocol inconsistencies or
failed sends stop the enabled session rather than silently dropping a segment.
The dispatcher overrides are specific to the pinned Pipecat version.

## HTTP connection-reuse experiment

`LLM_KEEPALIVE_EXPIRY_SECONDS` and `TTS_KEEPALIVE_EXPIRY_SECONDS` now default to
`60`, instead of the pinned SDK's 5-second idle expiry. Both accept positive,
finite seconds. The factory copies the SDK's exported `DEFAULT_CONNECTION_LIMITS`
and changes only `keepalive_expiry`; its 1000-connection and 100-idle-connection
caps, HTTP/TLS defaults, and request/connect timeouts stay as before. Retry policy
is separate: only the clinician LLM gets one native SDK retry, as described below.
The SDK defaults object is never modified globally.

This lets each session-owned client reuse an idle connection across longer
conversational gaps. It does not add pings, warm-up requests, provider calls,
shared cross-session clients, or keep a canceled/broken connection alive. Servers
and proxies can still close connections earlier. The first request still needs
a connection. This changes neither the STT WebSocket nor the existing 100 ms
TTS audio chunks and 1.5-second Smart Turn fallback.

The exact values are recorded under `configuration.llm.keepalive_expiry_seconds`
and `configuration.tts.keepalive_expiry_seconds`. Restart the server to load
these defaults; no private `.env` edit is needed. To restore only the previous
idle expiry for a comparison, restart with:

```bash
LLM_KEEPALIVE_EXPIRY_SECONDS=5 TTS_KEEPALIVE_EXPIRY_SECONDS=5 uv run --locked --extra voice python -m healthcare_voice_agent --serve --env-file .env
```

Pipeline construction also passes its session trace to both HTTP factories.
Public HTTPX2 request/response hooks attach the pinned httpcore2 trace callback.
The selected events in `events.jsonl` are:

- `http_request_start`: stage (`llm` or `tts`) and session-client request ordinal.
- `http_connection_phase`: observed TCP connection, TLS handshake, or Unix-socket
  connection duration and completion/failure. TCP timing includes the backend's
  connection work, potentially DNS; it is not a separate DNS measurement.
- `http_response_headers`: status code, elapsed time to response headers,
  observed TCP/TLS durations, and `connection_reuse` (`new`, `reused`, `unknown`).
  `reused` means a transport request-header event occurred without a setup event;
  no callback evidence means `unknown`, not automatic reuse. Header timing is
  not the full response duration, first token/audio time, or pure model inference.
- `http_request_failed`: a recognized phase failed, with a bounded `error_kind`
  (`timeout`, `transport`, or `unknown`) and whether headers arrived. A failure
  after HTTP 200 is not assumed to be successful stream completion.
- `http_request_cancelled`: a recognized cancellation exception, including before headers.
- `http_stream_closed`: recognized generator/stream closure, not provider-turn success.
- `http_response_body_complete` and `http_response_closed`: transport lifecycle
  events only. Neither establishes that a complete usable LLM reply was produced.

Classification inspects only an exception object's recognized class, never its
message, arguments, representation, or arbitrary class name. Unrecognized errors
remain failures with `error_kind=unknown`; none of these labels changes retries.

Only fixed event labels, allowlisted error categories, identifiers, status codes,
booleans, and timings are serialized. URLs, hosts, headers, bodies, trace callback info dictionaries, raw
exceptions, and SSL/connection objects are never copied to these events. Hooks
do not read or buffer the response body, enable SDK debug logs, or change error
propagation/retry behavior. Diagnostic sink failures do not fail HTTP requests.

Each front-desk/clinician session also records `flow_prompt_loaded`: hashes of the
loaded static role and native tool definitions, separate from the generic base
`prompt_loaded` hash. This fingerprints live Python modules, not newer source-file
bytes, so an unrestarted server can be distinguished from updated instructions.
Raw prompts, patient context and user messages are not included. This identifies
static prompts/contracts, not an entire source build or dynamic conversation.
Restart servers manually after edits before evaluating live behavior.

Offline tests exercise real HTTPX2/httpcore2 pooling against an in-memory backend
with all real sockets blocked. After 10 simulated idle seconds, a 60-second pool
reuses the connection and a 5-second pool replaces it. They also check actual
factory wiring, cleanup, failure propagation, secret-free phase timings, and
lazy streaming. This establishes pooling behavior, **not live latency savings**.

For a browser comparison, repeat fixed synthetic questions with 10–20-second
gaps after replies. Keep all other settings fixed. Compare the number of new
connections, TCP/TLS setup time, and existing end-to-end speech-start latency;
separate the first request and interrupted requests from ordinary follow-ups.
Do not shorten turn boundaries or change models within this comparison.

## Audio buffering and silence-fallback experiment

The general conversation profile retains these defaults from the earlier combined experiment:

The separate front-desk launcher now defaults only `TTS_AUDIO_CHUNK_MS` to `500`
for a larger PCM reservoir, while honoring an explicit environment override.
It leaves the 1.5-second Smart Turn setting unchanged. This scoped rollback has
a deterministic synthetic underflow regression, not live evidence that it fixes
all reported stuttering. See [the demo's buffering notes](frontdesk-demo.md#speech-buffering).

| Setting | Default | Previous baseline |
| --- | --- | --- |
| `TTS_AUDIO_CHUNK_MS` | `100` | `500` |
| `VOICE_SMART_TURN_STOP_SECONDS` | `1.5` | `3` |

The TTS factory overrides Pipecat's `chunk_size` property: at 24 kHz mono PCM16,
100 ms is 4,800 bytes instead of 24,000. This changes HTTP audio read buffering,
not request text, sentence aggregation, voice, instructions, speed, or PCM format.
The setting accepts whole milliseconds from 20 through 1000. Smaller buffers may
start output sooner but may also expose stalls. A 400 ms reduction in buffered
audio is **not** a guaranteed 400 ms wall-clock saving.

The pipeline passes `SmartTurnParams(stop_secs=1.5)` to the existing local
analyzer. This is its maximum continuous-silence fallback, not a mandatory wait
on every turn and not the VAD stop threshold. Resumed speech resets this silence
counter. A pause longer than the fallback can now be treated as a completed turn
sooner, including when the person intends to continue. The transcript-finality
checks, STT safety timeout, and global stuck-turn watchdog remain unchanged.

Both effective values are recorded in run/session `config.json`, respectively
under `configuration.tts.audio_chunk_ms` and
`configuration.voice.smart_turn_stop_seconds`. Restart the server to load them;
no private `.env` edit is needed when using the defaults.

```bash
uv run --locked --extra voice python -m healthcare_voice_agent --serve --env-file .env
```

To restore only these two settings to their previous values, while keeping the
transcript-finalization fix, restart with:

```bash
TTS_AUDIO_CHUNK_MS=500 VOICE_SMART_TURN_STOP_SECONDS=3 uv run --locked --extra voice python -m healthcare_voice_agent --serve --env-file .env
```

For this combined experiment, repeat the same synthetic questions and paused
speech in both modes. Reuse [the fixed latency cases](../evals/scenarios/latency-finalization-v1.json)
and include a longer thinking pause to inspect early replies. Listen for audio
gaps, clipped beginnings, and premature turn-taking. Record missing/interrupted
responses as well as latency. Keep TTS input text, models, voice, instructions,
and playback device fixed where possible. Since both settings change together,
an overall gain cannot be attributed to either setting alone. One-setting-at-a-time
runs can separate their effects later if needed.

Offline checks cover exact chunk sizes, unchanged request bodies and audio
bytes, streaming before completion, default/rollback configuration, analyzer
wiring, and the silence timer reset. They do not establish live latency gains,
audible smoothness, or learned turn-detection accuracy.

## TTS delivery instructions

The first naturalness change keeps `gpt-4o-mini-tts`, voice `coral`, the provider's
numeric speed default, sentence aggregation, PCM output, and turn handling
unchanged. `TTS_INSTRUCTIONS` now defaults to the fixed
`DEFAULT_TTS_INSTRUCTIONS` text in `config.py`. It requests warm conversational
intonation and clear, natural English while preserving the supplied text.
The former multilingual, accent-specific and code-switching directions have been
removed from active defaults. `TTS_LANGUAGE=en` is the current default.

The factory sends this through Pipecat's public
`OpenAITTSService.Settings(instructions=...)` field on every speech request.
There is no extra model call, transcript rewrite, or language-selection step.
These are synthesis instructions, not an evaluated quality improvement.

- Omit `TTS_INSTRUCTIONS` to use the built-in profile.
- Set it to custom style-only text to override the profile.
- Set it to `off`, `none`, `disabled`, or an empty value to omit instructions and
  reproduce the previous no-instructions setting.
- `tts-1` and `tts-1-hd` do not support instructions; configuration rejects them
  unless instructions are disabled, rather than silently dropping the setting.
- Exact instructions are saved under `configuration.tts.instructions` in both
  run and session config records. Treat this as non-secret configuration; never
  put credentials or patient information in it.

**Restart the server after changing provider settings or installing this change.**
A reconnect alone does not reload the frozen server-side provider configuration.
No `.env` edit or dependency reinstall is required to use the new default.

For a listening check, use the four fixed synthetic texts in
[`tts-naturalness-v1.json`](../evals/scenarios/tts-naturalness-v1.json). In the
browser, ask the assistant to repeat one test text exactly, then inspect
`tts_input` to confirm the actual text. A rephrased answer is not a matched-input
TTS comparison. The old v1 set includes historical multilingual experiments;
for the current English-only profile use its English cases. Check pronunciation,
rhythm, and whether any words were changed or dropped. Keep chunk boundaries and the
playback device fixed as well when making a controlled comparison.

To return to the previous setting without editing the private `.env`, restart
with an explicit process override:

```bash
TTS_INSTRUCTIONS=off uv run --locked --extra voice python -m healthcare_voice_agent --serve --env-file .env
```

Compare this against the default profile on the same fixed set, then review
before changing voices, models, speed, or wording. Offline tests check request
forwarding, exact input preservation, and instruction omission. They do not
measure audible naturalness or provider adherence to the instructions.

## Live-service gate

All three factories require `LIVE_API_ENABLED=true` and a non-empty private
`OPENAI_API_KEY`. No budget configuration is required. Legacy `API_BUDGET_USD`
environment values are ignored.

Defaults leave live services disabled. Without `--serve`, the CLI only validates
and records config. `--serve` checks the live flag and key before starting the
local server; provider sessions start when a browser negotiates its connection.

Automatic idle and session-duration cutoffs are disabled by default. Network,
startup, and cleanup timeouts remain enabled to recover from stalled operations.

## Timeout and cleanup semantics

- `LLM_TIMEOUT_SECONDS` and `TTS_TIMEOUT_SECONDS`: HTTP read/write/pool phase
  timeouts. They are not a total end-to-end deadline.
- `*_CONNECT_TIMEOUT_SECONDS`: HTTP connection timeout.
- `STT_WS_CLOSE_TIMEOUT_SECONDS`: WebSocket close handshake only. This does not
  configure connection establishment, endpointing, or a transcription deadline.
- HTTP SDK retries remain disabled for the baseline and front desk. The clinician
  LLM alone uses `max_retries=1`: at most two HTTP attempts for SDK-retryable
  connection/timeout/status failures. Pipecat's separate timeout retry remains
  disabled, so there is no second retry layer. No streamed response, completed
  tool, or whole turn is replayed by application code.
- After a usable, nonfatal clinician LLM error, the connection stays open and a
  fixed direct-TTS fallback asks the user to say "try again". Cancellation does
  not speak a fallback. See [clinician recovery](clinician-demo.md#recoverable-llm-errors).
  TTS retries and STT automatic reconnect remain disabled. Fatal/unusable errors
  and other profiles retain their existing stop behavior.
- Run records expose `provider_policy.llm_http_max_retries` and
  `provider_policy.tts_http_max_retries` separately. One retry may lengthen the
  wait; the existing phase timeouts/backoff are not a whole-turn deadline.

Pipecat 1.11.0 does not forward arbitrary HTTP timeout kwargs for its stock LLM
service. The local subclass uses its public `create_client` hook instead. TTS
accepts a configured HTTP client directly. OpenAI SDK 3.16.2 uses HTTPX2, so
both the HTTP client and Timeout type come from its public exports; legacy
`httpx.Timeout` objects are not mixed into that client. The LLM/TTS subclasses also close
their owned SDK clients in `cleanup()`; the stock services do not do so.
The session's worker owns service cleanup after startup. The assembly failure
path separately cleans up clients created before a later constructor fails.

## Install and test the real adapters

Run package installation in your normal Terminal, not a browser-agent process:

```bash
uv sync --locked --extra voice --no-cache --link-mode copy
uv run --locked --extra voice pytest
```

Avoiding the shared cache here prevents reusing any earlier quarantined downloads.
Do not disable macOS security checks or remove quarantine attributes globally.

The unit tests use explicit fake services to check wiring, timeouts, gates, and
cleanup. An additional integration test is skipped when Pipecat is absent. With
the voice extra installed, it constructs actual services with a dummy key,
blocks network connections, and cleans them up. It never starts their sessions
or performs inference. A constructor test is not a live API or voice-quality test.

## Adding another provider later

1. Add its dependency as an explicit optional extra.
2. Add its validated configuration and supported provider name.
3. Add dispatch/adapter construction inside `voice/providers.py`.
4. Implement its timeout, cancellation, and cleanup behavior explicitly.
5. Add offline constructor tests, then run the same frozen evaluation scenarios.

Do not assume an OpenAI-compatible endpoint supports the same streaming, tool,
language, or cancellation behavior. No provider changes require editing clinic
tool contracts or putting provider clients inside prompts/business logic.

## Verified interface references

- [Released Pipecat 1.11.0 OpenAI STT source](https://github.com/pipecat-ai/pipecat/blob/v1.11.0/src/pipecat/services/openai/stt.py)
- [Released Pipecat OpenAI LLM client hook](https://github.com/pipecat-ai/pipecat/blob/v1.11.0/src/pipecat/services/openai/base_llm.py)
- [Released Pipecat OpenAI TTS source](https://github.com/pipecat-ai/pipecat/blob/v1.11.0/src/pipecat/services/openai/tts.py)
- [OpenAI realtime transcription guide](https://developers.openai.com/api/docs/guides/realtime-transcription)
- [OpenAI GPT-4.1 mini model/snapshot](https://platform.openai.com/docs/models/gpt-4.1-mini)
- [OpenAI TTS instructions and voice guidance](https://developers.openai.com/api/docs/guides/text-to-speech)

## Self-hosted GPU adapters

See [GPU setup](gpu-models.md) for pinned runtime/model revisions and launchers.
The defaults above remain unchanged; `scripts/voice_gpu.sh` selects all three
new providers without modifying `.env`.

- Nemotron uses the actual NeMo-Speech.cpp WebSocket protocol, not OpenAI
  Realtime. It streams 16 kHz PCM during speech, flushes on local VAD stop, and
  waits for the ordered commit acknowledgement before emitting complete text.
  Ambiguous replacement-style partials are deliberately not displayed.
- Breeze sends the whole text with a voice-design `instruction` and reads 24 kHz
  mono PCM16 incrementally. It checks response headers and PCM alignment, keeps
  the native context alive without fake audio, and retains real first-audio metrics.
  The total bound is `TTS_INFERENCE_TIMEOUT_SECONDS`; connect timeout and
  `TTS_AUDIO_CHUNK_MS` also apply. Kokoro splitting, thread settings and SDK
  HTTP connection-timing hooks do not apply to Breeze's aiohttp transport.
- HybridDiffusion retains the OpenAI-compatible SSE/tool protocol through the
  pinned Pipecat service. Thinking is disabled through the checkpoint's chat
  template. `LLM_TIMEOUT_SECONDS` bounds the complete streamed turn as well as
  HTTP phases. Its server must use `qwen3_coder` for tool parsing; model/tool
  correctness has not been validated by protocol tests.

No retries, hosted fallbacks, automatic model downloads or dependency installs
are performed. Use synthetic data only and restart after changing providers.
