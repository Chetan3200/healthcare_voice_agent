# First latency change: committed-transcript finalization

## Scope

This guide describes the first, transcript-only patch. The later
[audio-buffer/silence-fallback experiment](providers.md#audio-buffering-and-silence-fallback-experiment)
changes two additional defaults. To isolate this first patch when comparing its
`STT_FINALIZE_TRANSCRIPTS` switch, explicitly keep `TTS_AUDIO_CHUNK_MS=500` and
`VOICE_SMART_TURN_STOP_SECONDS=3` in both modes.

This patch fixes the missing transcript-finalization signal in the pinned
Pipecat 1.11 OpenAI realtime STT adapter. It does **not** change STT/LLM/TTS models,
TTS voice or delivery instructions, language selection, VAD thresholds, Smart
Turn inference, HTTP timeouts, or session cutoffs. It does not turn off
`wait_for_transcript` or lower the STT p99 safety timeout.

Before this patch, completed OpenAI transcription items were emitted as
`TranscriptionFrame(finalized=False)`. Even when Smart Turn considered the user
finished, it generally waited until the STT safety deadline rather than using
the already-available completed text. Earlier traces commonly attributed about
0.5–0.8 seconds to this extra wait. That is a target, not a measured saving from
this patch. Slower LLM/TTS requests and genuinely incomplete turns remain.

## How it works

1. Local VAD stop reserves an audio commit before awaiting framework processing.
2. The actual commit send keeps that reservation's identity across resumed
   speech. WebSocket sends are serialized so backpressure cannot reorder them.
3. `input_audio_buffer.committed` binds the provider item ID to the pending
   commit, validating its predecessor chain.
4. Completed transcription items are buffered by ID. Completion order need not
   match speech order, and completion may arrive before its acknowledgement.
5. When the speaker is quiet and all currently reserved commits are complete,
   their text is released once, in commit order, as one frame. Only a relevant
   completed batch receives the fast-finalization signal.
6. The Smart Turn stop strategy rechecks acoustic generation and interim-item
   identity at consumption time. This covers a new VAD system frame overtaking
   an already-queued older transcript. The aggregator retains the older words,
   but those words cannot satisfy the newer segment's STT gate.

Partial transcripts are not treated as completed text. New speech clears the
older text from the stop strategy's eligibility state, not from conversation
aggregation. Smart Turn must still report completion. Duplicate completed items
and late deltas for already-completed items are not emitted again.

Malformed/unmatched results, inconsistent ACK order, reported transcription
failures, and send/stream failures stop the session instead of releasing a
known-incomplete batch. Tracking is bounded to 64 outstanding commits and a
512-item recent duplicate cache. Unknown completions can be provisionally held
only while sent commits await ACK; they cannot be released without a matching
acknowledgement.

## Explicit boundaries

- Requires one fresh transcription session, unique provider item IDs, ordered
  commit ACKs with a consistent `previous_item_id` chain, and server VAD disabled.
- The receive dispatcher and handler overrides depend on **Pipecat 1.11.0**.
  Recheck them before a dependency upgrade.
- This is not an ASR accuracy fix, a replacement for VAD, or a general guarantee
  against missed/false speech detections.
- The existing controller's five-second stuck-turn watchdog is unchanged. Its
  forced-stop behavior under a prolonged stall is not replaced by this patch.
- Offline protocol tests do not establish live backend compatibility or an
  end-to-end latency improvement. Verify both with a real browser run.

## Restart and compare

The default is `STT_FINALIZE_TRANSCRIPTS=true`; no private `.env` change is
needed. Restart the server, then reconnect the browser:

```bash
uv run --locked --extra voice python -m healthcare_voice_agent --serve --env-file .env
```

To restore the previous STT adapter and turn-strategy behavior without editing
`.env`, stop that server and launch a separate run with:

```bash
STT_FINALIZE_TRANSCRIPTS=false uv run --locked --extra voice python -m healthcare_voice_agent --serve --env-file .env
```

Use the same microphone, voice, model settings, and fixed utterances for both
runs. Compare the six cases in
[`latency-finalization-v1.json`](../evals/scenarios/latency-finalization-v1.json).
Do not silently tune thresholds or prompts based on failures within this set.

Record failures and missing/canceled responses as well as timed responses.
Report counts and median/range for the plain completed questions separately
from paused speech, corrections, and barge-in. These are server-observed speech
starts, not measurements at the listener's headphones.

## Trace evidence

The configuration records `stt.finalize_transcripts` and the voice-policy mode.
The `stt_final` trace now includes:

- `finalized`
- `speech_generation`
- ordered `commit_sequences` and `item_ids`
- per-piece `completion_received_at`

Use these with `user_turn_committed`, interruption events, and the existing
latency breakdown. A true flag at the producer is not on its own proof that the
consumer accepted an immediate stop; compare the downstream turn event too.
No raw provider result or audio is added to the trace.

Offline checks cover all completion-order permutations for three pieces,
completion-before-ACK, resumed/overlapping speech, queued stale frames, newer
interim items, duplicate events, empty results, failures, teardown, and rollback.
They exercise the actual Pipecat adapter methods and the real turn controller
and user aggregator with deterministic analyzers and no network.

## References

- [OpenAI realtime transcription and item ordering](https://developers.openai.com/api/docs/guides/realtime-transcription)
- [Pinned Pipecat OpenAI STT adapter](https://github.com/pipecat-ai/pipecat/blob/v1.11.0/src/pipecat/services/openai/stt.py)
