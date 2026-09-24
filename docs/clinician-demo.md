# Synthetic clinician assistant

For caller lines and presenter cues covering both agents, see
[voice demo conversations](voice-demo-conversations.md).

## Implemented

One native Pipecat Flows node, `clinician`, with the five required clinical tools
plus `open_case`, a session-only case-selection control:

| Tool | Authority and boundary |
|---|---|
| `open_case` | Exact case ID selects an authorized synthetic case; a name returns identity candidates only. No database writes. |
| `get_study` | Metadata only for the selected authorized case, including eligible model-result IDs/versions for a selected study. Multiple studies produce same-case candidates, not a guessed selection. |
| `get_model_result` | Stored completed predictions only. No inference or image interpretation. Scores are not clinical certainty. |
| `get_reviewed_report` | Reviewed reports only. No draft/model substitution; explicit eligible historical versions are labelled. |
| `get_appointments` | Read-only case appointments, filtered by time/type/status. `statuses=null` means all statuses, not just booked visits. |
| `search_clinic_instructions` | Exa search on allowed public domains, with provider-extracted passages and source URLs. No generated answers inside this tool. |

Every nullable argument is required: use `None` in Python and `null` in JSON.
All six share `status`, `data`, `error`, `warnings` and `meta` with request ID,
retrieval time, duration and applicable case-context version. Validated output
schemas and examples are under `docs/contracts/fracture-clinic-tool-*`.

The first four use the existing synthetic `clinic` database and parameterized
SELECTs on read-only connections. The server confirms an authorized synthetic
case through `open_case`, not at startup. No patient is preselected. Clinical
read arguments cannot change the selected case themselves. Name-only searches
return minimal authorized names/patient IDs/case IDs, never birth dates or clinical
evidence, and require clarification. The query and output schema omit DOBs.
The agent offers case IDs to distinguish matches and does not volunteer DOBs,
including after selection. An exact caller-supplied case ID needs no extra
confirmation or name/DOB verification.

`status=ok` with `data.state=selection_required` is only a candidate lookup, not
an opened case. The prompt and clinical tool descriptions require a completed
`open_case` round with `status=ok` and `data.state=selected` before clinical reads.
Parallel calls remain enabled for independent reads after their prerequisites.
Spoken dates use day + named month + year (e.g. “21st September 1996”), not ISO
separators; structured tool dates remain unchanged. This is a prompt instruction,
not a deterministic pronunciation filter or verified live speech behavior.
The prompt also requires exact normalization of the latest caller-supplied case
ID, explicit timezone labels, appointment-only retrieval for appointment-only
questions, and a separate answer for every requested missing field (including
end time). Overlapping bookings require scheduling staff to resolve; caller
preference is not proof. Status answers do not offer booking, cancellation or
rescheduling; explicit write requests are referred to the front desk.

Every selection attempt clears the old case and increments a context version;
failed or ambiguous switches cannot fall back to old records. Results from an
obsolete context are discarded. Pipecat's native RESET strategy reuses the same
node and removes previous patient evidence from model context. Only a successful
selection preserves the latest imperative user request for remaining reads. Failed
or ambiguous selections do not replay it: failures explain the error and await new
input; candidate lookups ask which returned case to open. This addresses the repeated
`open_case` loop without disabling parallel reads or retaining old case evidence.
Offline tests verify reset messages, not live-LLM compliance. Normal VAD interruption stops prior output. This is not a
played-audio ledger or a guarantee of live model/ASR correctness. One case is
active at a time; cross-patient comparisons are not supported.

This remains local demo access scoping, NOT production authentication. No clinical
database writes, migrations or seeding are needed for the existing database.
The optional legacy `CLINICIAN_CASE_ID` setting no longer preselects a case.
Selection schemas: `docs/contracts/clinician-case-selection-contract.json`.

The front desk stays separate: its own node, seven tools, booking service and
isolated database on port 55432. The clinician uses existing `DB_*` / `POSTGRES_*`
settings (5433 in this installation). Do not point it at real patient data.

## Run

Keep existing `.env` database/OpenAI values. Add `EXA_API_KEY` privately for web
search; do not paste the key in chat. Optional settings:

```dotenv
CLINICIAN_USER_ID=SYN-USER-CLIN
EXA_SEARCH_DOMAINS=nhs.uk,nice.org.uk,orthoinfo.aaos.org
```

```bash
./scripts/voice_clinician.sh --check
LIVE_API_ENABLED=true ./scripts/voice_clinician.sh
```

### Background-search demo (opt-in artificial delay)

```bash
CLINICIAN_DEMO_WEB_SEARCH_DELAY_SECONDS=15 LIVE_API_ENABLED=true ./scripts/voice_clinician.sh
```

The flag defaults to `0`, accepts 0–15 seconds, and delays only the first valid
`search_clinic_instructions` invocation per session before the real Exa call.
It uses asynchronous sleep, not a blocked worker/event loop. No question needs
an exact trigger phrase. Later searches and database reads remain undelayed.
The configured delay is recorded separately from real provider latency.

After three seconds, an unfinished read/search may produce exactly one fixed
notice, without another LLM call:

> Can I help you with anything else while the request is being processed?

A notice is skipped if work finishes or a new conversational turn starts before
it is dispatched. The timer is not a browser-audio latency guarantee. Tool work
uses native Pipecat tasks, not another agent or an external/persistent queue.
Native cancellation helpers are advertised for explicit user cancellation; there
is no comparison-specific or general-purpose job-management tool.

Results are installed through native callbacks with automatic inference disabled.
One gate resumes the existing LLM when no user turn, model response, foreground
batch or speech output is active. It waits for the output transport's ordered
TTS completion, not just text completion or an underrun-induced bot-stop signal.
A new foreground turn may consume ready results instead of a separate wake-up.
Case switches invalidate pending delivery and cancel native reads before reset;
session shutdown suppresses late output. The handler deadline is 29 seconds,
inside the existing native 30-second safety deadline; timeout results use the
same delivery path, not a retry. Bounded synchronous DB/HTTP work already running
in a thread may still finish after cancellation, but must not publish late output.

Barge-in retains a result exposed to an interrupted reply for the next turn.
After an LLM-error fallback, retained results wait for the next caller turn;
there is no hidden application retry. Native incomplete-turn waits remain active.
There is no exact audio-resume or proof-of-hearing guarantee. Model compliance and
browser audio still require a manual live test. No acknowledgement classifier,
VAD changes, audio-format changes, migration, server restart, or provider calls
are performed by enabling/configuring these local code paths.

The clinician launcher defaults to a one-hour session limit
(`VOICE_MAX_SESSION_SECONDS=3600`) and a separate 120-second idle timeout.
Explicit shell environment values override these launcher defaults. Restart the
server manually for a changed limit to apply; an already-running server retains
its startup configuration. The front-desk limits are unchanged.

Open `http://127.0.0.1:7863/client/` and Connect. Start with “Open case 1042.”
See [50 test prompts and expected outcomes](clinician-test-queries.md). The first command performs only a
read-only database/access check; it does not call Exa/OpenAI or validate keys.
The launcher does not install, migrate, seed or download models. Connecting the
browser opts into OpenAI usage; requesting guidance additionally uses Exa.
Without an Exa key the four database tools still work and web search returns an
explicit backend/configuration error, not an empty successful result.

The profile keeps English STT (`gpt-live-transcribe`), GPT-4.1 mini, Marin
(`gpt-4o-mini-tts`), 500-ms audio delivery chunks and full-reply TTS buffering.
Full-reply buffering improves cross-sentence coherence but delays first audio.
Native VAD/Smart Turn and finalized transcripts are retained. The clinician also
uses Pipecat's native incomplete-turn filtering, described below. No phrase
validator, extra confirmation loop, dialogue planner or audio-delivery gate was added.

## Spoken guidance style

The clinician prompt asks for short connected speech rather than written source
lists: usually two to four points and roughly 60-100 words, with more detail when
requested or needed for safety. Publishers are named naturally, without repeated
source brackets, Markdown links, URLs or numbered bullets in speech. Shared
advice may be combined; different patient groups, follow-up intervals and source
disagreements must stay distinct. Exact quotations must come from returned
passages and be separated from explanation with spoken cues such as "The leaflet
says" and "In plain English". Fictional style examples are explicitly not evidence.

A publisher/leaflet follow-up is still guidance, not case selection. The existing
search tool accepts an exact returned source URL in `document_id`, with a generic
query and `version=null`; multiple possible pages require clarification. This is
excerpt retrieval, not a claim to have read the entire page or verified clinic
policy. No search backend or audio/interruption behavior was changed for this
prompt update. Offline prompt assertions do not establish live speaking quality.

## Tool batches and model-version discovery

The clinician-only `ClinicianFlowManager` adapter uses Pipecat 1.11's existing
per-group completion state instead of forcing an LLM response after each result.
It also covers a fast validation error arriving before a sibling's in-progress
frame. Case-selection transitions are consumed once; unrelated ordinary tool
groups can continue. This is not a global conversation lock. The front desk uses
the unchanged native manager. Independent reads remain parallel. Clinician reads
and search now use native `cancel_on_interruption=False`; `open_case` still cancels
on interruption. A small delivery gate coordinates notices and result wake-ups;
it does not execute tool jobs, cache clinical data, or add a retry controller.
The adapter depends on pinned Pipecat internals and needs
its focused tests rerun before upgrading Pipecat.

This prevents replies from partial tool batches, not incorrect model-selected
calls themselves. The prompt/tool descriptions require a known study ID before
a dependent model/report lookup, discourage repeated successful reads, preserve
score meanings, and forbid inferring nonexistent history from one failed lookup.

`get_study.data.model_results` adds four metadata fields per eligible completed
result: `model_result_id`, `model_version`, `record_version`, and `is_current`.
The read-only query independently scopes authorization to the selected case/study
and shares the study transaction. No prediction text or scores are included.
An empty list means no eligible results; null/omitted in an older response means
not enumerated. Use a returned ID to retrieve a requested historical result with
`get_model_result`. Current lookup still uses the explicit current designation;
pending/failed current runs never silently fall back to historical results.
`get_study.data.current_model_result` separately exposes the designated active
current result's ID, model/record version, and inference status, including pending
and failed states. It contains no prediction text. Multiple active current records
return `RECORD_CONFLICT`. For a current request, use `model_result_id=null`; do not
choose the first completed historical ID. Case 1052 exposes pending V2 while V1
remains discoverable only for an explicitly requested historical lookup. No new
tool, database schema change or migration is involved.

Garbled clinical terms such as “risk caste” require brief clarification rather than
confidently switching topics. Clear case IDs and clear requests still do not require
extra confirmation. This is prompt guidance, not an added ASR/backchannel classifier.

`clinician_tool_call` trace entries link the provider tool-call ID to the existing
backend request ID. They record only nulls, recognized synthetic record-ID shapes,
and known generic case placeholders; other values are redacted. Patient-name and
web-query arguments are omitted. These diagnostics do not change accepted IDs.

## Recoverable LLM errors

The clinician LLM uses one native OpenAI SDK retry (`max_retries=1`): at most two
HTTP attempts for SDK-retryable connection, timeout and status failures. Pipecat's
separate timeout retry stays disabled. No application retry controller, whole-turn
replay, completed-tool replay, global lock or reconnect loop is added. Backend
outcomes such as `RECORD_NOT_FOUND` are tool results, not provider failures to retry.

If the request still fails and Pipecat considers the LLM usable/nonfatal, the
session stays connected and sends this fixed text directly to TTS, without
another LLM call:

> I'm still here, but I couldn't finish that request. Say 'try again' and I'll retry.

An ordered data frame discards any unfinished buffered reply before the fallback.
Cancellation similarly discards partial text before Pipecat's final response-end
frame can flush it. Native speech interruption, explicit tool cancellation and
independent tool parallelism are retained; interrupted speech does not generate
a fallback or cancel surviving async reads.
A failed incomplete response also cancels its pending native re-engagement timer,
so it cannot silently start another LLM request after the fallback. Successful
incomplete turns keep their normal native timing. The SDK does not retry a response
after streaming has begun: a mid-stream error uses the fallback without replaying
that stream. A later user request can continue with the existing conversation context.

The existing generic browser error notification is retained. Traces distinguish
`recoverable_llm_error` (`action="keep_session_open"`) from
`llm_error_fallback` (`action="queued_for_tts"`); queuing is not proof of playback.
If TTS itself fails, spoken delivery is not guaranteed and its existing error/stop
policy still applies. Fatal/unusable LLM errors and other profiles also retain
their previous stop behavior. TTS retries and STT reconnect remain disabled.

Offline tests cover retry success/exhaustion, non-retryable errors, cancelled
backoff, direct speech, incomplete-turn gating, partial-output discard, subsequent
requests, and a completed tool followed by SDK retry without tool replay. These
are mocked-provider tests, not live voice or LLM-compliance verification. The
one-hour clinician session limit applies after a manual server restart; the
separate 120-second idle timeout remains unchanged.

## Incomplete-turn filtering: enabled for the clinician

The clinician uses `FilterIncompleteUserTurnStrategies` around the existing
`SpeechGenerationTurnStopStrategy` and Smart Turn analyzer. The front-desk and
plain conversation profiles retain their previous turn strategies.

1. VAD and Smart Turn detect a possible end of speech; the finalized-transcript
   generation guard remains in place.
2. Smart Turn triggers the normal conversational LLM request. Pipecat adds its
   native completion instructions, including after a role/prompt update.
3. A complete verdict releases the normal answer. An incomplete verdict suppresses
   answer text and waits for continuation; earlier fragments remain in context.
4. Native defaults wait **5 seconds** for a short incomplete fragment or **10
   seconds** for a thinking pause, measured from the LLM verdict. If the user does
   not continue, Pipecat asks the LLM for a brief re-engagement response. Resumed
   speech cancels that pending re-prompt.

The clinician's native no-activity turn watchdog is **15 seconds**, rather than
its ordinary 5-second default, so it does not pre-empt the 10-second wait under
normal response timing. This is a bounded fallback, not a latency guarantee.
Other profiles retain the 5-second watchdog. These choices are recorded in each
run's `config.json`; no extra custom timer/controller or classifier was added.

A complete text turn uses one LLM request, not a separate classifier request
followed by an answer request. Continued fragments and timeout re-prompts can add
requests. Actual latency and classification accuracy remain unmeasured. The
feature depends on the LLM following the protocol; native Pipecat can fall back
to forwarding text if the model omits its completion marker. It is not a hard
semantic safety guarantee and does not prevent VAD-triggered backchannel interruptions.

Full-reply TTS is unchanged. Native completion markers are not sent to TTS, and
incomplete answer text does not produce a synthesis request.

**Focused verification:** 23 tests passed across the new filter integration
module, existing finalized-transcript tests, full-reply TTS tests and clinician
runtime tests. The new cases cover all three profile choices, complete responses,
short/long incomplete responses plus continuation, one timeout re-prompt,
role-update compatibility, tool calls without a text marker, Smart Turn's
incomplete verdict, stale completion after resumed speech and re-prompt
cancellation while speaking. Network access was blocked and LLM responses were
scripted; this validates native wiring/protocol handling, not live judgement.
Timeout tests use shortened test-only delays; production retains 5s/10s waits.

Native aggregator shutdown may flush unsent text into a final inference. The
Smart Turn no-inference check is therefore scoped to listening, before shutdown,
not a claim that Pipecat makes no request during disconnect cleanup.

No provider call, server restart, install, migration or model download was made.

## External guidance is not clinic policy

Exa web search is the agreed replacement for versioned clinic-document retrieval.
A separate versioned-document backend is not required under this revised scope.

The search contract is **2.0.0**; clinical read contracts remain **1.0.0**.
Dummy database document search has been replaced, not disguised as web retrieval:

- `document_id` means an exact source URL. Other versions of that page are not
  silently substituted. Explicit `version` requests return
  `DOCUMENT_VERSION_NOT_FOUND` because verified version retrieval is unsupported.
- Results have `version_scope="web_unversioned"`, `document_version=null`,
  `approval_status="not_verified"`, source URL, retrieval time, publication date
  when available, and an `EXTERNAL_GUIDANCE` warning.
- Passages are returned as supplied by Exa highlights, without an LLM summary.
  This does not independently prove the provider copied every passage accurately,
  that its indexed page is current, or that the institution approves a particular
  interpretation. Publication dates are not document versions.
- Default allowed domains: NHS, NICE and AAOS OrthoInfo. An allowed host is NOT
  evidence of clinic approval. Results outside configured domains are filtered.
- No case context or record text is automatically attached to web requests.
  The model is instructed to ask generic clinical questions and not send patient
  identifiers/private reports. This prompt is not a production PHI-loss-prevention
  system; the demo is restricted to synthetic patients.
- The assistant may summarize retrieved evidence with its limitations. The
  search tool does not generate medical advice, and web passages are not reviewed
  case evidence or a patient-specific treatment decision.

## Voice scenarios and current boundaries

| Scenario | Native support | Current profile / smallest next step |
|---|---|---|
| Continue talking while lookup runs | Native async calls and deferred result messages. | Enabled for clinician reads/search with a short processing notice and a session-local delivery gate. Original work survives speech interruptions; source/LLM/playback behavior still requires live retesting. |
| Correct January to February while retaining March | Per-call cancellation exists. | Ordinary speech no longer cancels clinician reads. The LLM can explicitly cancel individual native calls, but semantic supersession/retaining the correct sibling is not a deterministic guarantee. Case switches cancel all outstanding reads before context reset. |
| Interrupt a multi-step answer and resume at the next unheard instruction | Interruption frames and assistant-context tracking exist. | No played-instruction ledger. Generated/TTS-sent text is not proof of browser playback or hearing. Prompt asks where to resume when uncertain. Exact automatic continuation needs client playback progress aligned with instruction boundaries; long full-reply TTS is not sufficient. |
| Ignore “mm-hmm”/coughs but obey “stop” | VAD and turn strategies detect activity, not its clinical/conversational meaning. | No reliable acknowledgement/cough classifier is enabled. Word-count thresholds can wrongly ignore “stop”. A semantic/acoustic decision stage would add complexity and latency; no guaranteed solution is claimed. |
| Allow hesitation and thinking pauses | VAD + Smart Turn plus native `FilterIncompleteUserTurnStrategies` / `UserTurnCompletionLLMServiceMixin`. | Enabled for the clinician and offline-tested with scripted LLM verdicts. Live speech/LLM classification quality remains unverified; see the incomplete-turn section above. |

Cancellation suppresses a tool's awaited completion; a bounded synchronous
DB/HTTP operation already running in a worker thread may still finish. It does
not promise provider-side cancellation or avoided billing. Parallel work is not
by itself safe background-result delivery.

These findings come from the installed Pipecat 1.11.0 implementation. No new
semantic interruption model, background job framework, instruction state machine
or speculative parameter search was added. Background support uses native tool
tasks plus one notice/delivery gate; the native incomplete-turn watchdog is unchanged.

## Verification boundary

- Async delivery update: 568 bounded offline regression tests passed. New tests
  cover the one-shot delay, exact processing notice, native async overlap with an
  intervening question, one deferred continuation after output drain, native
  cancellation-tool advertisement/execution, deadline and case-reset hooks,
  interrupted/error delivery, and real Pipecat TTS plus `BaseOutputTransport`
  completion with mocked SDKs and synthetic in-memory PCM. Existing front-desk,
  retry/fallback, incomplete-turn, provider and session tests remain green.
  This is not live Exa, real-LLM compliance, browser playback or voice-quality proof.
- Batch/history update: 58 focused offline tests passed, including a mocked native
  OpenAI stream with a fast error and slow sibling, one continuation after both
  results, edge-first/edge-last transitions, unrelated groups during a pending
  transition, unchanged native-manager behavior and redacted diagnostics.
  Ten contract examples validated. Read-only runtime checks on cases 1051–1056
  verified discovery, current/historical separation, cross-case rejection and
  missing score/summary preservation. These checks made no provider calls or DB
  writes. The updated live model behavior still requires restart and retesting.

- Earlier identity/privacy check: 44 offline tests passed, including removal of DOB from
  the case-selection query/schema/payload and privacy/date/dependency instruction
  wiring. These do not verify live model compliance or pronunciation. They also
  cover startup without a case, exact selection, duplicate-name candidates, failed switches,
  stale read/selection rejection, native same-node context replacement, recoverable
  LLM errors and incomplete-turn integration. The existing synthetic database
  exposes 31 authorized cases; read-only checks verified the guide's principal
  missing/pending/draft/historical/conflict cases. New multi-case voice behavior
  has not been live-tested. No new migrations or seeds were applied.
- The [50-query guide](clinician-test-queries.md) is a manual coverage plan, not a
  measured 50-case voice evaluation. A frozen gold-versus-ASR benchmark, p50/p95
  browser-audio measurements and an agreed API budget remain outstanding. Exa is
  the agreed replacement for versioned-document retrieval, with source attribution
  and no invented versions or clinic-approval claims.

Earlier checks:

- 29 focused tests passed: clinical reads, Exa request/response mocks, required
  nullable arguments, case boundaries, native tool handlers, secret exclusion,
  contract examples and front-desk profile isolation.
- Actual read-only PostgreSQL calls succeeded for all four clinical tools on
  case 1042. Case mismatch and an inaccessible/nonexistent model result were
  rejected. No database writes were performed.
- Earlier in the work a broader unit run reported 530 passes and three
  booking/schema failures outside this clinician-focused check. This is NOT a
  claim that the repository-wide suite is clean.
- The Exa key is now configured when loading the current environment. The latest
  inspected run (`20260923T033358Z-d6cfb0b8`) still recorded no Exa key and made no
  guidance search call. Restart to load the key. Live Exa execution remains
  unverified; construction/parsing/filtering/error handling are mocked-tested.
- No demo restart, paid provider request, model download, migration or install
  was performed by the implementation checks. User-run traces show successful
  clinical reads and thinking-pause/re-prompt behavior, but premature tool use on
  “Compare the model result with…” remains a live limitation. The latest run had
  no service error, so it did not exercise recovery. New multi-case voice behavior
  and browser playback remain unverified.

Code map: `clinician/flow.py` (node/prompt), `clinician/runtime.py` (envelope and
versioned session context), `clinician/case_selection.py` (selection contract),
`clinician/records.py` (read-only queries),
`clinician/web_search.py` (Exa), `voice/clinician.py` (native flow adapter).
