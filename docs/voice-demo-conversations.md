# Voice demo conversations
A bounded script for the synthetic fracture clinic. It covers every available
tool, selected edge cases, natural delays, and interruptions. **Caller lines are
scripted user speech.** Replace bracketed placeholders with actual returned names or
times. Presenter cues and expected behavior are not speech and are not a
guarantee of verbatim model replies. “Agent example” lines are illustrative,
spoken only after matching successful tool results, not claimed live outputs.
## Facts and guardrails
- Checked on 23 September 2026, Asia/Kolkata; the clock is not frozen. Front desk: port `7862`, isolated
  DB port `55432`, revision `005`. Clinician: port `7863`.
- Front desk is fixed to case 1042, Arjun Sharma. Pre-existing `AP-1042-01` is a
  24 September 2026 10:30 AM fracture follow-up: never move or cancel it.
- Synthetic doctors: Dr Mira Desai (`SYN-CLIN-01`), Dr Dev Kapoor
  (`SYN-CLIN-02`), Dr Leena Rao (`SYN-CLIN-03`).
- Schedule was read-only checked as enabled through 6 October: weekdays 9 AM to
  5 PM, 30-minute slots, five types. Only live `search_availability` establishes
  availability.
- Create one new imaging booking with Dr Dev Kapoor. Move and cancel only that
  booking. Keep its live appointment ID/version private.
- Acknowledgement handling is unchanged: “mm-hmm” or “okay” can still interrupt
  speech. Do not expect automatic continuation or exact playback resume.
## Mandatory tool coverage
| Agent | Tool | Script step |
| --- | --- | --- |
| Front desk | `clinic_information` | FD-1 |
| Front desk | `search_availability` | FD-2, FD-4 |
| Front desk | `book_slot` | FD-3 |
| Front desk | `list_appointments` | FD-3, FD-5 |
| Front desk | `reschedule_appointment` | FD-4 |
| Front desk | `cancel_appointments` | FD-5 |
| Front desk | `operation_status` | FD-6 |
| Clinician | `open_case` | CL-1, CL-4 |
| Clinician | `get_study` | CL-2 |
| Clinician | `get_model_result` | CL-2, CL-5 |
| Clinician | `get_reviewed_report` | CL-2, CL-5 |
| Clinician | `get_appointments` | CL-3, CL-6 |
| Clinician | `search_clinic_instructions` | CL-7 |
## Run of show
1. Manually start front desk with
   `LIVE_API_ENABLED=true VOICE_MAX_SESSION_SECONDS=3600 ./scripts/voice_frontdesk_demo.sh`.
   The one-hour override is for this demo; its normal maximum is 10 minutes and
   idle timeout remains 120 seconds. These launch commands were not run here.
2. Run FD-1 through FD-6; capture only the new imaging booking’s live ID/version.
3. Manually start clinician with `LIVE_API_ENABLED=true ./scripts/voice_clinician.sh`.
   Its default maximum is one hour and idle timeout is 120 seconds. Use a fresh
   connection for CL-1. Check that the published slot range still covers your
   demo date; do not assume September fixtures remain upcoming indefinitely.
4. Run CL-1 through CL-7, then at most two optional branches.
## Front desk
### FD-1: directory, not availability
**Caller:** “What appointments and doctors can this clinic help with?”
**Expected behavior:** Call `clinic_information`; briefly name services and the
clinician directory, without claiming a date/time is open.
### FD-2: live imaging offers
**Caller:** “I need an imaging appointment with Dr Dev Kapoor on Monday morning next week. Give me two options.”
**Expected behavior:** Call `search_availability` for imaging, the resolved dates,
and `SYN-CLIN-02`. Offer only returned slots; if none, describe only that search
window and ask for a wider range or changed preference.
**Presenter cue:** Choose one returned human-readable time. Keep its `slot_id`
private.
**Agent example:** “I have [first returned time] or [second returned time]. Would either work?”
### FD-3: book the offered slot, then verify listing
**Caller:** “Book the [offered day] at [offered time] with Dr Dev Kapoor.”
**Expected behavior:** Call `book_slot` with the exact returned `slot_id`; report
success only after commitment, without a second confirmation request.
**Agent example:** “Your imaging appointment is booked for [confirmed date and time] with Dr Dev Kapoor.”
**Caller:** “Please list my appointments.”
**Expected behavior:** Call `list_appointments`; distinguish the new imaging
booking from `AP-1042-01`. Retain only the new booking’s ID/version for FD-4/5.
### FD-4: atomic move of the demo booking
**Caller:** “Move the imaging appointment we just booked to a later time on the same day with Dr Dev Kapoor.”
**Expected behavior:** Call `search_availability` for live matching imaging slots.
After the caller selects one, call `reschedule_appointment` with the new imaging
booking’s ID/version and the replacement `slot_id`. Do not separately cancel it
or alter `AP-1042-01`.
**Agent example:** “I’ve moved that imaging appointment to [confirmed new time]. I haven’t changed the fracture follow-up.”
**Presenter cue:** If a choice turn is needed, say: “The later one, please.”
### FD-5: cancel only the moved imaging booking
**Caller:** “Cancel the imaging appointment we just moved, but keep my fracture follow-up.”
**Expected behavior:** If needed call `list_appointments`; then call
`cancel_appointments` with only the current explicit pair for that imaging
booking. Report commitment, never cancellation of the fracture follow-up.
**Agent example:** “I’ve cancelled only the imaging appointment we just moved.”
### FD-6: status is not a history proof
**Caller:** “Please check the backend status of that last operation.”
**Expected behavior:** Call `operation_status`. After a completed success, it
returns `operation_state: none` and `last_result`. Say that limited result, not
that full history or lost-acknowledgement recovery was proven.
## Clinician
### CL-1: ambiguous name, selected ID
**Caller:** “Open Arjun Sharma’s case.”
**Expected behavior:** Call `open_case` as a name lookup. Arjun Sharma is
ambiguous; offer minimal identity choices and ask for a case ID, with no findings.
**Agent example:** “I found more than one matching case. Which case ID do you mean?”
**Caller:** “Case 1042.”
**Expected behavior:** Call `open_case` with the exact ID. Clinical reads wait
for a selected-case result.
### CL-2: metadata, model, and review are different evidence
**Caller:** “What imaging was done, what did the model predict, and what did the clinician review?”
**Expected behavior:** Call `get_study`, then use its eligible IDs for
`get_model_result` and `get_reviewed_report`. Baseline 1042’s reviewed report is
a left nondisplaced distal-radius fracture. Label model output unreviewed: fracture
presence 0.82 and dislocation 0.04 are synthetic normalized scores, not calibrated
clinical probabilities; 0.04 is not confidence of no dislocation.
### CL-3: appointment evidence
**Caller:** “When is the booked follow-up, with whom, and where?”
**Expected behavior:** Call `get_appointments`. State the fixture’s 24 September
2026, 10:30-10:50 AM IST fracture follow-up with Dr Dev Kapoor, Synthetic Fracture
Clinic, Room 2. Do not offer write actions; those go to front desk.
### CL-4: switch discards old context
**Caller:** “Sorry, I meant case 1043. Show the study details.”
**Expected behavior:** `open_case` 1043, clearing 1042, then `get_study` for 1043.
No 1042 clinical evidence may be reused.
### CL-5: disagreement
**Caller:** “Open case 1063. Do the model result and reviewed report agree?”
**Expected behavior:** Select with `open_case`, then retrieve study, model, and
report. State the disagreement: model suspects fracture; reviewed report says no
acute fracture. Do not invent consensus.
**Agent example:** “They disagree. The model flags a suspected fracture, while the reviewed report says no acute fracture. I’m keeping those two findings separate.”
### CL-6: recommendation is not booking
**Caller:** “Open case 1071. The report recommends a review in two weeks. Is it booked?”
**Expected behavior:** Select 1071; use reviewed-report evidence and
`get_appointments`. Explain that a recommendation is not a booking and this
scenario has none.
**Agent example:** “The report recommends a review, but I couldn’t find a booked appointment in this case’s records.”
### CL-7: public guidance boundary
**Caller:** “Find general public guidance on caring for a wrist cast. Name the sources.”
**Expected behavior:** Call `search_clinic_instructions` with a generic query;
no case is required and no patient identifiers are sent. Use only returned passages; identify them as public external guidance, not verified
clinic policy or patient-specific treatment. Surface an explicit unavailable/key
error rather than inventing sources.
**Caller:** “Use only the [returned source name] source you just mentioned. Read its relevant sentence exactly, then explain it simply.”
**Expected behavior:** Reuse the returned passage where sufficient. If another
lookup is needed, use the exact returned URL as `document_id`, a generic query,
and `version=null`. Do not call `open_case`. Keep quotation separate from explanation.
## Delays and interruptions
### Natural delay
**Presenter cue:** After FD-2 or CL-7, allow the lookup and full-reply TTS
buffering pause to be heard before judging latency. For a reproducible long lookup,
start the clinician server manually with
`CLINICIAN_DEMO_WEB_SEARCH_DELAY_SECONDS=15 LIVE_API_ENABLED=true ./scripts/voice_clinician.sh`.
Only the first valid web search per connection is artificially delayed. This is
not provider latency or a lost-write-acknowledgement simulation. Offline tests are
not live voice proof.

### Continue talking during a slow web search
**Setup:** Open a synthetic case first if the intervening question will use its records.
**Caller:** “Find public guidance on caring for a wrist cast.”
**Expected behavior after about three seconds of tool execution, if the conversation is free:**
“Can I help you with anything else while the request is being processed?”
**Caller:** “What imaging was done for this case?”
**Expected behavior:** Retrieve and answer the study question while the same web
search continues. Once the search is complete, finish any current answer/audio,
then briefly bring back its actual findings. No invented sources or clinic-policy
claims. The delay is asynchronous, and subsequent searches are not delayed.
**Variants:** Stay quiet and expect proactive result delivery; ask another question
while results arrive; explicitly cancel the search; switch case; disconnect.
No result should cut into active speech or be delivered after cancellation/reset.
The timer triggers the notice, not guaranteed browser audio at exactly three seconds.
### Read-only correction during interruption
**Caller:** “Open case 1042 and tell me the report.”
**Presenter cue:** Interrupt during lookup or answer.
**Caller:** “Stop. Wrong case. Open 1043 and read that report instead.”
**Expected behavior:** Stop old output; `open_case` 1043, then read only 1043.
Ordinary speech stops output but leaves native async reads running. The explicit
case switch cancels outstanding old reads and discards queued results before reset.
Do not promise an exact audio resume point.
### Interrupting a front-desk write
**Presenter cue:** Run this as an optional repeat after FD-6, first requesting a
fresh imaging offer. It may create another booking; reconcile and cancel only
that new booking afterwards. Never use the pre-existing fracture follow-up.
**Caller:** “Book the [freshly offered imaging time].”
**Presenter cue:** Immediately interrupt: “Stop.” This does not prove cancellation:
front-desk writes use `cancel_on_interruption=False` and shielded transactions can
finish.
**Caller:** “Did that booking go through? Check the operation status and list my imaging appointments. Don’t try booking again.”
**Expected behavior:** Use `operation_status` and `list_appointments` before a new
write. `none` alone is not proof that the interrupted booking failed, and
`last_result` might refer to an earlier operation if this request never started.
## Optional branches
| Caller line | Expected behavior |
| --- | --- |
| Front desk: “Are there imaging appointments after 6 PM on that same Monday?” | Search the explicit narrowed window. Explain only that no matching offers were found; do not claim all dates are full. |
| Front desk: “Before I decide whether to cancel, tell me what appointments I have. Don’t cancel anything.” | List appointments only. Discussion is not authorization to write. |
| Clinician, after CL-6: “Book that follow-up for next week.” | Explain the read-only boundary and direct booking to the front-desk assistant. Do not claim a booking or change records. |
| “Open case 1049. Tell me the exact examination time, side, and where I can view the image.” | `get_study`; explicitly state missing time, side, and image reference. |
| “Open case 1055. Give me the current prediction. Are there historical outputs?” | Do not substitute historical output for current; use study-discovered eligible IDs only for the history question. |
| “Open case 1070. Tell me the doctor’s name, room, and ending time.” | `get_appointments`; explicitly state missing clinician, location, and end time. |
| “Open case 1072. Is my imaging appointment the fracture follow-up?” | `get_appointments`; distinguish appointment types exactly. |
## Sources and verification boundaries
Grounded in `docs/frontdesk-demo.md`, `docs/clinician-demo.md`,
`docs/clinician-test-queries.md`, `demo/flow.py`, `demo/tools.py`, and launcher
scripts. The conversations have not been executed as a live voice demo. Doctor
names were separately updated and verified in both synthetic databases; no
clinical evidence, appointments, slots, schema, or acknowledgement behavior was
changed. No provider calls or server restarts were made for this preparation.
This script is not measured proof of live ASR, model choice, browser playback,
availability, or fault recovery.
Record actual transcript, tool results, and interruption outcomes during the demo.
