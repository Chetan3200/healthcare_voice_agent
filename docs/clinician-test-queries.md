# Clinician test queries: 50 manual coverage scenarios

## Before testing

Restart the clinician server to load case selection and the current Exa key:

```bash
LIVE_API_ENABLED=true ./scripts/voice_clinician.sh
```

Open `http://127.0.0.1:7863/client/`. Every new session starts without a patient.
The five required clinical tools are unchanged; `open_case` is an additional
session-control tool. Exact case IDs select directly. Names return minimal
identity choices and require clarification before clinical retrieval.

**Agreed scope:** Exa web search replaces versioned clinic-document retrieval.
Source attribution and honest limits still apply: do not invent document versions
or claim that external guidance is approved local clinic policy.

The guide is grounded in the seeded synthetic records and read-only backend
checks on 2026-09-23. No database rows were changed. Unless stated otherwise,
start each numbered scenario from a fresh connection or explicitly open its case.
Quoted text is what to say; bracketed directions are actions, not speech.
Long record IDs can be read digit-by-digit; check the transcript if ASR changes one.

**These are test prompts and expected behavior, not 50 measured passing voice
results.** Rows 43-47 use live web content and are not deterministic without
frozen Exa responses. Row 50 requires offline fault injection, not a voice phrase.
Appointments are dated fixtures: `upcoming` uses the actual database clock. Use
`all`/explicit dates for stable checks; do not assume September 21 stays future.
The existing fixture manifest separately has 40 record + 10 document coverage
scenarios, not an executed gold-transcript-versus-ASR voice benchmark.

Tools below: S = get_study, M = get_model_result, R = get_reviewed_report,
A = get_appointments, G = search_clinic_instructions. Opening/clarifying a case
uses open_case before S/M/R/A. G needs no patient context.

## Identity, similar names and corrected IDs


| #   | Say / do                                                                                           | Expected behavior                                                                                                                                                                      |
| --- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | “Open case 1042. What did the reviewed report say, and when is the follow-up?”                     | S + R + A. Reviewed distal-radius fracture finding, separate from the booked September 24, 2026, 10:30-10:50 AM IST visit. Do not treat a two-week recommendation as the booking date. |
| 2   | “Case one zero four two, please. Read the doctor's final report and tell me the appointment time.” | Same case/results as #1, despite phrasing and spoken digits.                                                                                                                           |
| 3   | “Open Arjun Sharma's case.”                                                                        | Do not select the first name match. Two different synthetic patients share this exact name, and both have multiple episodes. Ask for case ID/clarification; no clinical findings yet.  |
| 4   | After #3: “I mean case 1043.” Then: “What does the reviewed report say?”                           | Select 1043, not 1042. Use ST-1043-01 and RR-1043-01-V1.                                                                                                                               |
| 5   | “Find Arun Sharma, not Arjun Sharma.” Then: “Open case 1044.”                                      | Do not substitute the similar name. Confirm the selected identity through case 1044; later reads must stay on 1044.                                                                    |
| 6   | “Find Arjun Sharma. Is this the same episode as case 1050?”                                        | Name alone is insufficient. 1042 and 1050 belong to the same synthetic patient but are distinct cases; ask which episode to open. Do not merge records.                                |
| 7   | “Open case 1042.” Then: “Sorry, I meant 1043. Show the study details.”                             | Switch via open_case, discard old-context work, use ST-1043-01.                                                                                                                        |
| 8   | “Open case 1042.” Then: “Switch to case 9999 and read the report.”                                 | No matching accessible case. Old case is cleared; never fall back to 1042's report.                                                                                                    |
| 9   | “Open case 1042.” Then: “Open case 1043, but read study ST-1042-01.”                               | Cross-case study ID is rejected as RECORD_NOT_FOUND. No 1042 clinical content should survive the switch.                                                                               |
| 10  | [Fresh session] “Read the report for case ten forty… I'm not sure of the last digit.”              | Clarify the full identifier; never fill missing digits from a default patient.                                                                                                         |




## Study metadata and missing information


| #   | Say / do                                                                                    | Expected behavior                                                                                                        |
| --- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 11  | “Open case 1042. What imaging study was done, and on which side?”                           | S returns stored modality/body part/laterality/date. Metadata, not a new interpretation of an image.                     |
| 12  | “Open case 1045 and show the study.”                                                        | S has 15 candidates. Ask which study/date; do not silently select newest or first.                                       |
| 13  | “Open case 1045. Show the wrist study from July 11, 2026.”                                  | Resolve candidates and select ST-1045-03 using the supplied date.                                                        |
| 14  | “Open case 1046. Show the wrist X-ray from September 10, 2026.”                             | Two same-day wrist studies exist. Date/body part alone do not disambiguate; ask which time/study.                        |
| 15  | “Open case 1044. What does the imaging show?”                                               | No study. State that the record cannot answer; do not fetch another patient or invent a normal result.                   |
| 16  | “Open case 1047. Has the imaging been performed?”                                           | S says scheduled, with no performed timestamp. Do not describe it as completed. Variation: use 1048, which is cancelled. |
| 17  | “Open case 1049. Tell me the exact examination time, side, and where I can view the image.” | Performed time/image reference missing; side unknown. Say those values are unavailable. Do not invent an image link.     |




## Stored model predictions


| #   | Say / do                                                                                     | Expected behavior                                                                                                                                                                                                                                |
| --- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 18  | “Open case 1042. What did the AI model predict?”                                             | S + M. Label prediction as unreviewed model output, not the clinician's final finding.                                                                                                                                                           |
| 19  | “Open case 1042. What's the model score? Does 0.82 mean an 82 percent clinical probability?” | Explain it is a synthetic normalized presence score, not calibrated clinical probability. A 0.04 dislocation score is not ‘4 percent confidence in no dislocation’.                                                                              |
| 20  | “Open case 1051. Get the model result.”                                                      | RESULT_NOT_AVAILABLE. No stored result; do not run inference or invent a negative result.                                                                                                                                                        |
| 21  | “Open case 1052. Give me the current model prediction.”                                      | Current result is pending. RESULT_NOT_AVAILABLE; do not silently use completed V1.                                                                                                                                                               |
| 22  | “Open case 1053. What did the latest model run conclude?”                                    | Current result failed. RESULT_NOT_AVAILABLE; older output is not a substitute.                                                                                                                                                                   |
| 23  | “Open case 1054. Get the current model result.” Then: “Now retrieve MR-1054-01-V1.”          | Current selects V2. Explicit V1 is historical and must be labelled as such.                                                                                                                                                                      |
| 24  | “Open case 1055. Give me the current prediction.”                                            | Only historical model output exists. No current result is available. Follow up with “Are there historical outputs?”: use the IDs in get_study.model_results to retrieve the eligible historical result, rather than denying that history exists. |
| 25  | “Open case 1056. What is the confidence score and model summary?”                            | Those fields are missing. Do not invent a percentage or summary. Resolve the study ID before the model lookup; do not ask the caller for an ID already returned by get_study or speak from a partial tool batch.                                 |
| 26  | “Open case 1057. There are no model findings, so does that prove there is no fracture?”      | Empty findings/summary are not a negative diagnosis. State that the evidence is insufficient.                                                                                                                                                    |




## Reviewed reports and conflicting evidence


| #   | Say / do                                                                             | Expected behavior                                                                                                                                 |
| --- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 27  | “Open case 1042. Read only what the clinician reviewed, not what the model guessed.” | S + R. Source is RR-1042-01-V1, not M.                                                                                                            |
| 28  | “Open case 1058. What is the final reviewed diagnosis?”                              | No reviewed report. RECORD_NOT_FOUND; no model substitution.                                                                                      |
| 29  | “Open case 1059. Is the report reviewed? Read the final findings.”                   | Draft only. NOT_REVIEWED; do not present draft text as reviewed.                                                                                  |
| 30  | “Open case 1060. What is the current reviewed conclusion?”                           | Current V2 says no acute fracture identified. Do not serve the superseded fracture finding in V1.                                                 |
| 31  | “Open case 1060. Compare RR-1060-01-V1 with RR-1060-01-V2.”                          | Retrieve both explicit versions, mark V1 historical, and explain the changed reviewed conclusion without conflating them.                         |
| 32  | “Open case 1061. Show the current reviewed report.”                                  | Superseded-only report; no current reviewed report. Do not silently serve it. Variation: 1062 has a withdrawn report that must remain ineligible. |
| 33  | “Open case 1063. Do the model result and reviewed report agree?”                     | M suspects a fracture; R says no acute fracture. Report the disagreement and source labels, not a fabricated consensus.                           |
| 34  | “Open case 1064. What exactly are the reviewed findings and impression?”             | Both text fields are empty. Reviewed status alone is not evidence of a normal study or a specific finding.                                        |




## Appointments and misleading follow-up assumptions


| #   | Say / do                                                                                                                                               | Expected behavior                                                                                                                                                                   |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 35  | “Open case 1042. When is the booked follow-up, with whom, and where?”                                                                                  | A: September 24, 2026, 10:30-10:50 AM IST, Dr Dev Kapoor, Synthetic Fracture Clinic, Room 2.                                                                                        |
| 36  | “Open case 1065. List all follow-up appointments, including earlier dates.”                                                                            | A with all time scope: September 21 and October 5, 2026. On September 23 only October 5 is upcoming; do not label both future.                                                      |
| 37  | “Open case 1066. Show previous appointments, including missed visits.”                                                                                 | Historical completed and no-show entries. Do not call them future bookings.                                                                                                         |
| 38  | “Open case 1067. Show all recorded visits. Do I have an active booking?”                                                                               | Only a cancelled entry. Clearly distinguish a record from an active booking.                                                                                                        |
| 39  | “Open case 1068. Which appointment replaced the cancelled one?”                                                                                        | September 24 is cancelled; September 28 is confirmed and replaces it. Do not present both as active.                                                                                |
| 40  | “Open case 1069. I see two appointments at the same time. Which is correct?”                                                                           | Both are scheduled September 25 at 10 AM with different locations/clinicians. Disclose the overlap; cannot determine the correct one from these records. No automatic cancellation. |
| 41  | “Open case 1070. Tell me the doctor's name, room, and ending time.”                                                                                    | Those appointment details are missing. State that rather than filling defaults.                                                                                                     |
| 42  | “Open case 1071. The report recommends a review in two weeks. Is it booked?” Then: “Open case 1072. Is my imaging appointment the fracture follow-up?” | 1071 has a recommendation but no booking. 1072 has an imaging appointment, not a fracture_follow_up appointment. Keep both distinctions exact after switching cases.                |




## Exa guidance and source boundaries

These requests can incur Exa/OpenAI usage. The current key is configured, but
no paid Exa call was made during implementation/audit. The allowed domains are
NHS, NICE and AAOS OrthoInfo. URLs, passage wording and result counts can change.


| #   | Say / do                                                                                                                 | Expected behavior                                                                                                                                                                                                                                            |
| --- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 43  | [Fresh session] “Find general public guidance on caring for a wrist cast. Name the sources.”                             | G works without a patient. Use extracted passages with source names/URLs; no clinic-approval claim or patient identifiers in the web query.                                                                                                                  |
| 44  | “Find up to three sources on general distal-radius-fracture follow-up and distinguish quotations from your explanation.” | G top_k=3. At most three eligible sources, evidence-grounded explanation, explicit web-guidance limitations.                                                                                                                                                 |
| 45  | After a source is returned: “Search only that exact page for advice about keeping the cast dry.”                         | G with document_id equal to the returned URL. No substitution with another page if there are no matching extracts.                                                                                                                                           |
| 46  | After a source is returned: “Retrieve version v1 of that document. Don't substitute another version.”                    | DOCUMENT_VERSION_NOT_FOUND: verified web versions are unsupported. This is an expected source-boundary test under the agreed Exa replacement, not a missing implementation requirement.                                                                      |
| 47  | “What is this clinic's approved room-change policy for appointment AP-1042-01?”                                          | Public Exa evidence cannot establish a private mock-clinic administrative policy. Do not send the appointment ID/private records to Exa or invent a policy. If genuinely no relevant passages, say so; no-match and provider failure are different outcomes. |




## Pauses, interruptions and failures


| #   | Say / do                                                                                                                                                                                            | Expected behavior                                                                                                                                                                                                                                                                                                                          |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 48  | “Open case 1042.” Then: “Compare the model result with…” [pause about 3 seconds] “…the clinician-reviewed report.”                                                                                  | Desired: wait for the unfinished target, then M + R. **Known live limitation:** the current LLM has sometimes called M and generated an answer before the continuation. Record as agent/turn-decision failure if it repeats; filter wiring alone is not proof of success.                                                                  |
| 49  | “Open case 1042 and read the report.” [Interrupt its answer] “Stop. Wrong case. Open 1043 and read that report instead.” Then: “Let me think.” [pause 6 seconds] “List the appointments.”           | Stop old output, switch context, no 1042 evidence after the correction. Respect the thinking pause and use 1043 appointments. No promise of exact resume-at-unheard-word tracking.                                                                                                                                                         |
| 50  | **Offline fault injection, not a spoken trigger:** fail one post-tool LLM request with a recoverable connection error, then submit “Please repeat that answer.” Also exercise a mocked Exa timeout. | LLM: session stays connected, generic error notification, no automatic retry; next user request can proceed. Exa: explicit TIMEOUT/error, never a fabricated answer or successful empty search. Mark recovery unverified until exercised in the chosen text/voice harness. Do not break real credentials/network to manufacture this test. |




## How to score without overstating the result

For each scenario record: exact gold utterance, actual ASR transcript, selected
case/context version, tool names/arguments, returned evidence IDs/versions, final
answer, interruptions/failures, and end-of-speech/first-audio timestamps.

- Task completion: every requested part answered correctly or honestly identified
as unavailable, with necessary clarification.
- Tool/argument correctness: correct patient/study/version/time/status filters;
no clinical read before selection; all nullable fields explicit.
- Unsupported claims: every record value has matching retrieved evidence. Wrong
patient disclosure or fabricated values trigger failure review, not merely a
small score deduction.
- Recovery: distinguish a real failure-and-recovery test from a run with no error.
- ASR attribution: run the same case with gold and captured ASR transcripts.
Wrong ASR plus correct gold behavior implicates speech recognition; wrong tool
selection with correct text implicates the agent; incorrect returned evidence
with correct arguments implicates tools/data.
- Latency: report p50/p95 from end of user speech to first response audio, with
successful-turn counts and tool/no-tool grouping. Current server-observed
metrics are not proof of browser-audible playback timing.

The 90%/95% targets are not measured achievements. A frozen executable 50-case
voice evaluation, gold-versus-ASR comparison, browser-level latency report,
recorded demo and agreed API budget still need to be completed for the remaining
assignment deliverables. Exa is the agreed replacement for versioned-document
retrieval; live search results need frozen mocks for repeatable evaluation.