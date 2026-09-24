# Synthetic records for the mock fracture clinic

`seed.py` loads fixed, inspectable JSON fixtures into the existing Alembic-001
baseline tables. The fixture format remains revision `001`; loading is allowed
on database revisions `001` and additive booking revision `002`. It does not seed
booking schedules or drafts. Verify fixture scenario counts before creating
additional appointments, since new bookings can legitimately change those counts.
No LLM, imaging model, ASR, TTS or embedding API is called. All people,
clinical statements and model scores are fictional software-test data, not
medical advice, real patient records or measured model results.

## Run from the repository root

```bash
uv run python scripts/seed.py --dry-run
uv run python scripts/seed.py --apply
uv run python scripts/seed.py --verify-db
```

Only `--apply` inserts rows. See [README](../README.md) for setup.

## What is loaded

| Table | Rows |
|---|---:|
| patients | 12 |
| clinicians | 3 |
| app_users | 3 |
| cases | 31 |
| studies | 45 |
| model_results | 45 |
| model_findings | 43 |
| reviewed_reports | 43 |
| appointments | 14 |
| clinic_documents | 9 |
| document_versions | 10 |
| document_chunks | 30 |
| retrieval_indexes | 0 |
| chunk_embeddings | 0 |

Total: **288 valid rows**. The model-result count includes pending/failed/history
records; the report count includes drafts, superseded and withdrawn versions.
Not every row is eligible for a normal current-result tool response.

There are 40 record-coverage scenarios, 10 document-coverage scenarios, and 20
separate invalid-insert candidates. The 50 record/document entries are NOT the
assignment's completed 50-conversation voice evaluation suite. Tool behavior,
ASR quality, retrieval ranking, speech interruptions and end-to-end success are
not measured by these fixture tests.

## Useful cases to inspect in Antares

- `1042`: normal left-wrist study, model result, reviewed report and booked follow-up.
- `1042` / `1043`: similar case IDs; different patients with the same name.
- `1042` / `1050`: different injury episodes belonging to the same patient.
- `1044`: no studies or appointments.
- `1045`: fifteen studies.
- `1046`: two same-day wrist X-rays distinguishable by examination time.
- `1047` / `1048`: scheduled and cancelled examinations.
- `1049`: missing examination time and unknown side.
- `1051`: no model result.
- `1052` / `1053`: current pending/failed prediction with older completed output.
- `1054` / `1055`: current/history model versions, or historical-only output.
- `1056` / `1057`: missing model score/summary, or an empty completed result.
- `1058` / `1059`: missing reviewed report, or draft-only report.
- `1060`: current reviewed version changes the older conclusion.
- `1061` / `1062`: superseded-only or withdrawn-only reviewed report.
- `1063`: model predicts a fracture while the reviewed report says no acute fracture.
- `1064`: reviewed report with empty text sections.
- `1065`: two upcoming active follow-ups.
- `1066` / `1067`: past-only or cancelled-only appointments.
- `1068`: cancelled original and active replacement booking.
- `1069`: overlapping active follow-ups at different locations.
- `1070`: booking with missing optional details.
- `1071`: a follow-up recommendation but no booked appointment.
- `1072`: imaging is booked, but no fracture-clinic follow-up is booked.

Mock session users:
- `SYN-USER-CLIN`: active clinician, all seeded cases accessible.
- `SYN-USER-STAFF`: active staff, the same clinic-wide access.
- `SYN-USER-INACTIVE`: inactive account, no allowed cases in the helper view.

Documents include cast-care history/current versions, visit preparation and clinic
hours, expired/draft/withdrawn/future-effective versions, and deliberately
conflicting arrival-time guidance in two current document families. The conflicting
content concerns appointment logistics, not treatment instructions.

## Safe reruns and failure behavior

- All files have fixed IDs, values and dates. No randomness or current-clock data
  generation is used.
- The manifest fingerprints each fixture file with SHA-256. Unexpected file edits
  stop validation before database access.
- `--apply` runs in one PostgreSQL transaction. Missing rows are inserted; identical
  existing rows are skipped. A different existing value or other conflict aborts
  the operation rather than overwriting or deleting anything.
- Record-specific count checks protect intentional absences and ambiguities. For
  example, adding an extra study to case 1044 changes its no-study scenario and
  causes verification to fail. Unrelated extra records are not automatically deleted.
- If a connection is lost near commit, use `--verify-db` to establish the actual
  database state before rerunning. An error alone does not prove whether commit
  reached the server.
- There is no destructive reset or overwrite option.

Treat this fixture version as frozen for evaluation. If you deliberately change
fixture content, update its dataset version/manifest and associated expectations
as a reviewed change; do not edit the manifest merely to bypass an unexpected
checksum failure.

## Separate invalid-record checks

`fixtures/invalid_records.json` describes copies of normal rows with deliberately
invalid changes: nonexistent parents, cross-case links, duplicate current records,
missing required values, bad statuses/scores/timezones, and similar violations.
These candidates are never included in `--apply`.

After loading and verifying the normal records, you may run:

```bash
uv run python scripts/seed.py --check-invalid-db
```

This attempts each candidate inside a savepoint, expects a specific PostgreSQL
error code, and rolls back EVERY attempt, including unexpected successful inserts.
It then rolls back the whole check transaction. No invalid record is retained.
It requires the normal fixture baseline to be present and unchanged.

The checks establish that the database rejects corrupted data. They do not prove
that an agent handles a deliberately corrupted backend response correctly; that
is a later tool/agent test.

## Repeatable time and embeddings

The fixture reference time is **2026-09-17T09:00:00Z**. Scenario checks use this fixed
instant for past/upcoming and document-effective-date checks. The script does NOT
change your system clock or PostgreSQL clock. The later evaluation runner must
inject this same reference time into tools for repeatable tests. A live demo using
the actual clock will naturally classify dated appointments differently over time.

Document text and passages are loaded now, but embedding/index rows are empty.
Choose the embedding model, dimensions and API budget before generating vectors.
Metadata checks and verbatim-passage validation are not a semantic-search evaluation.

## Validation performed and tests

```bash
uv run pytest
```

The included 24 tests cover offline checksums, types, references, current-version
uniqueness, dates, scores, source passages, record boundaries and mocked
insert/skip/rollback behavior. They do not connect to a database.

At the original standalone script delivery, offline dry-run validation passed for all 288 valid rows and
all 92 record/document count checks; all 24 tests passed. The actual --apply,
--verify-db and --check-invalid-db modes had not been run against the live database.
