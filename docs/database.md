# Database design

Run setup and database commands from the repository root as documented in [README](../README.md).
The existing Alembic revision and synthetic fixture bytes are preserved.

## Database layout

IDs use strings of 1-64 characters, preserving leading zeros. Dates with times
use `TIMESTAMPTZ`. PostgreSQL stores the instant; appointments also retain an IANA
timezone name for display. Optional values use SQL `NULL` / Python `None`.

### People and access

- `patients`: patient ID, fictional name, optional birth date, synthetic-data flag,
  record revision and timestamps. Duplicate names are intentionally allowed.
- `clinicians`: fictional clinician ID and display name used on reports/bookings.
- `app_users`: mock session user ID, display name, role (`clinician` or `staff`),
  active flag and creation time. No patient link and no patient role.
- `cases`: case ID, patient link, description, status and opening/closing times.
- `authorized_cases` (a view, not a table): maps every active clinician/staff user
  to every case in the mock clinic. Inactive or unknown users get no cases.
- There is no `case_access` table or patient account role in revision 001.

Authentication is not implemented. For the local demo, the application can select
one seeded active clinician/staff user through a trusted mock session. The LLM
must not choose the user ID or role. The helper view is not row-level security and
does not prevent a privileged application from querying tables directly.

The five tool argument/response models do not change. Keep case-context checks,
clarification of ambiguous identifiers, and checks that a study/result/report
belongs to the requested case. Broad read access does not permit mixing cases.

### Studies and results

- `studies`: case link, performed time, modality, body part, side, views,
  acquisition status and optional image reference.
- `model_results`: study link, model name/version, processing status, stored summary,
  creation time, current/historical state, score description and limitations.
- `model_findings`: one predicted finding per row, linked to a model result;
  assessment plus optional normalized confidence in [0, 1].
- `reviewed_reports`: one report version per row, study link, review state,
  clinician/sign-off time, findings, impression, recommendation and prior-version link.
- `appointments`: case link, type, start/end, timezone, status, clinician/location,
  notes and replaced-appointment link.

Case/patient IDs on model/report responses are obtained by joining back through
`studies -> cases`, not copied independently into every table. This avoids storing
contradictory ownership information. The backend must still check requested IDs.

The tool response's `source.version` comes from the record revision converted to
a string. `source.evidence_id` is constructed from record type, ID and revision.
It is not a separate patient-data column. Request IDs, tool duration and
`case_context_version` belong in response/session traces, not clinical tables.

### Clinic documents and vectors

- `clinic_documents`: stable document family ID and internal name (`slug`).
- `document_versions`: exact version, title, full text, tags, approval/publication
  state, effective dates and current flag.
- `document_chunks`: exact passage text and section, linked to a document version.
- `retrieval_indexes`: a named search configuration containing embedding model,
  vector size, chunking version, cosine metric and relevance threshold.
- `chunk_embeddings`: one embedding per passage per search configuration.

The embedding column is `vector` without a fixed dimension. The generated dimension
and a database reference check ensure each vector matches its configuration's
specified size. No embedding size, model or threshold has been chosen yet, and no
embedding rows are created by setup. Zero vectors are rejected.

For a small document collection, exact vector search is sufficient initially.
Choose any faster approximate vector index only after model dimensions are agreed.
Do not compare embeddings from different models, even if their vector sizes match.
The `retrieval_indexes` name describes a frozen search configuration, not an
already-created approximate database index.

At retrieval time filter approved, published, effective/current document versions
before final result selection. A current flag alone is not enough: check dates.
Historical versions are allowed only when explicitly requested and still eligible.

## Edge cases supported by normal rows

- Unknown IDs (by querying an absent ID), duplicate names and similar IDs.
- Unknown/inactive app users, rather than patient-specific or per-case access denials.
- Patients with multiple cases and cases with zero, one or many studies.
- Scheduled/cancelled studies and missing optional values.
- Missing/pending/failed model results, multiple historical versions and empty findings.
- No report, drafts, superseded/withdrawn versions, or disagreement with model output.
- Missing/filtered-out appointments, cancellations, replacements and overlapping bookings.
- Missing guidance, older/future/draft/withdrawn documents and conflicting separate documents.

No report is represented by NO row. A pending report is a draft row. No appointments
means zero matching rows. These cases should not be represented using fake record
values such as a made-up appointment date.

## Rules enforced by SQL

- Parent records must exist; references do not cascade-delete clinical records.
- At most one result/report is designated current per study.
- At most one document version is designated current per document family.
- A report's prior version belongs to the same study.
- An appointment's replaced booking belongs to the same case.
- Reviewed reports have a reviewer and review time; drafts do not masquerade as reviewed.
- Completed model results have a generation time.
- Required fields, allowed statuses, confidence ranges, appointment timezone names
  and basic date ordering are checked.
- Passage/version links and embedding dimension/model-configuration links are checked.

A current pending/failed model result is intentionally allowed: the tool must return
`RESULT_NOT_AVAILABLE`, not use an old completed result. A current draft report
similarly means `NOT_REVIEWED`, not permission to fall back to an old reviewed report.

## Rules still required in Python

- Use a trusted active clinician/staff session for clinical reads and reject stale
  case context. No patient-ownership or per-case-grant check is needed.
- Keep published/reviewed text and its passages fixed; insert new versions rather
  than changing content under an existing evidence ID. SQL immutability triggers
  are not included in this initial schema.
- Update mutable record revisions/timestamps whenever their factual values change.
- Ensure report supersession moves to a later revision and replacement links have
  no cycles. SQL checks same-parent ownership, not full history ordering.
- Ensure each chunk is a verbatim part of its document and record complete manifests.
- Perform current/version/date filtering, correct missing/conflict errors and safe
  output shaping for the five tools; preserve model-versus-reviewed labels.

Constraint-breaking records (for example, two current reports or orphan studies)
should be separate invalid-data fixtures whose attempted insertion is expected to
fail and roll back. Do not disable normal database checks to load them. Tool handling
of deliberately corrupted backend responses can be tested separately later.

## Persistence and later changes

`docker compose stop` stops the DB; `docker compose start` starts it again.
`docker compose down` removes the container/network but retains this named volume.
Do not delete the volume unless you deliberately intend to erase the synthetic DB.

Schema changes now run only through Alembic, never Docker initialization scripts.
The previous draft SQL files were removed because no database had been created.
This consolidated revision 001 is for a fresh database; it is not a conversion
script for a manually created earlier draft schema.

Once revision 001 has been applied or shared as a deployed revision, leave it
unchanged and create a new revision for subsequent structural changes.
Revision `002` now adds the [front-desk booking tables](frontdesk-booking.md);
its implementation did not apply it to the live database. For the next change:

```bash
uv run alembic revision --rev-id 003 -m "describe_schema_change"
# Edit the new file's upgrade() and downgrade() functions.
uv run alembic upgrade head
uv run alembic current
```

Migrations currently use explicit PostgreSQL statements through Alembic's Python
API. ORM table models and `--autogenerate` are not configured yet; do not use that
flag until models/metadata have been added.

Useful read-only commands:

```bash
uv run alembic history
uv run alembic heads
```

To inspect the full initial SQL without connecting to a database or reading a
password, generate an offline SQL file:

```bash
uv run alembic upgrade head --sql > schema.sql
```

Do not manually execute this generated SQL alongside normal Alembic upgrades.
It is a preview, not a second migration owner.

WARNING: the initial revision's downgrade removes the clinic tables and their
contents. It is implemented for disposable development databases only and is not
part of normal setup. It deliberately leaves the public pgvector extension
installed, because the extension could be shared.

Alembic revisions version the database STRUCTURE. Record/report revisions and
clinic-document versions remain separate data fields and are not replaced by
Alembic's revision number.

Keep `.env` private and out of Git. Regenerating/changing the password setting does
not change the password stored in an existing database volume.

The image tag fixes the pgvector version and PostgreSQL major version, not an
immutable image digest. After runtime verification, record the exact image digest
and freeze tested Python dependencies for the final reproducible handover.

## Load the synthetic records

With the Python environment active, run:

```bash
uv run python scripts/seed.py --dry-run
uv run python scripts/seed.py --apply
uv run python scripts/seed.py --verify-db
```

The default mode is also a dry run. `--apply` inserts missing rows, skips identical
rows, and refuses to overwrite different existing records. No rows are deleted.
See [seeding.md](seeding.md) before running the optional invalid-record checks.

## Sources

- https://github.com/pgvector/pgvector
- https://hub.docker.com/r/pgvector/pgvector/tags
