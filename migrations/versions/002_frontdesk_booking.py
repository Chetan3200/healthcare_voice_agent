"""Add persisted front-desk slot, hold, confirmation and booking records.

Revision ID: 002
Revises: 001

This revision adds tables only. It does not alter the applied initial schema,
create clinic schedules, seed patient data or implement authentication/RLS.
All booking/controller/schedule writers must lock booking_guard row 1 within
one transaction before inspecting or changing availability. That application
protocol serializes overlapping clinician/patient checks across slot IDs;
unique keys alone protect exact slot identity, not arbitrary time overlaps.
"""

from alembic import op

revision: str = "002"
down_revision: str = "001"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE clinic.booking_guard (
            id INTEGER PRIMARY KEY CHECK (id = 1)
        )
    """)
    op.execute("INSERT INTO clinic.booking_guard (id) VALUES (1)")
    op.execute("""
        CREATE TABLE clinic.booking_clinics (
            clinic_id clinic.record_id PRIMARY KEY,
            timezone TEXT NOT NULL CHECK (clinic.valid_timezone(timezone))
        )
    """)
    op.execute("""
        CREATE TABLE clinic.booking_sessions (
            session_id UUID PRIMARY KEY,
            clinic_id clinic.record_id NOT NULL REFERENCES clinic.booking_clinics(clinic_id),
            user_id clinic.record_id NOT NULL REFERENCES clinic.app_users(user_id),
            case_id clinic.record_id REFERENCES clinic.cases(case_id),
            case_context_version INTEGER NOT NULL CHECK (case_context_version >= 1),
            conversation_version INTEGER NOT NULL CHECK (conversation_version >= 1),
            active BOOLEAN NOT NULL DEFAULT TRUE,
            current_draft_id UUID,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (current_draft_id IS NULL OR case_id IS NOT NULL)
        )
    """)
    op.execute("""
        CREATE INDEX booking_sessions_user_idx
        ON clinic.booking_sessions (user_id, clinic_id, active)
    """)
    op.execute("""
        CREATE TABLE clinic.booking_slots (
            slot_id UUID PRIMARY KEY,
            clinic_id clinic.record_id NOT NULL REFERENCES clinic.booking_clinics(clinic_id),
            appointment_type TEXT NOT NULL CHECK (appointment_type IN (
                'initial_consultation', 'fracture_follow_up', 'imaging', 'physiotherapy', 'other'
            )),
            starts_at TIMESTAMPTZ NOT NULL,
            ends_at TIMESTAMPTZ NOT NULL,
            clinician_id clinic.record_id NOT NULL REFERENCES clinic.clinicians(clinician_id),
            location TEXT NOT NULL CHECK (char_length(btrim(location)) > 0),
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            CONSTRAINT booking_slot_interval CHECK (ends_at > starts_at),
            CONSTRAINT booking_slot_identity UNIQUE (
                clinic_id, clinician_id, appointment_type, starts_at, ends_at
            )
        )
    """)
    op.execute("""
        CREATE INDEX booking_slots_search_idx
        ON clinic.booking_slots (clinic_id, appointment_type, starts_at, slot_id)
        WHERE enabled
    """)
    op.execute("""
        CREATE INDEX booking_slots_clinician_interval_idx
        ON clinic.booking_slots (clinician_id, starts_at, ends_at)
    """)
    op.execute("""
        CREATE TABLE clinic.booking_drafts (
            draft_id UUID PRIMARY KEY,
            booking_request_id UUID NOT NULL UNIQUE,
            session_id UUID NOT NULL REFERENCES clinic.booking_sessions(session_id),
            case_id clinic.record_id NOT NULL REFERENCES clinic.cases(case_id),
            patient_id clinic.record_id NOT NULL REFERENCES clinic.patients(patient_id),
            case_context_version INTEGER NOT NULL CHECK (case_context_version >= 1),
            slot_id UUID NOT NULL REFERENCES clinic.booking_slots(slot_id),
            state TEXT NOT NULL CHECK (state IN ('pending', 'booked', 'expired', 'failed')),
            failure_code TEXT,
            details JSONB NOT NULL CHECK (jsonb_typeof(details) = 'object'),
            details_hash TEXT NOT NULL CHECK (details_hash ~ '^[0-9a-f]{64}$'),
            hold_expires_at TIMESTAMPTZ NOT NULL,
            readback_version INTEGER,
            readback_at TIMESTAMPTZ,
            confirmation_version INTEGER,
            confirmation_event_id UUID,
            confirmed_at TIMESTAMPTZ,
            appointment_id clinic.record_id UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT booking_draft_appointment_case_fk
                FOREIGN KEY (appointment_id, case_id)
                REFERENCES clinic.appointments (appointment_id, case_id),
            CONSTRAINT booking_draft_hold_interval CHECK (hold_expires_at > created_at),
            CONSTRAINT booking_draft_update_order CHECK (updated_at >= created_at),
            CONSTRAINT booking_draft_failure_shape CHECK (
                (state = 'failed' AND failure_code IS NOT NULL AND failure_code IN (
                    'DRAFT_SUPERSEDED', 'SLOT_UNAVAILABLE', 'PATIENT_CONFLICT', 'CASE_NOT_BOOKABLE'
                )) OR (state <> 'failed' AND failure_code IS NULL)
            ),
            CONSTRAINT booking_draft_outcome_shape CHECK (
                (state = 'booked' AND appointment_id IS NOT NULL AND confirmation_version IS NOT NULL)
                OR (state <> 'booked' AND appointment_id IS NULL)
            ),
            CONSTRAINT booking_draft_readback_shape CHECK (
                (readback_version IS NULL AND readback_at IS NULL)
                OR (readback_version IS NOT NULL AND readback_version >= 1
                    AND readback_at IS NOT NULL AND readback_at >= created_at
                    AND readback_at < hold_expires_at)
            ),
            CONSTRAINT booking_draft_confirmation_shape CHECK (
                (confirmation_version IS NULL AND confirmation_event_id IS NULL AND confirmed_at IS NULL)
                OR (confirmation_version IS NOT NULL AND confirmation_version >= 1
                    AND confirmation_event_id IS NOT NULL AND confirmed_at IS NOT NULL
                    AND readback_version IS NOT NULL AND readback_at IS NOT NULL
                    AND confirmation_version > readback_version
                    AND confirmed_at >= readback_at AND confirmed_at < hold_expires_at)
            ),
            CONSTRAINT booking_draft_session_identity UNIQUE (session_id, draft_id),
            CONSTRAINT booking_draft_hold_identity UNIQUE (draft_id, slot_id, hold_expires_at),
            CONSTRAINT booking_draft_allocation_identity UNIQUE (
                draft_id, slot_id, booking_request_id, appointment_id
            )
        )
    """)
    op.execute("""
        ALTER TABLE clinic.booking_sessions
        ADD CONSTRAINT booking_session_current_draft_fk
        FOREIGN KEY (session_id, current_draft_id)
        REFERENCES clinic.booking_drafts (session_id, draft_id)
        DEFERRABLE INITIALLY DEFERRED
    """)
    op.execute("""
        CREATE INDEX booking_drafts_session_idx
        ON clinic.booking_drafts (session_id, created_at, draft_id)
    """)
    op.execute("""
        CREATE INDEX booking_drafts_pending_expiry_idx
        ON clinic.booking_drafts (hold_expires_at, draft_id)
        WHERE state = 'pending'
    """)
    op.execute("""
        CREATE UNIQUE INDEX booking_confirmation_event_once_idx
        ON clinic.booking_drafts (session_id, confirmation_event_id)
        WHERE confirmation_event_id IS NOT NULL
    """)
    op.execute("""
        CREATE TABLE clinic.booking_offers (
            session_id UUID NOT NULL REFERENCES clinic.booking_sessions(session_id),
            slot_id UUID NOT NULL REFERENCES clinic.booking_slots(slot_id),
            offered_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, slot_id)
        )
    """)
    op.execute("""
        CREATE TABLE clinic.booking_holds (
            slot_id UUID PRIMARY KEY REFERENCES clinic.booking_slots(slot_id),
            draft_id UUID NOT NULL UNIQUE REFERENCES clinic.booking_drafts(draft_id),
            expires_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT booking_hold_matches_draft_fk
                FOREIGN KEY (draft_id, slot_id, expires_at)
                REFERENCES clinic.booking_drafts (draft_id, slot_id, hold_expires_at)
                DEFERRABLE INITIALLY DEFERRED
        )
    """)
    op.execute("CREATE INDEX booking_holds_expiry_idx ON clinic.booking_holds (expires_at)")
    op.execute("""
        CREATE TABLE clinic.booking_allocations (
            slot_id UUID PRIMARY KEY REFERENCES clinic.booking_slots(slot_id),
            draft_id UUID NOT NULL UNIQUE REFERENCES clinic.booking_drafts(draft_id),
            booking_request_id UUID NOT NULL UNIQUE REFERENCES clinic.booking_drafts(booking_request_id),
            appointment_id clinic.record_id NOT NULL UNIQUE REFERENCES clinic.appointments(appointment_id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT booking_allocation_identity UNIQUE (
                draft_id, slot_id, booking_request_id, appointment_id
            ),
            CONSTRAINT booking_allocation_matches_draft_fk
                FOREIGN KEY (draft_id, slot_id, booking_request_id, appointment_id)
                REFERENCES clinic.booking_drafts (draft_id, slot_id, booking_request_id, appointment_id)
                DEFERRABLE INITIALLY DEFERRED
        )
    """)
    # A booked draft must have the matching allocation at commit; an unbooked
    # draft's NULL appointment_id deliberately exempts it (MATCH SIMPLE). Both
    # directions are deferred so the backend can perform its atomic writes in
    # either order without leaving a committed half-booking.
    op.execute("""
        ALTER TABLE clinic.booking_drafts
        ADD CONSTRAINT booking_draft_committed_allocation_fk
        FOREIGN KEY (draft_id, slot_id, booking_request_id, appointment_id)
        REFERENCES clinic.booking_allocations (draft_id, slot_id, booking_request_id, appointment_id)
        DEFERRABLE INITIALLY DEFERRED
    """)


def downgrade() -> None:
    """Remove only booking metadata, never pre-existing confirmed appointments.

    This deletes front-desk history/holds and disables its tools. Run only as an
    explicit administrative rollback after stopping all booking writers. No
    CASCADE: unknown dependencies cause a transaction failure rather than loss.
    """
    op.execute("ALTER TABLE clinic.booking_sessions DROP CONSTRAINT booking_session_current_draft_fk")
    op.execute("ALTER TABLE clinic.booking_drafts DROP CONSTRAINT booking_draft_committed_allocation_fk")
    op.execute("DROP TABLE clinic.booking_allocations")
    op.execute("DROP TABLE clinic.booking_holds")
    op.execute("DROP TABLE clinic.booking_offers")
    op.execute("DROP TABLE clinic.booking_drafts")
    op.execute("DROP TABLE clinic.booking_slots")
    op.execute("DROP TABLE clinic.booking_sessions")
    op.execute("DROP TABLE clinic.booking_clinics")
    op.execute("DROP TABLE clinic.booking_guard")
