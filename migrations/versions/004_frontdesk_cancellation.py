"""Add persisted standalone appointment cancellation receipts.

Revision ID: 004
Revises: 003

This revision adds cancellation intent and outcome storage only. It does not
cancel appointments, release allocations, seed data, or apply itself. The
backend uses the existing booking_guard row lock for every cancellation read or
write that participates in scheduling state.
"""
from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE clinic.booking_cancellations (
        cancellation_request_id UUID PRIMARY KEY,
        session_id UUID NOT NULL REFERENCES clinic.booking_sessions(session_id),
        case_id clinic.record_id NOT NULL REFERENCES clinic.cases(case_id),
        clinic_id clinic.record_id NOT NULL REFERENCES clinic.booking_clinics(clinic_id),
        case_context_version INTEGER NOT NULL CHECK (case_context_version >= 1),
        prepared_conversation_version INTEGER NOT NULL CHECK (prepared_conversation_version >= 1),
        source_appointment_id clinic.record_id NOT NULL,
        source_record_version INTEGER NOT NULL CHECK (source_record_version >= 1),
        source_snapshot JSONB NOT NULL CHECK (jsonb_typeof(source_snapshot) = 'object'),
        source_hash TEXT NOT NULL CHECK (source_hash ~ '^[0-9a-f]{64}$'),
        details_hash TEXT NOT NULL CHECK (details_hash ~ '^[0-9a-f]{64}$'),
        state TEXT NOT NULL CHECK (state IN ('pending', 'cancelled', 'expired', 'abandoned')),
        expires_at TIMESTAMPTZ NOT NULL,
        readback_version INTEGER,
        readback_at TIMESTAMPTZ,
        confirmation_version INTEGER,
        confirmation_event_id UUID,
        confirmed_at TIMESTAMPTZ,
        result_snapshot JSONB CHECK (result_snapshot IS NULL OR jsonb_typeof(result_snapshot) = 'object'),
        cancelled_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (source_appointment_id, case_id)
            REFERENCES clinic.appointments(appointment_id, case_id),
        CONSTRAINT booking_cancellation_expiry CHECK (expires_at > created_at),
        CONSTRAINT booking_cancellation_update_order CHECK (updated_at >= created_at),
        CONSTRAINT booking_cancellation_outcome_shape CHECK (
            (state = 'cancelled' AND result_snapshot IS NOT NULL AND cancelled_at IS NOT NULL)
            OR (state <> 'cancelled' AND result_snapshot IS NULL AND cancelled_at IS NULL)
        ),
        CONSTRAINT booking_cancellation_readback_shape CHECK (
            (readback_version IS NULL AND readback_at IS NULL)
            OR (readback_version IS NOT NULL AND readback_version >= 1
                AND readback_at IS NOT NULL AND readback_at >= created_at
                AND readback_at < expires_at)
        ),
        CONSTRAINT booking_cancellation_confirmation_shape CHECK (
            (confirmation_version IS NULL AND confirmation_event_id IS NULL AND confirmed_at IS NULL)
            OR (confirmation_version IS NOT NULL AND confirmation_version >= 1
                AND confirmation_event_id IS NOT NULL AND confirmed_at IS NOT NULL
                AND readback_version IS NOT NULL AND readback_at IS NOT NULL
                AND confirmation_version = readback_version + 1
                AND confirmed_at >= readback_at AND confirmed_at < expires_at)
        )
    )""")
    op.execute("""CREATE INDEX booking_cancellations_session_idx
        ON clinic.booking_cancellations (session_id, created_at, cancellation_request_id)""")
    op.execute("""CREATE UNIQUE INDEX booking_cancellation_pending_session_once_idx
        ON clinic.booking_cancellations (session_id) WHERE state = 'pending'""")
    op.execute("""CREATE UNIQUE INDEX booking_cancellation_confirmation_event_once_idx
        ON clinic.booking_cancellations (session_id, confirmation_event_id)
        WHERE confirmation_event_id IS NOT NULL""")


def downgrade():
    # Never erase audit/outcome receipts silently. An explicit administrative
    # cleanup is required before removing the cancellation feature.
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM clinic.booking_cancellations) THEN
            RAISE EXCEPTION 'Cannot downgrade cancellation while cancellation receipts exist';
        END IF;
    END $$""")
    op.execute("DROP TABLE clinic.booking_cancellations")
