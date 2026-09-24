"""Add atomic cancel-and-replace rescheduling with immutable booking receipts.

Revision ID: 003
Revises: 002

Existing allocation identities and their deferred reciprocal draft FKs remain.
Only active allocations uniquely occupy a slot; cancelled receipts stay queryable.
This migration never cancels an appointment or creates a replacement booking.
"""
from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE clinic.booking_allocations ADD COLUMN active BOOLEAN NOT NULL DEFAULT TRUE")
    op.execute("""UPDATE clinic.booking_allocations AS allocation
        SET active = appointment.status IN ('scheduled', 'confirmed')
        FROM clinic.appointments AS appointment
        WHERE appointment.appointment_id = allocation.appointment_id""")
    op.execute("ALTER TABLE clinic.booking_allocations DROP CONSTRAINT booking_allocations_pkey")
    op.execute("ALTER TABLE clinic.booking_allocations ADD CONSTRAINT booking_allocations_pkey PRIMARY KEY (draft_id)")
    op.execute("CREATE UNIQUE INDEX booking_active_slot_once_idx ON clinic.booking_allocations (slot_id) WHERE active")
    op.execute("""CREATE TABLE clinic.booking_reschedules (
        draft_id UUID PRIMARY KEY REFERENCES clinic.booking_drafts(draft_id),
        session_id UUID NOT NULL REFERENCES clinic.booking_sessions(session_id),
        case_id clinic.record_id NOT NULL REFERENCES clinic.cases(case_id),
        clinic_id clinic.record_id NOT NULL REFERENCES clinic.booking_clinics(clinic_id),
        source_appointment_id clinic.record_id NOT NULL,
        source_record_version INTEGER NOT NULL CHECK (source_record_version >= 1),
        source_snapshot JSONB NOT NULL CHECK (jsonb_typeof(source_snapshot) = 'object'),
        source_hash TEXT NOT NULL CHECK (source_hash ~ '^[0-9a-f]{64}$'),
        replacement_appointment_id clinic.record_id UNIQUE REFERENCES clinic.appointments(appointment_id),
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMPTZ,
        FOREIGN KEY (source_appointment_id, case_id)
            REFERENCES clinic.appointments(appointment_id, case_id),
        FOREIGN KEY (session_id, draft_id)
            REFERENCES clinic.booking_drafts(session_id, draft_id),
        CHECK ((replacement_appointment_id IS NULL AND completed_at IS NULL) OR
               (replacement_appointment_id IS NOT NULL AND completed_at IS NOT NULL AND completed_at >= created_at))
    )""")
    op.execute("CREATE INDEX booking_reschedules_source_idx ON clinic.booking_reschedules (source_appointment_id)")


def downgrade():
    # Historical reuse makes 002's slot primary key impossible. Fail before any
    # destructive DDL rather than discard receipts or silently alter bookings.
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM clinic.booking_reschedules) OR EXISTS (
            SELECT slot_id FROM clinic.booking_allocations GROUP BY slot_id HAVING count(*) > 1
        ) THEN RAISE EXCEPTION 'Cannot downgrade rescheduling while reschedule receipts or reused slots exist';
        END IF;
    END $$""")
    op.execute("DROP TABLE clinic.booking_reschedules")
    op.execute("DROP INDEX clinic.booking_active_slot_once_idx")
    op.execute("ALTER TABLE clinic.booking_allocations DROP CONSTRAINT booking_allocations_pkey")
    op.execute("ALTER TABLE clinic.booking_allocations ADD CONSTRAINT booking_allocations_pkey PRIMARY KEY (slot_id)")
    op.execute("ALTER TABLE clinic.booking_allocations DROP COLUMN active")
