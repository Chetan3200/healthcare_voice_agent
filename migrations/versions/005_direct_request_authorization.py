"""Allow exact direct user requests to authorize booking changes without readback.

Revision ID: 005
Revises: 004

Existing confirmation receipts retain their original readback-dependent shape.
Direct-request authorization is separately marked and still requires a complete,
expiring event receipt. This revision records schema only; it does not authorize,
book, reschedule, cancel, seed, or apply itself.
"""
from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


_DRAFT_OLD = """(confirmation_version IS NULL AND confirmation_event_id IS NULL AND confirmed_at IS NULL)
    OR (confirmation_version IS NOT NULL AND confirmation_version >= 1
        AND confirmation_event_id IS NOT NULL AND confirmed_at IS NOT NULL
        AND readback_version IS NOT NULL AND readback_at IS NOT NULL
        AND confirmation_version > readback_version
        AND confirmed_at >= readback_at AND confirmed_at < hold_expires_at)"""
_DRAFT_NEW = f"""(authorization_kind = 'confirmation' AND ({_DRAFT_OLD}))
    OR (authorization_kind = 'direct_request' AND confirmation_version IS NOT NULL
        AND confirmation_version >= 1 AND confirmation_event_id IS NOT NULL
        AND confirmed_at IS NOT NULL AND confirmed_at >= created_at
        AND confirmed_at < hold_expires_at)"""
_CANCELLATION_OLD = """(confirmation_version IS NULL AND confirmation_event_id IS NULL AND confirmed_at IS NULL)
    OR (confirmation_version IS NOT NULL AND confirmation_version >= 1
        AND confirmation_event_id IS NOT NULL AND confirmed_at IS NOT NULL
        AND readback_version IS NOT NULL AND readback_at IS NOT NULL
        AND confirmation_version = readback_version + 1
        AND confirmed_at >= readback_at AND confirmed_at < expires_at)"""
_CANCELLATION_NEW = f"""(authorization_kind = 'confirmation' AND ({_CANCELLATION_OLD}))
    OR (authorization_kind = 'direct_request' AND confirmation_version IS NOT NULL
        AND confirmation_version >= 1 AND confirmation_event_id IS NOT NULL
        AND confirmed_at IS NOT NULL AND confirmed_at >= created_at
        AND confirmed_at < expires_at)"""


def upgrade():
    for table, kind_constraint, confirmation_constraint in (
        ("booking_drafts", "booking_draft_authorization_kind", "booking_draft_confirmation_shape"),
        ("booking_cancellations", "booking_cancellation_authorization_kind", "booking_cancellation_confirmation_shape"),
    ):
        op.execute(f"ALTER TABLE clinic.{table} ADD COLUMN authorization_kind TEXT NOT NULL DEFAULT 'confirmation'")
        op.execute(f"ALTER TABLE clinic.{table} ADD CONSTRAINT {kind_constraint} "
                   "CHECK (authorization_kind IN ('confirmation', 'direct_request'))")
        op.execute(f"ALTER TABLE clinic.{table} DROP CONSTRAINT {confirmation_constraint}")
    op.execute(f"ALTER TABLE clinic.booking_drafts ADD CONSTRAINT booking_draft_confirmation_shape CHECK ({_DRAFT_NEW})")
    op.execute("ALTER TABLE clinic.booking_cancellations ADD CONSTRAINT "
               f"booking_cancellation_confirmation_shape CHECK ({_CANCELLATION_NEW})")


def downgrade():
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM clinic.booking_drafts WHERE authorization_kind = 'direct_request')
           OR EXISTS (SELECT 1 FROM clinic.booking_cancellations WHERE authorization_kind = 'direct_request') THEN
            RAISE EXCEPTION 'Cannot downgrade while direct-request authorization receipts exist';
        END IF;
    END $$""")
    for table, kind_constraint, confirmation_constraint, old_shape in (
        ("booking_drafts", "booking_draft_authorization_kind", "booking_draft_confirmation_shape", _DRAFT_OLD),
        ("booking_cancellations", "booking_cancellation_authorization_kind", "booking_cancellation_confirmation_shape", _CANCELLATION_OLD),
    ):
        op.execute(f"ALTER TABLE clinic.{table} DROP CONSTRAINT {confirmation_constraint}")
        op.execute(f"ALTER TABLE clinic.{table} DROP CONSTRAINT {kind_constraint}")
        op.execute(f"ALTER TABLE clinic.{table} DROP COLUMN authorization_kind")
        op.execute(f"ALTER TABLE clinic.{table} ADD CONSTRAINT {confirmation_constraint} CHECK ({old_shape})")
