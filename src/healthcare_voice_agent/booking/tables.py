"""SQLAlchemy Core tables for booking queries, not a schema-creation mechanism.

Production schema changes belong to Alembic revisions 002 through 005. The minimal mappings
of revision-001 tables below are query projections, NOT replacement definitions.
SQLite schema translation is used only by isolated offline transactional tests.
"""
from sqlalchemy import (
    Boolean, Column, Date, DateTime, ForeignKey, Integer, JSON, MetaData, String,
    Table, UniqueConstraint, Uuid, ForeignKeyConstraint, CheckConstraint, Index,
)

metadata = MetaData(schema="clinic")
patients = Table("patients", metadata,
    Column("patient_id", String(64), primary_key=True), Column("display_name", String, nullable=False))
clinicians = Table("clinicians", metadata,
    Column("clinician_id", String(64), primary_key=True), Column("display_name", String, nullable=False))
users = Table("app_users", metadata,
    Column("user_id", String(64), primary_key=True), Column("is_active", Boolean, nullable=False),
    Column("role", String, nullable=False))
cases = Table("cases", metadata,
    Column("case_id", String(64), primary_key=True),
    Column("patient_id", String(64), ForeignKey("clinic.patients.patient_id"), nullable=False),
    Column("status", String, nullable=False))
appointments = Table("appointments", metadata,
    Column("appointment_id", String(64), primary_key=True),
    Column("case_id", String(64), ForeignKey("clinic.cases.case_id"), nullable=False),
    Column("appointment_type", String, nullable=False),
    Column("starts_at", DateTime(timezone=True), nullable=False), Column("ends_at", DateTime(timezone=True)),
    Column("timezone", String, nullable=False), Column("status", String, nullable=False),
    Column("clinician_id", String(64), ForeignKey("clinic.clinicians.clinician_id")),
    Column("location", String), Column("record_version", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("appointment_id", "case_id"))

guard = Table("booking_guard", metadata, Column("id", Integer, primary_key=True))
clinics = Table("booking_clinics", metadata,
    Column("clinic_id", String(64), primary_key=True), Column("timezone", String, nullable=False))
sessions = Table("booking_sessions", metadata,
    Column("session_id", Uuid, primary_key=True),
    Column("clinic_id", String(64), ForeignKey("clinic.booking_clinics.clinic_id"), nullable=False),
    Column("user_id", String(64), ForeignKey("clinic.app_users.user_id"), nullable=False),
    Column("case_id", String(64), ForeignKey("clinic.cases.case_id")),
    Column("case_context_version", Integer, nullable=False),
    Column("conversation_version", Integer, nullable=False),
    Column("active", Boolean, nullable=False), Column("current_draft_id", Uuid),
    Column("created_at", DateTime(timezone=True), nullable=False))
slots = Table("booking_slots", metadata,
    Column("slot_id", Uuid, primary_key=True),
    Column("clinic_id", String(64), ForeignKey("clinic.booking_clinics.clinic_id"), nullable=False),
    Column("appointment_type", String, nullable=False),
    Column("starts_at", DateTime(timezone=True), nullable=False),
    Column("ends_at", DateTime(timezone=True), nullable=False),
    Column("clinician_id", String(64), ForeignKey("clinic.clinicians.clinician_id"), nullable=False),
    Column("location", String, nullable=False), Column("enabled", Boolean, nullable=False),
    UniqueConstraint("clinic_id", "clinician_id", "appointment_type", "starts_at", "ends_at", name="booking_slot_identity"))
drafts = Table("booking_drafts", metadata,
    Column("draft_id", Uuid, primary_key=True), Column("booking_request_id", Uuid, nullable=False, unique=True),
    Column("session_id", Uuid, ForeignKey("clinic.booking_sessions.session_id"), nullable=False),
    Column("case_id", String(64), ForeignKey("clinic.cases.case_id"), nullable=False),
    Column("patient_id", String(64), ForeignKey("clinic.patients.patient_id"), nullable=False),
    Column("case_context_version", Integer, nullable=False),
    Column("slot_id", Uuid, ForeignKey("clinic.booking_slots.slot_id"), nullable=False),
    Column("state", String, nullable=False), Column("failure_code", String),
    Column("details", JSON, nullable=False), Column("details_hash", String(64), nullable=False),
    Column("hold_expires_at", DateTime(timezone=True), nullable=False),
    Column("readback_version", Integer), Column("readback_at", DateTime(timezone=True)),
    Column("confirmation_version", Integer), Column("confirmation_event_id", Uuid),
    Column("confirmed_at", DateTime(timezone=True)),
    Column("authorization_kind", String, nullable=False, server_default="confirmation"),
    Column("appointment_id", String(64), unique=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False))
offers = Table("booking_offers", metadata,
    Column("session_id", Uuid, ForeignKey("clinic.booking_sessions.session_id"), primary_key=True),
    Column("slot_id", Uuid, ForeignKey("clinic.booking_slots.slot_id"), primary_key=True),
    Column("offered_at", DateTime(timezone=True), nullable=False))
holds = Table("booking_holds", metadata,
    Column("slot_id", Uuid, ForeignKey("clinic.booking_slots.slot_id"), primary_key=True),
    Column("draft_id", Uuid, ForeignKey("clinic.booking_drafts.draft_id"), nullable=False, unique=True),
    Column("expires_at", DateTime(timezone=True), nullable=False))
allocations = Table("booking_allocations", metadata,
    Column("slot_id", Uuid, ForeignKey("clinic.booking_slots.slot_id"), nullable=False),
    Column("draft_id", Uuid, ForeignKey("clinic.booking_drafts.draft_id"), primary_key=True, unique=True),
    Column("booking_request_id", Uuid, ForeignKey("clinic.booking_drafts.booking_request_id"), nullable=False, unique=True),
    Column("appointment_id", String(64), ForeignKey("clinic.appointments.appointment_id"), nullable=False, unique=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("active", Boolean, nullable=False, default=True, server_default="true"))

reschedules = Table("booking_reschedules", metadata,
    Column("draft_id", Uuid, ForeignKey("clinic.booking_drafts.draft_id"), primary_key=True),
    Column("session_id", Uuid, ForeignKey("clinic.booking_sessions.session_id"), nullable=False),
    Column("case_id", String(64), ForeignKey("clinic.cases.case_id"), nullable=False),
    Column("clinic_id", String(64), ForeignKey("clinic.booking_clinics.clinic_id"), nullable=False),
    Column("source_appointment_id", String(64), nullable=False),
    Column("source_record_version", Integer, nullable=False),
    Column("source_snapshot", JSON, nullable=False),
    Column("source_hash", String(64), nullable=False),
    Column("replacement_appointment_id", String(64), ForeignKey("clinic.appointments.appointment_id"), unique=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    ForeignKeyConstraint(["source_appointment_id", "case_id"],
                         ["clinic.appointments.appointment_id", "clinic.appointments.case_id"]),
    ForeignKeyConstraint(["session_id", "draft_id"],
                         ["clinic.booking_drafts.session_id", "clinic.booking_drafts.draft_id"]),
    CheckConstraint("source_record_version >= 1"),
    CheckConstraint("(replacement_appointment_id IS NULL AND completed_at IS NULL) OR "
                    "(replacement_appointment_id IS NOT NULL AND completed_at IS NOT NULL AND completed_at >= created_at)"))

cancellations = Table("booking_cancellations", metadata,
    Column("cancellation_request_id", Uuid, primary_key=True),
    Column("session_id", Uuid, ForeignKey("clinic.booking_sessions.session_id"), nullable=False),
    Column("case_id", String(64), ForeignKey("clinic.cases.case_id"), nullable=False),
    Column("clinic_id", String(64), ForeignKey("clinic.booking_clinics.clinic_id"), nullable=False),
    Column("case_context_version", Integer, nullable=False),
    Column("prepared_conversation_version", Integer, nullable=False),
    Column("source_appointment_id", String(64), nullable=False),
    Column("source_record_version", Integer, nullable=False),
    Column("source_snapshot", JSON, nullable=False),
    Column("source_hash", String(64), nullable=False),
    Column("details_hash", String(64), nullable=False),
    Column("state", String, nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("readback_version", Integer), Column("readback_at", DateTime(timezone=True)),
    Column("confirmation_version", Integer), Column("confirmation_event_id", Uuid),
    Column("confirmed_at", DateTime(timezone=True)),
    Column("authorization_kind", String, nullable=False, server_default="confirmation"),
    Column("result_snapshot", JSON), Column("cancelled_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    ForeignKeyConstraint(["source_appointment_id", "case_id"],
                         ["clinic.appointments.appointment_id", "clinic.appointments.case_id"]),
    CheckConstraint("case_context_version >= 1"),
    CheckConstraint("prepared_conversation_version >= 1"),
    CheckConstraint("source_record_version >= 1"),
    CheckConstraint("state IN ('pending', 'cancelled', 'expired', 'abandoned')"),
    CheckConstraint("expires_at > created_at", name="booking_cancellation_expiry"),
    CheckConstraint("updated_at >= created_at", name="booking_cancellation_update_order"),
    CheckConstraint("(state = 'cancelled' AND result_snapshot IS NOT NULL AND cancelled_at IS NOT NULL) OR "
                    "(state <> 'cancelled' AND result_snapshot IS NULL AND cancelled_at IS NULL)",
                    name="booking_cancellation_outcome_shape"),
    CheckConstraint("(readback_version IS NULL AND readback_at IS NULL) OR "
                    "(readback_version IS NOT NULL AND readback_version >= 1 AND readback_at IS NOT NULL "
                    "AND readback_at >= created_at AND readback_at < expires_at)",
                    name="booking_cancellation_readback_shape"),
    CheckConstraint("authorization_kind IN ('confirmation', 'direct_request')",
                    name="booking_cancellation_authorization_kind"),
    CheckConstraint("(authorization_kind = 'confirmation' AND ((confirmation_version IS NULL "
                    "AND confirmation_event_id IS NULL AND confirmed_at IS NULL) OR "
                    "(confirmation_version IS NOT NULL AND confirmation_version >= 1 "
                    "AND confirmation_event_id IS NOT NULL AND confirmed_at IS NOT NULL "
                    "AND readback_version IS NOT NULL AND readback_at IS NOT NULL "
                    "AND confirmation_version = readback_version + 1 "
                    "AND confirmed_at >= readback_at AND confirmed_at < expires_at))) OR "
                    "(authorization_kind = 'direct_request' AND confirmation_version IS NOT NULL "
                    "AND confirmation_version >= 1 AND confirmation_event_id IS NOT NULL "
                    "AND confirmed_at IS NOT NULL AND confirmed_at >= created_at "
                    "AND confirmed_at < expires_at)",
                    name="booking_cancellation_confirmation_shape"))


# Mirror production's portable constraints in offline SQLite fixtures. PostgreSQL
# JSONB type/shape, regexp and pg_timezone_names checks remain migration-only.
# Do not use this metadata to create or migrate a production database.
guard.append_constraint(CheckConstraint("id = 1"))
sessions.append_constraint(CheckConstraint("case_context_version >= 1"))
sessions.append_constraint(CheckConstraint("conversation_version >= 1"))
sessions.append_constraint(CheckConstraint("current_draft_id IS NULL OR case_id IS NOT NULL"))
sessions.append_constraint(ForeignKeyConstraint(
    ["session_id", "current_draft_id"],
    ["clinic.booking_drafts.session_id", "clinic.booking_drafts.draft_id"],
    name="booking_session_current_draft_fk", deferrable=True, initially="DEFERRED"))
slots.append_constraint(CheckConstraint("ends_at > starts_at", name="booking_slot_interval"))
slots.append_constraint(CheckConstraint("length(trim(location)) > 0"))
slots.append_constraint(CheckConstraint("appointment_type IN ('initial_consultation', 'fracture_follow_up', 'imaging', 'physiotherapy', 'other')"))
drafts.append_constraint(CheckConstraint("case_context_version >= 1"))
drafts.append_constraint(CheckConstraint("state IN ('pending', 'booked', 'expired', 'failed')"))
drafts.append_constraint(CheckConstraint("hold_expires_at > created_at", name="booking_draft_hold_interval"))
drafts.append_constraint(CheckConstraint("updated_at >= created_at", name="booking_draft_update_order"))
drafts.append_constraint(CheckConstraint(
    "(state = 'failed' AND failure_code IS NOT NULL AND failure_code IN "
    "('DRAFT_SUPERSEDED', 'SLOT_UNAVAILABLE', 'PATIENT_CONFLICT', 'CASE_NOT_BOOKABLE')) "
    "OR (state <> 'failed' AND failure_code IS NULL)", name="booking_draft_failure_shape"))
drafts.append_constraint(CheckConstraint(
    "(state = 'booked' AND appointment_id IS NOT NULL AND confirmation_version IS NOT NULL) "
    "OR (state <> 'booked' AND appointment_id IS NULL)", name="booking_draft_outcome_shape"))
drafts.append_constraint(CheckConstraint(
    "(readback_version IS NULL AND readback_at IS NULL) OR "
    "(readback_version IS NOT NULL AND readback_version >= 1 AND readback_at IS NOT NULL "
    "AND readback_at >= created_at AND readback_at < hold_expires_at)",
    name="booking_draft_readback_shape"))
drafts.append_constraint(CheckConstraint("authorization_kind IN ('confirmation', 'direct_request')",
    name="booking_draft_authorization_kind"))
drafts.append_constraint(CheckConstraint(
    "(authorization_kind = 'confirmation' AND ((confirmation_version IS NULL "
    "AND confirmation_event_id IS NULL AND confirmed_at IS NULL) OR "
    "(confirmation_version IS NOT NULL AND confirmation_version >= 1 "
    "AND confirmation_event_id IS NOT NULL AND confirmed_at IS NOT NULL "
    "AND readback_version IS NOT NULL AND readback_at IS NOT NULL "
    "AND confirmation_version > readback_version "
    "AND confirmed_at >= readback_at AND confirmed_at < hold_expires_at))) OR "
    "(authorization_kind = 'direct_request' AND confirmation_version IS NOT NULL "
    "AND confirmation_version >= 1 AND confirmation_event_id IS NOT NULL "
    "AND confirmed_at IS NOT NULL AND confirmed_at >= created_at "
    "AND confirmed_at < hold_expires_at)",
    name="booking_draft_confirmation_shape"))
drafts.append_constraint(ForeignKeyConstraint(
    ["appointment_id", "case_id"], ["clinic.appointments.appointment_id", "clinic.appointments.case_id"],
    name="booking_draft_appointment_case_fk"))
drafts.append_constraint(UniqueConstraint("session_id", "draft_id", name="booking_draft_session_identity"))
drafts.append_constraint(UniqueConstraint("draft_id", "slot_id", "hold_expires_at", name="booking_draft_hold_identity"))
drafts.append_constraint(UniqueConstraint("draft_id", "slot_id", "booking_request_id", "appointment_id",
    name="booking_draft_allocation_identity"))
drafts.append_constraint(ForeignKeyConstraint(
    ["draft_id", "slot_id", "booking_request_id", "appointment_id"],
    ["clinic.booking_allocations.draft_id", "clinic.booking_allocations.slot_id",
     "clinic.booking_allocations.booking_request_id", "clinic.booking_allocations.appointment_id"],
    name="booking_draft_committed_allocation_fk", deferrable=True, initially="DEFERRED"))
holds.append_constraint(ForeignKeyConstraint(
    ["draft_id", "slot_id", "expires_at"],
    ["clinic.booking_drafts.draft_id", "clinic.booking_drafts.slot_id", "clinic.booking_drafts.hold_expires_at"],
    name="booking_hold_matches_draft_fk", deferrable=True, initially="DEFERRED"))
allocations.append_constraint(UniqueConstraint("draft_id", "slot_id", "booking_request_id", "appointment_id",
    name="booking_allocation_identity"))
allocations.append_constraint(ForeignKeyConstraint(
    ["draft_id", "slot_id", "booking_request_id", "appointment_id"],
    ["clinic.booking_drafts.draft_id", "clinic.booking_drafts.slot_id",
     "clinic.booking_drafts.booking_request_id", "clinic.booking_drafts.appointment_id"],
    name="booking_allocation_matches_draft_fk", deferrable=True, initially="DEFERRED"))
Index("booking_sessions_user_idx", sessions.c.user_id, sessions.c.clinic_id, sessions.c.active)
Index("booking_slots_search_idx", slots.c.clinic_id, slots.c.appointment_type, slots.c.starts_at, slots.c.slot_id,
    postgresql_where=slots.c.enabled, sqlite_where=slots.c.enabled)
Index("booking_slots_clinician_interval_idx", slots.c.clinician_id, slots.c.starts_at, slots.c.ends_at)
Index("booking_drafts_session_idx", drafts.c.session_id, drafts.c.created_at, drafts.c.draft_id)
Index("booking_drafts_pending_expiry_idx", drafts.c.hold_expires_at, drafts.c.draft_id,
    postgresql_where=drafts.c.state == "pending", sqlite_where=drafts.c.state == "pending")
Index("booking_confirmation_event_once_idx", drafts.c.session_id, drafts.c.confirmation_event_id,
    unique=True, postgresql_where=drafts.c.confirmation_event_id.is_not(None),
    sqlite_where=drafts.c.confirmation_event_id.is_not(None))
Index("booking_holds_expiry_idx", holds.c.expires_at)

Index("booking_active_slot_once_idx", allocations.c.slot_id, unique=True,
    postgresql_where=allocations.c.active, sqlite_where=allocations.c.active)
Index("booking_reschedules_source_idx", reschedules.c.source_appointment_id)
Index("booking_cancellations_session_idx", cancellations.c.session_id,
      cancellations.c.created_at, cancellations.c.cancellation_request_id)
Index("booking_cancellation_pending_session_once_idx", cancellations.c.session_id, unique=True,
      postgresql_where=cancellations.c.state == "pending", sqlite_where=cancellations.c.state == "pending")
Index("booking_cancellation_confirmation_event_once_idx", cancellations.c.session_id,
      cancellations.c.confirmation_event_id, unique=True,
      postgresql_where=cancellations.c.confirmation_event_id.is_not(None),
      sqlite_where=cancellations.c.confirmation_event_id.is_not(None))
