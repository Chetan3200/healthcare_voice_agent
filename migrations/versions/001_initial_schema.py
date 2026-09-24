"""Create the clinician/staff-only mock clinic schema.

Revision ID: 001
Revises: None

This is the single initial revision. No database existed when the earlier draft
SQL revisions were consolidated. Alembic owns transaction and revision tracking.
The statements are explicit PostgreSQL DDL; no ORM models are required.
"""

from alembic import op


revision: str = "001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create the complete initial schema; Alembic controls the transaction."""
    op.execute(
        r"""
        -- Initial PostgreSQL schema owned by Alembic revision 001.
        -- Clinician/staff-only mock app; no patient login accounts or per-case grants.
        
        CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public
        """
    )

    op.execute(
        r"""
        CREATE SCHEMA clinic
        """
    )

    op.execute(
        r"""
        SET LOCAL search_path TO clinic, public
        """
    )

    op.execute(
        r"""
        CREATE DOMAIN record_id AS TEXT
            CHECK (char_length(VALUE) BETWEEN 1 AND 64)
        """
    )

    op.execute(
        r"""
        -- Shared people and access information. These are application identities, not
        -- PostgreSQL login roles. This schema does NOT implement authentication.
        CREATE TABLE patients (
            patient_id record_id PRIMARY KEY,
            display_name TEXT NOT NULL CHECK (char_length(btrim(display_name)) > 0),
            date_of_birth DATE,
            is_synthetic BOOLEAN NOT NULL DEFAULT TRUE CHECK (is_synthetic),
            record_version INTEGER NOT NULL DEFAULT 1 CHECK (record_version >= 1),
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    op.execute(
        r"""
        -- Deliberately no unique constraint on names.
        CREATE INDEX patients_name_idx ON patients (lower(display_name))
        """
    )

    op.execute(
        r"""
        CREATE TABLE clinicians (
            clinician_id record_id PRIMARY KEY,
            display_name TEXT NOT NULL CHECK (char_length(btrim(display_name)) > 0),
            is_synthetic BOOLEAN NOT NULL DEFAULT TRUE CHECK (is_synthetic),
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    op.execute(
        r"""
        CREATE TABLE app_users (
            user_id record_id PRIMARY KEY,
            display_name TEXT NOT NULL CHECK (char_length(btrim(display_name)) > 0),
            role TEXT NOT NULL CHECK (role IN ('clinician', 'staff')),
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    op.execute(
        r"""
        CREATE TABLE cases (
            case_id record_id PRIMARY KEY,
            patient_id record_id NOT NULL REFERENCES patients(patient_id),
            description TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('open', 'closed', 'archived')),
            opened_at TIMESTAMPTZ NOT NULL,
            closed_at TIMESTAMPTZ,
            record_version INTEGER NOT NULL DEFAULT 1 CHECK (record_version >= 1),
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (closed_at IS NULL OR closed_at >= opened_at)
        )
        """
    )

    op.execute(
        r"""
        CREATE INDEX cases_patient_idx ON cases(patient_id)
        """
    )

    op.execute(
        r"""
        -- All active clinician/staff users may access every case in this mock clinic.
        -- This view is a convenience for server queries, not authentication or RLS.
        -- The server supplies the trusted user ID; the LLM must not choose it.
        CREATE VIEW authorized_cases AS
        SELECT u.user_id, c.case_id, c.patient_id
        FROM app_users AS u
        CROSS JOIN cases AS c
        WHERE u.is_active AND u.role IN ('clinician', 'staff')
        """
    )

    op.execute(
        r"""
        -- get_study: a case may have zero, one, or many imaging examinations.
        CREATE TABLE studies (
            study_id record_id PRIMARY KEY,
            case_id record_id NOT NULL REFERENCES cases(case_id),
            performed_at TIMESTAMPTZ,
            modality TEXT NOT NULL CHECK (modality IN ('XR', 'CT', 'MRI', 'US', 'OTHER')),
            body_part TEXT NOT NULL CHECK (char_length(btrim(body_part)) > 0),
            laterality TEXT NOT NULL CHECK (
                laterality IN ('left', 'right', 'bilateral', 'not_applicable', 'unknown')
            ),
            views TEXT[] NOT NULL DEFAULT '{}',
            acquisition_status TEXT NOT NULL CHECK (
                acquisition_status IN ('scheduled', 'completed', 'cancelled')
            ),
            image_ref TEXT,
            record_version INTEGER NOT NULL DEFAULT 1 CHECK (record_version >= 1),
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    op.execute(
        r"""
        CREATE INDEX studies_case_date_idx ON studies(case_id, performed_at, study_id)
        """
    )

    op.execute(
        r"""
        -- get_model_result: stored mock predictions, not calls to a model.
        CREATE TABLE model_results (
            model_result_id record_id PRIMARY KEY,
            study_id record_id NOT NULL REFERENCES studies(study_id),
            model_name TEXT NOT NULL CHECK (char_length(btrim(model_name)) > 0),
            model_version TEXT NOT NULL CHECK (char_length(btrim(model_version)) > 0),
            inference_status TEXT NOT NULL CHECK (
                inference_status IN ('pending', 'completed', 'failed')
            ),
            review_status TEXT NOT NULL DEFAULT 'unreviewed' CHECK (review_status = 'unreviewed'),
            generated_at TIMESTAMPTZ,
            record_status TEXT NOT NULL DEFAULT 'active' CHECK (
                record_status IN ('active', 'superseded', 'withdrawn', 'archived')
            ),
            is_current BOOLEAN NOT NULL DEFAULT FALSE,
            summary TEXT,
            score_description TEXT,
            limitations TEXT[] NOT NULL DEFAULT '{}',
            record_version INTEGER NOT NULL DEFAULT 1 CHECK (record_version >= 1),
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (inference_status <> 'completed' OR generated_at IS NOT NULL),
            CHECK (NOT is_current OR record_status = 'active')
        )
        """
    )

    op.execute(
        r"""
        CREATE INDEX model_results_study_idx ON model_results(study_id)
        """
    )

    op.execute(
        r"""
        CREATE UNIQUE INDEX one_current_model_result_per_study
            ON model_results(study_id) WHERE is_current
        """
    )

    op.execute(
        r"""
        CREATE TABLE model_findings (
            model_result_id record_id NOT NULL REFERENCES model_results(model_result_id),
            finding_index INTEGER NOT NULL CHECK (finding_index >= 0),
            label TEXT NOT NULL CHECK (char_length(btrim(label)) > 0),
            assessment TEXT NOT NULL CHECK (
                assessment IN ('suspected', 'not_detected', 'indeterminate')
            ),
            confidence DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 1),
            PRIMARY KEY (model_result_id, finding_index)
        )
        """
    )

    op.execute(
        r"""
        -- get_reviewed_report: each report_id identifies a version, not a mutable family.
        -- The Python writer must insert a new row for revised reviewed content. This
        -- initial schema does not install UPDATE/DELETE immutability triggers.
        CREATE TABLE reviewed_reports (
            report_id record_id PRIMARY KEY,
            study_id record_id NOT NULL REFERENCES studies(study_id),
            record_version INTEGER NOT NULL CHECK (record_version >= 1),
            review_status TEXT NOT NULL CHECK (review_status IN ('draft', 'reviewed')),
            record_status TEXT NOT NULL DEFAULT 'active' CHECK (
                record_status IN ('active', 'superseded', 'withdrawn', 'archived')
            ),
            is_current BOOLEAN NOT NULL DEFAULT FALSE,
            reviewed_by_clinician_id record_id REFERENCES clinicians(clinician_id),
            reviewed_at TIMESTAMPTZ,
            findings_text TEXT NOT NULL DEFAULT '',
            impression_text TEXT NOT NULL DEFAULT '',
            follow_up_recommendation TEXT,
            supersedes_report_id record_id,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (study_id, record_version),
            UNIQUE (report_id, study_id),
            FOREIGN KEY (supersedes_report_id, study_id)
                REFERENCES reviewed_reports(report_id, study_id),
            CHECK (supersedes_report_id IS NULL OR supersedes_report_id <> report_id),
            CHECK (
                (review_status = 'draft' AND reviewed_by_clinician_id IS NULL AND reviewed_at IS NULL)
                OR (review_status = 'reviewed' AND reviewed_by_clinician_id IS NOT NULL AND reviewed_at IS NOT NULL)
            ),
            CHECK (NOT is_current OR record_status = 'active')
        )
        """
    )

    op.execute(
        r"""
        CREATE INDEX reviewed_reports_study_idx ON reviewed_reports(study_id)
        """
    )

    op.execute(
        r"""
        CREATE UNIQUE INDEX one_current_report_per_study
            ON reviewed_reports(study_id) WHERE is_current
        """
    )

    op.execute(
        r"""
        -- A current draft is allowed: the tool must say NOT_REVIEWED, not fall back to
        -- a historical reviewed report. A draft can also coexist as non-current while
        -- a previously signed report remains current.
        
        CREATE FUNCTION valid_timezone(value TEXT)
        RETURNS BOOLEAN LANGUAGE SQL STABLE
        AS $$
            SELECT EXISTS (
                SELECT 1 FROM pg_catalog.pg_timezone_names WHERE name = value
            );
        $$
        """
    )

    op.execute(
        r"""
        -- get_appointments: appointment identity survives status changes; record_version
        -- must be incremented by the writer when a factual value changes.
        CREATE TABLE appointments (
            appointment_id record_id PRIMARY KEY,
            case_id record_id NOT NULL REFERENCES cases(case_id),
            appointment_type TEXT NOT NULL CHECK (
                appointment_type IN (
                    'initial_consultation', 'fracture_follow_up', 'imaging', 'physiotherapy', 'other'
                )
            ),
            starts_at TIMESTAMPTZ NOT NULL,
            ends_at TIMESTAMPTZ,
            timezone TEXT NOT NULL CHECK (valid_timezone(timezone)),
            status TEXT NOT NULL CHECK (
                status IN ('scheduled', 'confirmed', 'cancelled', 'completed', 'no_show')
            ),
            clinician_id record_id REFERENCES clinicians(clinician_id),
            location TEXT,
            notes TEXT,
            replaces_appointment_id record_id,
            record_version INTEGER NOT NULL DEFAULT 1 CHECK (record_version >= 1),
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (appointment_id, case_id),
            FOREIGN KEY (replaces_appointment_id, case_id)
                REFERENCES appointments(appointment_id, case_id),
            CHECK (replaces_appointment_id IS NULL OR replaces_appointment_id <> appointment_id),
            CHECK (ends_at IS NULL OR ends_at >= starts_at)
        )
        """
    )

    op.execute(
        r"""
        CREATE INDEX appointments_case_date_idx ON appointments(case_id, starts_at, appointment_id)
        """
    )

    op.execute(
        r"""
        -- search_clinic_instructions: document family -> published versions -> passages.
        -- Published content/chunks must be treated as immutable by the Python writer:
        -- insert a new version instead of changing text under an existing evidence ID.
        CREATE TABLE clinic_documents (
            document_id record_id PRIMARY KEY,
            slug TEXT NOT NULL UNIQUE CHECK (char_length(btrim(slug)) > 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    op.execute(
        r"""
        CREATE TABLE document_versions (
            document_id record_id NOT NULL REFERENCES clinic_documents(document_id),
            version TEXT NOT NULL CHECK (char_length(version) BETWEEN 1 AND 32),
            title TEXT NOT NULL CHECK (char_length(btrim(title)) > 0),
            full_text TEXT NOT NULL CHECK (char_length(btrim(full_text)) > 0),
            tags TEXT[] NOT NULL DEFAULT '{}',
            approval_status TEXT NOT NULL CHECK (approval_status IN ('draft', 'approved')),
            publication_status TEXT NOT NULL CHECK (
                publication_status IN ('draft', 'published', 'withdrawn')
            ),
            published_at TIMESTAMPTZ,
            effective_from TIMESTAMPTZ,
            effective_to TIMESTAMPTZ,
            is_current BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (document_id, version),
            CHECK (
                publication_status <> 'published'
                OR (approval_status = 'approved' AND published_at IS NOT NULL AND effective_from IS NOT NULL)
            ),
            CHECK (
                NOT is_current
                OR (publication_status = 'published' AND approval_status = 'approved')
            ),
            CHECK (effective_to IS NULL OR (effective_from IS NOT NULL AND effective_to > effective_from)),
            CHECK (published_at IS NULL OR effective_from IS NULL OR published_at <= effective_from)
        )
        """
    )

    op.execute(
        r"""
        CREATE UNIQUE INDEX one_current_version_per_document
            ON document_versions(document_id) WHERE is_current
        """
    )

    op.execute(
        r"""
        CREATE TABLE document_chunks (
            chunk_id record_id PRIMARY KEY,
            document_id record_id NOT NULL,
            document_version TEXT NOT NULL,
            chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
            section TEXT NOT NULL,
            content TEXT NOT NULL CHECK (char_length(btrim(content)) > 0),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id, document_version)
                REFERENCES document_versions(document_id, version),
            UNIQUE (document_id, document_version, chunk_index)
        )
        """
    )

    op.execute(
        r"""
        -- Optional keyword support if hybrid search is selected later. No LLM/API usage.
        CREATE INDEX document_chunks_text_search_idx
            ON document_chunks USING GIN (to_tsvector('english', content))
        """
    )

    op.execute(
        r"""
        -- These tables remain EMPTY until an embedding model/configuration is agreed.
        -- One index_version fixes the model, dimensions, chunking and relevance threshold.
        CREATE TABLE retrieval_indexes (
            index_version TEXT PRIMARY KEY CHECK (char_length(btrim(index_version)) > 0),
            embedding_model TEXT NOT NULL CHECK (char_length(btrim(embedding_model)) > 0),
            embedding_dimensions INTEGER NOT NULL CHECK (embedding_dimensions BETWEEN 1 AND 16000),
            chunking_version TEXT NOT NULL,
            distance_metric TEXT NOT NULL DEFAULT 'cosine' CHECK (distance_metric = 'cosine'),
            min_similarity DOUBLE PRECISION NOT NULL CHECK (min_similarity BETWEEN -1 AND 1),
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (index_version, embedding_dimensions)
        )
        """
    )

    op.execute(
        r"""
        CREATE TABLE chunk_embeddings (
            index_version TEXT NOT NULL,
            chunk_id record_id NOT NULL REFERENCES document_chunks(chunk_id),
            embedding public.vector NOT NULL,
            embedding_dimensions INTEGER GENERATED ALWAYS AS (public.vector_dims(embedding)) STORED,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (index_version, chunk_id),
            FOREIGN KEY (index_version, embedding_dimensions)
                REFERENCES retrieval_indexes(index_version, embedding_dimensions),
            CHECK (public.vector_norm(embedding) > 0)
        )
        """
    )


def downgrade() -> None:
    """Delete clinic tables and their data. Never use on data you need to retain.

    Leave the public pgvector extension installed because it may be shared.
    No CASCADE: unexpected external dependencies cause a safe transaction failure.
    """
    op.execute("DROP VIEW clinic.authorized_cases")
    op.execute("DROP TABLE clinic.chunk_embeddings")
    op.execute("DROP TABLE clinic.retrieval_indexes")
    op.execute("DROP TABLE clinic.document_chunks")
    op.execute("DROP TABLE clinic.document_versions")
    op.execute("DROP TABLE clinic.clinic_documents")
    op.execute("DROP TABLE clinic.appointments")
    op.execute("DROP FUNCTION clinic.valid_timezone(text)")
    op.execute("DROP TABLE clinic.reviewed_reports")
    op.execute("DROP TABLE clinic.model_findings")
    op.execute("DROP TABLE clinic.model_results")
    op.execute("DROP TABLE clinic.studies")
    op.execute("DROP TABLE clinic.cases")
    op.execute("DROP TABLE clinic.app_users")
    op.execute("DROP TABLE clinic.clinicians")
    op.execute("DROP TABLE clinic.patients")
    op.execute("DROP DOMAIN clinic.record_id")
    op.execute("DROP SCHEMA clinic")
