"""Check connectivity and empty schema setup. Does not seed or alter the database."""

from pathlib import Path
import os

import psycopg
from alembic.config import Config
from alembic.script import ScriptDirectory
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TABLES = {
    "patients", "clinicians", "app_users", "cases",
    "studies", "model_results", "model_findings", "reviewed_reports", "appointments",
    "clinic_documents", "document_versions", "document_chunks", "retrieval_indexes",
    "chunk_embeddings",
    "booking_guard", "booking_clinics", "booking_sessions", "booking_slots",
    "booking_drafts", "booking_offers", "booking_holds", "booking_allocations",
    "booking_reschedules",
}


def main() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        raise SystemExit("Missing .env. Run uv run python scripts/setup_env.py first.")
    load_dotenv(env_path)
    expected_heads = set(
        ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini"))).get_heads()
    )
    required = ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")
    if any(not os.getenv(key) for key in required):
        raise SystemExit("Required local database settings are missing. Check .env privately.")

    try:
        port = int(os.getenv("DB_PORT", "5432"))
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise SystemExit("DB_PORT must be an integer from 1 to 65535.") from None

    try:
        with psycopg.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=port,
            dbname=os.environ["POSTGRES_DB"],
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            connect_timeout=5,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                extension = cursor.fetchone()
                if extension is None:
                    raise SystemExit("pgvector is not enabled. Run alembic upgrade head.")
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'clinic' AND table_type = 'BASE TABLE'"
                )
                actual_tables = {row[0] for row in cursor.fetchall()}
                missing = EXPECTED_TABLES - actual_tables
                if missing:
                    raise SystemExit("Missing schema tables: " + ", ".join(sorted(missing)))
                cursor.execute("SELECT to_regclass('public.alembic_version') IS NOT NULL")
                if not cursor.fetchone()[0]:
                    raise SystemExit("Alembic revision tracking is missing. Run alembic upgrade head.")
                cursor.execute("SELECT version_num FROM public.alembic_version ORDER BY version_num")
                versions = {row[0] for row in cursor.fetchall()}
                if versions != expected_heads:
                    raise SystemExit("The database is not at the latest packaged Alembic revision. Run alembic upgrade head.")
                if "case_access" in actual_tables or "schema_versions" in actual_tables:
                    raise SystemExit("Obsolete draft-schema tables are present; this package expects the new Alembic-managed schema.")
                cursor.execute(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                    "WHERE table_schema = 'clinic' AND table_name = 'app_users' "
                    "AND column_name = 'patient_id')"
                )
                if cursor.fetchone()[0]:
                    raise SystemExit("The old patient-account link is still present.")
                cursor.execute("SELECT to_regclass('clinic.authorized_cases') IS NOT NULL")
                if not cursor.fetchone()[0]:
                    raise SystemExit("The authorized_cases view is missing.")
                cursor.execute("SELECT count(*) FROM clinic.patients")
                patient_count = cursor.fetchone()[0]

        print("Database connection: OK")
        print(f"pgvector extension: {extension[0]}")
        print("Alembic revision: " + ", ".join(sorted(versions)))
        print(f"Expected clinic tables present: {len(EXPECTED_TABLES)}")
        print(f"Patient records: {patient_count}")
    except psycopg.Error:
        raise SystemExit(
            "Database connection/query failed. Check Docker status, Alembic migration status "
            "and local settings. Credentials have not been printed."
        ) from None


if __name__ == "__main__":
    main()
