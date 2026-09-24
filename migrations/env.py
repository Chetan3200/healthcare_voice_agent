"""Alembic environment for PostgreSQL/psycopg.

Online migration uses private .env settings. Offline SQL generation needs no
connection or credentials. There are no ORM models/autogeneration yet.
"""

from logging.config import fileConfig
from pathlib import Path
import os

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, pool


config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

ROOT = Path(__file__).resolve().parents[1]
target_metadata = None


def database_url() -> URL:
    env_path = ROOT / ".env"
    if not env_path.exists():
        raise RuntimeError("Missing .env. Run uv run python scripts/setup_env.py first.")
    load_dotenv(env_path)
    required = ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")
    if any(not os.getenv(key) for key in required):
        raise RuntimeError("Required database settings are missing. Check .env privately.")
    try:
        port = int(os.getenv("DB_PORT", "5432"))
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise RuntimeError("DB_PORT must be an integer from 1 to 65535.") from None

    return URL.create(
        drivername="postgresql+psycopg",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=port,
        database=os.environ["POSTGRES_DB"],
    )


def run_migrations_offline() -> None:
    context.configure(
        dialect_name="postgresql",
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table="alembic_version",
        version_table_schema="public",
        transactional_ddl=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(
        database_url(),
        poolclass=pool.NullPool,
        connect_args={"connect_timeout": 5},
        hide_parameters=True,
    )
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                version_table="alembic_version",
                version_table_schema="public",
                transactional_ddl=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
