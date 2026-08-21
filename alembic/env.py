import os
from logging.config import fileConfig

from sqlalchemy import pool

from alembic import context
from app.config import get_settings
from app.db import models  # noqa: F401 — registers every table on Base.metadata
from app.db.base import Base

config = context.config

# An explicit DSN for the migration tool, and the one place in this repository
# that will talk to a database other than Cloud SQL.
#
# It exists for the test database. The suite runs against a throwaway Postgres
# and needs the same schema, and the schema is defined by these migrations —
# `create_all` would build the tables but none of the append-only triggers, so
# the harness would be testing a database the application never runs on.
#
# The application itself never reads this. It is not a second configuration
# mode; it is an argument to `alembic`.
ALEMBIC_URL_OVERRIDE = os.environ.get("ALEMBIC_DATABASE_URL")

# Otherwise the URL comes from settings, not alembic.ini, so migrations and the
# application can never disagree about which database they are pointed at.
config.set_main_option(
    "sqlalchemy.url", ALEMBIC_URL_OVERRIDE or get_settings().sync_database_url
)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    from sqlalchemy import create_engine

    if ALEMBIC_URL_OVERRIDE:
        # A plain DSN, dialled directly. The test database.
        connectable = create_engine(ALEMBIC_URL_OVERRIDE, poolclass=pool.NullPool)
    else:
        # Cloud SQL through the connector, which supplies the connection — so
        # there is no host in the URL to dial. The connector rather than Cloud
        # Run's unix socket because `alembic upgrade head` has to run *before*
        # the revision that needs it exists, from wherever the operator is.
        settings = get_settings()

        from app.db.cloud_sql import sync_creator

        connectable = create_engine(
            settings.sync_database_url,
            creator=sync_creator(settings),
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
