"""Cloud SQL over the Python connector (ARCHITECTURE 19: one managed database).

Two engines need the same instance: the application's async one on asyncpg, and
Alembic's synchronous one on psycopg. Both are built here so the connector is
created once and the driver choice is the only difference between them.

Why the connector rather than the unix socket Cloud Run also offers: the socket
only exists inside a Cloud Run revision, and `alembic upgrade head` has to run
*before* the revision that needs it is deployed. The connector works from a
laptop and from Cloud Run, so the migration step and the running service reach
the database the same way.

Nothing here is imported unless `CLOUD_SQL_INSTANCE` is set, so a local run
never constructs a connector and never needs credentials.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import Settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _connector():
    """One connector per process.

    It owns a background refresh loop for the instance's ephemeral
    certificates; building a second would duplicate that work and the
    certificates along with it.
    """
    from google.cloud.sql.connector import Connector

    logger.info("opening Cloud SQL connector")
    return Connector()


def _ip_type(settings: Settings):
    from google.cloud.sql.connector import IPTypes

    return IPTypes.PRIVATE if settings.cloud_sql_ip_type.upper() == "PRIVATE" else IPTypes.PUBLIC


def async_creator(settings: Settings):
    """An `async_creator` for `create_async_engine`, over asyncpg."""

    async def connect():
        return await _connector().connect_async(
            settings.cloud_sql_instance,
            "asyncpg",
            user=settings.db_user,
            password=settings.db_password,
            db=settings.db_name,
            ip_type=_ip_type(settings),
        )

    return connect


def sync_creator(settings: Settings):
    """A `creator` for `create_engine`, over psycopg. Alembic's path."""

    def connect():
        return _connector().connect(
            settings.cloud_sql_instance,
            "psycopg",
            user=settings.db_user,
            password=settings.db_password,
            db=settings.db_name,
            ip_type=_ip_type(settings),
        )

    return connect


def close() -> None:
    """Release the connector and its refresh loop."""
    if _connector.cache_info().currsize:
        _connector().close()
        _connector.cache_clear()
