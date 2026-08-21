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


# A `Connector` binds to the event loop it was built on, and refuses a call
# from any other with `ConnectorLoopError`. One process-wide singleton is
# therefore wrong: a script that calls `asyncio.run()` twice gets a second loop
# and the cached connector rejects it. Keyed by loop instead, so each loop gets
# its own and the common case — one long-lived server loop — still builds one.
_async_connectors: dict[int, object] = {}


def _async_connector():
    """The connector bound to the running loop, built on first use."""
    import asyncio

    from google.cloud.sql.connector import Connector

    loop = asyncio.get_running_loop()
    connector = _async_connectors.get(id(loop))
    if connector is None:
        logger.info("opening Cloud SQL connector for this event loop")
        connector = Connector(loop=loop)
        _async_connectors[id(loop)] = connector
    return connector


@lru_cache(maxsize=1)
def _sync_connector():
    """The connector for Alembic's synchronous path.

    Separate from the async ones on purpose: this one runs its own background
    loop internally, and sharing it with a caller's loop is what produces the
    mismatch above.
    """
    from google.cloud.sql.connector import Connector

    logger.info("opening Cloud SQL connector (sync)")
    return Connector()


def _ip_type(settings: Settings):
    from google.cloud.sql.connector import IPTypes

    return IPTypes.PRIVATE if settings.cloud_sql_ip_type.upper() == "PRIVATE" else IPTypes.PUBLIC


def async_creator(settings: Settings, server_settings: dict[str, str]):
    """An `async_creator` for `create_async_engine`, over asyncpg.

    `server_settings` is passed **here**, not through the engine's
    `connect_args`. SQLAlchemy hands connection-making entirely to a creator
    and ignores `connect_args` when one is supplied, so settings left there are
    silently dropped — which is how `hnsw.iterative_scan` came back as `off` on
    the first Cloud SQL instance despite being configured. That setting is the
    difference between a filtered vector query returning rows and returning
    nothing at all, and it fails without an error.
    """

    async def connect():
        return await _async_connector().connect_async(
            settings.cloud_sql_instance,
            "asyncpg",
            user=settings.db_user,
            password=settings.db_password,
            db=settings.db_name,
            ip_type=_ip_type(settings),
            server_settings=dict(server_settings),
        )

    return connect


def sync_creator(settings: Settings):
    """A `creator` for `create_engine`, over pg8000. Alembic's path.

    pg8000 rather than psycopg: the connector's psycopg driver connects over a
    unix domain socket, which Windows does not support, and migrations are run
    from whatever laptop the operator happens to have. pg8000 uses TCP.
    """

    def connect():
        return _sync_connector().connect(
            settings.cloud_sql_instance,
            "pg8000",
            user=settings.db_user,
            password=settings.db_password,
            db=settings.db_name,
            ip_type=_ip_type(settings),
        )

    return connect


def close() -> None:
    """Release every connector and its certificate-refresh loop."""
    for connector in _async_connectors.values():
        connector.close()
    _async_connectors.clear()
    if _sync_connector.cache_info().currsize:
        _sync_connector().close()
        _sync_connector.cache_clear()
