from collections.abc import AsyncGenerator

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

# Connection settings the application depends on for correct query plans.
# Exported so the test harness can bind an engine that behaves identically — a
# suite on default settings exercises neither of these, which is exactly how
# the HNSW post-filter below went unnoticed.
SERVER_SETTINGS = {
    # Postgres defaults this to 4.0, a spinning-disk figure. On SSD
    # (Cloud SQL, and any modern local disk) it makes an index scan
    # look ~4x more expensive than it is, and the planner answers
    # vector queries with a sequential scan instead of the HNSW index.
    # Measured on a 5 000-chunk corpus: 183ms seq scan vs 1ms HNSW.
    # Set per-connection so the fix travels with the application
    # rather than depending on server configuration being applied.
    "random_page_cost": "1.1",
    # HNSW post-filters: it walks the graph for `ef_search` (40)
    # nearest candidates and only then applies the WHERE clause. Every
    # vector query here is filtered by something highly selective —
    # `user_id` on concepts, `paper_id` on chunks — so once the table
    # holds more than a few thousand rows the user's own nearest row
    # can fall outside those 40 and the query returns *nothing*. That
    # fails silently: canonicalization reads "no similar concept" and
    # creates a duplicate instead of adjudicating, so the cross-paper
    # edge the callback depends on is never written.
    # Measured on this schema: 4 misses in 6 probes at ~3 000 rows,
    # 0 in 6 with iterative scans on.
    # `strict_order` rather than `relaxed_order` because the
    # similarity value decides a threshold band, not just an order.
    "hnsw.iterative_scan": "strict_order",
}

def _build_engine():
    """One engine, reaching the database whichever way this deployment does.

    `SERVER_SETTINGS` travels either way — those settings are the difference
    between a correct query plan and a silently wrong one, and they must not
    depend on how the connection was established.
    """
    if settings.uses_cloud_sql:
        from app.db.cloud_sql import async_creator

        return create_async_engine(
            settings.database_url,
            echo=False,
            async_creator=async_creator(settings),
            connect_args={"server_settings": SERVER_SETTINGS},
            # Cloud SQL closes idle connections, and a pooled-but-dead one
            # surfaces as a failed turn rather than a reconnect.
            pool_pre_ping=True,
            pool_recycle=1800,
        )

    return create_async_engine(
        settings.database_url,
        echo=False,
        connect_args={"server_settings": SERVER_SETTINGS},
    )


engine = _build_engine()

async_session_factory = async_sessionmaker(engine, expire_on_commit=False)

# Without this, Alembic autogenerate emits unnamed constraints that later
# migrations cannot reliably drop.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
