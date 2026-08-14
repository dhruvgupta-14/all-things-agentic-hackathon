from collections.abc import AsyncGenerator

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=False,
    connect_args={
        "server_settings": {
            # Postgres defaults this to 4.0, a spinning-disk figure. On SSD
            # (Cloud SQL, and any modern local disk) it makes an index scan
            # look ~4x more expensive than it is, and the planner answers
            # vector queries with a sequential scan instead of the HNSW index.
            # Measured on a 5 000-chunk corpus: 183ms seq scan vs 1ms HNSW.
            # Set per-connection so the fix travels with the application
            # rather than depending on server configuration being applied.
            "random_page_cost": "1.1",
        }
    },
)

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
