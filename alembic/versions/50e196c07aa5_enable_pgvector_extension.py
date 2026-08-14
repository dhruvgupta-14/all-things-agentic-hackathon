"""enable pgvector extension

Revision ID: 50e196c07aa5
Revises: 
Create Date: 2026-08-14 19:08:26.628883

"""
from typing import Sequence, Union

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '50e196c07aa5'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Every vector(768) column and every HNSW index in the next migration
    # depends on this. gen_random_uuid() is core in PostgreSQL 13+, so pgcrypto
    # is not required.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP EXTENSION IF EXISTS vector")
