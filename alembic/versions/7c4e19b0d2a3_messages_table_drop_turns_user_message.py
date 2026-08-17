"""messages table; drop turns.user_message

Conversation history moves from ADK's session service into a table we own, so
it survives an instance being reclaimed and does not depend on the framework's
internal schema. `messages` becomes the single owner of conversation content,
which makes `turns.user_message` a duplicate — so it goes.

Revision ID: 7c4e19b0d2a3
Revises: 60a731298ee8
Create Date: 2026-08-15

"""
from collections.abc import Sequence

import pgvector.sqlalchemy  # noqa: F401 - keeps the Vector type importable
import sqlalchemy as sa
from alembic import op

revision: str = "7c4e19b0d2a3"
down_revision: str | Sequence[str] | None = "60a731298ee8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "messages",
        sa.Column(
            "message_id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("turn_id", sa.UUID(), nullable=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.SmallInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "char_length(content) BETWEEN 1 AND 32000",
            name=op.f("ck_messages_content_length"),
        ),
        sa.CheckConstraint("ordinal >= 0", name=op.f("ck_messages_ordinal_non_negative")),
        sa.CheckConstraint(
            "role IN ('user', 'assistant', 'summary')", name=op.f("ck_messages_role")
        ),
        sa.CheckConstraint(
            "token_count IS NULL OR token_count > 0",
            name=op.f("ck_messages_token_count_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name=op.f("fk_messages_session_id"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["turn_id"],
            ["turns.turn_id"],
            name=op.f("fk_messages_turn_id"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            name=op.f("fk_messages_user_id"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("message_id", name=op.f("pk_messages")),
        sa.UniqueConstraint(
            "session_id", "ordinal", name="uq_messages_session_id_ordinal"
        ),
    )
    # The retention sweep's only query.
    op.create_index("ix_messages_created_at", "messages", ["created_at"], unique=False)
    op.create_index(
        "ix_messages_user_id_created_at",
        "messages",
        ["user_id", sa.literal_column("created_at DESC")],
        unique=False,
    )

    # Immutable, not append-only. A transcript that can be rewritten is not a
    # transcript, so UPDATE is rejected — but DELETE has to remain possible or
    # the 30-day retention sweep and `ON DELETE CASCADE` from a deleted user
    # would both be blocked by their own safety net.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION reject_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'table % is immutable; content cannot be rewritten',
                TG_TABLE_NAME
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER messages_immutable
        BEFORE UPDATE ON messages
        FOR EACH ROW EXECUTE FUNCTION reject_update();
        """
    )

    # `messages` now owns conversation content outright. Keeping the column
    # would leave two places to look for what the reader said.
    op.drop_column("turns", "user_message")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "turns",
        sa.Column("user_message", sa.VARCHAR(length=8000), autoincrement=False, nullable=True),
    )
    op.execute("DROP TRIGGER IF EXISTS messages_immutable ON messages")
    op.execute("DROP FUNCTION IF EXISTS reject_update()")
    op.drop_index("ix_messages_user_id_created_at", table_name="messages")
    op.drop_index("ix_messages_created_at", table_name="messages")
    op.drop_table("messages")
