"""append-only tables admit a deliberate erasure path

Revision ID: 9a1f4c2b7e35
Revises: 7c4e19b0d2a3
Create Date: 2026-08-19

`turns`, `observations` and `quiz_attempts` each declare `ON DELETE CASCADE`
on `user_id` *and* carry a `BEFORE UPDATE OR DELETE` append-only trigger. Both
cannot hold: the cascade fires a DELETE, the trigger rejects it, and a user who
has ever taken a turn cannot be deleted at all.

    DELETE FROM users WHERE auth_subject='local-dev-user';
    ERROR:  table turns is append-only

The resolution keeps ARCHITECTURE 4.7's mandate — "BEFORE UPDATE OR DELETE
trigger raises" — for every ordinary path, and opens exactly one deliberate
door. A DELETE succeeds only inside a transaction that has explicitly set
`app.erasure`; UPDATE is refused unconditionally, always.

Why not mirror `messages` (block UPDATE, permit DELETE), which is the other
obvious option: `messages` is a transcript, and its own migration explains that
DELETE stays open because the *routine* 30-day retention sweep needs it. These
three tables are the audit trail the learner model is replayed from, and
nothing routine deletes from them. Dropping the DELETE block outright would
trade a mandated guarantee for a capability used once, by an operator, on
purpose — so the block stays and erasure asks for it by name.

That also matches how the original migration described the design: grants are
the primary control ("the app role holds only SELECT, INSERT") and the trigger
is what survives a role misconfiguration. A trigger that additionally blocks
the privileged path enforces more than the specification asked for, and the
declared cascade is what pays for it.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "9a1f4c2b7e35"
down_revision: str | None = "7c4e19b0d2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# `current_setting(..., true)` returns NULL rather than raising when the
# setting was never assigned, so the common case — nobody asked for erasure —
# falls straight through to the RAISE.
_GUARDED = """
CREATE OR REPLACE FUNCTION reject_mutation() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE'
       AND current_setting('app.erasure', true) = 'on' THEN
        RETURN OLD;
    END IF;

    RAISE EXCEPTION 'table % is append-only', TG_TABLE_NAME
        USING ERRCODE = 'restrict_violation';
END;
$$ LANGUAGE plpgsql;
"""

_UNGUARDED = """
CREATE OR REPLACE FUNCTION reject_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'table % is append-only', TG_TABLE_NAME
        USING ERRCODE = 'restrict_violation';
END;
$$ LANGUAGE plpgsql;
"""


def upgrade() -> None:
    # Replacing the function is enough: all three triggers already call it.
    op.execute(_GUARDED)


def downgrade() -> None:
    op.execute(_UNGUARDED)
