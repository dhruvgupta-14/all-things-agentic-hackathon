"""Full deletion of one user's data (ARCHITECTURE 19, and the 4.7 conflict).

`turns`, `observations` and `quiz_attempts` are append-only by trigger, so the
`ON DELETE CASCADE` each declares on `user_id` cannot fire on its own. This
module is the only thing that opens that door, and it opens it for exactly one
statement's worth of work.

The flag is transaction-local: it is set with `set_config(..., is_local => true)`
so it dies with the surrounding transaction whether that transaction commits or
rolls back, and it is cleared explicitly once the delete has run. There is no
path by which an ordinary request leaves it on.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User

logger = logging.getLogger(__name__)

ERASURE_SETTING = "app.erasure"


async def erase_user(session: AsyncSession, user_id: uuid.UUID) -> bool:
    """Delete a user and everything that cascades from them.

    Returns whether a row was actually removed. The caller owns the
    transaction: nothing here commits, so an erasure that must be audited
    elsewhere can be written in the same transaction and either both land or
    neither does.
    """
    # Deliberately not a bare `SET LOCAL`: set_config's third argument is the
    # is_local flag, which keeps this bound to the current transaction.
    await session.execute(
        text("SELECT set_config(:name, 'on', true)"), {"name": ERASURE_SETTING}
    )
    try:
        result = await session.execute(delete(User).where(User.user_id == user_id))
    finally:
        # The transaction would drop the setting anyway; clearing it here means
        # the window is one statement wide rather than however long the caller
        # holds the transaction open afterwards.
        await session.execute(
            text("SELECT set_config(:name, 'off', true)"), {"name": ERASURE_SETTING}
        )

    erased = bool(result.rowcount)
    if erased:
        logger.info("erased all data for user %s", user_id)
    return erased
