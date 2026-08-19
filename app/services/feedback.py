"""Explicit feedback that visibly changes the next turn (ARCHITECTURE 4.14, 19).

The named-track requirement is not "collect feedback" — it is that feedback
*changes behaviour*, and that the change is verifiable rather than asserted.
`feedback.applied_to_turn_id` is what makes it verifiable: the row records
which later turn was composed differently because of it, so "we listened" is a
join, not a claim.

Two kinds of feedback do different work:

  * `too_basic` / `too_advanced` / `style_preference` move a **standing
    preference** on `users.preferences`. They change every later turn until
    changed again.
  * `helpful` / `not_helpful` / `wrong` are a **record about one turn**. They
    are evidence, not a control, and they deliberately do not silently retune
    anything — a single thumbs-down is not a mandate to rewrite how someone is
    taught.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import EXPLANATION_STYLE, FEEDBACK_TYPE, Feedback, User

logger = logging.getLogger(__name__)

# `users.preferences.depth`, ordered. Feedback nudges one step at a time
# rather than jumping to an extreme: someone saying "too basic" once wants a
# notch up, not a research seminar.
DEPTH_LEVELS = ("introductory", "normal", "detailed", "expert")
DEFAULT_DEPTH = "normal"

MAX_COMMENT_CHARS = 2000


class FeedbackRejected(Exception):
    """Malformed feedback. Nothing is written."""


@dataclass(slots=True)
class RecordedFeedback:
    feedback_id: uuid.UUID
    feedback_type: str
    # Whether it moved a standing preference, as opposed to being recorded as
    # evidence about one turn.
    changed_preferences: bool
    depth: str


def _shift_depth(current: str, direction: int) -> str:
    try:
        index = DEPTH_LEVELS.index(current)
    except ValueError:
        index = DEPTH_LEVELS.index(DEFAULT_DEPTH)
    return DEPTH_LEVELS[max(0, min(len(DEPTH_LEVELS) - 1, index + direction))]


class FeedbackService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        user_id: uuid.UUID,
        feedback_type: str,
        target_turn_id: uuid.UUID | None = None,
        target_concept_id: uuid.UUID | None = None,
        comment: str | None = None,
        preferred_style: str | None = None,
    ) -> RecordedFeedback:
        if feedback_type not in FEEDBACK_TYPE:
            raise FeedbackRejected(f"unknown feedback type {feedback_type!r}")

        # The CHECK constraint requires exactly one target; refusing here gives
        # a usable error instead of an integrity violation.
        targets = [target_turn_id, target_concept_id]
        if sum(target is not None for target in targets) != 1:
            raise FeedbackRejected(
                "feedback must name exactly one of a turn or a concept"
            )

        user = await self._session.get(User, user_id)
        if user is None:  # pragma: no cover - the principal always exists
            raise FeedbackRejected("no such user")

        preferences = dict(user.preferences or {})
        depth = preferences.get("depth", DEFAULT_DEPTH)
        changed = False

        if feedback_type == "too_basic":
            depth = _shift_depth(depth, +1)
            preferences["depth"] = depth
            changed = True
        elif feedback_type == "too_advanced":
            depth = _shift_depth(depth, -1)
            preferences["depth"] = depth
            changed = True
        elif feedback_type == "style_preference":
            if preferred_style not in EXPLANATION_STYLE:
                raise FeedbackRejected(
                    "style_preference needs a style from the closed set"
                )
            preferences["preferred_style"] = preferred_style
            changed = True

        if changed:
            # Reassigned rather than mutated in place: SQLAlchemy does not see
            # in-place edits to a JSONB dict, and the update would be lost.
            user.preferences = preferences

        row = Feedback(
            user_id=user_id,
            feedback_type=feedback_type,
            target_turn_id=target_turn_id,
            target_concept_id=target_concept_id,
            comment=(comment or "").strip()[:MAX_COMMENT_CHARS] or None,
        )
        self._session.add(row)
        await self._session.flush()

        logger.info(
            "feedback recorded",
            extra={"feedback_type": feedback_type, "changed_preferences": changed},
        )
        return RecordedFeedback(
            feedback_id=row.feedback_id,
            feedback_type=feedback_type,
            changed_preferences=changed,
            depth=depth,
        )

    async def apply_pending(
        self, user_id: uuid.UUID, turn_id: uuid.UUID
    ) -> list[uuid.UUID]:
        """Stamp feedback that has not yet visibly changed anything.

        Called once the turn it influenced exists. Only the kinds that move a
        standing preference are stamped — those are the ones that actually
        changed this turn. A `not_helpful` on an earlier answer did not compose
        this one, and claiming it did would make `applied_to_turn_id` mean
        nothing.
        """
        pending = list(
            (
                await self._session.scalars(
                    select(Feedback).where(
                        Feedback.user_id == user_id,
                        Feedback.applied_to_turn_id.is_(None),
                        Feedback.feedback_type.in_(
                            ["too_basic", "too_advanced", "style_preference"]
                        ),
                    )
                )
            ).all()
        )
        for row in pending:
            row.applied_to_turn_id = turn_id
        return [row.feedback_id for row in pending]


def depth_instruction(preferences: dict | None) -> str | None:
    """How the standing preference reaches the agent.

    Returned as an instruction fragment rather than applied to the answer after
    the fact: rewriting a composed answer to be simpler is how you get an
    answer that no longer matches its citations.
    """
    preferences = preferences or {}
    depth = preferences.get("depth", DEFAULT_DEPTH)
    style = preferences.get("preferred_style")

    parts = []
    if depth == "introductory":
        parts.append(
            "This reader has asked for less depth. Build up from fundamentals "
            "and define terms as you use them."
        )
    elif depth == "detailed":
        parts.append(
            "This reader has asked for more depth. Do not simplify past the "
            "point of being useful, and keep the technical detail."
        )
    elif depth == "expert":
        parts.append(
            "This reader has asked for full technical depth. Assume fluency "
            "with the mathematics and skip the introductory framing entirely."
        )

    if style:
        parts.append(f"They have asked for {style} explanations where possible.")

    return " ".join(parts) if parts else None
