"""The single write path into learner memory (ARCHITECTURE 14.2).

`record_learning_signal` is the only tool that writes anything the learner
model is built from, so this is the narrowest and most guarded surface in the
system. The split is absolute:

    the model decides   that a signal occurred, which concept, which type
                        from the closed enum, and which style was in play
    the backend decides the weight, the resolution pairing, the arithmetic,
                        the canonical concept, and every row written

The model cannot name a user, a session or a turn — those are closed over by
the tool, never parameters — and it cannot set a score. There is deliberately
no `update_concept_score`: scores are computed from evidence, and a tool that
wrote one would hand the model the one number that has to stay auditable.

**Canonicalization here is exact-match-or-create, with no similarity merge.**
That is not a shortcut. HANDOFF 7.6 records that no similarity threshold
separates a true synonym from a true relation — measured, `variational
inference`/`variational autoencoder` scores 0.9263 while `ELBO`/`evidence
lower bound` scores 0.8595 — so the auto-merge band was removed entirely and
§16.3 has exactly two deterministic outcomes. Merging is the batched
adjudication pass's job at ingest, where one model call covers a whole paper.
Doing it here would add a model call to every turn that records a signal, on
the strength of a threshold that is known not to work.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    EXPLANATION_STYLE,
    SIGNAL_TYPE,
    Concept,
    Observation,
)
from app.ingestion.concepts import normalize_name
from app.services import embeddings
from app.services.embeddings import Embedder
from app.services.learner_state import LearnerState, derive, recompute, weight_for

logger = logging.getLogger(__name__)

MAX_CONCEPT_NAME_CHARS = 200
MAX_NOTE_CHARS = 500

# Which signal_source a signal_type implies. The model reports *what happened*;
# how much that is worth follows from the kind of evidence it is, and is not
# something the model gets a say in.
SOURCE_FOR_SIGNAL: dict[str, str] = {
    "explicit_confusion": "explicit",
    "explicit_understanding": "explicit",
    "implicit_confusion": "implicit",
    "applied_correctly": "implicit",
    "quiz_correct": "quiz",
    "quiz_partial": "quiz",
    "quiz_incorrect": "quiz",
    "user_stated_known": "user_stated",
    "user_stated_unknown": "user_stated",
    "reinforcement": "system",
}

# Signals that mean "they have got it now". Recording one closes the most
# recent open struggle for that concept — the pair is the valuable signal
# (ARCHITECTURE 10.1), and pairing it deterministically is what makes
# `effective_style` mean "this is what worked" rather than "this was last".
_RESOLVING_SIGNALS = frozenset(
    {"explicit_understanding", "user_stated_known", "quiz_correct", "applied_correctly"}
)

# Signals that open one.
_STRUGGLE_SIGNALS = frozenset(
    {
        "explicit_confusion",
        "implicit_confusion",
        "user_stated_unknown",
        "quiz_incorrect",
        "quiz_partial",
    }
)


class SignalRejected(Exception):
    """The signal was malformed. Nothing is written, and the turn continues.

    A rejected signal must not fail the turn: the reader asked a question and
    is owed an answer, whatever the agent got wrong about bookkeeping.
    """


@dataclass(slots=True)
class PendingSignal:
    """A validated signal, waiting for the turn row it belongs to.

    Observations are append-only and carry an FK to `turns`, and the turn row
    is not written until step 11 — so a signal recorded during the agent loop
    could neither reference its turn nor be updated to reference it later.
    Buffering is what preserves `observations.turn_id`, which ARCHITECTURE 4.11
    calls the thing that makes memory inspectable.

    It also means a turn that fails writes no learner memory at all: evidence
    about the reader should come only from exchanges that actually happened.
    """

    concept_id: uuid.UUID
    concept_name: str
    signal_type: str
    signal_source: str
    weight: float
    style_in_play: str | None
    note: str | None
    resolves_observation_id: uuid.UUID | None
    quiz_attempt_id: uuid.UUID | None
    # What the score will be once this lands. Computed with the same pure
    # function the write path uses, so the agent is not told one number and
    # the database given another.
    projected: LearnerState
    concept_created: bool = False


@dataclass(slots=True)
class RecordedSignal:
    concept_id: uuid.UUID
    concept_name: str
    observation_id: uuid.UUID
    state: LearnerState
    resolved_observation_id: uuid.UUID | None = None
    concept_created: bool = False


def _sanitise_name(raw: str) -> str:
    """Concept names come from the model, so they are treated as hostile.

    Collapsed, length-bounded, and stripped of control characters — a name is
    a label, and anything that looks like markup or an instruction has no
    business becoming one.
    """
    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", raw or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:MAX_CONCEPT_NAME_CHARS]


class SignalService:
    def __init__(
        self, session: AsyncSession, *, embedder: Embedder | None = None
    ) -> None:
        self._session = session
        self._embedder = embedder or embeddings.get_embedder()

    async def _recall_or_create(
        self,
        user_id: uuid.UUID,
        name: str,
        *,
        paper_id: uuid.UUID | None,
    ) -> tuple[Concept, bool]:
        normalized = normalize_name(name)

        existing = await self._session.scalar(
            select(Concept)
            .where(
                Concept.user_id == user_id,
                Concept.merged_into_id.is_(None),
                or_(
                    Concept.normalized_name == normalized,
                    Concept.aliases.overlap([name]),
                ),
            )
            .limit(1)
        )
        if existing is not None:
            # A concept first met in conversation and later found in a paper
            # should remember both. Provenance only ever grows.
            if paper_id is not None and paper_id not in (existing.source_paper_ids or []):
                existing.source_paper_ids = [*(existing.source_paper_ids or []), paper_id]
            return existing, False

        concept = Concept(
            user_id=user_id,
            canonical_name=name,
            normalized_name=normalized,
            source_paper_ids=[paper_id] if paper_id else [],
            embedding=self._embedder.embed_query(name),
        )
        self._session.add(concept)
        await self._session.flush()
        return concept, True

    async def _open_struggle(
        self, user_id: uuid.UUID, concept_id: uuid.UUID
    ) -> Observation | None:
        """The most recent struggle for this concept that nothing has resolved.

        Deliberately narrow: one open struggle is closed by one resolution, so
        a reader who stumbled twice and understood once still has an open
        struggle on the record.
        """
        resolved = select(Observation.resolves_observation_id).where(
            Observation.user_id == user_id,
            Observation.resolves_observation_id.isnot(None),
        )
        return await self._session.scalar(
            select(Observation)
            .where(
                Observation.user_id == user_id,
                Observation.concept_id == concept_id,
                Observation.signal_type.in_(list(_STRUGGLE_SIGNALS)),
                Observation.observation_id.notin_(resolved),
            )
            .order_by(Observation.observed_at.desc())
            .limit(1)
        )

    async def prepare(
        self,
        *,
        user_id: uuid.UUID,
        concept_name: str,
        signal_type: str,
        paper_id: uuid.UUID | None = None,
        style_in_play: str | None = None,
        note: str | None = None,
        quiz_attempt_id: uuid.UUID | None = None,
    ) -> PendingSignal:
        """Validate, canonicalize, weigh and pair — everything but the write.

        The concept *is* written here if it is new, because the pending signal
        has to name one. A turn that then fails rolls the whole transaction
        back, so no orphan survives.
        """
        if signal_type not in SIGNAL_TYPE:
            raise SignalRejected(f"unknown signal type {signal_type!r}")

        name = _sanitise_name(concept_name)
        if not name:
            raise SignalRejected("concept name was empty after sanitisation")

        # An unknown style is dropped rather than rejected: the observation is
        # still worth keeping, and the CHECK constraint would refuse the row.
        style = style_in_play if style_in_play in EXPLANATION_STYLE else None

        concept, created = await self._recall_or_create(
            user_id, name, paper_id=paper_id
        )

        source = SOURCE_FOR_SIGNAL.get(signal_type, "implicit")
        weight = weight_for(signal_type, source)

        resolves: uuid.UUID | None = None
        if signal_type in _RESOLVING_SIGNALS:
            struggle = await self._open_struggle(user_id, concept.concept_id)
            resolves = struggle.observation_id if struggle else None

        return PendingSignal(
            concept_id=concept.concept_id,
            concept_name=concept.canonical_name,
            signal_type=signal_type,
            signal_source=source,
            weight=weight,
            style_in_play=style,
            note=(note or "").strip()[:MAX_NOTE_CHARS] or None,
            resolves_observation_id=resolves,
            quiz_attempt_id=quiz_attempt_id,
            projected=await self._project(concept.concept_id, signal_type, weight, resolves),
            concept_created=created,
        )

    async def _project(
        self,
        concept_id: uuid.UUID,
        signal_type: str,
        weight: float,
        resolves: uuid.UUID | None,
    ) -> LearnerState:
        """What the score becomes once the pending signal lands.

        Uses the same pure `derive` the write path uses, over the stored
        evidence plus one unsaved row, so the number handed to the agent is the
        number that will be committed.
        """
        existing = list(
            (
                await self._session.scalars(
                    select(Observation)
                    .where(Observation.concept_id == concept_id)
                    .order_by(Observation.observed_at)
                )
            ).all()
        )
        provisional = Observation(
            concept_id=concept_id,
            signal_type=signal_type,
            signal_source=SOURCE_FOR_SIGNAL.get(signal_type, "implicit"),
            weight=weight,
            resolves_observation_id=resolves,
            observed_at=datetime.now(UTC),
        )
        return derive([*existing, provisional])

    async def commit(
        self,
        pending: PendingSignal,
        *,
        user_id: uuid.UUID,
        session_id: uuid.UUID | None,
        turn_id: uuid.UUID | None,
    ) -> RecordedSignal:
        """Write the buffered signal now that its turn exists."""
        observation = Observation(
            user_id=user_id,
            concept_id=pending.concept_id,
            session_id=session_id,
            turn_id=turn_id,
            signal_type=pending.signal_type,
            signal_source=pending.signal_source,
            weight=pending.weight,
            style_in_play=pending.style_in_play,
            resolves_observation_id=pending.resolves_observation_id,
            quiz_attempt_id=pending.quiz_attempt_id,
            note=pending.note,
        )
        self._session.add(observation)
        await self._session.flush()

        state = await recompute(self._session, pending.concept_id)
        await self._session.flush()

        logger.info(
            "learning signal recorded",
            extra={
                "concept_id": str(pending.concept_id),
                "signal_type": pending.signal_type,
                "weight": pending.weight,
                "resolved": bool(pending.resolves_observation_id),
            },
        )

        return RecordedSignal(
            concept_id=pending.concept_id,
            concept_name=pending.concept_name,
            observation_id=observation.observation_id,
            state=state,
            resolved_observation_id=pending.resolves_observation_id,
            concept_created=pending.concept_created,
        )

    async def record(
        self,
        *,
        user_id: uuid.UUID,
        concept_name: str,
        signal_type: str,
        session_id: uuid.UUID | None = None,
        turn_id: uuid.UUID | None = None,
        paper_id: uuid.UUID | None = None,
        style_in_play: str | None = None,
        note: str | None = None,
        quiz_attempt_id: uuid.UUID | None = None,
    ) -> RecordedSignal:
        """Prepare and commit in one step, for callers that already have a turn."""
        pending = await self.prepare(
            user_id=user_id,
            concept_name=concept_name,
            signal_type=signal_type,
            paper_id=paper_id,
            style_in_play=style_in_play,
            note=note,
            quiz_attempt_id=quiz_attempt_id,
        )
        return await self.commit(
            pending, user_id=user_id, session_id=session_id, turn_id=turn_id
        )
