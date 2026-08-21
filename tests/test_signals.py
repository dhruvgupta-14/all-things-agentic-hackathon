"""The only model-reachable write into learner memory.

`record_learning_signal` is the narrowest surface in the system, so these tests
are about what the model cannot do as much as what it can: it cannot set a
score, cannot name another reader, and cannot get a malformed signal past
validation. It also cannot stop memory accumulating by simply not calling —
that is what the backstop is for.
"""

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Concept, Observation, Session, Turn, User
from app.services.learner_state import SOURCE_WEIGHT, weight_for
from app.services.signals import SignalRejected, SignalService
from tests.fakes import HashingEmbedder


async def _user(session: AsyncSession) -> User:
    user = User(auth_subject=f"signal-test-{uuid.uuid4()}")
    session.add(user)
    await session.flush()
    return user


async def _turn(session: AsyncSession, user: User) -> Turn:
    conversation = Session(user_id=user.user_id)
    session.add(conversation)
    await session.flush()
    turn = Turn(session_id=conversation.session_id, user_id=user.user_id, ordinal=0)
    session.add(turn)
    await session.flush()
    return turn


def _service(session: AsyncSession) -> SignalService:
    return SignalService(session, embedder=HashingEmbedder())


# --------------------------------------------------------------------------
# Validation — what the model cannot get past
# --------------------------------------------------------------------------


async def test_an_unknown_signal_type_is_rejected(db_session: AsyncSession):
    user = await _user(db_session)

    with pytest.raises(SignalRejected):
        await _service(db_session).record(
            user_id=user.user_id,
            concept_name="ELBO",
            signal_type="they_seemed_delighted",
        )


async def test_an_empty_concept_name_is_rejected(db_session: AsyncSession):
    user = await _user(db_session)

    with pytest.raises(SignalRejected):
        await _service(db_session).record(
            user_id=user.user_id,
            concept_name="   \n\t ",
            signal_type="explicit_confusion",
        )


async def test_a_rejected_signal_writes_nothing(db_session: AsyncSession):
    user = await _user(db_session)

    with pytest.raises(SignalRejected):
        await _service(db_session).record(
            user_id=user.user_id, concept_name="ELBO", signal_type="nonsense"
        )

    written = await db_session.scalar(
        select(func.count())
        .select_from(Observation)
        .where(Observation.user_id == user.user_id)
    )
    assert written == 0


async def test_concept_names_are_sanitised_and_bounded(db_session: AsyncSession):
    """Names come from the model, so they are treated as hostile."""
    user = await _user(db_session)

    recorded = await _service(db_session).record(
        user_id=user.user_id,
        concept_name="  Varia\x00tional\n\n  lower   bound  " + "x" * 400,
        signal_type="explicit_confusion",
    )

    assert "\x00" not in recorded.concept_name
    assert "\n" not in recorded.concept_name
    assert len(recorded.concept_name) <= 200


async def test_an_unknown_style_is_dropped_not_fatal(db_session: AsyncSession):
    """The observation is still worth keeping; the CHECK would refuse the row."""
    user = await _user(db_session)

    recorded = await _service(db_session).record(
        user_id=user.user_id,
        concept_name="ELBO",
        signal_type="explicit_confusion",
        style_in_play="interpretive_dance",
    )

    observation = await db_session.get(Observation, recorded.observation_id)
    assert observation.style_in_play is None


# --------------------------------------------------------------------------
# The backend assigns the weight
# --------------------------------------------------------------------------


async def test_the_weight_comes_from_the_table_not_the_caller(
    db_session: AsyncSession,
):
    user = await _user(db_session)

    recorded = await _service(db_session).record(
        user_id=user.user_id,
        concept_name="ELBO",
        signal_type="explicit_confusion",
    )

    observation = await db_session.get(Observation, recorded.observation_id)
    assert observation.signal_source == "explicit"
    assert observation.weight == pytest.approx(
        weight_for("explicit_confusion", "explicit")
    )


async def test_the_signal_source_is_derived_not_supplied(db_session: AsyncSession):
    """The model says what happened; what kind of evidence that is follows."""
    user = await _user(db_session)
    service = _service(db_session)

    quiz = await service.record(
        user_id=user.user_id, concept_name="A", signal_type="quiz_correct"
    )
    hint = await service.record(
        user_id=user.user_id, concept_name="B", signal_type="implicit_confusion"
    )

    assert (await db_session.get(Observation, quiz.observation_id)).signal_source == "quiz"
    assert (
        await db_session.get(Observation, hint.observation_id)
    ).signal_source == "implicit"


# --------------------------------------------------------------------------
# Resolution pairing
# --------------------------------------------------------------------------


async def test_understanding_closes_the_open_struggle(db_session: AsyncSession):
    """ARCHITECTURE 10.1 — the pair is the valuable signal, linked by the
    backend rather than inferred later."""
    user = await _user(db_session)
    service = _service(db_session)

    struggle = await service.record(
        user_id=user.user_id,
        concept_name="ELBO",
        signal_type="explicit_confusion",
        style_in_play="formal",
    )
    resolution = await service.record(
        user_id=user.user_id,
        concept_name="ELBO",
        signal_type="explicit_understanding",
        style_in_play="numerical",
    )

    assert resolution.resolved_observation_id == struggle.observation_id

    concept = await db_session.get(Concept, resolution.concept_id)
    assert concept.effective_style == "numerical"


async def test_a_struggle_is_only_closed_once(db_session: AsyncSession):
    user = await _user(db_session)
    service = _service(db_session)

    await service.record(
        user_id=user.user_id, concept_name="ELBO", signal_type="explicit_confusion"
    )
    first = await service.record(
        user_id=user.user_id, concept_name="ELBO", signal_type="explicit_understanding"
    )
    second = await service.record(
        user_id=user.user_id, concept_name="ELBO", signal_type="explicit_understanding"
    )

    assert first.resolved_observation_id is not None
    assert second.resolved_observation_id is None


async def test_resolution_does_not_reach_across_concepts(db_session: AsyncSession):
    user = await _user(db_session)
    service = _service(db_session)

    await service.record(
        user_id=user.user_id, concept_name="ELBO", signal_type="explicit_confusion"
    )
    elsewhere = await service.record(
        user_id=user.user_id,
        concept_name="Forward Process",
        signal_type="explicit_understanding",
    )

    assert elsewhere.resolved_observation_id is None


async def test_resolution_does_not_reach_across_users(db_session: AsyncSession):
    mine = await _user(db_session)
    theirs = await _user(db_session)
    service = _service(db_session)

    await service.record(
        user_id=theirs.user_id, concept_name="ELBO", signal_type="explicit_confusion"
    )
    mine_resolution = await service.record(
        user_id=mine.user_id, concept_name="ELBO", signal_type="explicit_understanding"
    )

    assert mine_resolution.resolved_observation_id is None


# --------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------


async def test_the_same_concept_twice_is_one_row(db_session: AsyncSession):
    user = await _user(db_session)
    service = _service(db_session)

    first = await service.record(
        user_id=user.user_id, concept_name="ELBO", signal_type="explicit_confusion"
    )
    again = await service.record(
        user_id=user.user_id, concept_name="  elbo  ", signal_type="reinforcement"
    )

    assert first.concept_id == again.concept_id
    assert first.concept_created is True
    assert again.concept_created is False


async def test_concepts_are_created_per_user(db_session: AsyncSession):
    """Concepts are user-scoped, not a global ontology (ARCHITECTURE 4.9)."""
    mine = await _user(db_session)
    theirs = await _user(db_session)
    service = _service(db_session)

    a = await service.record(
        user_id=mine.user_id, concept_name="ELBO", signal_type="explicit_confusion"
    )
    b = await service.record(
        user_id=theirs.user_id, concept_name="ELBO", signal_type="explicit_confusion"
    )

    assert a.concept_id != b.concept_id


# --------------------------------------------------------------------------
# Buffering — provenance survives
# --------------------------------------------------------------------------


async def test_a_prepared_signal_writes_nothing_until_committed(
    db_session: AsyncSession,
):
    user = await _user(db_session)
    service = _service(db_session)

    pending = await service.prepare(
        user_id=user.user_id, concept_name="ELBO", signal_type="explicit_confusion"
    )

    observations = await db_session.scalar(
        select(func.count())
        .select_from(Observation)
        .where(Observation.user_id == user.user_id)
    )
    assert observations == 0
    assert pending.weight == pytest.approx(SOURCE_WEIGHT["explicit"])


async def test_the_projected_score_is_the_one_that_gets_committed(
    db_session: AsyncSession,
):
    """The agent is told a number; that number has to be the truth."""
    user = await _user(db_session)
    turn = await _turn(db_session, user)
    service = _service(db_session)

    pending = await service.prepare(
        user_id=user.user_id, concept_name="ELBO", signal_type="explicit_confusion"
    )
    recorded = await service.commit(
        pending,
        user_id=user.user_id,
        session_id=turn.session_id,
        turn_id=turn.turn_id,
    )

    assert recorded.state.raw_score == pytest.approx(pending.projected.raw_score)
    assert recorded.state.confidence == pytest.approx(pending.projected.confidence)


async def test_a_committed_signal_keeps_its_provenance(db_session: AsyncSession):
    """ARCHITECTURE 4.11 calls `turn_id` what makes memory inspectable."""
    user = await _user(db_session)
    turn = await _turn(db_session, user)
    service = _service(db_session)

    pending = await service.prepare(
        user_id=user.user_id, concept_name="ELBO", signal_type="explicit_confusion"
    )
    recorded = await service.commit(
        pending,
        user_id=user.user_id,
        session_id=turn.session_id,
        turn_id=turn.turn_id,
    )

    observation = await db_session.get(Observation, recorded.observation_id)
    assert observation.turn_id == turn.turn_id
    assert observation.session_id == turn.session_id


# --------------------------------------------------------------------------
# The backstop
# --------------------------------------------------------------------------


async def test_a_backstop_row_moves_the_clock_without_moving_the_score(
    db_session: AsyncSession,
):
    """It records that the concept came up, and claims nothing more."""
    user = await _user(db_session)
    service = _service(db_session)

    await service.record(
        user_id=user.user_id,
        concept_name="ELBO",
        signal_type="explicit_understanding",
    )
    concept_id = (
        await service.record(
            user_id=user.user_id, concept_name="ELBO", signal_type="reinforcement"
        )
    ).concept_id

    concept = await db_session.get(Concept, concept_id)
    assert concept.understanding_score == pytest.approx(1.0, abs=0.01)
    assert concept.evidence_count == 2
    assert concept.last_reinforced_at is not None
