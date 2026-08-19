"""The scoring constants, and the two narratives the architecture works out.

HANDOFF 5.2 left the weight table, the score formula, the decay and the
callback gap undecided. They are decided in `app/services/learner_state.py`,
and the two examples the architecture document works end to end are the
calibration: if a constant moves, one of these fails and the document and the
code have to be reconciled deliberately.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Concept, Observation, Session, Turn, User
from app.services.learner_state import (
    CALLBACK_MIN_TURN_GAP,
    CONFIDENCE_FLOOR,
    SIGNAL_TARGETS,
    SOURCE_WEIGHT,
    LearnerState,
    confidence_for,
    decay_factor,
    derive,
    is_callback_candidate,
    recompute,
    target_for,
    weight_for,
)

BASE = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _observation(
    signal_type: str,
    signal_source: str,
    *,
    style: str | None = None,
    resolves: uuid.UUID | None = None,
    at: datetime = BASE,
) -> Observation:
    """An unsaved Observation — `derive` is pure and needs no database."""
    return Observation(
        observation_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        concept_id=uuid.uuid4(),
        signal_type=signal_type,
        signal_source=signal_source,
        weight=weight_for(signal_type, signal_source),
        style_in_play=style,
        resolves_observation_id=resolves,
        observed_at=at,
    )


# --------------------------------------------------------------------------
# The documented narratives
# --------------------------------------------------------------------------


def test_architecture_10_1_struggle_resolved_by_a_numerical_example():
    """ARCHITECTURE 10.1 step 6: score 0.35, confidence 0.7, style numerical.

    The reader is confused while a concept is explained formally, the system
    re-explains numerically, and they understand. These exact numbers appear
    in the document's own diagram.
    """
    struggle = _observation("explicit_confusion", "explicit", style="formal")
    resolution = _observation(
        "explicit_understanding",
        "explicit",
        style="numerical",
        resolves=struggle.observation_id,
        at=BASE + timedelta(minutes=2),
    )

    state = derive([struggle, resolution])

    assert state.raw_score == pytest.approx(0.35, abs=0.005)
    assert state.confidence == pytest.approx(0.70, abs=0.005)
    assert state.effective_style == "numerical"
    assert state.evidence_count == 2


def test_architecture_12_the_same_concept_read_days_later():
    """ARCHITECTURE 12: decayed score 0.31, confidence 0.72 — still trusted.

    Confidence must NOT decay with the score. If it did, a concept would fall
    below the confidence floor at roughly the same time its score became low
    enough to be worth mentioning, and the cross-paper callback could never
    fire on exactly the stale concepts it exists to surface.
    """
    struggle = _observation("explicit_confusion", "explicit", style="formal")
    resolution = _observation(
        "explicit_understanding",
        "explicit",
        style="numerical",
        resolves=struggle.observation_id,
    )
    state = derive([struggle, resolution])

    later = BASE + timedelta(days=5.3)
    decayed = state.score_at(later)

    assert decayed == pytest.approx(0.31, abs=0.005)
    assert state.confidence >= CONFIDENCE_FLOOR
    assert is_callback_candidate(decayed, state.confidence)


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------


def test_quiz_is_the_highest_weight_class():
    """ARCHITECTURE 11 step 8 says so in as many words."""
    assert SOURCE_WEIGHT["quiz"] == max(SOURCE_WEIGHT.values())
    assert weight_for("quiz_correct", "quiz") > weight_for(
        "explicit_understanding", "explicit"
    )


def test_inferred_signals_weigh_less_than_stated_ones():
    assert weight_for("implicit_confusion", "implicit") < weight_for(
        "explicit_confusion", "explicit"
    )


def test_applying_a_concept_outweighs_a_vague_hint():
    """Both are inferred; only one is a demonstration."""
    assert weight_for("applied_correctly", "implicit") > weight_for(
        "implicit_confusion", "implicit"
    )


def test_every_signal_type_has_a_target_and_a_usable_weight():
    from app.db.models import SIGNAL_SOURCE, SIGNAL_TYPE

    for signal_type in SIGNAL_TYPE:
        assert signal_type in SIGNAL_TARGETS, signal_type
        assert 0.0 <= SIGNAL_TARGETS[signal_type] <= 1.0

    # Total over the whole enum product: nonsense pairs must not raise, since
    # the schema permits them and a crash here would fail a turn.
    for signal_type in SIGNAL_TYPE:
        for source in SIGNAL_SOURCE:
            assert 0.0 <= weight_for(signal_type, source) <= 1.0

    # Every source that represents an actual observation carries real weight.
    for source in SIGNAL_SOURCE:
        if source != "system":
            assert weight_for("explicit_understanding", source) > 0.0


def test_the_backstop_cannot_manufacture_confidence():
    """The reinforcement backstop fires when nothing was actually observed.

    Volume of non-evidence must not become evidence. At any positive weight it
    would: three backstop rows would sit exactly on the 0.3 confidence floor
    and ten would clear it comfortably, so a concept nobody ever demonstrated
    anything about would be reported with confidence.
    """
    state = derive([_observation("reinforcement", "system") for _ in range(10)])

    assert state.raw_score is None, "exposure is not a claim about understanding"
    assert state.confidence == 0.0
    assert not is_callback_candidate(state.raw_score, state.confidence)


def test_the_backstop_still_resets_the_decay_clock():
    """Which is what "reinforcement" means, and why the row is worth writing.

    It carries no evidentiary weight, but it records that the concept came up
    again — so the score it already had stops decaying from that moment.
    """
    revisited = BASE + timedelta(days=40)
    state = derive(
        [
            _observation("explicit_understanding", "explicit", at=BASE),
            _observation("reinforcement", "system", at=revisited),
        ]
    )

    assert state.last_reinforced_at == revisited
    assert state.evidence_count == 2
    # The understanding signal alone still sets the score.
    assert state.raw_score == pytest.approx(1.0, abs=0.005)
    assert state.score_at(revisited) == pytest.approx(1.0, abs=0.005)


# --------------------------------------------------------------------------
# Score arithmetic
# --------------------------------------------------------------------------


def test_assisted_understanding_scores_below_unassisted():
    """ARCHITECTURE 10.1: the resolution is the signal, not an erasure.

    Understanding that arrived because the system re-explained must not score
    the same as understanding that was there all along.
    """
    prior = uuid.uuid4()
    assert target_for("explicit_understanding", resolves_prior=True) < target_for(
        "explicit_understanding"
    )

    unassisted = derive([_observation("explicit_understanding", "explicit")])
    assisted = derive(
        [_observation("explicit_understanding", "explicit", resolves=prior)]
    )
    assert assisted.raw_score < unassisted.raw_score


def test_the_score_is_order_independent():
    """Replay must be reproducible — the whole point of an evidence trail."""
    observations = [
        _observation("quiz_incorrect", "quiz", at=BASE),
        _observation("explicit_understanding", "explicit", at=BASE + timedelta(days=1)),
        _observation("implicit_confusion", "implicit", at=BASE + timedelta(days=2)),
    ]

    forward = derive(observations)
    backward = derive(list(reversed(observations)))

    assert forward.raw_score == pytest.approx(backward.raw_score)
    assert forward.confidence == pytest.approx(backward.confidence)


def test_no_evidence_is_not_a_score_of_zero():
    """Nothing observed must read as unknown, not as "does not understand"."""
    state = derive([])

    assert state.raw_score is None
    assert state.confidence == 0.0
    assert state.score_at(BASE) is None
    assert not is_callback_candidate(state.raw_score, state.confidence)


def test_confidence_saturates_rather_than_growing_without_bound():
    ten = confidence_for(10.0)
    hundred = confidence_for(100.0)

    assert 0.0 < confidence_for(0.8) < ten < hundred < 1.0


# --------------------------------------------------------------------------
# Decay
# --------------------------------------------------------------------------


def test_decay_is_monotonic_and_floored():
    fresh = decay_factor(BASE, BASE)
    a_month = decay_factor(BASE, BASE + timedelta(days=30))
    a_year = decay_factor(BASE, BASE + timedelta(days=365))

    assert fresh == 1.0
    assert a_month == pytest.approx(0.5, abs=0.01)
    assert a_year < a_month
    # Floored, so months-old concepts stay ordered by what was observed rather
    # than all collapsing to zero and becoming equally good callback targets.
    assert a_year >= 0.25


def test_decay_never_amplifies_a_score():
    assert decay_factor(BASE, BASE - timedelta(days=10)) == 1.0


def test_a_stored_score_is_undecayed():
    """ARCHITECTURE 17: the cache holds the raw value; decay is a read-time
    concern, so the row cannot become wrong merely by sitting there."""
    state = LearnerState(
        raw_score=0.8,
        confidence=0.9,
        evidence_count=3,
        effective_style=None,
        last_reinforced_at=BASE,
    )

    assert state.raw_score == 0.8
    assert state.score_at(BASE + timedelta(days=30)) == pytest.approx(0.4, abs=0.01)


# --------------------------------------------------------------------------
# The callback gate
# --------------------------------------------------------------------------


def test_a_low_score_we_are_unsure_about_is_not_a_callback():
    """ARCHITECTURE 10.2 step 4 — "a reason to ask, not to announce"."""
    assert not is_callback_candidate(0.1, CONFIDENCE_FLOOR - 0.01)
    assert is_callback_candidate(0.1, CONFIDENCE_FLOOR)


def test_a_well_understood_concept_is_not_a_callback():
    assert not is_callback_candidate(0.9, 0.95)


def test_the_callback_gap_leaves_room_in_a_short_conversation():
    """Small enough that the documented demo can show one, large enough that
    it does not fire every turn."""
    assert 2 <= CALLBACK_MIN_TURN_GAP <= 10


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


async def _concept_with_user(session: AsyncSession) -> tuple[User, Concept]:
    user = User(auth_subject=f"learner-test-{uuid.uuid4()}")
    session.add(user)
    await session.flush()

    concept = Concept(
        user_id=user.user_id,
        canonical_name="Variational lower bound",
        normalized_name=f"variational lower bound {uuid.uuid4()}",
    )
    session.add(concept)
    await session.flush()
    return user, concept


async def test_recompute_writes_the_cache_from_the_evidence(db_session: AsyncSession):
    user, concept = await _concept_with_user(db_session)

    conversation = Session(user_id=user.user_id)
    db_session.add(conversation)
    await db_session.flush()
    turn = Turn(session_id=conversation.session_id, user_id=user.user_id, ordinal=0)
    db_session.add(turn)
    await db_session.flush()

    struggle = Observation(
        user_id=user.user_id,
        concept_id=concept.concept_id,
        turn_id=turn.turn_id,
        signal_type="explicit_confusion",
        signal_source="explicit",
        weight=weight_for("explicit_confusion", "explicit"),
        style_in_play="formal",
    )
    db_session.add(struggle)
    await db_session.flush()

    db_session.add(
        Observation(
            user_id=user.user_id,
            concept_id=concept.concept_id,
            turn_id=turn.turn_id,
            signal_type="explicit_understanding",
            signal_source="explicit",
            weight=weight_for("explicit_understanding", "explicit"),
            style_in_play="numerical",
            resolves_observation_id=struggle.observation_id,
        )
    )
    await db_session.flush()

    state = await recompute(db_session, concept.concept_id)
    await db_session.flush()

    assert state.raw_score == pytest.approx(0.35, abs=0.005)
    assert concept.understanding_score == pytest.approx(0.35, abs=0.005)
    assert concept.score_confidence == pytest.approx(0.70, abs=0.005)
    assert concept.evidence_count == 2
    assert concept.effective_style == "numerical"
    assert concept.last_reinforced_at is not None


async def test_recompute_leaves_an_explicit_override_alone(db_session: AsyncSession):
    """A correction from the reader outranks inference (ARCHITECTURE 4.9)."""
    user, concept = await _concept_with_user(db_session)
    concept.user_override_score = 0.9
    await db_session.flush()

    db_session.add(
        Observation(
            user_id=user.user_id,
            concept_id=concept.concept_id,
            signal_type="quiz_incorrect",
            signal_source="quiz",
            weight=weight_for("quiz_incorrect", "quiz"),
        )
    )
    await db_session.flush()

    await recompute(db_session, concept.concept_id)
    await db_session.flush()

    assert concept.user_override_score == 0.9
    assert concept.understanding_score == pytest.approx(0.0, abs=0.005)


async def test_recompute_is_scoped_to_one_concept(db_session: AsyncSession):
    """Another concept's evidence must not leak into this one's score."""
    user, concept = await _concept_with_user(db_session)

    other = Concept(
        user_id=user.user_id,
        canonical_name="Reparameterization trick",
        normalized_name=f"reparameterization trick {uuid.uuid4()}",
    )
    db_session.add(other)
    await db_session.flush()

    db_session.add(
        Observation(
            user_id=user.user_id,
            concept_id=other.concept_id,
            signal_type="quiz_correct",
            signal_source="quiz",
            weight=weight_for("quiz_correct", "quiz"),
        )
    )
    await db_session.flush()

    state = await recompute(db_session, concept.concept_id)

    assert state.raw_score is None
    assert state.evidence_count == 0
