"""LearnerStateService — weights, score arithmetic, decay, style ranking.

This is the `[D]` half of learner memory (ARCHITECTURE 2.2, 3, 14.2). The model
classifies a signal into the closed vocabulary and nothing else: the weight it
carries, the arithmetic it feeds, and the decay applied afterwards are all
decided here, by lookup, so a score is always reproducible from the
`observations` rows that produced it.

HANDOFF 5.2 listed the constants below as undecided. They are chosen here, and
they are not free parameters — the architecture document works two examples
end to end, and both are reproduced exactly:

  * 10.1  a struggle resolved by a numerical example leaves
          score 0.35, confidence 0.7, effective_style 'numerical'
  * 12    the same concept read days later scores 0.31 and is still
          trusted at confidence 0.72

`test_learner_state.py` asserts both, so a change to any constant that breaks
the documented narrative fails the suite rather than the demo.

Two structural rules from ARCHITECTURE 17 shape everything here:

  * the score is **computed from observations**; `concepts.understanding_score`
    is only a cache
  * **decay is applied at read time** from `last_reinforced_at`, never written
    down — which is what keeps the cache from ever being wrong-by-staleness
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Concept, Observation

# --------------------------------------------------------------------------
# Weights — HANDOFF 5.2, the `(signal_type, signal_source)` table
# --------------------------------------------------------------------------
# Weight is *evidentiary strength*: how much this observation should count. It
# is deliberately separate from what the observation says about the learner,
# which is the target value below. Conflating them would make "confidently
# confused" and "vaguely confident" indistinguishable.
#
# The ordering of the classes is the load-bearing part, not the exact decimals:
#
#   quiz         a measured answer against a stored rubric. ARCHITECTURE 11
#                puts quiz signals at "the highest weight class" explicitly
#   user_stated  the learner's own claim about themselves — authoritative
#                about their state, but not a demonstration of it
#   explicit     said in so many words during the conversation
#   implicit     inferred from phrasing. Weak on purpose: this is the class
#                most likely to be wrong, and it is the one the model has the
#                most latitude to over-report
#   system       zero. See below — this one is not a small weight, it is none.
#
# `system` carries no evidentiary weight at all, and a small weight would be
# the wrong answer rather than a cautious one. The reinforcement backstop
# (ARCHITECTURE 14.2) fires precisely when the agent recorded *nothing*, so a
# system row asserts only that the concept came up. Give it any positive
# weight and it accumulates: at 0.10, three backstop rows reach the 0.3
# confidence floor exactly and ten sail past it, at which point a concept
# nobody ever demonstrated anything about is being reported with confidence.
# Volume of non-evidence must not become evidence.
#
# What a reinforcement row does do is move `last_reinforced_at`, which resets
# the decay clock — which is what the word means. It keeps memory accumulating
# without letting exposure masquerade as attainment.
SOURCE_WEIGHT: dict[str, float] = {
    "quiz": 1.00,
    "user_stated": 0.90,
    "explicit": 0.80,
    "implicit": 0.40,
    "system": 0.00,
}

# Pairs whose weight does not follow from the source class alone. Applying a
# concept correctly is inferred rather than stated, but it is the strongest
# evidence a conversation can produce short of a graded answer — someone who
# uses a concept correctly has demonstrated it, whatever they say about
# themselves. It would be wrong to score it as a weak implicit hint.
SIGNAL_WEIGHTS: dict[tuple[str, str], float] = {
    ("applied_correctly", "implicit"): 0.75,
}


def weight_for(signal_type: str, signal_source: str) -> float:
    """The deterministic weight for one observation.

    Total by construction: the enum pair space is 10x5 and most combinations
    are nonsense that no caller should be able to turn into a crash. An
    unlisted pair falls back to its source class, which is the conservative
    reading — the source is what bounds how much the signal can be trusted.
    """
    pair = SIGNAL_WEIGHTS.get((signal_type, signal_source))
    if pair is not None:
        return pair
    return SOURCE_WEIGHT.get(signal_source, SOURCE_WEIGHT["implicit"])


# --------------------------------------------------------------------------
# Target values — what a signal claims the understanding *is*
# --------------------------------------------------------------------------
SIGNAL_TARGETS: dict[str, float] = {
    "explicit_confusion": 0.00,
    "implicit_confusion": 0.15,
    "explicit_understanding": 1.00,
    "quiz_correct": 1.00,
    "quiz_partial": 0.50,
    "quiz_incorrect": 0.00,
    "applied_correctly": 1.00,
    "user_stated_known": 1.00,
    "user_stated_unknown": 0.00,
    # Exposure, not attainment. Listed for completeness and inert in practice:
    # the backstop writes it with `signal_source = 'system'`, which carries
    # zero weight, so this value never reaches the average. A concept carried
    # only by backstop rows reads as *unknown*, which is the truth about it.
    "reinforcement": 0.60,
}

# An understanding that arrived *because* the system re-explained is not the
# same as one that was there all along, and the difference is exactly what
# `resolves_observation_id` records. Scoring assisted understanding at full
# marks would erase the struggle from the learner's history the moment it was
# resolved — the opposite of what ARCHITECTURE 10.1 calls the valuable signal.
ASSISTED_UNDERSTANDING_TARGET = 0.70

_POSITIVE_UNDERSTANDING = ("explicit_understanding", "user_stated_known")


def target_for(signal_type: str, *, resolves_prior: bool = False) -> float:
    """What this signal says the understanding is, on 0..1."""
    if resolves_prior and signal_type in _POSITIVE_UNDERSTANDING:
        return ASSISTED_UNDERSTANDING_TARGET
    return SIGNAL_TARGETS.get(signal_type, 0.50)


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------
# confidence = W / (W + CONFIDENCE_HALF_WEIGHT), where W is total evidence
# weight. Saturating rather than linear: the difference between no evidence and
# one good signal must matter far more than the difference between the ninth
# and the tenth.
#
# 0.7 is fixed by ARCHITECTURE 10.1 — two explicit signals (W = 1.6) must read
# as 0.70, and 1.6 / 2.3 = 0.696.
CONFIDENCE_HALF_WEIGHT = 0.70

# Mirrors the partial-index predicate on `concepts`
# (`WHERE ... score_confidence >= 0.3`). The floor lives in the schema so a low
# score we are unsure about cannot surface as a claim; it is repeated here so
# application-side filters agree with the index rather than drifting from it.
CONFIDENCE_FLOOR = 0.30


def confidence_for(total_weight: float) -> float:
    if total_weight <= 0:
        return 0.0
    return total_weight / (total_weight + CONFIDENCE_HALF_WEIGHT)


# --------------------------------------------------------------------------
# Decay
# --------------------------------------------------------------------------
# Knowledge fades, so an unreinforced score falls. ARCHITECTURE 12 shows a
# concept last seen "days later" reading 0.31 against the 0.35 it was recorded
# at, which fixes both the direction and roughly the rate: a 30-day half-life
# puts that reading at about five days elapsed.
SCORE_HALF_LIFE_DAYS = 30.0

# Decay expresses staleness, not evidence of failure. Left unbounded it drives
# every months-old concept toward zero, at which point they are all maximally
# weak, all equally good callback candidates, and the ranking that picks
# between them carries no information. The floor keeps old scores ordered by
# what was actually observed.
SCORE_DECAY_FLOOR = 0.25


def decay_factor(last_reinforced_at: datetime | None, now: datetime) -> float:
    """Multiplier on the raw score, from elapsed time alone."""
    if last_reinforced_at is None:
        return 1.0
    if last_reinforced_at.tzinfo is None:
        last_reinforced_at = last_reinforced_at.replace(tzinfo=UTC)

    elapsed_days = (now - last_reinforced_at).total_seconds() / 86400.0
    if elapsed_days <= 0:
        return 1.0

    return max(SCORE_DECAY_FLOOR, 0.5 ** (elapsed_days / SCORE_HALF_LIFE_DAYS))


# --------------------------------------------------------------------------
# The callback gate — ARCHITECTURE 10.2 step 4, 12 steps 4-6
# --------------------------------------------------------------------------
# "Weak" for the purposes of a proactive callback. Both worked examples (0.35
# fresh, 0.31 decayed) sit below this, so the documented demo fires; a concept
# scoring at or above it is not something to interrupt the reader about.
WEAK_SCORE_BELOW = 0.40

# Minimum turns since the last callback. ARCHITECTURE 12 caps callbacks at one
# per turn and requires a gap without naming it. Five is a session's worth of
# breathing room: frequent enough that a genuine connection still lands in a
# short demo conversation, rare enough that the feature reads as a considered
# interjection rather than a tic. Suppression is recorded, never silent, so if
# this is wrong the evidence for changing it is in `turns`.
CALLBACK_MIN_TURN_GAP = 5


def is_callback_candidate(score: float | None, confidence: float | None) -> bool:
    """Weak *and* confident — a low score we are unsure about is not evidence.

    ARCHITECTURE 10.2 is explicit that "a low score we are unsure about is a
    reason to ask, not to announce", which is why this is an AND and why the
    floor matches the index predicate.
    """
    if score is None or confidence is None:
        return False
    return score < WEAK_SCORE_BELOW and confidence >= CONFIDENCE_FLOOR


# --------------------------------------------------------------------------
# Derivation
# --------------------------------------------------------------------------
_SMALLINT_MAX = 32767


@dataclass(frozen=True)
class LearnerState:
    """The derived view of one concept. All of it recomputable from evidence."""

    raw_score: float | None
    confidence: float
    evidence_count: int
    effective_style: str | None
    last_reinforced_at: datetime | None

    def score_at(self, now: datetime) -> float | None:
        """The score as of `now`, with decay applied — the read-time value."""
        if self.raw_score is None:
            return None
        return self.raw_score * decay_factor(self.last_reinforced_at, now)


def derive(observations: list[Observation]) -> LearnerState:
    """Fold an evidence trail into a score, confidence and style.

    A weighted mean rather than a running update: replaying the same rows must
    give the same answer in any order, and an update rule that folds each
    observation into the previous score cannot promise that.
    """
    if not observations:
        return LearnerState(None, 0.0, 0, None, None)

    total_weight = 0.0
    weighted_target = 0.0
    # How often each style resolved a struggle, and when it last did.
    style_counts: dict[str, int] = defaultdict(int)
    style_last_seen: dict[str, datetime] = {}
    last_reinforced: datetime | None = None

    for observation in observations:
        resolves_prior = observation.resolves_observation_id is not None
        weight = observation.weight
        target = target_for(observation.signal_type, resolves_prior=resolves_prior)

        total_weight += weight
        weighted_target += weight * target

        observed_at = observation.observed_at
        if observed_at is not None and observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=UTC)
        if observed_at is not None and (
            last_reinforced is None or observed_at > last_reinforced
        ):
            last_reinforced = observed_at

        # ARCHITECTURE 17: effective style is derived from resolutions alone.
        # A style that merely happened to be in play when someone understood
        # something proves nothing; a style that *resolved a struggle* is the
        # only evidence that it teaches this person.
        if resolves_prior and observation.style_in_play:
            style = observation.style_in_play
            style_counts[style] += 1
            seen_at = observed_at or datetime.min.replace(tzinfo=UTC)
            if seen_at > style_last_seen.get(style, datetime.min.replace(tzinfo=UTC)):
                style_last_seen[style] = seen_at

    raw_score = weighted_target / total_weight if total_weight > 0 else None

    # Most resolutions wins; the most recent breaks a tie, because a style that
    # worked lately is better evidence than one that worked once, long ago.
    effective_style = None
    if style_counts:
        effective_style = max(
            style_counts, key=lambda s: (style_counts[s], style_last_seen[s])
        )

    return LearnerState(
        raw_score=raw_score,
        confidence=confidence_for(total_weight),
        evidence_count=min(len(observations), _SMALLINT_MAX),
        effective_style=effective_style,
        last_reinforced_at=last_reinforced,
    )


async def recompute(session: AsyncSession, concept_id: uuid.UUID) -> LearnerState:
    """Re-derive one concept's cache from its observations and store it.

    The cache columns exist so the memory views and the callback gate do not
    replay the evidence trail on every read. `understanding_score` is stored
    **undecayed**: decay belongs to read time (ARCHITECTURE 17), and writing a
    decayed value down would make the row mean something different depending
    on when it was last recomputed.

    `user_override_score` is not touched. An explicit correction from the
    reader outranks inference and is never silently overwritten.
    """
    observations = list(
        (
            await session.scalars(
                select(Observation)
                .where(Observation.concept_id == concept_id)
                .order_by(Observation.observed_at)
            )
        ).all()
    )

    state = derive(observations)

    concept = await session.get(Concept, concept_id)
    if concept is not None:
        concept.understanding_score = state.raw_score
        concept.score_confidence = state.confidence
        concept.evidence_count = state.evidence_count
        concept.effective_style = state.effective_style
        concept.last_reinforced_at = state.last_reinforced_at

    return state
