"""The 14 tables of the Research Paper Reading Companion.

Layout follows ARCHITECTURE section 4. Two conventions are load-bearing:

* Enumerations are TEXT + CHECK rather than native enums, so widening a
  vocabulary is a one-line migration instead of a type rewrite.
* ``user_id`` leads every composite index, so an index scan can never *begin*
  by touching another user's rows.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    REAL,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

EMBEDDING_DIM = 768

PROCESSING_STATUS = ("queued", "processing", "ready", "partially_ready", "failed")
SECTION_ROLE = (
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "references",
    "appendix",
    "unknown",
)
ACTIVITY = ("FREE", "EXPLAINING", "QUIZ_PENDING", "AWAITING_CLARIFICATION")
EXPLANATION_STYLE = (
    "formal",
    "intuitive",
    "numerical",
    "analogical",
    "visual_verbal",
    "code",
    "contrastive",
)
SIGNAL_TYPE = (
    "explicit_confusion",
    "implicit_confusion",
    "explicit_understanding",
    "quiz_correct",
    "quiz_partial",
    "quiz_incorrect",
    "applied_correctly",
    "user_stated_known",
    "user_stated_unknown",
    "reinforcement",
)
SIGNAL_SOURCE = ("explicit", "implicit", "quiz", "user_stated", "system")
RELATIONSHIP_TYPE = (
    "prerequisite_of",
    "component_of",
    "specialisation_of",
    "contrasts_with",
    "equivalent_notation",
    "co_occurs_with",
)
DISCOVERY_METHOD = ("embedding", "model", "user_stated")
GRADE = ("correct", "partial", "incorrect")
FEEDBACK_TYPE = (
    "helpful",
    "not_helpful",
    "too_basic",
    "too_advanced",
    "wrong",
    "style_preference",
)
GROUNDING_STATUS = ("grounded", "no_evidence", "degraded", "n/a")


def _in(column: str, values: tuple[str, ...]) -> str:
    rendered = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({rendered})"


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )


def _now() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


# --------------------------------------------------------------------------
# 4.1 users
# --------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("auth_subject", name="uq_users_auth_subject"),
        Index(
            "uq_users_email",
            "email",
            unique=True,
            postgresql_where=text("email IS NOT NULL"),
        ),
    )

    user_id: Mapped[uuid.UUID] = _pk()
    auth_subject: Mapped[str] = mapped_column(String(128), nullable=False)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    preferences: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    is_demo_account: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = _now()


# --------------------------------------------------------------------------
# 4.3 papers  (defined before user_paper_access, which references it)
# --------------------------------------------------------------------------
class Paper(Base):
    __tablename__ = "papers"
    __table_args__ = (
        CheckConstraint(_in("processing_status", PROCESSING_STATUS), name="processing_status"),
        CheckConstraint("year IS NULL OR (year BETWEEN 1900 AND 2100)", name="year_range"),
        CheckConstraint("page_count IS NULL OR page_count > 0", name="page_count_positive"),
        CheckConstraint(
            "extractable_text_ratio IS NULL OR (extractable_text_ratio BETWEEN 0 AND 1)",
            name="extractable_ratio_range",
        ),
        Index(
            "ix_papers_processing_status_live",
            "processing_status",
            postgresql_where=text("processing_status NOT IN ('ready', 'failed')"),
        ),
    )

    paper_id: Mapped[uuid.UUID] = _pk()
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    title: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    authors: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    year: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    page_count: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    processing_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'queued'")
    )
    processing_phase: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    extractable_text_ratio: Mapped[float | None] = mapped_column(REAL, nullable=True)
    unreadable_pages: Mapped[list[int] | None] = mapped_column(
        ARRAY(SmallInteger), nullable=True
    )
    security_findings: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    concept_candidates: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = _now()


# --------------------------------------------------------------------------
# 4.2 user_paper_access — the authorization boundary
# --------------------------------------------------------------------------
class UserPaperAccess(Base):
    __tablename__ = "user_paper_access"
    __table_args__ = (
        Index(
            "ix_user_paper_access_user_id_last_opened_at",
            "user_id",
            text("last_opened_at DESC"),
            postgresql_where=text("revoked_at IS NULL"),
        ),
        Index("ix_user_paper_access_paper_id", "paper_id"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.paper_id", ondelete="CASCADE"),
        primary_key=True,
    )
    added_at: Mapped[datetime] = _now()
    nickname: Mapped[str | None] = mapped_column(String(200), nullable=True)
    last_opened_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


# --------------------------------------------------------------------------
# 4.4 sections
# --------------------------------------------------------------------------
class Section(Base):
    __tablename__ = "sections"
    __table_args__ = (
        UniqueConstraint("paper_id", "ordinal", name="uq_sections_paper_id_ordinal"),
        CheckConstraint(_in("section_role", SECTION_ROLE), name="section_role"),
        CheckConstraint("ordinal >= 0", name="ordinal_non_negative"),
        CheckConstraint(
            "page_start > 0 AND page_end >= page_start", name="page_span_valid"
        ),
        Index("ix_sections_paper_id_section_role", "paper_id", "section_role"),
    )

    section_id: Mapped[uuid.UUID] = _pk()
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.paper_id", ondelete="CASCADE"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    heading: Mapped[str | None] = mapped_column(String(500), nullable=True)
    section_path: Mapped[str] = mapped_column(String(200), nullable=False)
    section_role: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'unknown'")
    )
    page_start: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    page_end: Mapped[int] = mapped_column(SmallInteger, nullable=False)


# --------------------------------------------------------------------------
# 4.5 chunks — the retrieval and citation atom
# --------------------------------------------------------------------------
class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("paper_id", "ordinal", name="uq_chunks_paper_id_ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_non_negative"),
        CheckConstraint(
            "char_length(content) BETWEEN 1 AND 8000", name="content_length"
        ),
        CheckConstraint("token_count IS NULL OR token_count > 0", name="token_count_positive"),
        CheckConstraint(
            "page_start > 0 AND page_end >= page_start", name="page_span_valid"
        ),
        # Unembedded and non-indexable chunks never enter the ANN index.
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_where=text("is_indexable AND embedding IS NOT NULL"),
        ),
        Index(
            "ix_chunks_paper_id_indexable",
            "paper_id",
            postgresql_where=text("is_indexable"),
        ),
    )

    chunk_id: Mapped[uuid.UUID] = _pk()
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.paper_id", ondelete="CASCADE"),
        nullable=False,
    )
    # A chunk belongs to exactly one section: the no-crossing rule as an FK.
    section_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sections.section_id", ondelete="CASCADE"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    # NOT NULL by design — a chunk without a page is uncitable.
    page_start: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    page_end: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    is_indexable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )


# --------------------------------------------------------------------------
# 4.9 concepts — the learner model
# --------------------------------------------------------------------------
class Concept(Base):
    __tablename__ = "concepts"
    __table_args__ = (
        CheckConstraint("merged_into_id <> concept_id", name="no_self_merge"),
        CheckConstraint(
            "understanding_score IS NULL OR (understanding_score BETWEEN 0 AND 1)",
            name="understanding_score_range",
        ),
        CheckConstraint(
            "score_confidence IS NULL OR (score_confidence BETWEEN 0 AND 1)",
            name="score_confidence_range",
        ),
        CheckConstraint(
            "user_override_score IS NULL OR (user_override_score BETWEEN 0 AND 1)",
            name="user_override_score_range",
        ),
        CheckConstraint("evidence_count >= 0", name="evidence_count_non_negative"),
        CheckConstraint(
            f"effective_style IS NULL OR {_in('effective_style', EXPLANATION_STYLE)}",
            name="effective_style",
        ),
        # A merged-away concept keeps its name without blocking the survivor.
        Index(
            "uq_concepts_user_id_normalized_name",
            "user_id",
            "normalized_name",
            unique=True,
            postgresql_where=text("merged_into_id IS NULL"),
        ),
        Index("ix_concepts_aliases", "aliases", postgresql_using="gin"),
        Index("ix_concepts_source_paper_ids", "source_paper_ids", postgresql_using="gin"),
        Index(
            "ix_concepts_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_where=text("merged_into_id IS NULL"),
        ),
        # The confidence floor lives inside the index predicate, so "low score we
        # are unsure about" can never surface as a claim.
        Index(
            "ix_concepts_user_id_understanding_score",
            "user_id",
            "understanding_score",
            postgresql_where=text("merged_into_id IS NULL AND score_confidence >= 0.3"),
        ),
    )

    concept_id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    canonical_name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(200), nullable=False)
    aliases: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")
    )
    description: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    understanding_score: Mapped[float | None] = mapped_column(REAL, nullable=True)
    score_confidence: Mapped[float | None] = mapped_column(REAL, nullable=True)
    evidence_count: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("0")
    )
    effective_style: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_paper_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, server_default=text("'{}'::uuid[]")
    )
    first_seen_at: Mapped[datetime] = _now()
    last_reinforced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    user_override_score: Mapped[float | None] = mapped_column(REAL, nullable=True)
    # Reversible merge — the absorbed row survives.
    merged_into_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("concepts.concept_id", ondelete="SET NULL"),
        nullable=True,
    )


# --------------------------------------------------------------------------
# 4.12 quizzes  (defined before sessions, which references pending_quiz_id)
# --------------------------------------------------------------------------
class Quiz(Base):
    __tablename__ = "quizzes"
    __table_args__ = (
        # Mandatory grounding: a quiz not anchored in the paper tests world
        # knowledge rather than reading.
        CheckConstraint(
            "array_length(grounding_chunk_ids, 1) >= 1", name="grounding_required"
        ),
        Index(
            "ix_quizzes_user_id_concept_id_created_at",
            "user_id",
            "concept_id",
            text("created_at DESC"),
        ),
    )

    quiz_id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    concept_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("concepts.concept_id", ondelete="CASCADE"),
        nullable=False,
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.paper_id", ondelete="CASCADE"),
        nullable=False,
    )
    question: Mapped[str] = mapped_column(String(2000), nullable=False)
    rubric: Mapped[dict] = mapped_column(JSONB, nullable=False)
    grounding_chunk_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False
    )
    created_at: Mapped[datetime] = _now()


# --------------------------------------------------------------------------
# 4.6 sessions
# --------------------------------------------------------------------------
class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        CheckConstraint(_in("activity", ACTIVITY), name="activity"),
        # The state and its payload cannot disagree.
        CheckConstraint(
            "(activity = 'QUIZ_PENDING') = (pending_quiz_id IS NOT NULL)",
            name="quiz_pending_consistency",
        ),
        CheckConstraint("turn_count >= 0", name="turn_count_non_negative"),
        Index("ix_sessions_user_id_started_at", "user_id", text("started_at DESC")),
        Index(
            "ix_sessions_last_activity_at",
            "last_activity_at",
            postgresql_where=text("ended_at IS NULL"),
        ),
    )

    session_id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    active_paper_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.paper_id", ondelete="SET NULL"),
        nullable=True,
    )
    activity: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'FREE'")
    )
    pending_quiz_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("quizzes.quiz_id", ondelete="SET NULL"),
        nullable=True,
    )
    active_concept_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("concepts.concept_id", ondelete="SET NULL"),
        nullable=True,
    )
    last_callback_turn: Mapped[int | None] = mapped_column(Integer, nullable=True)
    turn_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    started_at: Mapped[datetime] = _now()
    last_activity_at: Mapped[datetime] = _now()
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


# --------------------------------------------------------------------------
# 4.7 turns — append-only
# --------------------------------------------------------------------------
class Turn(Base):
    __tablename__ = "turns"
    __table_args__ = (
        UniqueConstraint("session_id", "ordinal", name="uq_turns_session_id_ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_non_negative"),
        CheckConstraint(_in("grounding_status", GROUNDING_STATUS), name="grounding_status"),
        CheckConstraint(
            f"explanation_style IS NULL OR {_in('explanation_style', EXPLANATION_STYLE)}",
            name="explanation_style",
        ),
        # A proactive callback cannot be recorded without a memory read to
        # ground it in.
        CheckConstraint(
            "callback_concept_id IS NULL OR memory_read", name="callback_requires_memory"
        ),
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0", name="input_tokens_non_negative"
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="output_tokens_non_negative",
        ),
        CheckConstraint("latency_ms IS NULL OR latency_ms >= 0", name="latency_non_negative"),
        Index("ix_turns_session_id_created_at", "session_id", "created_at"),
        Index("ix_turns_user_id_created_at", "user_id", text("created_at DESC")),
    )

    turn_id: Mapped[uuid.UUID] = _pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    # Denormalized deliberately — every hot query filters by user.
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    paper_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.paper_id", ondelete="SET NULL"),
        nullable=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    user_message: Mapped[str | None] = mapped_column(String(8000), nullable=True)
    agent_action: Mapped[str | None] = mapped_column(String(64), nullable=True)
    explanation_style: Mapped[str | None] = mapped_column(Text, nullable=True)
    callback_concept_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("concepts.concept_id", ondelete="SET NULL"),
        nullable=True,
    )
    # Suppression is a feature and is measured.
    callback_suppressed_reason: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    memory_read: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    grounding_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'n/a'")
    )
    tools_called: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = _now()


# --------------------------------------------------------------------------
# 4.8 turn_retrievals — retrieval set and citations in one table
# --------------------------------------------------------------------------
class TurnRetrieval(Base):
    __tablename__ = "turn_retrievals"
    __table_args__ = (
        CheckConstraint("rank >= 1", name="rank_positive"),
        CheckConstraint("similarity BETWEEN -1 AND 1", name="similarity_range"),
        CheckConstraint(
            "(citation_marker IS NOT NULL) = was_cited", name="marker_matches_cited"
        ),
        Index(
            "ix_turn_retrievals_turn_id_cited",
            "turn_id",
            postgresql_where=text("was_cited"),
        ),
        Index("ix_turn_retrievals_chunk_id", "chunk_id"),
    )

    turn_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("turns.turn_id", ondelete="CASCADE"),
        primary_key=True,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
        primary_key=True,
    )
    rank: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    similarity: Mapped[float] = mapped_column(REAL, nullable=False)
    retrieval_query: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    # Set by the deterministic verifier after generation.
    was_cited: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    citation_marker: Mapped[str | None] = mapped_column(String(16), nullable=True)


# --------------------------------------------------------------------------
# 4.10 concept_relationships
# --------------------------------------------------------------------------
class ConceptRelationship(Base):
    __tablename__ = "concept_relationships"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "source_concept_id",
            "target_concept_id",
            "relationship_type",
            name="uq_concept_relationships_edge",
        ),
        CheckConstraint(
            "source_concept_id <> target_concept_id", name="no_self_edge"
        ),
        CheckConstraint(_in("relationship_type", RELATIONSHIP_TYPE), name="relationship_type"),
        CheckConstraint(_in("discovery_method", DISCOVERY_METHOD), name="discovery_method"),
        CheckConstraint("confidence BETWEEN 0 AND 1", name="confidence_range"),
        Index("ix_concept_relationships_user_id_source", "user_id", "source_concept_id"),
        # Reverse traversal: "which weak concept blocks the most downstream
        # understanding".
        Index("ix_concept_relationships_user_id_target", "user_id", "target_concept_id"),
    )

    relationship_id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    source_concept_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("concepts.concept_id", ondelete="CASCADE"),
        nullable=False,
    )
    target_concept_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("concepts.concept_id", ondelete="CASCADE"),
        nullable=False,
    )
    relationship_type: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(REAL, nullable=False)
    discovery_method: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("turns.turn_id", ondelete="SET NULL"),
        nullable=True,
    )
    discovered_at: Mapped[datetime] = _now()


# --------------------------------------------------------------------------
# 4.13 quiz_attempts — append-only
# --------------------------------------------------------------------------
class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"
    __table_args__ = (
        UniqueConstraint("quiz_id", "attempt_no", name="uq_quiz_attempts_quiz_id_attempt_no"),
        CheckConstraint(f"grade IS NULL OR {_in('grade', GRADE)}", name="grade"),
        # Null grade iff grading failed — we never guess a grade.
        CheckConstraint(
            "(grade IS NULL) = (grading_error IS NOT NULL)", name="grade_or_error"
        ),
        CheckConstraint(
            "grader_confidence IS NULL OR (grader_confidence BETWEEN 0 AND 1)",
            name="grader_confidence_range",
        ),
        CheckConstraint("attempt_no >= 1", name="attempt_no_positive"),
    )

    attempt_id: Mapped[uuid.UUID] = _pk()
    quiz_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("quizzes.quiz_id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("turns.turn_id", ondelete="SET NULL"),
        nullable=True,
    )
    answer_text: Mapped[str] = mapped_column(String(4000), nullable=False)
    grade: Mapped[str | None] = mapped_column(Text, nullable=True)
    missing_elements: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    grader_confidence: Mapped[float | None] = mapped_column(REAL, nullable=True)
    grading_error: Mapped[str | None] = mapped_column(String(200), nullable=True)
    attempt_no: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    created_at: Mapped[datetime] = _now()


# --------------------------------------------------------------------------
# 4.11 observations — raw, immutable, append-only
# --------------------------------------------------------------------------
class Observation(Base):
    __tablename__ = "observations"
    __table_args__ = (
        CheckConstraint(_in("signal_type", SIGNAL_TYPE), name="signal_type"),
        CheckConstraint(_in("signal_source", SIGNAL_SOURCE), name="signal_source"),
        CheckConstraint("weight BETWEEN 0 AND 1", name="weight_range"),
        CheckConstraint(
            f"style_in_play IS NULL OR {_in('style_in_play', EXPLANATION_STYLE)}",
            name="style_in_play",
        ),
        CheckConstraint(
            "resolves_observation_id <> observation_id", name="no_self_resolution"
        ),
        # The derivation query: every score recomputation reads exactly this.
        Index(
            "ix_observations_user_id_concept_id_observed_at",
            "user_id",
            "concept_id",
            text("observed_at DESC"),
        ),
        Index(
            "ix_observations_concept_id_resolutions",
            "concept_id",
            postgresql_where=text("resolves_observation_id IS NOT NULL"),
        ),
        Index("ix_observations_turn_id", "turn_id"),
        Index("ix_observations_user_id_observed_at", "user_id", text("observed_at DESC")),
    )

    observation_id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    concept_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("concepts.concept_id", ondelete="CASCADE"),
        nullable=False,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.session_id", ondelete="SET NULL"),
        nullable=True,
    )
    # Provenance — this is what makes memory inspectable.
    turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("turns.turn_id", ondelete="SET NULL"),
        nullable=True,
    )
    signal_type: Mapped[str] = mapped_column(Text, nullable=False)
    signal_source: Mapped[str] = mapped_column(Text, nullable=False)
    # Assigned deterministically by lookup, never by the model.
    weight: Mapped[float] = mapped_column(REAL, nullable=False)
    style_in_play: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolves_observation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("observations.observation_id", ondelete="SET NULL"),
        nullable=True,
    )
    quiz_attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("quiz_attempts.attempt_id", ondelete="SET NULL"),
        nullable=True,
    )
    note: Mapped[str | None] = mapped_column(String(500), nullable=True)
    observed_at: Mapped[datetime] = _now()


# --------------------------------------------------------------------------
# 4.14 feedback
# --------------------------------------------------------------------------
class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint(_in("feedback_type", FEEDBACK_TYPE), name="feedback_type"),
        # Exactly one target.
        CheckConstraint(
            "(target_turn_id IS NOT NULL)::int + (target_concept_id IS NOT NULL)::int = 1",
            name="exactly_one_target",
        ),
        Index("ix_feedback_user_id_created_at", "user_id", text("created_at DESC")),
    )

    feedback_id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    feedback_type: Mapped[str] = mapped_column(Text, nullable=False)
    target_turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("turns.turn_id", ondelete="SET NULL"),
        nullable=True,
    )
    target_concept_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("concepts.concept_id", ondelete="SET NULL"),
        nullable=True,
    )
    comment: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    # Which later turn changed as a result — makes "feedback changes behaviour"
    # verifiable rather than asserted.
    applied_to_turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("turns.turn_id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = _now()


APPEND_ONLY_TABLES = ("turns", "observations", "quiz_attempts")
