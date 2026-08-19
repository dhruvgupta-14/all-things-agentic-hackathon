"""The ADK agent, built fresh for one turn and discarded.

ADK is the runtime, not a store. Its session service is in-memory and lives for
the duration of a single turn: we hydrate it from the `messages` table, run the
agent, take the final response, and throw the whole thing away
(ARCHITECTURE 17). Nothing ADK holds outlives the request, so history survives
an instance being reclaimed and no framework table is ever created.

The agent is constructed per turn rather than once at import because its tools
close over that turn's authorization scope. A long-lived agent would need scope
passed as an argument, which is exactly the thing §13 forbids.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field

from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent.instructions import build_instruction
from app.agent.tools import (
    TurnToolContext,
    build_generate_quiz,
    build_get_concept_context,
    build_record_learning_signal,
    build_retrieve_learner_memory,
    build_retrieve_paper_context,
    build_scope_guard,
)
from app.config import get_settings
from app.services.messages import HistoryMessage

logger = logging.getLogger(__name__)

APP_NAME = "paper-companion"

# Each iteration of the agent loop is a real model request. Observed in spike
# S-1: left uncapped, the agent searched three times for one question. The cap
# bounds both latency on camera and spend against a per-day request quota.
MAX_ITERATIONS = 3


# Vertex returns 429 RESOURCE_EXHAUSTED under shared-capacity pressure and 503
# UNAVAILABLE on transient faults, and both usually clear on an immediate
# retry — observed repeatedly during live runs, including mid-demo. A failed
# turn is a far worse outcome than a slower one, so these are retried; anything
# else is a real fault and still fails the turn closed.
# One retry, not two. Each attempt re-runs the entire agent loop, which is
# 30-50s of model time, so a generous budget turns a 40s turn into a 2-minute
# one — measured at 116s with three attempts under sustained rate limiting.
# The failure this exists for is a single 429 that clears immediately, and one
# retry catches that; a second only makes a bad case worse.
MAX_TRANSIENT_ATTEMPTS = 2
TRANSIENT_BACKOFF_SECONDS = 2

_TRANSIENT_MARKERS = (
    "resource_exhausted",
    "429",
    "unavailable",
    "503",
    "deadline_exceeded",
    "504",
)


def _is_transient(exc: Exception) -> bool:
    """Worth waiting out, as opposed to worth reporting.

    Matched on the message because ADK wraps the transport's exception types,
    so the class alone does not distinguish "the model is busy" from "the
    prompt was malformed".
    """
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


async def _reset_for_retry(context: TurnToolContext) -> None:
    """Undo what a failed attempt's tool calls left behind.

    `memory_seen` is deliberately *not* cleared: the unconditional prefetch put
    it there before the agent ran, so clearing it would make the turn report
    `memory_read = False` for a read that genuinely happened.

    A quiz is the one tool with state outside this context. If a failed attempt
    asked one, the reader never saw the question — leaving the session in
    QUIZ_PENDING would grade their next message against a question that was
    never put to them.
    """
    context.retrieved.clear()
    context.queries.clear()
    context.tools_called.clear()
    context.pending_signals.clear()

    if context.quiz_asked is not None and context.conversation is not None:
        logger.warning(
            "undoing a quiz asked by a failed attempt", extra={"quiz_id": str(context.quiz_asked)}
        )
        context.conversation.activity = "FREE"
        context.conversation.pending_quiz_id = None
        if context.session is not None:
            from app.db.models import Quiz

            stale = await context.session.get(Quiz, context.quiz_asked)
            if stale is not None:
                await context.session.delete(stale)
            await context.session.flush()
        context.quiz_asked = None


class AgentUnavailable(Exception):
    """The model could not be reached. The turn fails rather than inventing."""


@dataclass(slots=True)
class AgentOutcome:
    text: str
    tools_called: list[str] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None


def _configure_transport() -> None:
    """Point the ADK/GenAI client at whichever backend is configured.

    ADK reads these from the environment rather than from a client we pass, so
    this is the one place the transport choice is applied.
    """
    settings = get_settings()
    if settings.vertex_project:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
        os.environ["GOOGLE_CLOUD_PROJECT"] = settings.vertex_project
        os.environ["GOOGLE_CLOUD_LOCATION"] = settings.vertex_location
    elif settings.gemini_api_key:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"
        os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key


def build_agent(
    context: TurnToolContext,
    paper_title: str | None,
    *,
    memory_summary: str | None = None,
    callback_hint: str | None = None,
    depth_hint: str | None = None,
) -> LlmAgent:
    """One agent, scoped to one turn.

    Memory tools are only attached when the services behind them exist, so a
    turn running without learner memory offers the agent nothing it cannot
    actually do rather than a tool that always returns an apology.
    """
    settings = get_settings()

    tools = [build_retrieve_paper_context(context)]
    if context.memory is not None:
        tools.append(build_retrieve_learner_memory(context))
        tools.append(build_get_concept_context(context))
    if context.signals is not None:
        tools.append(build_record_learning_signal(context))
    if context.quizzes is not None and context.conversation is not None:
        tools.append(build_generate_quiz(context))

    return LlmAgent(
        name="reading_companion",
        model=settings.gemini_model,
        instruction=build_instruction(
            paper_title,
            memory_summary=memory_summary,
            callback_hint=callback_hint,
            depth_hint=depth_hint,
        ),
        tools=tools,
        before_tool_callback=build_scope_guard(context),
    )


def _history_to_contents(history: list[HistoryMessage]) -> list[types.Content]:
    """Turn stored messages into ADK's content list.

    A `summary` row stands in for a stretch of older conversation, so it is
    presented as user-side context rather than as something the agent said.
    """
    contents: list[types.Content] = []
    for message in history:
        if message.role == "assistant":
            role = "model"
            text = message.content
        elif message.role == "summary":
            role = "user"
            text = f"[earlier in this conversation] {message.content}"
        else:
            role = "user"
            text = message.content
        contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    return contents


async def run_turn(
    *,
    context: TurnToolContext,
    history: list[HistoryMessage],
    user_message: str,
    paper_title: str | None,
    session_key: str,
    memory_summary: str | None = None,
    callback_hint: str | None = None,
    depth_hint: str | None = None,
) -> AgentOutcome:
    """Run one turn and return the composed draft.

    The draft is unverified: citation markers in it are the model's claims
    until `citations.verify` matches them against what was retrieved.
    """
    _configure_transport()

    agent = build_agent(
        context,
        paper_title,
        memory_summary=memory_summary,
        callback_hint=callback_hint,
        depth_hint=depth_hint,
    )
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent, app_name=APP_NAME, session_service=session_service
    )

    # Hydrate from PostgreSQL. This session exists only for this call, and is
    # rebuilt from the `messages` table on every turn.
    adk_session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=str(context.user_id),
        session_id=session_key,
    )
    for content in _history_to_contents(history):
        await session_service.append_event(
            adk_session,
            Event(author=content.role, content=content),
        )

    parts: list[str] = []
    input_tokens = output_tokens = None

    for attempt in range(1, MAX_TRANSIENT_ATTEMPTS + 1):
        # Each attempt re-runs the whole loop, so the accumulators it fills
        # have to start empty. Leaving them would renumber citation markers
        # against a retrieval set containing both attempts' passages — the one
        # thing in this system that must never be approximate.
        await _reset_for_retry(context)
        parts = []
        iterations = 0

        try:
            async for event in runner.run_async(
                user_id=str(context.user_id),
                session_id=session_key,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=user_message)]
                ),
            ):
                usage = getattr(event, "usage_metadata", None)
                if usage is not None:
                    input_tokens = getattr(usage, "prompt_token_count", None)
                    output_tokens = getattr(usage, "candidates_token_count", None)

                if event.is_final_response() and event.content and event.content.parts:
                    parts.append(
                        "".join(part.text or "" for part in event.content.parts)
                    )
                else:
                    iterations += 1
                    if iterations > MAX_ITERATIONS * 4:
                        # A runaway loop costs real money and real latency.
                        # Stop and compose from whatever has been gathered.
                        logger.warning("agent loop exceeded its iteration budget")
                        break
            break
        except Exception as exc:
            if attempt < MAX_TRANSIENT_ATTEMPTS and _is_transient(exc):
                delay = TRANSIENT_BACKOFF_SECONDS * attempt
                logger.warning(
                    "agent run hit a transient model error, retrying in %ss "
                    "(attempt %s/%s): %s",
                    delay,
                    attempt,
                    MAX_TRANSIENT_ATTEMPTS,
                    str(exc)[:200],
                )
                await asyncio.sleep(delay)
                continue
            logger.exception("agent run failed")
            raise AgentUnavailable(str(exc)) from exc

    return AgentOutcome(
        text="".join(parts).strip(),
        tools_called=list(context.tools_called),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
