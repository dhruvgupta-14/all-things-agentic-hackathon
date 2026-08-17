"""The SSE contract for a turn.

One place defines the wire format, and both the pipeline and the SPA are built
against it. Event names come from ARCHITECTURE 15; the payload shapes are
defined here.

Order for a normal turn:

    state* -> token* -> citations -> memory_used -> done

`state` events are emitted during the deterministic phases, before any token
exists. That matters: citations are verified *before* streaming begins
(ARCHITECTURE 9.2 steps 9 and 12), so the first token is several seconds out,
and a pane that sits blank until then looks broken. The phase events also make
the agent's tool choices visible, which is the point of the debug strip.

`error` may replace the tail of any stream. `done` is always last unless the
connection drops.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

# Phases a turn passes through, in order. Named for a reader, not for the code.
TurnPhase = Literal[
    "started",
    "retrieving",
    "consulting_memory",
    "composing",
    "verifying",
    "persisted",
]


class SSEEvent(BaseModel):
    """Base class carrying the wire encoding."""

    event: str

    def encode(self) -> str:
        """Serialise to the `event:`/`data:` framing SSE requires."""
        body = self.model_dump(mode="json", exclude={"event"})
        return f"event: {self.event}\ndata: {json.dumps(body)}\n\n"


class StateEvent(SSEEvent):
    """Progress. Emitted before tokens, and once more when the turn lands."""

    event: Literal["state"] = "state"
    phase: TurnPhase
    activity: str
    # What the agent has chosen to call so far — the visible trace of agency.
    tools_called: list[str] = Field(default_factory=list)


class TokenEvent(SSEEvent):
    """A slice of the verified answer.

    The text has already passed citation verification, so a marker that
    appears here will resolve. Nothing streamed is ever retracted.
    """

    event: Literal["token"] = "token"
    text: str


class CitationPayload(BaseModel):
    """Everything the overlay needs to open the source without another call."""

    marker: str
    chunk_id: str
    paper_id: str
    section_path: str
    page_start: int
    page_end: int
    similarity: float


class CitationsEvent(SSEEvent):
    event: Literal["citations"] = "citations"
    citations: list[CitationPayload] = Field(default_factory=list)


class MemoryRecordPayload(BaseModel):
    concept_id: str
    name: str
    understanding_score: float | None = None
    score_confidence: float | None = None
    effective_style: str | None = None


class MemoryUsedEvent(SSEEvent):
    """Which learner memory informed this turn. Empty is a valid answer."""

    event: Literal["memory_used"] = "memory_used"
    memory: list[MemoryRecordPayload] = Field(default_factory=list)


class DoneEvent(SSEEvent):
    event: Literal["done"] = "done"
    turn_id: str
    grounding_status: str
    latency_ms: int


class ErrorEvent(SSEEvent):
    """A typed code, never a stack trace (ARCHITECTURE 4.3, 4.7)."""

    event: Literal["error"] = "error"
    code: str
    message: str


def encode(event: SSEEvent) -> str:
    return event.encode()


def decode(raw: str) -> list[dict[str, Any]]:
    """Parse an SSE stream back into events. For tests and the debug strip."""
    events: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        name: str | None = None
        data: str | None = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        if name and data is not None:
            events.append({"event": name, **json.loads(data)})
    return events
