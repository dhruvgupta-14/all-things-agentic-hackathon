"""The agent's tools, and the boundary they cannot reach across.

`retrieve_paper_context` (ARCHITECTURE 14.2) is the only tool in this slice.
Its contract splits cleanly:

    the agent decides   what to search for, which section role, how many
    the backend decides whose data, which papers, the relevance floor, dedup,
                        and the post-retrieval assertion

That split is enforced structurally rather than by instruction. The tool is a
closure over a per-turn context, so `user_id` and `paper_scope` are **not
parameters** — there is no argument through which the model could name another
reader's papers, successful prompt injection or not (ARCHITECTURE 13.1).

`before_tool_callback` is the audited checkpoint on top of that: it records
every call and refuses any attempt to smuggle scope in through arguments.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SECTION_ROLE
from app.services.retrieval import RetrievalService, RetrievedChunk

logger = logging.getLogger(__name__)

# Arguments a tool may never accept from the model. Present as a tripwire: if
# one ever appears, the tool signature has drifted into something unsafe.
FORBIDDEN_TOOL_ARGS = frozenset(
    {"user_id", "paper_id", "paper_scope", "session_id", "turn_id", "sql", "path"}
)

MAX_QUERY_CHARS = 500
MAX_TOP_K = 10


class ToolScopeViolation(Exception):
    """The model tried to supply scope. Fails the turn closed."""


@dataclass
class TurnToolContext:
    """Everything the tools need, none of it reachable by the model."""

    session: AsyncSession
    user_id: uuid.UUID
    paper_scope: list[uuid.UUID]
    retrieval: RetrievalService
    # Filled as the agent works; the pipeline reads these afterwards.
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)

    def marker_for(self, chunk_id: uuid.UUID) -> str | None:
        for index, chunk in enumerate(self.retrieved, start=1):
            if chunk.chunk_id == chunk_id:
                return f"[{index}]"
        return None


def build_retrieve_paper_context(context: TurnToolContext):
    """Create the tool bound to one turn's authorization scope."""

    async def retrieve_paper_context(
        query: str,
        section_role: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Search the reader's open paper for passages relevant to a query.

        Args:
            query: What to look for, in natural language. 1-500 characters.
            section_role: Optionally restrict to one part of the paper, such as
                'method' or 'results'. Omit to search the whole paper.
            top_k: How many passages to return, 1-10.

        Returns:
            Passages, each with the citation marker to use when referring to it.
        """
        context.tools_called.append("retrieve_paper_context")

        cleaned = (query or "").strip()[:MAX_QUERY_CHARS]
        if not cleaned:
            return {"passages": [], "note": "Empty query; nothing was searched."}

        role = section_role if section_role in SECTION_ROLE else None
        limit = max(1, min(int(top_k or 5), MAX_TOP_K))

        results = await context.retrieval.retrieve(
            cleaned,
            paper_scope=context.paper_scope,
            top_k=limit,
            section_role=role,
        )
        context.queries.append(cleaned)

        # Markers are positional over everything retrieved this turn, so a
        # second search continues the numbering rather than restarting it and
        # silently repointing [1] at different text.
        passages = []
        for chunk in results:
            if not any(seen.chunk_id == chunk.chunk_id for seen in context.retrieved):
                context.retrieved.append(chunk)
            passages.append(
                {
                    "marker": context.marker_for(chunk.chunk_id),
                    "text": chunk.content,
                    "section": chunk.section_path,
                    "section_role": chunk.section_role,
                    "pages": chunk.citation_locator,
                }
            )

        if not passages:
            return {
                "passages": [],
                "note": (
                    "Nothing in this paper matched. Tell the reader the paper "
                    "does not appear to cover this rather than answering from "
                    "general knowledge."
                ),
            }
        return {"passages": passages}

    return retrieve_paper_context


def build_scope_guard(context: TurnToolContext):
    """The `before_tool_callback` audit checkpoint (ARCHITECTURE 13.1).

    Scope is already unreachable — it is closed over, not passed — so this
    exists to record what the agent chose and to fail loudly if a tool ever
    grows an argument that could carry authorization.
    """

    def before_tool(tool, args: dict[str, Any], tool_context) -> dict | None:
        smuggled = FORBIDDEN_TOOL_ARGS.intersection(args or {})
        if smuggled:
            logger.error(
                "SECURITY: model supplied scope-bearing tool arguments",
                extra={"tool": getattr(tool, "name", "?"), "args": sorted(smuggled)},
            )
            raise ToolScopeViolation(
                f"{sorted(smuggled)} may not be supplied by the model"
            )

        logger.info(
            "tool call",
            extra={
                "tool": getattr(tool, "name", "?"),
                "user_id": str(context.user_id),
                "papers_in_scope": len(context.paper_scope),
            },
        )
        # None means "proceed with the real tool".
        return None

    return before_tool
