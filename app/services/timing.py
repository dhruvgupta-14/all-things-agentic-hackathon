"""Where a turn's wall-clock actually goes.

HANDOFF 6.3 records 56-70s per turn with the causes listed as a guess:
"multiple tool searches, model latency, and full generation completing before
verification". This exists so that stops being a guess. It is deliberately
permanent rather than a throwaway script — the same breakdown is what you need
in Cloud Logging when a deployed turn is slow, and by then the script is gone.

Cost is one `perf_counter()` per span, so it is left on in production. The
breakdown is logged as structured fields and surfaced on the debug strip; it is
not persisted, because `turns.latency_ms` is the durable metric and a schema
change to store a breakdown would be a migration for a diagnostic.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TurnTimings:
    """Named spans within one turn, in milliseconds."""

    spans: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    _started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Time a block, accumulating if the name repeats.

        Repeats are the interesting case: three `retrieve_paper_context` calls
        show up as one span with a count of three, which is what makes "the
        agent searched three times" legible as a cost rather than a footnote.
        """
        begin = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - begin) * 1000
            self.spans[name] = self.spans.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    def record(self, name: str, milliseconds: float) -> None:
        """Add a span measured elsewhere — inside a tool, say."""
        self.spans[name] = self.spans.get(name, 0.0) + milliseconds
        self.counts[name] = self.counts.get(name, 0) + 1

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000

    def breakdown(self) -> dict[str, dict[str, float | int]]:
        total = self.total_ms
        return {
            name: {
                "ms": round(elapsed, 1),
                "calls": self.counts.get(name, 1),
                "pct": round(100 * elapsed / total, 1) if total else 0.0,
            }
            for name, elapsed in sorted(
                self.spans.items(), key=lambda item: -item[1]
            )
        }

    def log(self, turn_id: str | None = None) -> None:
        breakdown = self.breakdown()
        total = round(self.total_ms, 1)
        # Unaccounted time is the useful number when a breakdown looks fine and
        # the turn is still slow — it is where the next span needs to go.
        measured = sum(item["ms"] for item in breakdown.values())
        logger.info(
            "turn timing total=%.0fms unaccounted=%.0fms %s",
            total,
            total - measured,
            " ".join(
                f"{name}={item['ms']:.0f}ms/{item['calls']}" for name, item in breakdown.items()
            ),
            extra={"turn_id": turn_id, "total_ms": total, "breakdown": breakdown},
        )
