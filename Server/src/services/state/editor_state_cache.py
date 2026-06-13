"""Shared editor-state cache for gate decisions (one source, no per-call fan-out).

Every gate decision reads this cache instead of polling Unity directly: the
snapshot is fetched through the canonical ``editor_state`` resource at most
once per TTL per instance, so any number of concurrently parked calls share a
single poll stream.

The cache also holds short-lived *exclusive-edge ownership* records: when a
session triggers an exclusive editor transition (compile, play enter) — either
through a declared exclusive tool or detected post-hoc after arbitrary-code
execution — the session's display name is recorded here so later parked calls
can attribute the busy state to its owner.

Everything fails open: a fetch error behaves as "no snapshot", and callers
must treat a ``None`` snapshot as permission to proceed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Snapshot freshness window. Parked calls re-poll on this cadence; faster
# pollers coalesce onto the cached snapshot.
DEFAULT_TTL_SECONDS = 0.5

# A stale snapshot younger than this still answers when a refresh fails
# (e.g. the bridge briefly not responding); older misses return None.
STALE_GRACE_SECONDS = 5.0

# Upper bound on a single editor-state fetch. A slow fetch (e.g. waiting out a
# domain-reload session resolve) must not eat a parked call's entire budget —
# timing out is a cache miss, and cache misses fail open.
FETCH_TIMEOUT_SECONDS = 5.0

# Exclusive-edge ownership records expire on their own; there is no cleanup.
EDGE_TTL_SECONDS = 30.0


@dataclass
class _CacheEntry:
    state: dict[str, Any]
    fetched_at: float  # time.monotonic()


@dataclass
class _ExclusiveEdge:
    owner: str
    kind: str
    recorded_at: float  # time.monotonic()


def _instance_key(unity_instance: str | None) -> str:
    return unity_instance or "default"


class EditorStateCache:
    """Process-local shared snapshot of the editor state, keyed by instance."""

    def __init__(self, ttl_s: float = DEFAULT_TTL_SECONDS):
        self._ttl_s = float(ttl_s)
        self._entries: dict[str, _CacheEntry] = {}
        self._edges: dict[str, _ExclusiveEdge] = {}

    def reset(self, ttl_s: float | None = None) -> None:
        """Clear all cached state (test seam)."""
        self._entries.clear()
        self._edges.clear()
        if ttl_s is not None:
            self._ttl_s = float(ttl_s)

    async def get(
        self,
        ctx,
        unity_instance: str | None,
        max_age_s: float | None = None,
    ) -> dict[str, Any] | None:
        """Return the shared editor-state snapshot for an instance.

        Refreshes through the ``editor_state`` resource when the cached
        snapshot is older than the TTL (or ``max_age_s`` when given; pass 0.0
        to force a refresh). Returns ``None`` on a miss — callers fail open.
        """
        key = _instance_key(unity_instance)
        ttl = self._ttl_s if max_age_s is None else float(max_age_s)
        entry = self._entries.get(key)
        now = time.monotonic()
        if entry is not None and (now - entry.fetched_at) <= ttl:
            return entry.state

        state = await self._fetch(ctx)
        if state is None:
            # Refresh failed: a recent-enough stale snapshot still answers.
            if entry is not None and (now - entry.fetched_at) <= STALE_GRACE_SECONDS:
                return entry.state
            return None

        self._entries[key] = _CacheEntry(state=state, fetched_at=time.monotonic())
        await self._notify_play_lease(unity_instance, state)
        return state

    async def _notify_play_lease(
        self, unity_instance: str | None, state: dict[str, Any]
    ) -> None:
        """Feed fresh snapshots to the play lease (acquire on human play-enter,
        renew while playing, clear on play exit). Never raises."""
        try:
            from services.state.play_lease import play_lease_manager

            await play_lease_manager.observe_editor_state(unity_instance, state)
        except Exception as exc:
            logger.debug(
                "editor_state_cache: play-lease observation skipped: %r", exc
            )

    async def _fetch(self, ctx) -> dict[str, Any] | None:
        """One bounded editor-state fetch. Any failure is a miss (fail open)."""
        try:
            # Lazy import: editor_state pulls in transport modules.
            from services.resources.editor_state import get_editor_state

            response = await asyncio.wait_for(
                get_editor_state(ctx), timeout=FETCH_TIMEOUT_SECONDS
            )
            if hasattr(response, "model_dump"):
                response = response.model_dump()
            if not isinstance(response, dict) or not response.get("success", False):
                return None
            data = response.get("data")
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.debug("editor_state_cache: fetch failed (fail-open): %r", exc)
            return None

    # ------------------------------------------------------------------
    # Exclusive-edge ownership (compile / play-transition attribution)
    # ------------------------------------------------------------------
    def record_exclusive_edge(
        self,
        unity_instance: str | None,
        owner: str,
        kind: str = "compile",
    ) -> None:
        """Record which session owns the current exclusive editor transition."""
        if not owner:
            return
        self._edges[_instance_key(unity_instance)] = _ExclusiveEdge(
            owner=owner, kind=kind, recorded_at=time.monotonic()
        )

    def get_exclusive_edge_owner(self, unity_instance: str | None) -> str | None:
        """Display name of the unexpired exclusive-edge owner, if any."""
        key = _instance_key(unity_instance)
        edge = self._edges.get(key)
        if edge is None:
            return None
        if (time.monotonic() - edge.recorded_at) > EDGE_TTL_SECONDS:
            self._edges.pop(key, None)
            return None
        return edge.owner


# Global singleton (simple, process-local) — same pattern as
# external_changes_scanner.
editor_state_cache = EditorStateCache()
