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

It additionally tracks *blocking-state persistence*: for each instance, how
long the same blocking editor condition (compiling, domain-reload pending,
play-mode transition) has been continuously observed by gate polls. A record
re-anchors to "now" when continuity lapses (no observation within
``BLOCKING_CONTINUITY_WINDOW_SECONDS`` — the state may have cleared and
recurred unseen) and is dropped the moment a poll sees the reason inactive.
The gate uses these ages, alongside the bridge-reported state-start
timestamps, to detect phantom transitions that never end.

Everything fails open: a fetch error behaves as "no snapshot", and callers
must treat a ``None`` snapshot as permission to proceed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
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

# Lifetime of an optimistic "tests pending" marker pinned when a test run is
# dispatched, bridging the window before the bridge snapshot publishes
# ``tests.is_running``. Cleared early once a snapshot confirms the run; the
# TTL is only the backstop for a dispatch whose run never materializes.
TESTS_PENDING_TTL_SECONDS = 15.0

# Maximum gap between observations of the same blocking reason for the
# persistence record to stay continuous. Parked calls poll on a sub-second
# cadence and busy-retry loops re-poll within ~1s, so anything beyond this
# means nobody was watching — the state may have cleared and recurred unseen,
# and the record re-anchors rather than over-counting.
BLOCKING_CONTINUITY_WINDOW_SECONDS = 10.0


@dataclass
class _CacheEntry:
    state: dict[str, Any]
    fetched_at: float  # time.monotonic()


@dataclass
class _ExclusiveEdge:
    owner: str
    kind: str
    recorded_at: float  # time.monotonic()


@dataclass
class _BlockingObservation:
    first_observed: float  # time.monotonic()
    last_observed: float  # time.monotonic()


@dataclass
class TestsPending:
    owner: str | None
    recorded_at: float  # time.monotonic()


def _instance_key(unity_instance: str | None) -> str:
    return unity_instance or "default"


class EditorStateCache:
    """Process-local shared snapshot of the editor state, keyed by instance."""

    def __init__(self, ttl_s: float = DEFAULT_TTL_SECONDS):
        self._ttl_s = float(ttl_s)
        self._entries: dict[str, _CacheEntry] = {}
        self._edges: dict[str, _ExclusiveEdge] = {}
        self._blocking: dict[str, dict[str, _BlockingObservation]] = {}
        self._tests_pending: dict[str, TestsPending] = {}
        # Per-instance edge-event signals (MCPC-030). A pushed editor-edge
        # event sets the instance's asyncio.Event so any park loop awaiting it
        # wakes immediately instead of waiting out its bounded poll. The signal
        # is purely a "re-check now" nudge — the poll remains the source of
        # truth — so it can be coalesced and auto-reset without losing
        # correctness: a missed signal only costs the poll's normal latency.
        self._event_signals: dict[str, asyncio.Event] = {}

    def reset(self, ttl_s: float | None = None) -> None:
        """Clear all cached state (test seam)."""
        self._entries.clear()
        self._edges.clear()
        self._blocking.clear()
        self._tests_pending.clear()
        self._event_signals.clear()
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

    # ------------------------------------------------------------------
    # Tests-pending marker (dispatch -> snapshot race bridging)
    # ------------------------------------------------------------------
    def mark_tests_pending(
        self,
        unity_instance: str | None,
        owner: str | None = None,
    ) -> None:
        """Pin an optimistic "a test run was just dispatched" marker.

        Compile-risk gate checks treat the marker as ``running_tests`` until
        a snapshot confirms the run (which clears it) or the marker's TTL
        elapses — closing the 1-2s window between the run_tests dispatch and
        ``tests.is_running`` becoming visible in the shared snapshot.
        """
        self._tests_pending[_instance_key(unity_instance)] = TestsPending(
            owner=owner, recorded_at=time.monotonic()
        )

    def clear_tests_pending(self, unity_instance: str | None) -> None:
        self._tests_pending.pop(_instance_key(unity_instance), None)

    def tests_pending(self, unity_instance: str | None) -> TestsPending | None:
        """The unexpired tests-pending marker for an instance, if any."""
        key = _instance_key(unity_instance)
        pending = self._tests_pending.get(key)
        if pending is None:
            return None
        if (time.monotonic() - pending.recorded_at) > TESTS_PENDING_TTL_SECONDS:
            self._tests_pending.pop(key, None)
            return None
        return pending

    # ------------------------------------------------------------------
    # Blocking-state persistence (phantom-transition detection support)
    # ------------------------------------------------------------------
    def observe_blocking_states(
        self,
        unity_instance: str | None,
        active_reasons: Iterable[str],
    ) -> dict[str, float]:
        """Update persistence records from one gate poll; return ages.

        ``active_reasons`` is the set of blocking reasons the current snapshot
        shows for the instance. Active reasons accumulate continuously
        observed age (seconds since first observation, re-anchored when the
        continuity window lapses); reasons absent from the set are forgotten,
        so a state that clears and later recurs starts a fresh record.
        """
        key = _instance_key(unity_instance)
        now = time.monotonic()
        active = set(active_reasons)
        records = self._blocking.get(key)
        if records is None:
            if not active:
                return {}
            records = {}
            self._blocking[key] = records

        for reason in list(records):
            if reason not in active:
                del records[reason]

        ages: dict[str, float] = {}
        for reason in active:
            observation = records.get(reason)
            if (
                observation is None
                or (now - observation.last_observed) > BLOCKING_CONTINUITY_WINDOW_SECONDS
            ):
                observation = _BlockingObservation(
                    first_observed=now, last_observed=now
                )
                records[reason] = observation
            else:
                observation.last_observed = now
            ages[reason] = now - observation.first_observed

        if not records:
            self._blocking.pop(key, None)
        return ages

    # ------------------------------------------------------------------
    # Edge-event signalling (MCPC-030)
    # ------------------------------------------------------------------
    def _signal_for(self, key: str) -> asyncio.Event:
        signal = self._event_signals.get(key)
        if signal is None:
            signal = asyncio.Event()
            self._event_signals[key] = signal
        return signal

    def apply_edge_event(
        self,
        unity_instance: str | None,
        event_name: str,
    ) -> None:
        """Record a bridge-pushed editor-edge event and wake parked waiters.

        Two effects, both best-effort:

        1. Optimistically *relax* the cached snapshot's blocking flags for
           edges that mean a blocking state has ended (compile finished,
           domain reload done, play transition complete). It never *sets* a
           blocking flag — a lost or stale event must never manufacture a
           wedge — so the worst case for a spurious clear is one premature
           re-poll, which immediately re-reads the true state.
        2. Set the instance's edge signal so any park loop awaiting it wakes
           now and re-polls, rather than waiting out its bounded sleep.

        Never raises.
        """
        key = _instance_key(unity_instance)
        try:
            self._relax_blocking_flags(key, event_name)
        except Exception as exc:
            logger.debug(
                "editor_state_cache: edge-event flag relax skipped: %r", exc
            )
        # Coalesce: setting an already-set Event is a no-op; waiters that are
        # not yet awaiting still observe the set flag on their next wait, and
        # the park loop clears it on consumption so it cannot permanently
        # latch (see consume_edge_signal).
        self._signal_for(key).set()

    def _relax_blocking_flags(self, key: str, event_name: str) -> None:
        """Clear cached blocking flags an edge implies are over (never sets)."""
        entry = self._entries.get(key)
        if entry is None:
            return
        state = entry.state
        if not isinstance(state, dict):
            return

        if event_name in ("compile_finished", "domain_reload_done"):
            compilation = state.get("compilation")
            if isinstance(compilation, dict):
                if compilation.get("is_compiling") is True:
                    compilation["is_compiling"] = False
                if compilation.get("is_domain_reload_pending") is True:
                    compilation["is_domain_reload_pending"] = False
        elif event_name in ("entered_play", "exited_play"):
            play_mode = (state.get("editor") or {}).get("play_mode")
            if isinstance(play_mode, dict) and play_mode.get("is_changing") is True:
                play_mode["is_changing"] = False

    async def wait_for_edge_event(
        self,
        unity_instance: str | None,
        timeout_s: float,
    ) -> bool:
        """Wait up to ``timeout_s`` for an edge event, racing the bounded sleep.

        Returns True if an edge event fired within the window, False on
        timeout. Either way the signal is auto-reset on return so the next
        wait starts clean and a single push can't permanently latch the loop
        awake. Event loss is harmless: the caller's poll cadence is the
        backstop, so on a False return it simply re-polls as before.
        """
        key = _instance_key(unity_instance)
        signal = self._signal_for(key)
        try:
            await asyncio.wait_for(signal.wait(), timeout=max(0.0, timeout_s))
            fired = True
        except asyncio.TimeoutError:
            fired = False
        except Exception as exc:
            logger.debug(
                "editor_state_cache: edge-event wait failed (fail-open): %r", exc
            )
            fired = False
        # Auto-reset so the signal can't leak or permanently latch.
        signal.clear()
        return fired


# Global singleton (simple, process-local) — same pattern as
# external_changes_scanner.
editor_state_cache = EditorStateCache()
